import torch
import sys
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from utils.prompter import Prompter
from peft import PeftModel

# Define paths and model configurations
train_data_path = "../generate_dataset/alpaca_format/squad_train_hybrid_retrieve.json"
base_model = "meta-llama/Llama-2-7b-chat-hf"
lora_weights = "../alpaca-lora/results/squad/hybrid_retrieve"

# Check for GPU availability
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Class for Language Model Head
class LMHeadModel:
    global device

    def __init__(self, model_name, weights=None, load_in_8bit=True):
        # Initialize the model and the tokenizer.
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model = model
        if weights is not None:
            self.model = PeftModel.from_pretrained(
                model,
                weights,
                torch_dtype=torch.float16,
            )
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.last_output = ""
        self.prompter = Prompter("")
        # Set up model configurations
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)  # Enable Torchscript compilation

    def generate_prompt(self, example):
        # Generate a prompt based on the input example
        instruction = example["instruction"]
        input_text = example["input"]
        return self.prompter.generate_prompt(instruction, input_text)

    def get_predictions(self, sentence):
        # Generate model predictions for the given sentence
        inputs = self.tokenizer(sentence, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=0.8,
            top_k=200,
        )
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids.to("cuda"),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=20,
            )
        self.last_output = outputs

    def get_first_token_probability(self, id):
        # Get the probability of the first token
        predictions = self.transform(self.last_output.scores)
        first_token_probability = torch.nn.functional.softmax(
            predictions[0, 0, :], dim=-1)[id].item()
        return first_token_probability

    def transform(self, input_tensor):
        # Concatenate and reshape the input tensor
        if not isinstance(input_tensor, tuple) or not all(isinstance(t, torch.Tensor) for t in input_tensor):
            raise ValueError("Invalid input. Ensure the input is a tuple of PyTorch tensors.")
        concatenated_tensor = torch.cat(input_tensor, dim=0)
        reshaped_tensor = concatenated_tensor.view(1, len(input_tensor), -1)
        return reshaped_tensor

    def get_output(self):
        # Decode and get the output text
        s = self.last_output.sequences[0]
        output_decoded = self.tokenizer.decode(s)
        output = self.prompter.get_response(output_decoded)
        return output

    def get_last_word_probabilities(self, top_k=500):
        # Get the probabilities of the last word candidates
        predictions = self.transform(self.last_output.scores)
        next_token_candidates_tensor = predictions[0, -1, :]
        topk_candidates_indexes = torch.topk(
            next_token_candidates_tensor, top_k).indices.tolist()
        all_candidates_probabilities = torch.nn.functional.softmax(
            next_token_candidates_tensor, dim=-1)
        topk_candidates_probabilities = \
            all_candidates_probabilities[topk_candidates_indexes].tolist()
        topk_candidates_tokens = \
            [self.tokenizer.decode([idx]).strip() for idx in topk_candidates_indexes]
        return list(zip(topk_candidates_tokens, topk_candidates_indexes, topk_candidates_probabilities))

    def get_first_word_probabilities(self, top_k=500):
        # Get the probabilities of the first word candidates
        predictions = self.transform(self.last_output.scores)
        first_word_candidates_tensor = predictions[0, 0, :]
        topk_candidates_indexes = torch.topk(
            first_word_candidates_tensor, top_k).indices.tolist()
        all_candidates_probabilities = torch.nn.functional.softmax(
            first_word_candidates_tensor, dim=-1)
        topk_candidates_probabilities = \
            all_candidates_probabilities[topk_candidates_indexes].tolist()
        topk_candidates_words = \
            [self.tokenizer.decode([idx]).strip() for idx in topk_candidates_indexes]
        return list(zip(topk_candidates_words, topk_candidates_indexes, topk_candidates_probabilities))
