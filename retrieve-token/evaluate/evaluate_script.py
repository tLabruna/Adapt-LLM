import os
import argparse
import torch
import sys
import json
import ast
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from utils.prompter import Prompter
from peft import PeftModel
from tqdm import tqdm
from datetime import datetime
from gen_probability import LMHeadModel

# Check for GPU availability
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Check for MPS availability
try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

# Function to generate an example based on mode
def get_example(content, passages_from, mode, k, use_ret_token):
    if mode == "always_retrieve":
        example = {
            "instruction": "answer the question Q given the context C",
            "input": f"Q: {content['question']}\nC: {content[passages_from]}",
        }
    elif mode == "never_retrieve":
        example = {
            "instruction": "answer the question Q",
            "input": f"Q: {content['question']}",
        }
    elif mode == "hybrid_retrieve":
        if k == 0:
            example = {
                "instruction": "answer the question Q. If you need help answer <RET> to get the context",
                "input": f"Q: {content['question']}"
            }
        elif k == 1:
            if use_ret_token:
                example = {
                    "instruction": "answer the question Q given the context C",
                    "input": f"Q: {content['question']}\nC: {content[args.passages_from]}",
                }
            else:
                    example = {
                    "instruction": "answer the question Q",
                    "input": f"Q: {content['question']}",
                }
    return example

# Function to determine if context is needed
def context_needed(threshold, from_score, content, output, model):
    statistics = {}
    use_ret_token = False
    if isinstance(threshold, float):
        if from_score:
            score = content["s_pop"]
            if score < args.threshold:
                if "<RET>" not in output:
                    statistics["non_ret_in_ret"] = 1
                statistics["ret"] = 1
                use_ret_token = True
                return True, statistics, use_ret_token
            else:
                statistics["non_ret"] = 1
                if "<RET>" in output:
                    statistics["ret_in_non_ret"] = 1
                    return True, statistics, use_ret_token
        else:
            probability_of_RET = model.get_first_token_probability(29966)
            if "<RET>" in output:
                statistics["ret_llama"] = 1
                statistics["tot_probability_ret"] = probability_of_RET
            else:
                statistics["non_ret_llama"] = 1
                statistics["tot_probability_non_ret"] = probability_of_RET
            if probability_of_RET > args.threshold:
                if "<RET>" not in output:
                    statistics["non_ret_in_ret"] = 1
                statistics["ret"] = 1
                use_ret_token = True
                return True, statistics, use_ret_token
            else:
                statistics["non_ret"] = 1
                if "<RET>" in output:
                    statistics["ret_in_non_ret"] = 1
                    return True, statistics, use_ret_token
    else:
        if "<RET>" in output:
            statistics["ret"] = 1
            use_ret_token = True
            return True, statistics, use_ret_token
        statistics["non_ret"] = 1
    return False, statistics, use_ret_token

# Function to check correctness of prediction
def correct_prediction(content, dataset, output, use_ret_token):
    statistics = {}
    is_correct = False
    possible_answers = content['possible_answers']

    if not isinstance(possible_answers, list):
        possible_answers = [possible_answers]

    if dataset == "squad":
        possible_answers = ast.literal_eval(possible_answers[0])
    for pa in possible_answers:
        if pa.lower() in output.lower():
            is_correct = True
    if is_correct:
        if use_ret_token:
            statistics["ret_correct"] = 1
        else:
            statistics["non_ret_correct"] = 1

    statistics["total_correct"] = 1 if is_correct else 0

    return is_correct, statistics

# Function to retrieve instance information
def retrieve_instance(content, output, is_correct, use_ret_token, passages_from):
    return {
        "question": str(content['question']),
        "possible_answers": str(content['possible_answers']),
        "prediction": str(output),
        "is_correct": bool(is_correct),
        "use_ret_token": use_ret_token,
        "passage_retrieve": content[passages_from],
        "passage_from": passages_from
    }

# Main evaluation function
def evaluate(args, load_8bit=True):
    # Initialize the language model
    base_model = args.base_model or os.environ.get("BASE_MODEL", "")
    assert base_model, "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    if args.from_score:
        model = LMHeadModel(base_model, load_in_8bit=load_8bit)
    else:
        model = LMHeadModel(base_model, weights=args.lora, load_in_8bit=load_8bit)

    # Load the dataset
    with open(args.data_path, "r") as json_file:
        dataset = json.load(json_file)

    retrieve_dataset = {}
    n_steps = {
        "always_retrieve": 1,
        "never_retrieve": 1,
        "hybrid_retrieve": 2
    }
    statistics = {
        "hh": 0,  # number of questions
        "ret": 0,
        "non_ret": 0,
        "ret_correct": 0,
        "non_ret_correct": 0,
        "ret_in_non_ret": 0,
        "non_ret_in_ret": 0,
        "tot_probability_non_ret": 0,
        "non_ret_llama": 0,
        "tot_probability_ret": 0,
        "ret_llama": 0,
        "total_correct": 0
    }

    # Iterate over each instance in the dataset
    for key_id, content in tqdm(dataset.items(), total=len(dataset), file=sys.stdout):
        if statistics["hh"] % 100 == 0 and statistics["hh"] != 0:
            print(f"{statistics['hh']}/{len(dataset)}. Time: {datetime.now()}. Acc: {statistics['total_correct'] / (statistics['hh'])}. RET in non-RET: {statistics['ret_in_non_ret']}. Non-RET in RET: {statistics['non_ret_in_ret']}.")
            if statistics["ret"] > 0:
                print(f"Ret: {statistics['ret']}. Ret Acc: {statistics['ret_correct'] / (statistics['ret'])}.")
            if statistics["non_ret"] > 0:
                print(f"Non Ret: {statistics['non_ret']}. Non Ret Acc: {statistics['non_ret_correct'] / (statistics['non_ret']+1)}.")
            if statistics["non_ret_llama"] > 0:
                print(f"Avg Probability Non-RET: {statistics['tot_probability_non_ret'] / statistics['non_ret_llama']}.")
            if statistics["ret_llama"] > 0:
                print(f"Avg Probability RET: {statistics['tot_probability_ret'] / statistics['ret_llama']}.")
            print(statistics)

        use_ret_token = False

        # Iterate over hybrid mode steps
        for k in range(n_steps[args.mode]):

            # Create example
            example = get_example(content, args.passages_from, args.mode, k, use_ret_token)

            # Create prompt
            prompt = model.generate_prompt(example)

            # Compute the prediction
            model.get_predictions(prompt)

            # Get the prediction output
            output = model.get_output()

            # If Hybrid, decide if we need the context
            if args.mode == "hybrid_retrieve" and k == 0:
                need_context, statistics_temp, use_ret_token = context_needed(args.threshold, args.from_score, content, output, model)
                for stat in statistics_temp:
                    statistics[stat] += statistics_temp[stat]

                # If we need the context, we go to the next cycle
                if need_context:
                    continue

            # Evaluate if the prediction output is correct
            is_correct, statistics_temp = correct_prediction(content, args.dataset, output, use_ret_token)

            for stat in statistics_temp:
                statistics[stat] += statistics_temp[stat]

            retrieve_dataset[int(key_id)] = retrieve_instance(content, output, is_correct, use_ret_token, args.passages_from)

            break

        statistics["hh"] += 1

    if args.mode == "hybrid_retrieve":
        if statistics["ret"] > 0:
            print(f"Ret: {statistics['ret']}. Ret Correct: {statistics['ret_correct']}. Acc: {statistics['ret_correct'] / statistics['ret']}.")
        if statistics["non_ret"] > 0:
            print(f"Non Ret: {statistics['non_ret']}. Non Ret Correct: {statistics['non_ret_correct']}. Acc: {statistics['non_ret_correct'] / statistics['non_ret']}.")
        print(f"Number of times that returns RET for a non RET threshold: {statistics['ret_in_non_ret']}")

    # Save the results
    with open(f"{args.out_dir}/{os.path.basename(args.data_path)[:-4]}_passages_from_{args.passages_from}_predictions.json", "w") as json_file:
        json.dump(retrieve_dataset, json_file)

    with open(f"{args.out_dir}/{os.path.basename(args.data_path)[:-4]}_passages_from_{args.passages_from}_results.txt", "w") as f:
        f.write(f"Accuracy: {statistics['total_correct'] / len(dataset)}")

    print(statistics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--lora", type=str)
    parser.add_argument("--lora_always", type=str)
    parser.add_argument("--lora_never", type=str)
    parser.add_argument("--hybrid_flex", type=str)
    parser.add_argument("--passages_from", type=str)
    parser.add_argument("--threshold", type=float, nargs='?')
    parser.add_argument("--from_score", type=bool, nargs='?')

    args, _ = parser.parse_known_args()

    # Define dataset splits
    dataset_splits = {"nq": ['dev'], 'popqa': ["test"], 'popqa_test': ["test"], "squad": ["dev"]}

    tmp = args.data_path
    for split in dataset_splits[args.dataset]:
        args.data_path = f"{tmp}/{args.dataset}_{split}.json"
        evaluate(args, split)
