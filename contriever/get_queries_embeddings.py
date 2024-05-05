from src.contriever import Contriever
from transformers import AutoTokenizer
import json
import pandas as pd
from tqdm import tqdm
import os
import pickle
import sys

# Load Contriever model and tokenizer
model = Contriever.from_pretrained("facebook/contriever")
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")

batch_size = 10

# Function to split list into batches
def chunks(lst, batch_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

# Check command line argument for dataset type
if len(sys.argv) != 2:
    print("Usage: python combined_script.py [nq | squad]")
    sys.exit(1)

arg = sys.argv[1]
if arg not in ["nq", "squad"]:
    print("Invalid argument. Use 'nq' or 'squad'.")
    sys.exit(1)

# Define splits based on dataset type
if arg == "nq":
    splits = ["test", "train"]
elif arg == "squad":
    splits = ["dev", "train"]

for split in splits:
    # Construct file pattern
    pattern = f"biencoder-{arg}-{split}_predictions.json"
    extract_questions_path = os.path.join("../lit-llama/data/alpaca", pattern)

    # Load dataset from JSON file
    with open(extract_questions_path, "r") as json_file:
        dataset = json.load(json_file)

    list_dataset = [[key, content] for key, content in dataset.items()]

    question_document_pairs = {}
    for c in tqdm(chunks(list_dataset, batch_size), total=len(dataset)//batch_size):
        questions = [content['question'] for _, content in c]
        
        # Tokenize questions
        inputs = tokenizer(questions, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: inputs[key].to(model.device) for key in inputs}

        # Get embeddings from Contriever model
        embeddings = model(**inputs)

        _vector = embeddings.detach().cpu().numpy()

        # Populate question-document pairs
        for k in range(_vector.shape[0]):
            question_document_pairs[c[k][0]] = {
                "question": c[k][1]["question"],
                "question_embedding": _vector[k],
                "possible_answers": c[k][1]["possible_answers"],
                "gold_passage": c[k][1]["passage_text"],
                "contriever_passage": None,
                "contriver_score": None,
            }

    # Save question-document pairs to a pickle file
    with open(f"new_dataset/{os.path.basename(extract_questions_path)[:-5]}_pickled.obj", "wb") as f:
        pickle.dump(question_document_pairs, f)
