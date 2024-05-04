from src.contriever import Contriever
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import os
import faiss
import torch
from faiss import write_index, read_index

model = Contriever.from_pretrained("facebook/contriever").cuda()
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")


# Cambia el manejo de lectura del archivo CSV
with open('psgs_w100_2.tsv', 'r', encoding='utf-8') as f:
    # Divide cada línea en tabuladores
    lines = f.read().splitlines()
    lines = [line.split('\t') for line in lines]
    df = pd.DataFrame(lines, columns=['id', 'title', 'text'])

wikipedia = list(df['text'])

print("wikpedia length:", len(wikipedia))

# Create Faiss index
vector_dimension = 768
index = faiss.IndexFlatL2(vector_dimension)

model = model.to('cuda:0')

res = faiss.StandardGpuResources()
index = faiss.index_factory(vector_dimension, "Flat", faiss.METRIC_L2)
index = faiss.index_cpu_to_gpu(res, 0, index)  # Mueve el índice a la GPU 0

batch_size = 64

def chunks(lst, batch_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

# get Wikipedia embeddings
for text_batch in tqdm(chunks(wikipedia, batch_size), total=len(wikipedia) // batch_size):
    inputs = tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt", max_length=512, truncation_strategy='longest_first')
    inputs = {key: value.to('cuda:0') for key, value in inputs.items()}  # Mueve los tensores de entrada a GPU
    inputs['input_ids'] = inputs['input_ids'].cuda()
    inputs['attention_mask'] = inputs['attention_mask'].cuda()

    with torch.no_grad():
        embeddings = model(**inputs).detach().cpu().numpy()

    
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

write_index(index, "wikipedia.index")

print(index.ntotal)
