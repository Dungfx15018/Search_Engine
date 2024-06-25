import torch
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset

import os

# Set environment variable to allow duplicate OpenMP runtime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

model = AutoModel.from_pretrained('vinai/phobert-base')
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')



def get_embedding(item):
    tokens = tokenizer(item['text'], return_tensors='pt', truncation=True, padding='max_length', max_length=128)

    with torch.no_grad():
        output = model(**tokens)
    embedding = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    return {'embedding': embedding}

class ReRanker():
    def __init__(self):
        self.model = model
        self.tokenizer = tokenizer
    def rank(self, query, docs):

        # Load các docs từ TF-IDF và chuyển thành Hugging Dataset
        # Thực hiện tìm vector tương đồng với query
        # Trả về kết quả
        dataset = Dataset.from_list(docs)

        embedding_datasets = dataset.map(get_embedding,batched=True)

        embedding_datasets.add_faiss_index(column="embedding")

        question = {'text': query}

        question_embedding = get_embedding(question)['embedding']

        scores, samples = embedding_datasets.get_nearest_examples ('embedding',question_embedding,k=10)

        return samples['text']
