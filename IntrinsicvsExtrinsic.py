import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize

tokenizer  = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

def get_embeddings(tokens):
    embeddings = []
    for token in tokens:
        try:
            inputs = tokenizer(token, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                token_embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            embeddings.append(token_embeddings)
        except ValueError:
            continue  
    return embeddings

def calculate_eai(prompt_embeddings, response_embeddings):
    similarity_scores = []
    for prompt_token in prompt_embeddings:
        for response_token in response_embeddings:
            similarity_score = cosine_similarity(prompt_token.reshape(1, -1), response_token.reshape(1, -1))
            similarity_scores.append(similarity_score[0][0])
    eai = np.mean(similarity_scores)    
    return eai

def calculate_eai_from_csv(csv_file):
    data = pd.read_csv(csv_file)
    
    for index, row in data.iterrows():
        prompt = row['prompt']
        response = row['response']

        prompt_tokens = word_tokenize(prompt)
        response_tokens = word_tokenize(response)
        
        embedding_prompt = get_embeddings(prompt_tokens)
        embedding_response = get_embeddings(response_tokens)
        
        if len(embedding_prompt) > 0 and len(embedding_response) > 0:
            eai_score = calculate_eai(embedding_prompt, embedding_response)
            
            print(f'Prompt: {prompt}')
            print(f'Response: {response}')
            print(f'EAI Score: {eai_score}')
            print()
        else:
            print(f"Skipping pair {index}: Unable to tokenize prompt or response.")

csv_file = "/Users/eddielee/Desktop/UNI/year3/LLM/hallucinations.csv"
calculate_eai_from_csv(csv_file)
