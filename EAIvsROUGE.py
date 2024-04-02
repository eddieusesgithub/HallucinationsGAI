import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel, pipeline
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

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

def calculate_rouge(prompt, response):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(prompt, response)
    rouge_1_f1 = scores['rouge1'].fmeasure
    rouge_2_f1 = scores['rouge2'].fmeasure
    rouge_l_f1 = scores['rougeL'].fmeasure
    
    return rouge_1_f1, rouge_2_f1, rouge_l_f1

import pandas as pd

def calculate_metrics_from_csv(csv_file):
    data = pd.read_csv(csv_file)
    
    for index, row in data.iterrows():
        prompt = row['article']
        summary = summarizer(prompt, max_length=250, min_length=100, do_sample=False)
        
        prompt_tokens = word_tokenize(prompt)
        response_tokens = word_tokenize(summary[0]['summary_text'])
        
        embedding_prompt = get_embeddings(prompt_tokens)
        embedding_response = get_embeddings(response_tokens)
        
        if len(embedding_prompt) > 0 and len(embedding_response) > 0:
            eai_score = calculate_eai(embedding_prompt, embedding_response)
            rouge_value = calculate_rouge(prompt, summary[0]['summary_text'])
            
            print(f'Prompt: {prompt}')
            print(f'Response: {summary[0]["summary_text"]}')
            print(f'EAI Score: {eai_score}')
            print(f'Rouge Score: {rouge_value}')
            print()
        else:
            print(f"Skipping pair {index}: Unable to tokenize prompt or response.")

csv_file = "/path/to/directory/articles.csv"
calculate_metrics_from_csv(csv_file)
