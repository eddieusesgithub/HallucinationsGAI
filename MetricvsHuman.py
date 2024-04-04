import torch
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2Tokenizer, GPT2Model

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

def get_embeddings(tokens):
    embeddings = []
    for token in tokens:
        inputs = tokenizer(token, return_tensors="pt")
        with torch.no_grad():   
            outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        embeddings.append(token_embeddings)
    return embeddings

def calculate_eai(prompt, response):
    similarity_scores = []
    for prompt_token in prompt:
        for response_token in response:
            similarity_score = cosine_similarity(prompt_token, response_token)
            similarity_scores.append(similarity_score) 
    eai = np.mean(similarity_scores)
    return eai

def calculate_rouge1(prompt, response):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(prompt, response)
    rouge_1_f1 = scores['rouge1'].fmeasure

    return rouge_1_f1
    
def calculate_rouge2(prompt, response):
    scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
    scores = scorer.score(prompt, response)
    rouge_2_f1 = scores['rouge2'].fmeasure

    return rouge_2_f1

def calculate_rougeL(prompt, response):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(prompt, response)
    rouge_l_f1 = scores['rougeL'].fmeasure
    
    return rouge_l_f1

def calculate_bleu(prompt, response):
    reference = [prompt.split()]
    candidate = response.split()
    bleu_score = sentence_bleu(reference, candidate, weights = (1, 0))
    return bleu_score

def calculate_meteor(prompt, response):
    prompt_tokens = prompt.split() 
    response_tokens = response.split()  
    meteor_score_value = meteor_score([prompt_tokens], response_tokens)
    return meteor_score_value

def calculate_metrics_and_correlation(csv_file):
    data = pd.read_csv(csv_file)
    prompt_responses = zip(data['prompt'], data['response'], data['value'])

    k = 1
    value_values = []
    eai_values = []
    rouge1_values = []
    rouge2_values = []
    rougeL_values = []
    bleu_values = []
    meteor_values = []

    for prompt, response, value in prompt_responses:
        prompt_tokens = word_tokenize(prompt)
        response_tokens = word_tokenize(response)
        
        embedding_prompt = get_embeddings(prompt_tokens)
        embedding_response = get_embeddings(response_tokens)
        eai_score = calculate_eai(embedding_prompt, embedding_response)
        rouge1_score = calculate_rouge1(prompt, response)
        rouge2_score = calculate_rouge2(prompt, response)
        rougeL_score = calculate_rougeL(prompt, response)
        bleu_score = calculate_bleu(prompt, response)
        meteor_score = calculate_meteor(prompt, response)

        value_values.append(value)
        eai_values.append(eai_score)
        rouge1_values.append(rouge1_score)
        rouge2_values.append(rouge2_score)
        rougeL_values.append(rougeL_score)
        bleu_values.append(bleu_score)
        meteor_values.append(meteor_score)

        k += 1

    print(f"ROUGE-1 Pearson correlation coefficient: {np.corrcoef(value_values, rouge1_values)[0, 1]}")
    print(f"ROUGE-2 Pearson correlation coefficient: {np.corrcoef(value_values, rouge2_values)[0, 1]}")
    print(f"ROUGE-L Pearson correlation coefficient: {np.corrcoef(value_values, rougeL_values)[0, 1]}")
    print(f"BLEU Pearson correlation coefficient: {np.corrcoef(value_values, bleu_values)[0, 1]}")
    print(f"METEOR Pearson correlation coefficient: {np.corrcoef(value_values, meteor_values)[0, 1]}")
    print(f"EAI Pearson correlation coefficient: {np.corrcoef(value_values, eai_values)[0, 1]}")                                 
csv_file = "/Users/eddielee/Desktop/UNI/year3/LLM/PR5.csv" 

calculate_metrics_and_correlation(csv_file)