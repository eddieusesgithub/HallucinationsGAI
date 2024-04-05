import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

tokenizer = AutoTokenizer.from_pretrained("unicamp-dl/translation-pt-en-t5")
model = AutoModelForSeq2SeqLM.from_pretrained("unicamp-dl/translation-pt-en-t5")

tokenizer1  = BertTokenizer.from_pretrained('bert-base-uncased')
model1 = BertModel.from_pretrained('bert-base-uncased')

def get_embeddings(tokens):
    embeddings = []
    for token in tokens:
        try:
            inputs = tokenizer1(token, return_tensors="pt")
            with torch.no_grad():
                outputs = model1(**inputs)
                token_embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            embeddings.append(token_embeddings)
        except ValueError:
            continue  
    return embeddings

def translate(text, source_lang="pt", target_lang="en"):
    input_text = f"translate {source_lang} to {target_lang}: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(input_ids=input_ids, max_length=128, num_return_sequences=1)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def calculate_eai(prompt_embeddings, response_embeddings):
    similarity_scores = []
    for prompt_token in prompt_embeddings:
        for response_token in response_embeddings:
            similarity_score = cosine_similarity(prompt_token.reshape(1, -1), response_token.reshape(1, -1))
            similarity_scores.append(similarity_score[0][0])
    eai = np.mean(similarity_scores)    
    return eai

def calculate_bleu(prompt, response):
    reference = [prompt.split()]
    candidate = response.split()
    bleu_score = sentence_bleu(reference, candidate, weights = (1, 0))
    return bleu_score

def calculate_metrics_from_csv(csv_file):
    data = pd.read_csv(csv_file)
    
    for index, row in data.iterrows():
        prompt = row['portuguese']
        ideal = row['english']
        translation = translate(prompt)

        ideal_tokens = word_tokenize(ideal)
        response_tokens = word_tokenize(translation)
        
        embedding_ideal = get_embeddings(ideal_tokens)
        embedding_response = get_embeddings(response_tokens)
        
        if len(embedding_ideal) > 0 and len(embedding_response) > 0:
            eai_score = calculate_eai(embedding_ideal, embedding_response)
            bleu_value = calculate_bleu(ideal, translation)
            
            print(f'Ideal translation: {ideal}')
            print(f'Generated translation: {translation}')
            print(f'EAI Score: {eai_score}')
            print(f'BLEU Score: {bleu_value}')
            print()
        else:
            print(f"Skipping pair {index}: Unable to tokenize prompt or response.")

csv_file = "/Users/eddielee/Desktop/UNI/year3/LLM/translations.csv"
calculate_metrics_from_csv(csv_file)
