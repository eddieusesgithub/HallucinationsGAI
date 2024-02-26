

import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device('cpu')
print(device)

block_size = 64     #hyperparameters
batch_size = 32
max_iters = 300
learning_rate = 3e-4
eval_iters = 100

n_embd = 384
n_head = 8
n_layer = 8
dropout = 0.2

with open("/path/to/directory/vocab.txt", 'r', encoding = 'utf-8') as f:
    text = f.read()
chars = sorted(set(text))
vocab_size = len(chars)

string_to_int = { ch:i for i, ch in enumerate(chars) }
int_to_string = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

class Head(nn.Module):
    """ one head of self-attention """   
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """ 
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """  
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

# Define the GPTLanguageModel
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)        #decoders
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)       #final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)       #make it so softmax can work with the info

        self.apply(self._init_weights)
         
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
        
    
    def forward(self, index, targets=None):
        B,T = index.shape
   
        max_length = min(index.size(1), 32)

        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(max_length, device=device))
        x = tok_emb + pos_emb.unsqueeze(0) 
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape 
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, index, max_new_tokens):
        max_length = min(index.size(1), 32)
        for _ in range(max_new_tokens):
            # Get only the newly generated tokens
            index_cond = index[:, -max_length:]

            logits, loss = self.forward(index_cond)

            # Check if logits has valid dimensions
            if logits.dim() < 2 or logits.size(1) == 0:
                print("Error: Logits tensor has an invalid size along dimension 1.")
                break

            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index[:, max_length:]  # Return only the newly generated tokens

model = GPTLanguageModel(vocab_size)
print('loading model parameters...')
with open('model-01.pkl', 'rb') as f:
    model = pickle.load(f)
print('loaded model parameters successfully')
m = model.to(device)

def calculate_eai(embedding_prompt, embedding_response):
    embedding_prompt_2d = embedding_prompt.reshape(-1, embedding_prompt.shape[-1])
    embedding_response_2d = embedding_response.reshape(-1, embedding_response.shape[-1])

    similarity_scores = cosine_similarity(embedding_prompt_2d, embedding_response_2d)

    eai = np.mean(similarity_scores)
    return eai

cumulative_sai = 0
k=1
while True:
    prompt = input("Prompt:\n")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)

    # Generate response from the model
    generated_chars = m.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist()
    generated_text = decode(generated_chars)

    # Calculate embeddings for prompt and generated response
    embedding_prompt = m.token_embedding_table(context.unsqueeze(0)).detach().numpy()
    embedding_response = m.token_embedding_table(torch.tensor(generated_chars, dtype=torch.long, device=device)).detach().numpy()

    # Calculate SAI
    sai_score = calculate_sai(embedding_prompt, embedding_response)

    # Calculating rolling SAI average
    cumulative_sai = cumulative_sai + sai_score
    sai_average = cumulative_sai/k
    k=k+1
    # Display results
    print(f'Response number: {k-1}')
    print(f'Completion:\n{generated_text}')
    print(f'SAI Score: {sai_score}')
    print(f'Rolling SAI average: {sai_average}')
    
