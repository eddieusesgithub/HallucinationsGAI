import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import time

def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds"

i=0
start_time = time.time()
while i<100:    #can set this to as many iterations as you want 
    
    print(f"running iteration {i+1}")
    device = torch.device('mps')
    device = 'mps' if torch.backends.mps.is_built() else '.cpu'
    print(device)

    block_size = 64     #hyperparameters
    batch_size = 32
    max_iters = 301
    learning_rate = 3e-4
    eval_iters = 100

    n_embd = 384
    n_head = 8
    n_layer = 8
    dropout = 0.2
    
    with open("/path/to/directory/vocab.txt", 'r', encoding = 'utf-8') as f:    #I omit the directory to my personal computer
        text = f.read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    
    string_to_int = { ch:i for i, ch in enumerate(chars) }
    int_to_string = { i:ch for i, ch in enumerate(chars) }
    encode = lambda s: [string_to_int[c] for c in s]
    decode = lambda l: ''.join([int_to_string[i] for i in l])
    
    data = torch.tensor(encode(text), dtype=torch.long)
    
    #the following two functions accomodate for the fact it is infeasible to open extremely large text files on a computer's ram
    def get_random_chunk(split):
        #the output_train.txt and output_val.txt files are produced from the cleaningopenwebdata.py file
        filename = "/path/to/directory/output_train.txt" if split == 'train' else "/path/to/directory/output_val.txt"
        with open(filename, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access = mmap.ACCESS_READ) as mm:
                file_size = len(mm)
                start_pos = random.randint(0, (file_size) - block_size*batch_size)
                
                mm.seek(start_pos)
                block = mm.read(block_size*batch_size-1)
                
                decoded_block = block.decode('utf-8', errors = 'ignore').replace('\r', '')
                
                data = torch.tensor(encode(decoded_block), dtype=torch.long)
        return data
    
    def get_batch(split):
        data = get_random_chunk(split)
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x,y = x.to(device), y.to(device)
        return x,y
    
    @torch.no_grad()
    def estimate_loss():
    
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X,Y = get_batch(split)
                logits, loss = model(X,Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    
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
            wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 #scaling by 1/sqrt(len(keys))
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #masking with lower triangular matrix
            wei = F.softmax(wei, dim = -1)
            wei = self.dropout(wei) #scaled dot product
            v = self.value(x) 
            out = wei @ v #matmul
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
    
    class GPTLanguageModel(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)        
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
            self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(n_embd)       
            self.lm_head = nn.Linear(n_embd, vocab_size)  
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
            
            tok_emb = self.token_embedding_table(index)
            pos_emb = self.position_embedding_table(torch.arange(T, device = device))
            x = tok_emb + pos_emb 
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
            for _ in range(max_new_tokens):
                logits, _ = self.forward(index)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                index_next = torch.multinomial(probs, num_samples=1)
                index = torch.cat((index, index_next), dim=1)
            return index
    
    model = GPTLanguageModel(vocab_size)
    print('loading model parameters...')
    with open('model-01.pkl', 'rb') as f:
        model = pickle.load(f)
    print('loaded model parameters successfully')
    m = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    iter_start_time = time.time()
    for iter in range(max_iters):
        if iter % eval_iters == 0 or iter == max_iters:
            losses = estimate_loss()
            print(f"step: {iter}, train loss {losses['train']:.4f}, val loss: {losses['val']:.4f}" )
            iter_end_time=time.time()
            iter_time = iter_end_time - iter_start_time
            iter_start_time=time.time()
            print("iteration time:", format_time(iter_time))
        xb, yb = get_batch('train')
    
        logits, loss = model.forward(xb,yb)
        optimizer.zero_grad(set_to_none = True)
        loss.backward()
        optimizer.step()
    
    print(loss.item())
    
    with open('model-01.pkl', 'wb') as f:
        pickle.dump(model, f)
    print('model saved')
    i=i+1
    
end_time = time.time()
elapsed_time = end_time-start_time

print(f"{i} training iterations completed")     #these statements provide progress updates in the training process
print("Elapsed training time:", format_time(elapsed_time)) 
    
