import torch
import torch.nn as nn
from torch.nn import functional as F

block_size = 64
batch_size = 256
max_iters = 5000
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_emb_d = 384
n_head = 6
n_layer = 6
dropout = .2

torch.manual_seed(7)

with open('mahabharata.txt') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  data = train_data if split == "train" else val_data 
  ix = torch.randint(len(data)-block_size, (batch_size, ))
  x = torch.stack([data[i: i+block_size] for i in ix])
  y = torch.stack([data[i+1: i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_emb_d, head_size, bias = False) 
        self.query = nn.Linear(n_emb_d, head_size, bias = False)
        self.value = nn.Linear(n_emb_d, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # shape: (batch_size, context_size, head_size)
        q = self.query(x) # shape: (batch_size, context_size, head_size)
        wei = q @ k.transpose(-2, -1) * C**-0.5# (B, T, 16) @ (B, 16, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__ ()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_emb_d, n_emb_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out
    
   
class FeedForward(nn.Module):
    
    def __init__(self, n_emb_d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb_d, 4 * n_emb_d),
            nn.ReLU(),
            nn.Linear(4 * n_emb_d, n_emb_d),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
   
    def __init__(self, n_emb_d, n_head):
        super().__init__()
        head_size = n_emb_d // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_emb_d)
        self.ln1 = nn.LayerNorm(n_emb_d)
        self.ln2 = nn.LayerNorm(n_emb_d)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_emb_d) 
    self.position_embedding_table = nn.Embedding(block_size, n_emb_d)
    # self.blocks = nn.Sequential(
    #         Block(n_emb_d, n_head=4),
    #         Block(n_emb_d, n_head=4),
    #         Block(n_emb_d, n_head=4)
    #         nn.LayerNorm(n_emb_d)
    #     )   
    self.blocks = nn.Sequential(*[Block(n_emb_d, n_head=n_head) for _ in range(n_layer)])
    # self.sa_heads = MultiHeadAttention(4, n_emb_d//4)
    # self.ffwd = FeedForward(n_emb_d)
    self.ln_f = nn.LayerNorm(n_emb_d)
    self.lm_head = nn.Linear(n_emb_d, vocab_size) #language_modelling_head

  def forward(self, idx, targets=None):
    B, T = idx.shape

    token_embeddings = self.token_embedding_table(idx) # shape: (batch_size, context_length_size, number_of_embedding_dimensions)
    pos_embeddings = self.position_embedding_table(torch.arange(T, device=device)) # shape: (context_length_size, number_of_embedding_dimensions)
    x = token_embeddings + pos_embeddings # shape: (batch_size, context_length_size, vocabulary_size). broadcasting involved
    # x = self.sa_heads(x)
    # x = self.ffwd(x)
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x) # shape: (batch_size, context_length_size, vocabulary_size)
    
    if targets is None: 
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    # generate max_new_tokens new indices and concatenate to idx
    # idx is (B, T) array of indices. row is number of batches, and column is context length
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      logits, loss = self(idx_cond)
      # we only need the last value in the sequence to generate the next sequence, in this particular model
      logits = logits[:, -1, :] 
      # getting the probabilites from the logits
      probs = F.softmax(logits, dim = -1)
      # sampling from the distrbution
      idx_next = torch.multinomial(probs, num_samples = 1)
      idx = torch.cat((idx, idx_next), dim = 1)
    return idx
  
model = BigramModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for iter in range(max_iters):
  
  if iter%eval_interval == 0:
    losses = estimate_loss()
    print(f"step: {iter} train loss:{losses['train']:.4f} val loss:{losses['val']:.4f}")

  xb, yb = get_batch('train')
  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens = 10000)[0].tolist()))