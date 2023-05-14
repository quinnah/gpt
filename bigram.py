import torch # we use PyTorch: https://pytorch.org
import torch.nn as nn
from torch.nn import functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iters = 3000
eval_interval = 300
eval_iters = 200
n_embed = 32



# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))
# let's look at the first 1000 characters
print(text[:1000])

# get all unique characters in text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# create mapping from chars to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print(encode("hii there"))
print(decode(encode("hii there")))

# let's now encode the entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000]) # the 1000 characters we looked at earier will to the GPT look like this

# Split data into training and validation sets
n=int(0.9*len(data)) 
train_data = data[:n] # first 90% will be train, rest val
val_data = data[n:]

block_size = 8
train_data[:block_size + 1]

# mini batches of multiple text
# process batches
x = train_data[:block_size] # inputs to transformer
y = train_data[1:block_size+1] # next block size chars
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")


torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")
    
print(xb) # our input to the transformer

torch.manual_seed(1337)

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

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # very thin wrapper around tensor of shape vocab_size x vocab_size
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        # logits: 
        tok_embed = self.token_embedding_table(idx) # (B, T, C)
        pos_embed = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_embed + pos_embed # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)
        # pick out row of embedding table
        # predict score for next chars in the sequence
        if targets is None:
            loss = None
        # evaluate loss function
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # stretch to 2-d
            targets = targets.view(B*T) # one-dimensional
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on last time step
            logits = logits[:, -1, :] # becomes B,C
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
# logits, loss = m(xb, yb) # parameters: inputs, targets
# print(logits.shape)
# print(loss)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(context, max_new_tokens=500)[0].tolist()) 

# # version 1
# xbow = torch.zeros((B,T,C))
# for b in range(B):
#     for t in range(T):
#         xprev = x[b,:t+1]
#         xbow[b,t] = torch.mean(xprev, 0)


# # version 2:matrix multiply
# wei = torch.tril(torch.ones(T,T))
# wei = wei / wei.sum(1, keepdim=True)
# xbow2 = wei @ x
# torch.allclose(xbow, xbow2)

# # version 3: adding softmax
# tril = torch.tril(torch.ones(T, T))
# wei = torch.zeros((T,T))
# wei = wei.masked_fill(tril == 0, float('-inf')) 
# wei = F.softmax(wei, dim=-1)
# xbow3 = wei @ x
# torch.allclose(xbow, xbow3)

# version 4: self-attention
torch.manual_seed(1337)
B,T,C = 4, 8, 32 # batch, time, channels
x = torch.randn(B,T,C)

# single head performs self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x) # (B, T, 16)
q = query(x) # (B, T, 16)
wei = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) -> (B, T, T)

tril = torch.tril(torch.ones(T,T))
# wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf')) 
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v
# out = wei @ x

out.shape
