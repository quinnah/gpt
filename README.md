Goal: Token interaction: take information from previous tokens

Version 1: averaging past contexts with for loops
- Calculate avg of all vectors in previous tokens and current token
- extremely lossy

Version 2: use matrix multiplication for weighted aggregation

Version 3: Adding softmax
- Affinities between zeros are data-dependent (start looking at each other)
- Normalize and sum (aggregate)

Version 4:
- Gather information from the past in data-dependent way
- Every single token at each position emits two vectors: query and key
- Different tokens will find different other tokens
- Get dot product between keys and queries

Step 1. Implement single head that performs self-attention
- Query and key dot-product -> high affinity


- "self-attention" just means that the keys and values are produced from the same source as queries. In "cross-attention", the queries still get produced from x, but the keys and values come from some other, external source (e.g. an encoder module)
- Cross-attention: read information from other nodes 
- Used when there’s a separate source of nodes we want to pull information

- Have implemented masked multi-attention.. now need to implement feed forward

Layer normalization
- Gaussian distribution
- Rows will be normalized
- Add and norm is applied after the transformation, but in recent years more common to apply layer norm before the transformation

Scaling up the model
- Introduce n layer: how many layers of blocks
- Add dropout before returning back to residual pathway

Dropout after calculating affinities and softmax
Dropout = 0.2 -> 20% of intermediate calculations are disabled and dropped to 0