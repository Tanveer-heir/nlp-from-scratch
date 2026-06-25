# Tiny GPT with a Hand-Rolled KV Cache, from Scratch

> A from-scratch, decoder-only GPT (~10.8M parameters) trained as a character-level
> language model, paired with a **KV cache implemented by hand** (not via a library
> shortcut) for efficient autoregressive generation — plus a numerical proof that the
> cache is correct, and a benchmark showing why it matters.

This README is the long-form companion to **`tiny_gpt_kv_cache.ipynb`**. It explains every
concept the notebook implements, in the order the notebook implements it, with references
to the actual classes, functions, and test results produced when the notebook is run. Treat
it as study notes you can return to months later without re-deriving everything from the code.

---

## Table of Contents

1. [What's in the repo](#1-whats-in-the-repo)
2. [Quick start](#2-quick-start)
3. [GPT vs. encoder–decoder Transformers](#3-gpt-vs-encoderdecoder-transformers)
4. [Character-level language modeling](#4-character-level-language-modeling)
5. [Dataset preparation](#5-dataset-preparation)
6. [Tokenization](#6-tokenization)
7. [Input–target pair creation](#7-inputtarget-pair-creation)
8. [Model architecture overview](#8-model-architecture-overview)
9. [The decoder block](#9-the-decoder-block)
10. [Causal self-attention](#10-causal-self-attention)
11. [Feed-forward network (MLP)](#11-feed-forward-network-mlp)
12. [Layer normalization](#12-layer-normalization)
13. [Residual connections](#13-residual-connections)
14. [Parameter count](#14-parameter-count)
15. [Training process](#15-training-process)
16. [Text generation](#16-text-generation)
17. [Greedy vs. sampling](#17-greedy-vs-sampling)
18. [Temperature scaling](#18-temperature-scaling)
19. [Top-k sampling](#19-top-k-sampling)
20. [Why naive GPT inference is slow](#20-why-naive-gpt-inference-is-slow)
21. [KV cache: the idea](#21-kv-cache-the-idea)
22. [KV cache: the implementation](#22-kv-cache-the-implementation)
23. [Why only keys and values are cached (not queries)](#23-why-only-keys-and-values-are-cached-not-queries)
24. [KV cache correctness test](#24-kv-cache-correctness-test)
25. [Complexity analysis and benchmark results](#25-complexity-analysis-and-benchmark-results)
26. [Modern LLM inference](#26-modern-llm-inference)
27. [Known limitations and extension ideas](#27-known-limitations-and-extension-ideas)
28. [Key takeaways](#28-key-takeaways)
29. [Final mental model](#29-final-mental-model)

---

## 1. What's in the repo

| File | Purpose |
|---|---|
| `tiny_gpt_kv_cache.ipynb` | The full notebook: data, model, training, KV cache, correctness check, benchmark, generation |
| `README.md` | This document |

The notebook downloads its own data (Tiny Shakespeare) on first run, so no separate dataset
file is required. Running it top-to-bottom produces: a trained model, a loss curve, a
correctness proof for the KV cache, a cached-vs-uncached generation speed benchmark, and
sample generated text.

---

## 2. Quick start

```bash
pip install torch matplotlib jupyter
jupyter notebook tiny_gpt_kv_cache.ipynb
```

Run all cells in order. On a GPU, the default config trains in a couple of minutes. On CPU,
either be patient or reduce `max_steps` in the training-hyperparameters cell (Section 15
below) for a faster, lower-quality smoke test.

**Default model config** (Section 14 of the notebook):

```python
block_size = 256   # context length
n_layer    = 6
n_head     = 6
n_embd     = 384
dropout    = 0.1
```

This gives **10,786,560 parameters (~10.8M)** — inside the 10–30M target range. A
commented-out "bigger config" (`n_layer=8, n_head=8, n_embd=448`) reaches **~19.5M**
parameters if you have a GPU and want better samples.

---

## 3. GPT vs. encoder–decoder Transformers

The original Transformer (Vaswani et al., 2017) has two stacks:

```
Source Sentence → Encoder → Decoder → Output Sentence
```

The encoder reads the *whole* source sequence at once (no causal masking — every position
can see every other position). The decoder then generates the output sequence
autoregressively, attending both to its own previous outputs (causal self-attention) and
to the encoder's output (cross-attention).

GPT throws away the encoder and cross-attention entirely:

```
Previous Tokens → Decoder-only stack → Next Token
```

This is the architecture used by GPT-2/3/4, LLaMA, Mistral, Gemma, Qwen, and — relevant
here — by `TinyGPT`, the class this notebook builds. It has:

- ✔ No encoder
- ✔ No cross-attention
- ✔ Only causal self-attention

Everything in this notebook (the `CausalSelfAttention`, `Block`, and `TinyGPT` classes) is
exclusively the decoder side of that original diagram.

---

## 4. Character-level language modeling

The notebook tokenizes at the **character** level rather than the word or subword level.
`"Romeo"` becomes five separate tokens: `R`, `o`, `m`, `e`, `o`. This means:

- The vocabulary is tiny: **65 unique characters** for the Tiny Shakespeare corpus (letters,
  punctuation, space, newline).
- There's no BPE/tokenizer training step — `vocab_size` falls directly out of `set(text)`.
- The model has to learn spelling and structure from raw characters, which makes it a
  slower learner per-token than a subword model, but it keeps the notebook's focus on the
  *architecture and KV cache*, not on tokenization engineering. Swapping in a subword
  vocabulary later requires no changes to `TinyGPT` itself — only to the
  `encode`/`decode`/`stoi`/`itos` data-prep cells.

---

## 5. Dataset preparation

The dataset is **Tiny Shakespeare**, ~1.1MB / 1,115,394 characters, downloaded directly
inside the notebook from:

```
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

It's split:

```python
n = int(0.9 * len(data))
train_data = data[:n]   # 1,003,854 tokens
val_data   = data[n:]   #   111,540 tokens
```

a standard 90/10 train/validation split.

---

## 6. Tokenization

Every unique character gets an integer ID, built from the sorted set of characters seen in
the corpus:

```python
chars = sorted(list(set(text)))
vocab_size = len(chars)          # 65
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
```

`encode("Hello, world!")` → list of 65-way integer IDs; `decode(ids)` inverts it exactly
(the notebook asserts `decode(encode(s)) == s` as a sanity check immediately after defining
both functions). The model itself never sees characters — only these integers, which feed
into an `nn.Embedding` lookup table.

---

## 7. Input–target pair creation

Language modeling is framed as **next-character prediction**. For a window of
`block_size` characters, the input `x` and target `y` are the same slice of text, offset
by one position:

```python
def get_batch(split, block_size, batch_size, device):
    d = train_data if split == "train" else val_data
    ix = torch.randint(0, len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
```

Concretely, for the string `"hello"`:

```
input:  h e l l
target: e l l o
```

At every position simultaneously, the model is trained to predict the *next* character —
not just the final one. A sequence of length `block_size` yields `block_size` separate
next-token predictions in a single forward pass, which is what makes Transformer training
so parallelizable compared to RNNs.

---

## 8. Model architecture overview

```
Input token IDs
      │
      ▼
Token Embedding  ──┐
                    ├─►  (sum)
Position Embedding ─┘
      │
      ▼
  Block × n_layer        (each: LN → Attn → residual → LN → MLP → residual)
      │
      ▼
  Final LayerNorm
      │
      ▼
  Linear head → vocabulary logits
```

No encoder, no cross-attention — exactly the decoder-only shape from Section 3. In code,
this is the `TinyGPT` class:

```python
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer=6, n_head=6, n_embd=384, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
```

**Position embeddings matter:** attention itself has no notion of order — it would treat
`"AB"` and `"BA"` identically without them. `pos_emb` is a learned vector per absolute
position, added to the token embedding before the first block.

One detail that becomes important later (Section 22): `forward()` takes a `start_pos`
argument so that, during incremental decoding, a single new token can still be told *which*
absolute position it occupies:

```python
def forward(self, idx, targets=None, kv_caches=None, start_pos=0):
    B, T = idx.shape
    positions = torch.arange(start_pos, start_pos + T, device=idx.device)
    x = self.tok_emb(idx) + self.pos_emb(positions)[None, :, :]
    ...
```

---

## 9. The decoder block

Each `Block` is:

```
Input
  │
LayerNorm
  │
Causal Self-Attention
  │
Residual Add
  │
LayerNorm
  │
MLP
  │
Residual Add
```

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x, kv_cache=None):
        x = x + self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + self.mlp(self.ln2(x))
        return x
```

This block is repeated `n_layer` times to build depth. Real models scale this number up
substantially — GPT-2 Small uses 12 blocks, LLaMA-7B uses 32. The default config in this
notebook uses 6.

Note that `kv_cache` is threaded straight through the block into the attention sublayer —
the MLP never needs caching, since it has no notion of other positions (see Section 11).

---

## 10. Causal self-attention

The decoder must never see future tokens. For the sequence `"I love ?"`, when predicting
`?` the model may only condition on `"I love"`, never on tokens that come after the
position being predicted.

This is enforced with a **causal mask**: for a `T × T` attention-score matrix, every entry
where the key position is *after* the query position is set to `-inf` before the softmax,
so its probability becomes exactly zero:

```
mask (lower-triangular, True = allowed):
1 0 0
1 1 0
1 1 1
```

In the notebook this is `CausalSelfAttention`, built from raw `nn.Linear` layers (no
`nn.MultiheadAttention`, no `torch.nn.Transformer`):

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv_proj(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        ...
```

This single module has **two execution paths**, controlled by whether `kv_cache` is `None`:

- **`kv_cache is None`** → the *training / full-sequence* path: standard `T × T` masked
  attention over the whole chunk at once. This is what Sections 8–13 use for training.
- **`kv_cache is not None`** → the *incremental decoding* path, covered in Section 22.

This dual-path design means the exact same module is used for both training and inference
— it's the `kv_cache` argument that switches behavior, not a different class.

---

## 11. Feed-forward network (MLP)

After attention mixes information *between* token positions, each position is passed
independently through a small two-layer network:

```python
class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))
```

`Linear → GELU → Linear`, expanding to 4× the embedding width and projecting back down —
the standard GPT-style ratio. The MLP has **no notion of sequence position**: it applies
the identical transformation to every token independently. All cross-position information
mixing happens exclusively in attention; the MLP's job is to increase the expressive power
of each token's own representation once attention has gathered context for it.

**Why GELU and not ReLU?** ReLU hard-zeroes every negative input. GELU instead smoothly
suppresses negative values rather than clipping them outright, which empirically trains
better in Transformers. Modern LLMs use GELU or its descendants (e.g. SwiGLU) almost
universally instead of plain ReLU.

---

## 12. Layer normalization

Without normalization, activations in a deep network can grow or shrink uncontrollably as
they pass through many layers, destabilizing training. `nn.LayerNorm` re-centers and
rescales each token's feature vector to have zero mean and unit variance (with learned
scale/shift) before it enters a sublayer.

This notebook uses **pre-norm**: LayerNorm is applied *before* each sublayer, not after:

```
LayerNorm → Attention → Residual Add → LayerNorm → MLP → Residual Add
```

This is exactly what `Block.forward` does (Section 9): `self.ln1(x)` feeds into attention,
`self.ln2(x)` feeds into the MLP, and the *un-normalized* `x` is what the residual connects
to. Pre-norm is what GPT-2, GPT-3, and effectively all modern decoder-only models use,
because it trains substantially more stably than the original post-norm Transformer,
especially as depth increases.

---

## 13. Residual connections

Rather than replacing the input with the sublayer's output, the block *adds* the sublayer's
output to the input:

```
Output = x + Attention(LN(x))
Output = x + MLP(LN(x))
```

instead of `Output = Attention(x)`. This "skip connection" gives gradients a direct path
backward through the network, independent of how many layers deep the signal has to travel
— without it, training networks as deep as modern Transformers (dozens to hundreds of
layers) would be impractical. Both `+` signs in `Block.forward` (Section 9) are this
mechanism.

---

## 14. Parameter count

Running the model-instantiation cell with the default config:

```python
block_size = 256
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1

model = TinyGPT(vocab_size, block_size, n_layer, n_head, n_embd, dropout)
print(model.num_params())
```

prints:

```
Model parameters: 10,786,560 (10.79M)
```

squarely inside the requested 10–30M range. The notebook also includes a commented
alternative:

```python
# n_layer = 8
# n_head = 8
# n_embd = 448
```

which comes out to **~19.5M parameters** — useful if you have a GPU and want a stronger
model without leaving the target range.

---

## 15. Training process

Standard language-model training pipeline:

```
Input tokens → TinyGPT → vocabulary logits → CrossEntropyLoss → backprop → AdamW step
```

```python
optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95)
)
```

with a **linear warmup → cosine decay** learning-rate schedule (warmup avoids instability
from large gradients while weights are still near their random initialization; cosine decay
lets the optimizer "settle" later in training) and **gradient norm clipping** (clip to 1.0)
for additional stability:

```python
for step in range(max_steps):
    xb, yb = get_batch("train", block_size, batch_size, device)
    logits, loss = model(xb, targets=yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    scheduler.step()
```

Loss is computed inside `TinyGPT.forward` itself when `targets` is provided:

```python
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
```

Validation loss is tracked alongside training loss every `eval_interval` steps (averaged
over `eval_iters` batches, to reduce noise), and both curves are plotted at the end of
training so you can visually confirm the model is actually learning and not overfitting.

In a fast CPU smoke-test run with a deliberately tiny config (2 layers, 64-dim, 150 steps),
training loss dropped from **4.18 → 3.04** in under 20 seconds — the loss-decreasing
trend you should expect to see, just much more pronounced with the full default config and
more steps.

---

## 16. Text generation

Generation starts from a prompt and proceeds **autoregressively** — one new token at a
time, each new token appended to the growing sequence before predicting the next:

```
"ROMEO:" → predict → "H" → append → "ROMEO:H" → predict → "e" → append → "ROMEO:He" → ...
```

repeated until `max_new_tokens` new characters have been generated. This is implemented in
the notebook's `generate()` function (distinct from `TinyGPT.forward`, which only does a
single forward pass) — see Section 22 for the cached version.

---

## 17. Greedy vs. sampling

Two ways to turn the model's output probabilities into an actual next token:

- **Greedy decoding** — always pick the single highest-probability token. Deterministic
  and stable, but prone to repetitive, "stuck" text.
- **Sampling** — draw randomly from the predicted probability distribution
  (`torch.multinomial`). If the model predicts `dog: 0.6, cat: 0.3, car: 0.1`, sampling will
  usually pick `dog` but occasionally picks `cat` or `car`, producing more varied,
  natural-sounding text.

The notebook's `generate()` always samples (never strictly greedy), but pushes the
distribution toward greedy or toward uniform via temperature (Section 18) and restricts
the candidate set via top-k (Section 19).

---

## 18. Temperature scaling

Temperature divides the logits before the softmax:

```python
logits_step = logits_step / temperature
probs = F.softmax(logits_step, dim=-1)
```

- **Low temperature (e.g. 0.5)** → sharper distribution → closer to greedy, more
  deterministic, more repetitive.
- **High temperature (e.g. 2.0)** → flatter distribution → closer to uniform, more random,
  more prone to nonsensical output.

The notebook's sampling example uses `temperature=0.8`, a mild sharpening that keeps output
mostly coherent while still varying run to run.

---

## 19. Top-k sampling

Rather than letting every token in the (65-character) vocabulary be eligible for sampling,
top-k restricts the candidate pool to only the `k` most likely tokens at each step:

```python
v, _ = torch.topk(logits_step, top_k)
logits_step[logits_step < v[:, [-1]]] = float("-inf")
```

Everything outside the top `k` is masked to `-inf` before the softmax, so it gets exactly
zero probability mass. This prevents sampling from occasionally selecting wildly unlikely
tokens that would derail the generated text, while still preserving randomness among the
genuinely plausible candidates. The notebook uses `top_k=40` in its generation example.

---

## 20. Why naive GPT inference is slow

Suppose the prompt is `"Hello"` and we want to generate `"!"`, then `" "`, then `"H"`, etc.
Without any optimization, **every single generation step reprocesses the entire sequence
generated so far from scratch**:

```
step 1: forward("Hello")
step 2: forward("Hello!")
step 3: forward("Hello! ")
step 4: forward("Hello! H")
...
```

Each step redoes all the work of every previous step — recomputing keys and values for
tokens whose representations can never change again, every single time. As the generated
sequence grows, the cost of *each individual step* grows too, and the total cost across all
steps compounds. For long generations or long conversations, this becomes the dominant cost
of inference. This is exactly the `use_cache=False` path benchmarked in Section 25.

---

## 21. KV cache: the idea

Look again at attention:

```
Attention(Q, K, V) = softmax(QKᵀ / √d) V
```

For a given decoder layer, the **key** and **value** vectors at position *i* depend only on
the hidden state at position *i* — which, once that position has been processed, never
changes again. Only the **query** for the newest token is new at each step; every previous
position's keys and values are exactly the same as they were the last time they were
computed.

So instead of recomputing `K` and `V` for every old position at every step, **compute them
once, store them, and reuse them**. This stored memory is the **KV cache**. At each new
step, the model only has to:

1. Compute `Q`, `K`, `V` for the **single new token**.
2. Append the new `K`, `V` onto the cache.
3. Attend the new `Q` against the *entire* cached `K`/`V` (old + new).

No previously-finished computation is ever redone.

---

## 22. KV cache: the implementation

This is implemented directly inside `CausalSelfAttention.forward`, as the second of its two
execution paths (Section 10). The cache itself is just a plain Python dict per layer,
holding two tensors:

```python
def new_kv_cache(self):
    return [dict(k=None, v=None) for _ in range(self.n_layer)]
```

and the cached-attention branch:

```python
if kv_cache is not None:
    T_past = 0 if kv_cache.get("k") is None else kv_cache["k"].size(2)

    if kv_cache.get("k") is not None:
        k = torch.cat([kv_cache["k"], k], dim=2)   # append new K onto cached K
        v = torch.cat([kv_cache["v"], v], dim=2)   # append new V onto cached V
    kv_cache["k"] = k
    kv_cache["v"] = v

    T_q = q.size(2)   # new query positions in THIS call (often just 1)
    T_k = k.size(2)   # total cached length so far

    att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

    if T_q > 1:
        # needed during "prefill" (multi-token prompt fed through the cache
        # for the first time) — a single decode step (T_q == 1) needs no
        # mask at all, since every cached key is already in the past.
        q_pos = torch.arange(T_past, T_past + T_q, device=x.device).unsqueeze(1)
        k_pos = torch.arange(0, T_k, device=x.device).unsqueeze(0)
        causal = k_pos <= q_pos
        att = att.masked_fill(~causal, float("-inf"))

    att = F.softmax(att, dim=-1)
    out = att @ v
```

**Worked example**, mirroring the notebook's `generate()`:

1. **Prefill.** Feed the whole prompt (say `"ROMEO:"`, 6 characters) through the model in
   *one* forward call, with an empty cache. This populates `kv_cache["k"]`/`["v"]` for all 6
   positions in every layer, and `T_q = 6` here — so the `if T_q > 1` branch above kicks in
   to mask correctly within the prefill chunk. We only need the logits at the *last*
   position to predict the first new character.
2. **Step.** Sample the first new character, e.g. `"H"`. Feed *only* that one new token
   (`T_q = 1`) back into the model, passing `start_pos=6` so its positional embedding is
   correct. Inside attention, its `K`/`V` get appended to the cache (now length 7), and its
   query attends against all 7 cached keys — no mask needed since `T_q == 1`.
3. **Repeat.** Each subsequent step costs exactly one token's worth of `Q`/`K`/`V`
   projection and one row of attention against the ever-growing cache — never reprocessing
   earlier tokens.

The notebook's `generate()` function wires this up explicitly:

```python
if use_cache:
    kv_caches = model.new_kv_cache()
    logits, _ = model(prompt, kv_caches=kv_caches, start_pos=0)   # prefill
    cur_len = prompt.size(1)
    next_logits = logits[:, -1, :]

for _ in range(max_new_tokens):
    ...
    next_id = torch.multinomial(F.softmax(logits_step, dim=-1), num_samples=1)
    idx = torch.cat([idx, next_id], dim=1)
    if use_cache:
        logits, _ = model(next_id, kv_caches=kv_caches, start_pos=cur_len)  # single-token step
        cur_len += 1
        next_logits = logits[:, -1, :]
```

---

## 23. Why only keys and values are cached (not queries)

Attention is `softmax(QKᵀ)V`. Conceptually:

- The **query** represents *"what is this token looking for?"* — and that question is
  different for every newly generated token. There is nothing to cache, because the query
  for token *t+1* has no relationship to the query for token *t* that would make reuse
  meaningful.
- The **key** and **value** for a given position represent *"what this position offers to
  anyone attending to it"* — and that answer, once computed, is fixed forever. Position 3's
  key and value are identical no matter whether we're currently generating token 4 or token
  400.

So the cache only ever needs to grow `K` and `V` by one row (per new token); `Q` is freshly
computed every step and never stored.

---

## 24. KV cache correctness test

A KV cache implementation can run without crashing and still be **silently wrong** — for
example, my first draft of `CausalSelfAttention` skipped masking entirely whenever
`kv_cache` was passed, reasoning that "new tokens only attend to the past." That's true for
single-token decode steps, but **false during prefill**: when the prompt's multiple tokens
are processed together for the first time, query position 0 in that chunk must *not* be
allowed to see key position 2. Catching this requires comparing the cached implementation's
output against a known-correct baseline — never just trusting that it works because it
ran without an exception.

The notebook's test (`check_kv_cache_correctness`) does exactly this:

1. Run a normal full forward pass with `kv_cache=None` → ground-truth logits.
2. Run the *same* input sequence through the cached path: prefill the first few tokens in
   one call, then feed the remaining tokens **one at a time**, each time extending the
   cache.
3. Concatenate the cached-path logits and diff them against the ground truth.

```python
max_diff = (logits_full - logits_cached).abs().max().item()
assert max_diff < 1e-3, "KV-cache output diverges from the uncached forward pass!"
```

Running this in the notebook gives:

```
max |full - cached| logit difference: 2.980e-07
PASS: KV-cache output matches the full forward pass.
```

A difference on the order of `1e-7` is floating-point noise, not a bug — this is the
expected result for a correct implementation, and it's the bar every cache implementation
should be held to before it's trusted for real generation.

---

## 25. Complexity analysis and benchmark results

**Without KV cache:** generating token *t* requires recomputing attention, MLP, and every
layer's transformation for *all* `t` previous tokens — and this entire computation is
thrown away and redone from scratch at step *t+1*. Total redundant work compounds
quadratically with the number of tokens generated.

**With KV cache:** each new token only requires projecting `Q`/`K`/`V` for *that one token*
and attending it against the cache; no earlier computation is ever repeated.

The notebook's `benchmark_generation()` measures this directly, running both
`use_cache=True` and `use_cache=False` for the same prompt and token counts. A fast CPU run
(2-layer toy model, deliberately small for quick iteration) produced:

| tokens generated | cached | uncached | speedup |
|---|---|---|---|
| 50  | 0.056s | 0.088s | 1.56× |
| 100 | 0.067s | 0.164s | 2.46× |
| 200 | 0.066s | 0.333s | 5.08× |

The trend is the important part, not the absolute numbers: the speedup **grows** as the
generated sequence gets longer, because the uncached path's wasted work grows with it while
the cached path's per-step cost stays roughly flat. At the notebook's full default config
(10.8M params, `block_size=256`) and longer generations, this gap is even larger.

---

## 26. Modern LLM inference

Every production LLM uses KV caching during inference — GPT-3/4, LLaMA, Mistral, Gemma,
Qwen, and Claude all rely on it. Without it, interactive chat would be dramatically slower:
every new token in a long conversation would force the model to reprocess the *entire*
conversation history from the very first message, every single time it speaks.

What's implemented in this notebook — store `K`/`V` per layer, append new entries, attend
new queries against the full cache — is the same core mechanism real inference engines
(vLLM, llama.cpp, Hugging Face `transformers` with `use_cache=True`, etc.) use under the
hood. Production systems add substantial engineering on top (memory paging across many
concurrent requests, eviction policies, quantized cache storage, attention-variant tricks to
shrink the cache itself) — but the fundamental correctness requirement is identical to the
one verified in Section 24: **the cached computation must be mathematically indistinguishable
from recomputing everything from scratch.**

---

## 27. Known limitations and extension ideas

The cache implementation here is deliberately minimal for clarity. Things it does *not*
handle, and how you'd extend it:

- **Context overflow.** Once the cache reaches `block_size`, the notebook's `generate()`
  simply stops extending rather than evicting old entries. A **sliding-window cache** would
  drop the oldest cached positions to keep generating indefinitely, at the cost of the model
  losing access to the earliest context.
- **Cache memory footprint.** Storing full `K`/`V` for every layer and every position scales
  memory linearly with sequence length, which is the real bottleneck for very long contexts
  in production. **Multi-query attention (MQA)** and **grouped-query attention (GQA)** —
  sharing `K`/`V` across multiple query heads — are the standard real-world fix, and would
  be a natural next step to add to `CausalSelfAttention`.
- **Batched generation with mixed prompt lengths.** This implementation assumes a single
  prompt length per batch. Real serving systems batch many different conversations with
  different lengths simultaneously, which requires padding and an attention mask that
  interacts correctly with the cache.
- **Scale.** Bumping `n_layer`/`n_embd` toward the upper end of the 10–30M range, plus more
  training steps on a GPU, will produce visibly more coherent Shakespeare-flavored samples
  than the quick CPU smoke test described above.

---

## 28. Key takeaways

- ✔ GPT is a decoder-only Transformer: no encoder, no cross-attention, only causal
  self-attention.
- ✔ It predicts one token at a time, conditioned only on past tokens (the causal mask
  enforces this exactly, not approximately).
- ✔ Character-level modeling is a simple, tokenizer-free educational setup; the same
  `TinyGPT` class works unchanged with a subword vocabulary.
- ✔ Each decoder block is `LayerNorm → Attention → residual → LayerNorm → MLP → residual`,
  using **pre-norm** for training stability.
- ✔ Training uses teacher forcing (every position's true next character is known during
  training) and cross-entropy loss.
- ✔ Generation is autoregressive: sample one token, append it, repeat.
- ✔ Greedy decoding is deterministic but repetitive; sampling with temperature and top-k
  trades some determinism for more natural variety.
- ✔ Naive inference recomputes the entire sequence at every generation step — wasteful and
  the root cause of slow autoregressive decoding.
- ✔ A KV cache stores each layer's keys and values once and reuses them, since they never
  change once computed; only the query is ever new.
- ✔ Queries are never cached, because each newly generated token asks a genuinely new
  question.
- ✔ **A KV cache must be checked numerically against a full forward pass** — it's easy to
  write one that runs without error yet computes the wrong thing (e.g. missing the causal
  mask during multi-token prefill, as happened during development of this very notebook).
- ✔ The benefit of caching grows with sequence length, exactly as measured in Section 25.

---

## 29. Final mental model

```
Training
────────────────────────────────────────────
Entire sequence (length T)
        │
        ▼
TinyGPT forward (kv_cache = None)
        │
        ▼
Predict next token AT EVERY POSITION simultaneously
        │
        ▼
CrossEntropyLoss → backprop → AdamW step


Inference (use_cache=True)
────────────────────────────────────────────
Prompt
        │
        ▼
Prefill: one forward call, kv_cache populated for every prompt position
        │
        ▼
Sample next token from last position's logits
        │
        ▼
Feed ONLY the new token back in, start_pos = current length
        │
        ▼
Attention: new Q attends against (cached K,V) + (new K,V)
        │
        ▼
Append new K,V to cache permanently
        │
        ▼
Predict next token → repeat
```

This notebook is meant to bridge the gap between understanding Transformer layers on paper
and understanding, concretely, how real-world decoder-only LLMs — GPT, LLaMA, Mistral,
Gemma, Qwen, Claude — generate text efficiently in production: by never doing the same work
twice.
