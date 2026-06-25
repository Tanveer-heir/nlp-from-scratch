# Multi-Head Self-Attention from Scratch
### Pure NumPy — No PyTorch, No TensorFlow, No Autograd

> Build the complete attention block in isolation, get every masking variant mathematically airtight, and understand every tensor shape transformation — before wiring it into a full Transformer.

Once you understand this, you understand the computational heart of **BERT, GPT, T5, LLaMA, Gemma, Mistral, and Qwen**.

---

## Why This Exists

Most tutorials show you the formula and then hand you a `nn.MultiheadAttention` call. This project does the opposite — every matrix multiply, every reshape, every mask operation, every gradient-relevant design decision is written explicitly so there is nowhere for confusion to hide.

---

## Notebook Structure

| # | Section | What you learn |
|---|---------|---------------|
| 0 | Imports & reproducibility | Setup |
| 1 | Intuition — library analogy | Query / Key / Value mental model |
| 2 | Stable softmax | Why subtract max before exp |
| 3 | Scaled dot-product attention | The core formula, shapes, why √dₖ |
| 4a | Padding mask | Block `<PAD>` tokens, shape broadcasting |
| 4b | Causal mask | Upper-triangle, decoder auto-regression |
| 4c | Combined mask | Logical OR, decoder self-attention |
| 5 | Multi-Head Self-Attention class | Full implementation, split/merge heads |
| 6 | Verbose shape trace | Every intermediate tensor printed |
| 7 | 6 correctness checks | Verify masking is airtight |
| 8 | Attention heatmaps | Encoder vs decoder attention patterns |
| 9 | Cross-attention | Q from decoder, K/V from encoder |
| 10 | Summary & next steps | What to stack on top |

---

## The Problem Attention Solves

RNNs read one token at a time — information from early tokens must survive many steps to influence later ones. Long-range dependencies degrade.

```
The animal didn't cross the road because it was tired.
```

To resolve `it → animal`, an RNN must carry that signal across 6 tokens. Attention solves this by letting every token **directly** access every other token in a single operation:

```
Instead of:    I → love → NLP   (sequential)
Attention:     I
               love              (all at once)
               NLP
```

---

## Core Formula

```
Attention(Q, K, V) = softmax( Q Kᵀ / √dₖ ) · V
```

Every design decision in this project traces back to this one line.

### Query, Key, Value

Each token embedding is projected three ways with learned matrices:

```
Q = X · Wq     "What information do I need?"
K = X · Wk     "What information do I represent?"
V = X · Wv     "What information should I provide?"
```

The dot product `Q · Kᵀ` measures how relevant each key is to each query. Softmax turns those scores into weights. The weighted sum of values is the output — a context-aware representation of each token.

---

## Why Scale by √dₖ?

For Q and K drawn from Normal(0, 1) with dimension dₖ:

```
Var(Q · K) = dₖ
```

Without scaling, large dₖ → large dot products → softmax becomes near one-hot → one token gets weight ≈ 1, all others ≈ 0 → **gradients vanish**.

Dividing by √dₖ restores unit variance regardless of embedding size:

```
Var( Q·K / √dₖ ) = 1
```

This is the entire reason for the scale factor.

---

## Why Multiple Heads?

A single attention head can only learn **one type of relationship** per forward pass. Multiple heads attend to different aspects simultaneously:

```
Head 1 → subject–verb agreement
Head 2 → pronoun–antecedent coreference
Head 3 → positional proximity
Head 4 → long-range semantic dependency
```

Each head operates on a lower-dimensional subspace (`d_model / h`), so total compute stays constant. Heads never communicate during attention — only their outputs are concatenated.

---

## Architecture

```
Input X: (batch, seq, d_model)
        │
        ├─ X @ Wq  →  Q: (batch, seq, d_model)
        ├─ X @ Wk  →  K: (batch, seq, d_model)
        └─ X @ Wv  →  V: (batch, seq, d_model)
                │
                ▼   reshape + transpose
        Q: (batch, h, seq, d_k)     d_k = d_model / h
        K: (batch, h, seq, d_k)
        V: (batch, h, seq, d_v)
                │
                ▼   scaled dot-product (all heads in parallel — no loop)
        scores:  (batch, h, seq_q, seq_k)
                │
                ▼   apply mask  (−1e9 on forbidden positions)
                ▼   softmax     (weights sum to 1 per row)
                ▼   @ V
        context: (batch, h, seq, d_v)
                │
                ▼   transpose + reshape  (merge heads)
        concat:  (batch, seq, d_model)
                │
                ▼   @ Wo  (output projection)
        output:  (batch, seq, d_model)
```

### Tensor shapes at a glance

| Tensor | Shape | Notes |
|--------|-------|-------|
| Input X | `(batch, seq, d_model)` | Token embeddings |
| Q, K, V (full) | `(batch, seq, d_model)` | After linear projection |
| Q, K, V (split) | `(batch, h, seq, d_k)` | After split into heads |
| Attention scores | `(batch, h, seq_q, seq_k)` | Raw dot products |
| Attention weights | `(batch, h, seq_q, seq_k)` | After softmax |
| Context (per head) | `(batch, h, seq, d_v)` | Weighted sum of V |
| Context (merged) | `(batch, seq, d_model)` | Heads concatenated |
| Output | `(batch, seq, d_model)` | After Wo projection |

---

## The Split / Merge Head Dance

This is the most mechanically tricky part:

```python
# Split: (batch, seq, d_model) → (batch, h, seq, d_k)
x = x.reshape(batch, seq, h, d_k)   # insert head dimension
x = x.transpose(0, 2, 1, 3)         # move h before seq

# Merge: (batch, h, seq, d_v) → (batch, seq, d_model)
x = x.transpose(0, 2, 1, 3)         # move seq before h
x = x.reshape(batch, seq, h * d_v)  # flatten last two dims
```

There is no loop over heads — a single batched matrix multiply handles all `h` heads simultaneously.

---

## Masking — The Most Critical Part

Masking is what makes attention usable in practice. Three distinct mask types, each solving a different problem:

### Convention

```
mask = True   →  FORBIDDEN  →  score += −1e9  →  softmax weight ≈ 0
mask = False  →  ALLOWED    →  score unchanged
```

---

### Padding Mask

**Problem**: batches require equal-length sequences. Shorter sequences are padded with `<PAD>`:

```
Sentence A:  ["the", "king", "rules", "<PAD>", "<PAD>"]
Sentence B:  ["paris", "is", "great", "city",  "<PAD>"]
```

Without masking, `<PAD>` tokens receive non-zero attention weight and corrupt every context vector — the model learns from noise.

**Shape**: `(batch, 1, 1, seq_k)`

The `1, 1` dims broadcast automatically over all heads and all query positions. Every query avoids every pad key.

```python
mask = (token_ids == pad_id)              # (batch, seq)
mask = mask[:, np.newaxis, np.newaxis, :] # (batch, 1, 1, seq)
```

---

### Causal (Look-Ahead) Mask

**Problem**: the decoder generates tokens left to right. During training, the full target sequence is fed in parallel (teacher forcing). But each position must only attend to **past positions** — not future ones.

If position 3 sees position 5 during training, it just copies the answer. At inference, position 5 doesn't exist yet — the model breaks.

**Solution**: mask the strict upper triangle of the attention score matrix.

```
Position:   0   1   2   3   4
       0  [ ✓   ✗   ✗   ✗   ✗ ]   sees only itself
       1  [ ✓   ✓   ✗   ✗   ✗ ]   sees 0, 1
       2  [ ✓   ✓   ✓   ✗   ✗ ]   sees 0, 1, 2
       3  [ ✓   ✓   ✓   ✓   ✗ ]   sees 0..3
       4  [ ✓   ✓   ✓   ✓   ✓ ]   sees all
```

**Shape**: `(1, 1, seq, seq)` — identical for every batch item and every head.

```python
mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)  # strict upper triangle
mask = mask[np.newaxis, np.newaxis, :, :]              # (1, 1, seq, seq)
```

---

### Combined Mask (Decoder Self-Attention)

The decoder's self-attention needs **both** simultaneously. Combined with logical OR:

```
combined[i, j] = causal[i, j]  OR  padding[j]
```

A position is blocked if it is either in the future **or** a pad token.

**Shape**: `(batch, 1, seq, seq)`

```
               Padding mask      Causal mask       Combined mask
              ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
              │ · · · ✗ ✗  │   │ · ✗ ✗ ✗ ✗  │   │ · ✗ ✗ ✗ ✗  │
              │ · · · ✗ ✗  │ OR│ · · ✗ ✗ ✗  │ = │ · · ✗ ✗ ✗  │
              │ · · · ✗ ✗  │   │ · · · ✗ ✗  │   │ · · · ✗ ✗  │
              │ · · · ✗ ✗  │   │ · · · · ✗  │   │ · · · ✗ ✗  │ ← PAD row
              │ · · · ✗ ✗  │   │ · · · · ·  │   │ · · · ✗ ✗  │ ← PAD row
              └─────────────┘   └─────────────┘   └─────────────┘
```

### Mask shape cheatsheet

```
Padding mask   (batch, 1, 1, seq_k)   broadcast over h and seq_q
Causal mask    (1, 1, seq, seq)        broadcast over batch and h
Combined mask  (batch, 1, seq, seq)    explicit for all query positions
```

### Where each mask is used

| Layer | Mask type |
|-------|-----------|
| Encoder self-attention | Padding mask only |
| Decoder self-attention | Combined mask (causal + padding) |
| Decoder cross-attention | Padding mask on encoder sequence only |

---

## Cross-Attention

In the full Transformer, the decoder has a cross-attention layer that connects it to the encoder:

```
Q  ←  decoder state    (what the decoder is currently generating)
K  ←  encoder output   (the encoded source sequence)
V  ←  encoder output

scores = Q @ Kᵀ / √dₖ     shape: (batch, h, tgt_seq, src_seq)
```

This lets each decoder position look at the most relevant encoder positions. The padding mask here is built from the **encoder** sequence (ignore encoder `<PAD>`). There is **no causal mask** — the decoder is allowed to see the entire encoder output.

---

## 6 Built-in Correctness Checks

The notebook runs these automatically after each mask type:

| Check | What it verifies |
|-------|-----------------|
| 1 | PAD positions receive exactly `0.0` attention weight |
| 2 | Causal upper triangle is exactly `0.0` |
| 3 | Causal lower triangle has non-zero weights (past is attended to) |
| 4 | Every attention weight row sums to `1.0` |
| 5 | Combined mask blocks both PAD and future simultaneously |
| 6 | Forward pass is deterministic (same input → same output) |

---

## Computational Complexity

Self-attention compares every token with every other token:

```
Attention matrix size:  N × N
Time complexity:        O(N²·d)
Memory complexity:      O(N²)
```

This quadratic cost is why long-context Transformers are hard to scale. Variants that address this include FlashAttention (IO-aware recomputation), Sparse Attention (attend to a subset of positions), Longformer (local + global attention), and Performer (kernel approximation of softmax).

---

## Projection Weights

| Matrix | Shape | Role |
|--------|-------|------|
| Wq | `(d_model, d_model)` | Projects input to queries |
| Wk | `(d_model, d_model)` | Projects input to keys |
| Wv | `(d_model, d_model)` | Projects input to values |
| Wo | `(d_model, d_model)` | Mixes information across heads |

All four are learnable. Initialised with Xavier uniform: `scale = √(6 / (fan_in + fan_out))`.

---

## What to Stack on Top

This notebook isolates the attention block completely. To build a full Transformer, add these layers in order:

```
[This notebook]  Multi-Head Self-Attention
        +
Layer Normalisation   (pre-norm is more stable than post-norm)
        +
Residual connection   (output = x + sublayer(x))
        +
Position-wise FFN     Linear(d_model, d_ff) → ReLU → Linear(d_ff, d_model)
        +
Residual + LayerNorm
        +
Positional Encoding   (sinusoidal or learned)
        ×N            (stack N encoder/decoder blocks)
        =
Full Transformer
```

The masking logic built here transfers **unchanged** to any Transformer implementation.

---

## Installation & Usage

```bash
pip install numpy matplotlib scikit-learn
jupyter notebook multihead_attention_scratch.ipynb
```

Run all cells top to bottom. No GPU required — pure NumPy throughout.

---

## Dependencies

```
numpy
matplotlib
scikit-learn   (PCA, used in companion Word2Vec notebook)
```

---

## Key Takeaways

- Multi-Head Attention is scaled dot-product attention run in parallel across `h` subspaces
- Scaling by √dₖ is essential — without it, softmax saturates and gradients vanish
- The split/merge head operation is a reshape + transpose, not a loop
- Padding mask shape `(batch, 1, 1, seq_k)` broadcasts over all heads and query positions automatically
- Causal mask is `np.triu(..., k=1)` — strictly above the diagonal
- Combined mask = causal OR padding — used only in decoder self-attention
- Cross-attention uses the same code path; only the source of Q vs K/V differs
- All four weight matrices (Wq, Wk, Wv, Wo) are learnable
- The entire block is O(N²) in sequence length — the fundamental scaling challenge of Transformers

---

*Part of the NLP from Scratch series*
*Part 1 — Word2Vec (Skip-gram + Negative Sampling)*
*Part 2 — BPE Tokenizer*
*Part 3 — Multi-Head Self-Attention ← you are here*
