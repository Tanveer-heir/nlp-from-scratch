# Mini Transformer Encoder–Decoder — From Scratch

> A complete, from-scratch implementation of the original Transformer architecture in PyTorch — no `nn.Transformer`, no `nn.MultiheadAttention` — trained on a toy sequence-to-sequence task (sequence reversal).

**Notebook:** `mini_transformer_seq2seq.ipynb`

This project builds every piece of a Transformer encoder–decoder by hand so you can see exactly what happens at each step: embeddings, positional encoding, scaled dot-product attention, multi-head attention, feed-forward sublayers, masking, the full encoder/decoder stacks, teacher-forced training, and greedy autoregressive inference. The same ideas power BERT, GPT, T5, and LLaMA — this is the minimal, fully-readable version of the mechanism underneath all of them.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Overall Architecture](#3-overall-architecture)
4. [Configuration & Hyperparameters](#4-configuration--hyperparameters)
5. [Token Embeddings](#5-token-embeddings)
6. [Positional Encoding](#6-positional-encoding)
7. [Query, Key, and Value](#7-query-key-and-value)
8. [Scaled Dot-Product Attention](#8-scaled-dot-product-attention)
9. [Multi-Head Attention](#9-multi-head-attention)
10. [Feed-Forward Network](#10-feed-forward-network)
11. [Encoder](#11-encoder)
12. [Decoder](#12-decoder)
13. [Attention Masks](#13-attention-masks)
14. [Putting It Together: Seq2SeqTransformer](#14-putting-it-together-seq2seqtransformer)
15. [Training Process](#15-training-process)
16. [Teacher Forcing](#16-teacher-forcing)
17. [Greedy Decoding](#17-greedy-decoding)
18. [Results](#18-results)
19. [Attention Visualization](#19-attention-visualization)
20. [Mathematical Summary](#20-mathematical-summary)
21. [Repository / Notebook Structure](#21-repository--notebook-structure)
22. [How to Run](#22-how-to-run)
23. [Ideas to Extend This Project](#23-ideas-to-extend-this-project)
24. [Key Takeaways](#24-key-takeaways)

---

## 1. Introduction

Transformers were introduced in:

> **"Attention Is All You Need"** — Vaswani et al., 2017

Unlike RNNs or LSTMs, which process tokens one at a time in sequence, Transformers process **all tokens simultaneously** using **self-attention**. This enables:

- Efficient parallel computation (no sequential bottleneck during training)
- Better modeling of long-range dependencies (any token can directly attend to any other token, regardless of distance)
- The architectural backbone for nearly every modern large language model

This project implements every major component **manually**, using only `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, and raw tensor operations — deliberately avoiding PyTorch's built-in `nn.Transformer` and `nn.MultiheadAttention` so nothing is hidden behind a black box.

---

## 2. Problem Statement

Instead of a large-scale task like machine translation,

```
English → French
```

the notebook trains the model on **sequence reversal**:

```
Input:   5 2 8 1
Output:  1 8 2 5
```

### Why this task?

- It's genuinely **seq2seq** — it exercises real encoder–decoder cross-attention, not just a fixed positional remap.
- It requires **no dataset** — random examples are generated on the fly, infinitely.
- It's **easy to verify**: we know exactly what correct behavior looks like, both in the output sequence and in the attention pattern that should produce it (decoder step `t` should attend most to encoder position `L - 1 - t`).
- It forces the model to learn:
  - token identity
  - positional information
  - autoregressive sequence generation

without the complexity (or training time) of a real NLP corpus.

---

## 3. Overall Architecture

```
                Source Sequence
                      │
              Token Embedding
                      │
          Positional Encoding
                      │
                Encoder Stack  (N layers)
                      │
              Encoder Memory
                      │
         ┌────────────┴─────────────┐
         │                           │
 Target Tokens                Cross-Attention
   (shifted right,                  │
    teacher-forced)         Decoder Stack (N layers)
         │                          │
   Token Embedding                  │
         │                          │
   Positional Encoding              │
         │                          │
         └──────────────┬───────────┘
                         │
                  Linear Layer (→ vocab size)
                         │
                Softmax over Vocabulary
                         │
                Predicted Next Token
```

The **encoder** reads the entire source sequence and produces a contextualized representation ("memory") for every source position. The **decoder** generates the output one token at a time, at each step attending both to what it has generated so far (masked self-attention) and to the encoder's memory (cross-attention).

---

## 4. Configuration & Hyperparameters

The model is intentionally **small** ("mini") so it trains in well under a minute on CPU, while still containing every architectural piece of a full Transformer.

| Hyperparameter | Value | Meaning |
|---|---|---|
| `d_model` | 64 | Embedding / hidden dimension |
| `num_heads` | 4 | Number of attention heads (each head has dimension 16) |
| `num_layers` | 2 | Number of encoder layers **and** number of decoder layers |
| `d_ff` | 128 | Inner dimension of the feed-forward sublayer |
| `max_len` | 14 | Max sequence length the positional encoding table supports |
| `dropout` | 0.1 | Dropout applied after attention, FFN, and positional encoding |
| `vocab_size` | 13 | 10 digit tokens (0–9) + `<pad>`, `<bos>`, `<eos>` |

Special tokens: `PAD = 0`, `BOS = 1`, `EOS = 2`. Digit tokens are offset by `+3` so they don't collide with the special tokens (digit `0` → token id `3`, digit `9` → token id `12`).

**Total trainable parameters: ~169,933** — small enough to train from scratch in seconds-to-minutes, large enough to perfectly learn the task.

---

## 5. Token Embeddings

Neural networks can't operate on raw integers or text — each token is converted into a dense vector via a lookup table (`nn.Embedding`).

```
token id 7   →   [0.14, -1.23, 0.51, ..., 64 values]
```

With `d_model = 64`, every token becomes a 64-dimensional vector. In this implementation, the encoder and decoder each have their **own** embedding table (`Encoder.embedding`, `Decoder.embedding`), and embeddings are scaled by `√d_model` before positional encoding is added — a detail from the original paper that balances the relative magnitude of the embedding vs. the positional signal.

---

## 6. Positional Encoding

Self-attention alone has **no built-in notion of order** — it treats the input as a *set*, not a *sequence*. Without positional information,

```
A B C
```

and

```
C B A
```

would look identical to the attention mechanism. So a deterministic, sinusoidal positional signal is **added** to each token embedding:

For even dimensions:
$$PE_{(pos,\ 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

For odd dimensions:
$$PE_{(pos,\ 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Each `(2i, 2i+1)` dimension pair oscillates at a different frequency, so every position gets a unique vector "fingerprint," and nearby positions get similar fingerprints — which helps the model generalize smoothly across position. The notebook precomputes this table once (up to `max_len`) as a registered buffer (`PositionalEncoding`), and includes a heatmap visualization of the resulting pattern.

---

## 7. Query, Key, and Value

This is the core mechanism behind self-attention. Every embedded token is projected into three different learned spaces:

```
Q = X · W_Q
K = X · W_K
V = X · W_V
```

where `X` is the input embedding matrix and `W_Q`, `W_K`, `W_V` are learned weight matrices (implemented as `nn.Linear` layers inside `MultiHeadAttention`). They start randomly initialized (Xavier init in this implementation) and are updated via backpropagation during training.

**Why three separate projections?** Think of a library search:

| Vector | Role | Analogy |
|---|---|---|
| Query | "What am I looking for?" | The search you type into the catalog |
| Key | "What information do I represent?" | The catalog entry for each book |
| Value | "What information should I provide?" | The actual contents of the book |

The same token plays all three roles, but through three different linear lenses — this is what lets a single token simultaneously "advertise" itself differently depending on whether it's being matched against (`Key`), searched for (`Query`), or read out (`Value`).

**Worked example.** Suppose an embedding `[1, 0, 2, 1]` and a weight matrix

```
W_Q =
[[1, 0],
 [0, 1],
 [1, 1],
 [0, 1]]
```

Then `Query = Embedding × W_Q = [3, 3]`. Keys and Values are computed by the exact same procedure with their own weight matrices.

---

## 8. Scaled Dot-Product Attention

Attention determines *which tokens matter to which other tokens*. The computation has four steps:

**Step 1 — Similarity scores.** Every query is compared against every key via dot product:
$$\text{Scores} = QK^\top$$
A higher dot product means higher similarity between that query/key pair.

**Step 2 — Scale.**
$$\text{Scores} = \frac{QK^\top}{\sqrt{d_k}}$$
Without this scaling, dot products grow large in magnitude as `d_k` grows, pushing softmax into a saturated regime with near-zero gradients. Dividing by `√d_k` keeps the scores in a well-behaved range.

**Step 3 — Softmax.**
$$\text{Weights} = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)$$
Each row now sums to 1 and represents a probability distribution over "how much to attend to" each position. Example: raw scores `[3, 6, 2]` become roughly `[0.045, 0.909, 0.046]` after softmax — the model has effectively picked one dominant position to attend to.

**Step 4 — Weighted sum of Values.**
$$\text{Output} = \text{Weights} \cdot V$$

Putting it together:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

**Why is the output multiplied by Values and not Keys?** Keys exist purely to compute the matching score against a query — they never carry information into the output. Values carry the actual content. Back to the library analogy: your query is matched against the **catalog** (Keys) to find relevant books, but what you actually *read* is the **book content** (Values), not the catalog entry.

Implemented as the standalone function `scaled_dot_product_attention()`, with an optional `mask` argument that sets disallowed positions to `−1e9` before the softmax (see [§13](#13-attention-masks)).

---

## 9. Multi-Head Attention

Rather than computing attention once with the full `d_model`-dimensional vectors, the model splits into multiple smaller subspaces ("heads") and runs attention in each, in parallel:

```
d_model = 64, num_heads = 4   →   each head operates on 64 / 4 = 16 dimensions
```

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \cdot W_O$$
$$\text{head}_i = \text{Attention}(QW^Q_i,\ KW^K_i,\ VW^V_i)$$

Each head can specialize in a different kind of relationship — for instance, one head might focus on local/positional patterns while another focuses on longer-range content matches. The outputs of all heads are concatenated and passed through a final linear projection (`W_O`).

**Implementation detail:** instead of `h` separate small linear layers, the `MultiHeadAttention` class uses **one** full-size `d_model → d_model` linear layer for Q, K, and V respectively, then reshapes/transposes the result into `(num_heads, d_k)` chunks (`split_heads`) and merges them back afterward (`combine_heads`). This is mathematically identical to separate per-head projections but requires only a single matmul per projection — faster in practice.

This module is reused for **three different purposes** in the full model:
1. Encoder self-attention (`Q = K = V` = source sequence)
2. Decoder masked self-attention (`Q = K = V` = target sequence so far)
3. Decoder–encoder cross-attention (`Q` = decoder states, `K = V` = encoder memory)

---

## 10. Feed-Forward Network

After attention mixes information **across** positions, each position is passed independently through an identical 2-layer MLP:

$$\text{FFN}(x) = \max(0,\ xW_1 + b_1)\,W_2 + b_2$$

```
64 (d_model) → 128 (d_ff) → 64 (d_model)
```

Conceptually: **attention routes information between positions**; the **feed-forward network processes the information at each position** independently, giving the model extra representational capacity after context has been gathered.

---

## 11. Encoder

Each encoder layer (`EncoderLayer`) consists of two sub-blocks, each wrapped in a residual connection and LayerNorm ("post-norm," matching the original paper):

```
Input
  │
  ├──────────────┐
  │              │
Multi-Head        │
Self-Attention     │
  │              │
  ▼              │
  + ◄────────────┘     (residual connection)
  │
LayerNorm
  │
  ├──────────────┐
  │              │
Feed Forward      │
  │              │
  ▼              │
  + ◄────────────┘     (residual connection)
  │
LayerNorm
  │
Output
```

Self-attention here lets **every source token attend to every other source token** — there's no masking restriction in the encoder beyond ignoring padding. The full `Encoder` stacks `N` of these layers (here `N = 2`) on top of the embedded + positionally-encoded source sequence, producing the contextualized "memory" that the decoder will query via cross-attention.

---

## 12. Decoder

Each decoder layer (`DecoderLayer`) has **three** sub-blocks (vs. the encoder's two):

**1. Masked Self-Attention.** The target sequence attends to itself, but with a causal mask so position `t` cannot see positions `> t`:

```
<BOS>  1  8  ?
```

While predicting the token at `?`, everything after it is hidden. This is what makes the model genuinely autoregressive rather than just copying from a future token it isn't allowed to see at inference time.

**2. Cross-Attention.** Queries come from the decoder; **keys and values come from the encoder's output**. This is the literal "encoder–decoder" connection — at every decoder step, the model looks back at the *entire* source sequence and decides which positions are relevant for producing the current output token.

```
Queries  ←  Decoder
Keys/Values  ←  Encoder
```

**3. Feed-Forward.** Identical structure to the encoder's FFN.

Each sub-block has its own residual connection + LayerNorm. The full `Decoder` stacks `N` of these layers, fed the embedded + positionally-encoded target sequence, the causal+padding mask, and the encoder's output.

---

## 13. Attention Masks

Two distinct masks are used, both implemented to broadcast across all attention heads.

**Padding mask** (`make_padding_mask`) — marks real tokens as `1`/allowed and `<pad>` tokens as `0`/disallowed:

```
tokens:  1   2   3  PAD PAD
mask:    1   1   1   0   0
```

Used in (a) encoder self-attention, so padding doesn't influence real tokens, and (b) cross-attention, so the decoder never attends to source padding.

**Causal mask** (`make_causal_mask`) — a lower-triangular matrix preventing any position from attending to future positions:

```
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

Used only in decoder self-attention. The notebook combines the causal mask with the target's own padding mask (`make_decoder_self_mask`) via element-wise AND, since a position must be both a real token *and* not in the future to be attended to.

---

## 14. Putting It Together: Seq2SeqTransformer

```
src tokens --> Encoder --> encoder memory
                                  │
tgt tokens --> Decoder (self-attn + cross-attn to memory) --> Linear --> logits over vocab
```

The `Seq2SeqTransformer` class wires the `Encoder`, `Decoder`, and a final `generator` linear layer (`d_model → vocab_size`) together. It exposes:
- `encode(src, src_mask)` — run the encoder once
- `decode(tgt, enc_out, tgt_mask, memory_mask)` — run the decoder + output projection
- `forward(src, tgt)` — full training-time forward pass (builds both masks internally, encodes, then decodes the entire shifted target sequence at once)

Weights are initialized with Xavier uniform initialization, which tends to stabilize early Transformer training.

---

## 15. Training Process

**Loss:** `nn.CrossEntropyLoss` with `ignore_index=PAD`, so padded positions contribute nothing to the loss or its gradients.

**Optimizer:** Adam (`betas=(0.9, 0.98)`, `eps=1e-9` — the standard Transformer recipe), with a linear warmup schedule (`LambdaLR`) over the first 200 steps before settling at a constant learning rate of `3e-4`.

**Gradient clipping:** gradient norms are clipped to `1.0` to prevent instability from occasional large updates.

Backpropagation updates every learnable matrix in the network — embeddings, `W_Q`/`W_K`/`W_V`/`W_O` in every attention block, both FFN layers, and all LayerNorm affine parameters.

Each epoch runs both a training pass and a validation pass (`run_epoch`), tracking per-token loss and per-token accuracy (ignoring padding) for both.

---

## 16. Teacher Forcing

During training, the decoder is **never** asked to use its own (possibly wrong) predictions as input — it's always fed the correct previous tokens. Given a full target sequence `[<bos>, 1, 8, 2, 5, <eos>]`:

| | Sequence |
|---|---|
| Decoder **input** | `<bos>, 1, 8, 2, 5` (everything except the last token) |
| Decoder **label** | `1, 8, 2, 5, <eos>` (everything except the first token — shifted by one) |

At every position, the decoder input up to and including that position is used to predict the *next* token — exactly what the causal mask permits, and exactly the setup needed for valid autoregressive generation at inference time. This dramatically speeds up training convergence compared to feeding back the model's own predictions during training.

---

## 17. Greedy Decoding

At inference time, there is no ground-truth target — the sequence must be generated autoregressively, one token at a time:

```
Start with <BOS>
   │
   ▼
Encode source once
   │
   ▼
Feed current decoder input → take logits at last position → argmax (greedy pick)
   │
   ▼
Append predicted token to decoder input
   │
   ▼
Repeat until <EOS> is generated or max length is reached
```

This uses the exact same `encode()`/`decode()` methods as training — only the *driving loop* differs (one token at a time vs. all-at-once teacher forcing). Implemented as `greedy_decode()` in the notebook.

---

## 18. Results

Training for 15 epochs on this toy task (with the hyperparameters in [§4](#4-configuration--hyperparameters)) produces:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 1.977 | 0.289 | 1.038 | 0.601 |
| 5 | 0.459 | 0.834 | 0.090 | 0.972 |
| 10 | 0.159 | 0.947 | 0.006 | 0.999 |
| 15 | 0.081 | 0.974 | 0.001 | 1.000 |

**Exact-match sequence accuracy** (full predicted sequence == full target sequence):

| Test set | Accuracy |
|---|---|
| 15 random hand-inspected examples | **15 / 15** |
| 500 random sequences, lengths 3–8 (in training range) | **100.00%** |
| 500 random sequences, lengths 9–11 (longer than training) | **4.60%** |

The sharp drop on longer-than-trained sequences is a genuine and instructive result, not a bug: it illustrates a real limitation of this setup — sinusoidal positional encodings and learned attention patterns here don't automatically generalize to lengths well outside the training distribution.

---

## 19. Attention Visualization

The most convincing evidence that the model learned a real *algorithm* — not just memorized patterns — is to inspect its **cross-attention weights**. For sequence reversal, decoder output step `t` should attend most strongly to encoder input position `(L - 1 - t)`, producing a clean **anti-diagonal** pattern:

```
                  Input
              4   7   2   9

      9       .   .   .   X
Out   2       .   .   X   .
      7       .   X   .   .
      4       X   .   .   .
```

The notebook's `MultiHeadAttention` module stashes its attention weights (`self.attn_weights`) on every forward pass, so after running inference, `plot_cross_attention()` pulls the last decoder layer's cross-attention weights (averaged across heads) and renders them as a heatmap. The actual visualization in the notebook confirms this anti-diagonal alignment, which is strong evidence the model is genuinely using attention to solve the task rather than guessing.

---

## 20. Mathematical Summary

```
Embedding:           X
Queries:             Q = X · W_Q
Keys:                K = X · W_K
Values:              V = X · W_V

Attention scores:    QKᵀ
Scaling:             QKᵀ / √d_k
Softmax:             softmax(QKᵀ / √d_k)
Attention output:    softmax(QKᵀ / √d_k) · V

Multi-head:          Concat(head_1, ..., head_h) · W_O

Feed-forward:        max(0, xW_1 + b_1)·W_2 + b_2

Encoder sublayer:    LayerNorm(x + SelfAttn(x))
                     LayerNorm(x + FFN(x))

Decoder sublayer:    LayerNorm(x + MaskedSelfAttn(x))
                     LayerNorm(x + CrossAttn(x, enc_out))
                     LayerNorm(x + FFN(x))

Output:              softmax(Linear(decoder_output))
```

---

## 21. Repository / Notebook Structure

The notebook (`mini_transformer_seq2seq.ipynb`) is organized into the following sections, each with a markdown explanation followed by runnable code:

1. Imports & configuration
2. Toy dataset (sequence reversal) + vocabulary, `Dataset`/`DataLoader`/`collate_fn`
3. Positional encoding (+ heatmap visualization)
4. Scaled dot-product attention (standalone function)
5. Multi-head attention (`MultiHeadAttention` class)
6. Position-wise feed-forward network (`PositionWiseFeedForward`)
7. Encoder layer & encoder stack (`EncoderLayer`, `Encoder`)
8. Decoder layer & decoder stack (`DecoderLayer`, `Decoder`)
9. Masks: padding mask, causal mask, combined decoder mask (+ visualization)
10. Full model (`Seq2SeqTransformer`)
11. Training loop (`run_epoch`) + loss/accuracy curves
12. Greedy decoding (`greedy_decode`)
13. Evaluation on random test sequences (`evaluate_examples`, `exact_match_accuracy`)
14. Attention visualization (`plot_cross_attention`)

---

## 22. How to Run

**Requirements:** `torch`, `matplotlib` (standard library otherwise: `math`, `random`, `copy`).

```bash
pip install torch matplotlib
jupyter notebook mini_transformer_seq2seq.ipynb
```

Run all cells top to bottom. On CPU, the full 15-epoch training run completes in well under a minute given the small model size (~170K parameters). No GPU or external dataset is required — all training data is generated on the fly.

---

## 23. Ideas to Extend This Project

- **Harder toy tasks:** sorting a sequence, copying, a character-level "addition" task (`"23+45"` → `"68"`), or translation between two small synthetic "languages."
- **Beam search** instead of greedy decoding — typically improves accuracy on harder tasks where the single best next-token choice isn't always globally optimal.
- **Pre-norm instead of post-norm:** move `LayerNorm` to *before* each sub-layer (`x = x + SubLayer(LayerNorm(x))`) — this is what most modern large language models use, and tends to train more stably at scale.
- **Label smoothing** in the loss function — a regularization trick from the original paper.
- **Length generalization:** investigate why accuracy collapses on sequences longer than the training range, and experiment with relative or learned positional encodings as a fix.
- **Scale up:** increase `d_model`, `num_heads`, and `num_layers` to see how convergence speed and capacity change on harder tasks.

---

## 24. Key Takeaways

- ✔ Transformers replace recurrence with attention, enabling full parallelism over the sequence.
- ✔ Token order is preserved using positional encoding, since attention itself is order-agnostic.
- ✔ Queries, Keys, and Values are learned linear projections of the same embeddings, each playing a different role.
- ✔ Attention scores are computed as `QKᵀ`, scaled by `√d_k` to keep softmax well-behaved.
- ✔ Softmax converts raw scores into a probability distribution over positions.
- ✔ The attention output is a weighted sum of **Values** — Keys are used only for matching, never for carrying information into the output.
- ✔ Multi-head attention lets the model learn several different relationships in parallel, in different subspaces.
- ✔ The encoder processes the entire source sequence at once; the decoder generates the output autoregressively, one token at a time.
- ✔ Padding masks and causal masks together ensure attention never looks at filler tokens or the future.
- ✔ Teacher forcing (feeding the correct shifted target sequence during training) dramatically speeds up convergence versus feeding the model's own predictions.
- ✔ Greedy decoding generates output token-by-token at inference time, using the same encode/decode methods as training.
- ✔ Visualizing cross-attention weights is a simple, convincing way to confirm the model learned a genuine algorithmic alignment — not just memorized noise.
- ✔ This project faithfully implements the original Transformer architecture end-to-end and is a solid foundation for understanding modern models such as BERT, GPT, T5, and LLaMA.
