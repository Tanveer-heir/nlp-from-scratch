# Seq2Seq with Bahdanau Attention — from Scratch in PyTorch

A from-scratch PyTorch implementation of an **Encoder–Decoder Sequence-to-Sequence model with Bahdanau (Additive) Attention** — no `nn.MultiheadAttention`, no seq2seq libraries. Every matrix multiply in the attention mechanism is written out explicitly, trained on a toy task, and verified by visualizing the learned attention weights.

📓 Notebook: [`seq2seq_bahdanau_attention.ipynb`](./seq2seq_bahdanau_attention.ipynb)

---

## Overview

Sequence-to-Sequence models transform an input sequence into an output sequence. Real-world applications include:

- Machine Translation
- Text Summarization
- Question Answering
- Dialogue Systems
- Speech Recognition
- Text Generation

This project uses a **toy sequence-reversal task** instead of real translation data. It's a deliberate simplification: the task is trivial, but it produces a clean, known, visualizable alignment between input and output positions, which makes it easy to *see* whether attention has actually learned something correct.

```text
Input:  g j a a b i e
Output: e i b a a j g
```

The encoder, attention module, and decoder are all task-agnostic — swap in a parallel corpus loader and the same code trains a real translator.

---

## Motivation

### The vanilla Seq2Seq bottleneck

A classic encoder-decoder compresses the *entire* input sequence into a single fixed-length vector, which the decoder must then unpack the whole output from:

```text
Input Sequence
      ↓
   Encoder
      ↓
Context Vector   ← single fixed-size vector, regardless of input length
      ↓
   Decoder
      ↓
Output Sequence
```

A 50-token sequence and a 5-token sequence get squeezed into the same size vector. Performance degrades sharply as sequences get longer, because early information gets diluted or overwritten by the time the encoder reaches the end.

### The Bahdanau attention fix

Bahdanau, Cho & Bengio (2014) proposed keeping **every encoder hidden state** around, and letting the decoder compute a fresh, dynamically-weighted combination of all of them at **every** decoding step:

```text
Encoder Outputs (one vector per input token)
      ↓
  Attention   ← recomputed at every decoder step
      ↓
Context Vector   ← changes every step
      ↓
   Decoder
      ↓
 Next Token
```

Instead of forcing one vector to represent the whole sequence, the decoder gets to *ask a question* at every step — "given what I've generated so far, which input tokens matter right now?" — and the answer is a soft, differentiable lookup over the source sequence. This is the direct conceptual ancestor of Query/Key/Value attention in Transformers: Bahdanau attention is **additive attention** (scores via a small feedforward net + tanh), while Transformers use **dot-product attention** ($QK^\top$) — the softmax-weighted-sum idea is identical.

---

## Architecture

```text
Input Sequence
       ↓
 Embedding Layer
       ↓
Bidirectional Encoder GRU
       ↓
  Encoder Outputs  ──────────┐
       ↓                     │
 (decoder hidden state) ──→ Bahdanau Attention
                              ↓
                        Context Vector
                              ↓
                         Decoder GRU
                              ↓
                         Linear Layer
                              ↓
                            Softmax
                              ↓
                         Output Token
                              ↓
                    Repeat until <eos>
```

---

## Dataset

A toy **sequence reversal** task, generated on the fly (no external data needed):

```text
Input:  g j a a b i e
Target: e i b a a j g
```

- Vocabulary: lowercase letters `a`–`j` (10 symbols)
- Sequence length: randomly sampled between 5 and 10 tokens per example
- Special tokens: `<pad>` (id 0), `<sos>` (id 1), `<eos>` (id 2) → vocab size **13**
- 4,000 training examples / 400 validation examples, regenerated randomly each run

This is a stand-in for any (source, target) sequence task — replace `ReverseDataset`/`make_example()` with a real parallel-corpus loader and the rest of the pipeline (encoder, attention, decoder, training loop) is unchanged.

---

## Model Components

### 1. Embedding Layer

Converts token IDs into dense, learnable vector representations (`emb_dim = 32`).

```text
"a"  →  [-0.13, 0.82, -0.45, ...]
```

### 2. Bidirectional Encoder GRU

Processes the input sequence in both directions and concatenates the results, so every encoder output carries both left- and right-context:

```text
Forward:   a → b → c → d → e
Backward:  a ← b ← c ← d ← e
```

```python
encoder_outputs: (batch_size, src_len, 2 * hidden_dim)   # hidden_dim = 64
```

Padded positions are skipped entirely via `pack_padded_sequence` / `pad_packed_sequence`, so the GRU never wastes computation on `<pad>` tokens.

The encoder's final forward/backward hidden states are concatenated and projected through a `Linear + tanh` layer to seed the decoder's initial hidden state.

### 3. Bahdanau Attention

Computes how relevant each encoder hidden state is for the current decoding step.

**Score function** — for decoder state $s_{t-1}$ and encoder state $h_i$:

$$e_{t,i} = v^\top \tanh(W_s s_{t-1} + W_h h_i)$$

**Attention weights** — softmax over all source positions (padding masked to $-\infty$ first):

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}$$

**Context vector** — weighted sum of encoder outputs:

$$c_t = \sum_i \alpha_{t,i}\, h_i$$

```python
attn_weights: (batch_size, src_len)          # sums to 1 across src_len
context:      (batch_size, 2 * hidden_dim)
```

### 4. Decoder

At each timestep, the decoder combines the previous token's embedding with a freshly-computed attention context:

```text
Previous token embedding ─┐
                           ├─→ GRU → Hidden State ─┐
   Attention Context ──────┘                       │
                                                     ├─→ Linear → Softmax → Predicted Token
   Attention Context ─────────────────────────────┘
   Previous token embedding ───────────────────────┘
```

The output layer sees the GRU output, the attention context, *and* the input embedding directly (a "deep output" layer), giving it more direct signal than the hidden state alone.

Because attention recomputes the context at every step from the *current* hidden state, the decoder is run one timestep at a time rather than as a single multi-step RNN call.

---

## Complete Forward Pipeline

```text
Input Sequence
      ↓
  Embedding
      ↓
Bidirectional Encoder GRU
      ↓
Encoder Outputs
      ↓
  Attention  ←── decoder hidden state (previous step)
      ↓
Context Vector
      ↓
 Decoder GRU
      ↓
 Linear Layer
      ↓
Vocabulary Distribution
      ↓
 Predicted Token
      ↓
Repeat until <eos>
```

---

## Teacher Forcing

During training, the decoder is fed the *true* previous target token rather than its own prediction, with some probability `teacher_forcing_ratio`:

```text
Ground truth: e i b a a j g

Predicting "b":  input fed in is the true previous token "i",
                 not whatever the model itself predicted.
```

This notebook **linearly decays** the teacher-forcing ratio from 0.5 → 0.1 across training, leaning on ground truth early (when the model's own predictions are mostly noise) and increasingly relying on its own predictions later, better matching what it faces at inference time.

---

## Loss Function

```python
criterion = nn.CrossEntropyLoss(ignore_index=PAD)
```

Compares predicted token distributions against the true next token at every position, ignoring `<pad>` positions so padding never contributes to the gradient.

---

## Training Pipeline

```text
Input Sequence
      ↓
   Encoder
      ↓
Encoder Outputs
      ↓
   Decoder + Attention  (one step at a time)
      ↓
Predicted Token Distribution
      ↓
Cross-Entropy Loss
      ↓
Backpropagation
      ↓
Optimizer Step (Adam) + Gradient Clipping
```

Gradients are clipped to a max norm of 1.0 — standard practice for RNNs, which are prone to exploding gradients.

---

## Inference Pipeline (Greedy Decoding)

Unlike training, inference has no ground truth — the decoder feeds its own previous prediction back into itself at every step:

```text
<sos>
  ↓
Decoder + Attention → token₁
  ↓
Decoder + Attention → token₂
  ↓
   ...
  ↓
Decoder + Attention → <eos>
```

Generation stops once `<eos>` is produced (or `max_len` is reached).

---

## Attention Visualization

For a reversal task the correct alignment is known in advance: output step $t$ should attend almost entirely to input position $T-1-t$. The notebook plots the real learned attention matrix with actual token labels on both axes — here's the model's attention on a held-out example after training:

```text
                j     d     d     f     j     i   <eos>
   j         ▓▓▓▓▓   ·     ·     ·     ·     ·     ·
   d           ·   ▓▓▓▓▓   ·     ·     ·     ·     ·
   d           ·     ·   ▓▓▓▓▓   ·     ·     ·     ·
   f           ·     ·     ·   ▓▓▓▓▓   ·     ·     ·
   j           ·     ·     ·     ·   ▓▓▓▓▓   ·     ·
   i           ·     ·     ·     ·     ·   ▓▓▓▓▓    ·
 <eos>         ·     ·     ·     ·     ·     ·     ▓▓
```

A clean **anti-diagonal band** — the model learned, purely from data, exactly the alignment a reversal task requires, with no explicit supervision on *what* to attend to, only on the final output sequence. The notebook's actual output is a continuous-valued heatmap (matplotlib `viridis`), not this ASCII approximation.

---

## Results

Trained for 10 epochs on 4,000 generated examples (batch size 64, Adam, lr = 1e-3):

| Epoch | Train Loss | Val Loss |
|------:|-----------:|---------:|
| 1     | 2.1267     | 1.7098   |
| 2     | 1.1969     | 0.5347   |
| 3     | 0.3401     | 0.0925   |
| 5     | 0.0715     | 0.0177   |
| 10    | 0.0112     | 0.0017   |

**Validation accuracy:** 64/64 (100%) exact-match sequence reversal on a held-out batch after training.

```text
src=jddfji       target=ijfddj       pred=ijfddj      ✓
src=eheccjidfj   target=jfdijccehe   pred=jfdijccehe  ✓
src=ghhbfdja     target=ajdfbhhg     pred=ajdfbhhg    ✓
```

---

## Hyperparameters

```python
EMB_DIM    = 32
ENC_HIDDEN = 64
DEC_HIDDEN = 64
ATTN_DIM   = 32

Optimizer  = Adam (lr=1e-3)
Loss       = CrossEntropyLoss(ignore_index=PAD)
Teacher Forcing Ratio = 0.5 → 0.1 (linear decay across epochs)
Gradient Clipping = max_norm 1.0
```

**Total trainable parameters: 99,213**

---

## Shapes Throughout the Network

| Stage | Shape |
|---|---|
| Input tokens | `(batch_size, src_len)` |
| After embedding | `(batch_size, src_len, emb_dim)` |
| Encoder outputs | `(batch_size, src_len, 2 × hidden_dim)` |
| Attention weights | `(batch_size, src_len)` — sums to 1 |
| Context vector | `(batch_size, 2 × hidden_dim)` |
| Decoder step output (logits) | `(batch_size, vocab_size)` |

---

## Features

- Bidirectional GRU encoder with packed-sequence handling for variable-length inputs
- Bahdanau additive attention with explicit padding mask
- GRU decoder with a "deep output" layer (sees hidden state + context + embedding)
- Teacher forcing with linear decay
- Masked cross-entropy loss + gradient clipping
- Greedy decoding for inference
- Attention-weight heatmap visualization with real token labels
- Pure PyTorch — no seq2seq or attention libraries used

---

## Repository Structure

```text
.
├── seq2seq_bahdanau_attention.ipynb   # full notebook: concept, code, training, viz
└── README.md
```

---

## Getting Started

```bash
pip install torch numpy matplotlib
jupyter notebook seq2seq_bahdanau_attention.ipynb
```

Run all cells top to bottom — the dataset is generated on the fly, so no external data download is required. Training the toy task to convergence takes well under a minute on CPU.

---

## Extending to Real Translation

- Replace `ReverseDataset` / `make_example()` with a real parallel-corpus loader (e.g. tokenized sentence pairs from a dataset like Multi30k), with separate source/target vocabularies
- Increase `EMB_DIM` / `ENC_HIDDEN` / `DEC_HIDDEN`, stack more GRU/LSTM layers
- Replace greedy decoding with beam search for better output quality
- Add subword tokenization (e.g. BPE) so the vocabulary generalizes beyond a fixed token set
- This attention mechanism is exactly what motivated the Transformer's self-attention — once this notebook feels intuitive, the original Transformer paper's attention section should feel very familiar

---

## References

- Bahdanau, D., Cho, K., & Bengio, Y. (2014). *Neural Machine Translation by Jointly Learning to Align and Translate.* [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
- Sutskever, I., Vinyals, O., & Le, Q. (2014). *Sequence to Sequence Learning with Neural Networks.* [arXiv:1409.3215](https://arxiv.org/abs/1409.3215)

---

## Summary

This project implements the classical Seq2Seq encoder-decoder architecture with Bahdanau attention, entirely from scratch in PyTorch. The encoder produces a contextual representation for every input token, attention dynamically selects which of those representations matter at each decoding step, and the decoder generates the output sequence one token at a time — conditioning on a context vector that changes every step rather than a single fixed summary.

This architecture was one of the first major breakthroughs in neural machine translation and laid the conceptual foundation for the attention mechanisms used in modern Transformers.
