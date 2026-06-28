# Lab 14 -- Decoding Strategies from Scratch

A complete, self-contained notebook that builds a tiny GPT from scratch and implements every major decoding strategy, then visualises how each one changes output diversity, probability distributions, and generation behaviour.

---

## What This Lab Covers

Once a Transformer predicts probabilities for every possible next token, how do we choose which token to generate? A Transformer never directly outputs text. It outputs logits (unnormalized scores) for every token in the vocabulary. Those logits are converted into probabilities via Softmax, and a decoding strategy decides which token to emit. This lab implements and compares five such strategies.

The Transformer architecture stays unchanged throughout. Only the final decoding algorithm varies.

---

## Generation Pipeline

```
Input Text
    |
    v
Tokenization
    |
    v
Embedding
    |
    v
Transformer
    |
    v
Vocabulary Logits
    |
    v
Softmax
    |
    v
Probability Distribution
    |
    v
Decoding Strategy
    |
    v
Generated Token
```

---

## Notebook Sections

| # | Section | Description |
|---|---|---|
| 1 | Imports and Reproducibility | PyTorch, NumPy, Matplotlib, seed=42 |
| 2 | Tiny Dataset | Character-level corpus, vocab size 32 |
| 3 | Tiny GPT | Full decoder-only Transformer, ~110K params |
| 4 | Training | AdamW, CosineAnnealing, 600 steps |
| 5 | Greedy Decoding | argmax at every step |
| 6 | Beam Search | Width=4, length penalty, ranked beams |
| 7 | Temperature Sampling | Logit scaling by T |
| 8 | Top-K Sampling | Keep K highest-prob tokens |
| 9 | Top-P (Nucleus) Sampling | Keep smallest set covering P mass |
| 10 | Distribution Visualisation | 6-panel bar charts with entropy annotations |
| 11 | Diversity Measurement | Unique outputs, TTR, entropy across 50 runs |
| 12 | Diversity Dashboard | Horizontal bars + TTR vs Entropy scatter |
| 13 | Temperature Sweep | Continuous curve from T=0.05 to T=2.5 |
| 14 | Top-K Sweep | K from 1 to 50 |
| 15 | Top-P Sweep | p from 0.1 to 1.0 |
| 16 | Token Probability Heatmap | Probability mass across decoding steps |
| 17 | Side-by-Side Comparison | All strategies on the same prompts |
| 18 | Combined Strategy | Temperature + Top-K + Top-P pipeline |
| 19 | Summary Cheat Sheet | When to use each strategy |

---

## The Five Decoding Strategies

### 1. Greedy Decoding

Always picks the token with the highest probability.

```
predicted = argmax(P(w | context))
```

Pros: fast, deterministic, reproducible.
Cons: repetitive, cannot recover from bad early choices.

---

### 2. Beam Search

Maintains B candidate sequences simultaneously. At each step every beam is expanded over the full vocabulary, and the top-B sequences by cumulative log-probability are kept.

```
score(w_1..t) = sum of log P(w_i | w_<i)
```

Pros: finds stronger sequences than greedy.
Cons: still deterministic, computationally heavier.

---

### 3. Temperature Sampling

Divides logits by T before Softmax.

```
P_T(w) = exp(z_w / T) / sum_v exp(z_v / T)
```

- T close to 0 -- near-deterministic, peaky distribution
- T = 1 -- original model distribution
- T greater than 1 -- flatter, more random, more creative

---

### 4. Top-K Sampling

Keeps only the K most probable tokens. Everything else is set to negative infinity before Softmax.

```
K=1   -->  greedy
K=V   -->  unrestricted sampling
```

Typical values in practice: 20 to 100.

---

### 5. Top-P (Nucleus) Sampling

Keeps the smallest set of tokens whose cumulative probability reaches or exceeds P, then samples from that set.

Example with P=0.90:

```
cat   0.45
dog   0.25  --> cumulative 0.70
frog  0.18  --> cumulative 0.88
lion  0.07  --> cumulative 0.95  (threshold crossed)
```

Nucleus = {cat, dog, frog, lion}

Pros: adapts nucleus size to distribution sharpness, widely used in production LLMs.

---

## Diversity Metrics

Three quantitative metrics are used instead of subjective comparison.

**Unique Outputs** -- how many distinct sequences appear across 50 repeated runs with the same prompt.

**Type-Token Ratio (TTR)** -- ratio of unique characters to total characters. Higher means richer vocabulary usage.

**Character Entropy** -- Shannon entropy of the character distribution across all generated text. Higher means more randomness.

---

## Observed Diversity Ranking

From least to most diverse:

```
Greedy
    |
Beam Search
    |
Low Temperature (T ~ 0.3)
    |
Small Top-K (K ~ 5)
    |
Moderate Top-P (p ~ 0.8)
    |
Large Top-K (K ~ 50)
    |
Temperature = 1.0
    |
Top-P = 1.0
    |
High Temperature (T > 1.5)
```

---

## Combined Strategy (Production Pattern)

Modern LLMs such as GPT-style models typically chain three filters:

```
Logits
    |
Temperature scaling
    |
Top-K filter
    |
Top-P (nucleus) filter
    |
Random sampling
```

Typical parameters:

```python
temperature = 0.9
top_k       = 50
top_p       = 0.92
```

This combination balances coherence and creativity and is implemented in Cell 18.

---

## Strategy Reference Table

| Strategy | Deterministic | Diversity | Speed | Best For |
|---|:---:|---|---|---|
| Greedy | Yes | Very low | Fastest | Factual QA, deterministic generation |
| Beam Search | Yes | Low | Slow | Translation, summarisation |
| Temperature | No | Adjustable | Fast | Creative writing |
| Top-K | No | Medium | Fast | Controlled randomness |
| Top-P | No | High, adaptive | Fast | Open-ended generation |
| Combined | No | High and tunable | Fast | Production LLMs |

---

## Model Architecture

The TinyGPT used throughout:

- Token embedding + learned positional embedding
- 2 Transformer blocks (multi-head self-attention + feed-forward)
- Causal (masked) attention so the model cannot peek at future tokens
- LayerNorm before each sub-layer (Pre-LN style)
- Weight tying between the embedding matrix and the output projection
- Approximately 110K parameters

Training uses cross-entropy loss on next-character prediction, AdamW with cosine annealing, for 600 steps.

---

## Requirements

```
torch
numpy
matplotlib
nbformat
jupyter
```

Install with:

```bash
pip install torch numpy matplotlib nbformat jupyter
```

---

## Key Takeaways

The Transformer produces a probability distribution. The decoding strategy determines whether the output is deterministic, repetitive, diverse, or creative. Changing only the decoding algorithm -- without retraining the model -- produces dramatically different text. The combined strategy (Temperature + Top-K + Top-P) is the practical recommendation for most generation tasks.
