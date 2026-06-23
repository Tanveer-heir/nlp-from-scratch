# BiLSTM-CRF for Named Entity Recognition

> **Sequence labeling done right** — contextual representations from a Bidirectional LSTM, globally consistent predictions from a hand-implemented CRF, and efficient inference via Viterbi decoding. No `torchcrf`. Every algorithm written from scratch.

---

## Why This Architecture?

A plain classifier predicts each token's tag **independently**. It has no way to know that `I-PER` after `O` is structurally invalid, or that `B-PER → I-LOC` is a type mismatch. It gets individual tokens right but produces tag *sequences* that violate BIO constraints.

The **BiLSTM-CRF** fixes this at two levels:

| Component | Role |
|-----------|------|
| **BiLSTM** | Reads context from both directions; gives every token a rich representation of its surroundings |
| **CRF** | Scores entire sequences using a learned transition matrix; enforces tag-to-tag consistency |
| **Viterbi** | Finds the globally optimal sequence at inference time (not just the best tag per position) |

---

## The BIO Tagging Scheme

Each token gets exactly one label:

```
Barack  Obama  visited  Berlin  yesterday
B-PER   I-PER     O     B-LOC      O
```

- `B-X` — **B**egins an entity of type X  
- `I-X` — **I**nside a continuing entity of type X (must follow `B-X` or `I-X`)  
- `O`   — **O**utside any entity

The CRF learns the BIO constraints automatically from data — they are never hard-coded (except for the `START`/`END` boundary tokens).

---

## Architecture

```
Input tokens
     │
     ▼
Embedding Layer          token index → dense vector  (vocab_size × embed_dim)
     │
     ▼
BiLSTM                   forward + backward LSTM, concat hidden states
     │                   output shape: (batch, seq_len, 2 × hidden_dim)
     ▼
Linear Projection        (2 × hidden_dim) → num_tags   ← emission scores
     │
     ▼
CRF Layer
  ├─ Training  →  Forward Algorithm   →  log Z  →  NLL loss
  └─ Inference →  Viterbi Algorithm   →  best tag sequence
```

---

## Mathematical Foundation

### Sequence Score

For a tag sequence **y** = (y₀, y₁, …, y_{T-1}):

```
score(y) = T[START → y₀]
         + Σ_{t=0}^{T-1} E[t, yₜ]          ← emission terms
         + Σ_{t=1}^{T-1} T[y_{t-1} → yₜ]  ← transition terms
         + T[y_{T-1} → END]
```

- **E[t, k]** — how strongly the BiLSTM thinks token `t` carries tag `k`  
- **T[i, j]** — the CRF's learned score for transitioning from tag `i` to tag `j`

### Training Objective — Negative Log-Likelihood

```
Loss = log Z − score(y*)

where  Z = log Σ_{all y} exp(score(y))   ← log-partition function
```

Minimising NLL pushes the score of the correct sequence up and the scores of all other sequences down.

---

## Core Algorithms (hand-implemented, no `torchcrf`)

### Forward Algorithm — Computing log Z

Naively summing over all sequences is **exponential** in sequence length. The Forward Algorithm computes log Z in **O(T × K²)** using dynamic programming:

```
α[0, k]  =  T[START → k]  +  E[0, k]                        # initialise

α[t, k]  =  logsumexp_j ( α[t-1, j]  +  T[j, k] )  +  E[t, k]   # recurrence

log Z    =  logsumexp_k ( α[T-1, k]  +  T[k → END] )         # finalise
```

**Numerical stability** — the log-sum-exp trick prevents floating-point overflow:

```
logsumexp(a) = max(a) + log Σ exp(a − max(a))
```

Subtracting the maximum keeps all `exp()` arguments ≤ 0.

### Viterbi Algorithm — Best Sequence at Inference

Identical recurrence to the Forward Algorithm, with two changes:

| | Forward | Viterbi |
|--|---------|---------|
| Operator | `logsumexp` | `max` |
| Extra bookkeeping | — | store `argmax` as back-pointer |

```
δ[0, k]  =  T[START → k]  +  E[0, k]

δ[t, k]  =  max_j ( δ[t-1, j]  +  T[j, k] )  +  E[t, k]
                └── save argmax as back-pointer ──┘

best_path  =  backtrack from argmax at T-1
```

Both algorithms run in **O(T × K²)** — efficient for real-world tag sets.

---

## Implementation Details

### CRF Class

```python
class CRF(nn.Module):
    # Learnable transition matrix: (num_tags × num_tags)
    # Hard constraints baked in at init:
    #   transitions[:, START] = -10000   # nothing → START
    #   transitions[END, :]   = -10000   # END → nothing

    def _score_sentence(emissions, tags)   # gold sequence score
    def _forward_alg(emissions)            # log Z via DP
    def neg_log_likelihood(emissions, tags)# log Z − score(gold)
    def viterbi_decode(emissions)          # best path + score
```

### BiLSTM-CRF Class

```python
class BiLSTMCRF(nn.Module):
    # Embedding → Dropout → BiLSTM → Dropout → Linear → CRF

    def forward(x, tags, lengths)   # returns NLL loss (training)
    def predict(x, lengths)         # returns Viterbi tag sequences (inference)
```

Sequences are packed with `pack_padded_sequence` so the LSTM never processes padding tokens.

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding dim | 64 |
| LSTM hidden dim | 128 |
| LSTM layers | 1 |
| Dropout | 0.3 |
| Optimizer | Adam (lr=1e-3, wd=1e-4) |
| LR scheduler | StepLR (step=30, γ=0.5) |
| Gradient clip | 5.0 |
| Epochs | 80 |

---

## Notebook Walkthrough

| Section | Contents |
|---------|----------|
| **1 — Setup** | Imports, seeds, device selection |
| **2 — Dataset** | 10 BIO-tagged toy sentences, vocabulary building, `word2idx` / `tag2idx` |
| **3 — DataLoader** | `NERDataset`, `collate_fn` with padding, `pack_padded_sequence` |
| **4 — CRF Layer** | `_score_sentence`, `_forward_alg`, `viterbi_decode` — fully annotated |
| **5 — BiLSTM-CRF** | Full model combining all components |
| **6 — Instantiation** | Model summary, parameter count |
| **7 — Training** | Training loop with gradient clipping and LR scheduling |
| **8 — Loss Curve** | Matplotlib plot of NLL over epochs |
| **9 — Evaluation** | Token-level accuracy on training set |
| **10 — Predictions** | Qualitative entity extraction on sample sentences |
| **11 — Transition Matrix** | Heatmap of learned tag-to-tag transition scores |
| **12 — Viterbi Trace** | Step-by-step DP table printed for one sentence |
| **13 — Summary** | Key concepts, equations, complexity table |

---

## Example Output

**Input:**
```
Barack Obama visited Berlin yesterday
```

**Predicted:**
```
Barack      B-PER  ←
Obama       I-PER  ←
visited     O
Berlin      B-LOC  ←
yesterday   O
```

**Extracted entities:**
```python
{"Person": ["Barack Obama"], "Location": ["Berlin"]}
```

---

## Transition Matrix (learned)

The CRF learns that certain transitions are valid and others are not:

| Transition | Learned Score | Interpretation |
|------------|:-------------:|----------------|
| `B-PER → I-PER` | high positive | valid continuation |
| `B-LOC → I-LOC` | high positive | valid continuation |
| `O → I-PER` | very negative | can't continue without beginning |
| `B-PER → I-LOC` | very negative | type mismatch |
| `START → I-PER` | −10000 (hard) | I- can't start a sequence |

The heatmap in Section 11 makes these patterns visible.

---

## Complexity

```
Forward Algorithm:  O(T × K²)
Viterbi Decoding:   O(T × K²)

T = sequence length
K = number of tags
```

Compare to the naive exponential O(K^T) of enumerating all sequences.

---

## Requirements

```
torch >= 1.10
matplotlib
numpy
```

No external NLP libraries required. No `torchcrf`.

---

## References

- Lample et al. (2016) — *Bidirectional LSTM-CRF Models for Sequence Labeling* — [arXiv:1603.01360](https://arxiv.org/abs/1603.01360)
- Lafferty et al. (2001) — *Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data* — [link](https://repository.upenn.edu/cis_papers/159/)

---

## Project Structure

```
bilstm_crf_ner.ipynb   ← complete implementation + explanations
README.md              ← this file
```
