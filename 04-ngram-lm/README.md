# 📚 N-Gram Language Models with Smoothing
### Laplace Smoothing · Kneser-Ney Smoothing · Perplexity — From Scratch

> Statistical language models estimate the probability of word sequences using n-gram counts. MLE suffers from zero probabilities, Laplace smoothing fixes this but over-smooths, and Kneser-Ney provides a principled redistribution of probability mass — achieving superior perplexity and becoming the gold standard of classical language modeling.

---

## 📌 What You'll Learn

1. What an N-gram Language Model is and *why* it works
2. How to estimate probabilities using **Maximum Likelihood Estimation (MLE)**
3. The **zero-probability problem** and why smoothing is necessary
4. **Laplace (Add-1) Smoothing** — mechanics + worked example
5. **Kneser-Ney Smoothing** — the state-of-the-art classic, step by step
6. **Perplexity** — how we measure model quality, computed by hand
7. Full Python implementation with visualisations and reusable classes

---

## 🗂️ Project Structure

```
ngram_lm_smoothing.ipynb
│
├── 1. Corpus Preparation
├── 2. N-Gram Counting
├── 3. Maximum Likelihood Estimation (MLE)
├── 4. Zero Probability Problem
├── 5. Laplace (Add-1) Smoothing
├── 6. Kneser-Ney Smoothing
├── 7. Probability Visualization
├── 8. Perplexity Evaluation
└── 9. Reusable Language Model Classes
```

---

## 🧠 1. What is a Language Model?

A **Language Model (LM)** assigns a probability to a sequence of words:

$$P(w_1, w_2, \ldots, w_n)$$

**Example:**

```
the cat sat on the mat      → High probability  ✅ (natural sentence)
mat the sat cat on          → Low  probability  ❌ (unlikely sequence)
```

Using the **Chain Rule of Probability**:

$$P(w_1 w_2 \cdots w_n) = P(w_1)\;P(w_2|w_1)\;P(w_3|w_1 w_2)\;\cdots$$

Conditioning on the entire history is computationally intractable, so we apply the **Markov Assumption**.

---

## 🔗 2. The Bigram (Markov) Assumption

Assume each word depends only on the **one previous word**:

$$P(w_i \mid w_1 \cdots w_{i-1}) \approx P(w_i \mid w_{i-1})$$

So the sentence `the cat sat on the mat` becomes:

$$P(\text{the}) \cdot P(\text{cat}|\text{the}) \cdot P(\text{sat}|\text{cat}) \cdot P(\text{on}|\text{sat}) \cdot P(\text{the}|\text{on}) \cdot P(\text{mat}|\text{the})$$

| Model | Conditions on | Example |
|-------|-------------|---------|
| **Unigram** | nothing | $P(\text{dog})$ |
| **Bigram** | 1 previous word | $P(\text{dog} \mid \text{the})$ |
| **Trigram** | 2 previous words | $P(\text{dog} \mid \text{chase the})$ |
| **N-gram** | n−1 previous words | general case |

---

## 📊 3. N-Gram Counting

Given corpus:

```
<s> the cat sat </s>
<s> the cat ate </s>
```

Bigrams extracted:

```
(the, cat)
(cat, sat)
(cat, ate)
```

Bigram counts:

```python
('the', 'cat') : 2
('cat', 'sat') : 1
('cat', 'ate') : 1
```

Context (unigram) counts:

```python
'the' : 2
'cat' : 2
```

---

## 📈 4. Maximum Likelihood Estimation (MLE)

We estimate probabilities by **counting** in the training corpus:

$$P_{MLE}(w_i \mid w_{i-1}) = \frac{C(w_{i-1},\; w_i)}{C(w_{i-1})}$$

### Worked Example

```
C(the, cat) = 2,  C(the) = 2   →   P(cat | the) = 2/2 = 1.0
C(cat, sat) = 1,  C(cat) = 2   →   P(sat | cat) = 1/2 = 0.5
```

### MLE Code

```python
def mle_bigram_prob(word, context, bigram_counts, unigram_counts):
    num = bigram_counts.get((context, word), 0)
    den = unigram_counts.get((context,), 0)
    return num / den if den > 0 else 0.0
```

---

## 🚨 5. The Zero-Probability Problem

If a bigram **never appeared** in training, MLE gives it probability **0**.

### Example

Test sentence: `the dog sat on the mat`

```
C(the, dog) = 0   →   P(dog | the) = 0/2 = 0
```

This causes two disasters:

1. **The entire sentence gets probability 0**, even if it's perfectly grammatical — because `anything × 0 = 0`
2. **Perplexity becomes ∞** (log(0) is undefined)

**Solution: Smoothing** — steal a small amount of probability mass from seen events and redistribute it to unseen ones.

---

## 🔵 6. Laplace (Add-1) Smoothing

The simplest fix: **pretend every n-gram was seen at least once** by adding 1 to every count.

$$P_{Laplace}(w_i \mid w_{i-1}) = \frac{C(w_{i-1},\; w_i) + 1}{C(w_{i-1}) + V}$$

Where $V$ = vocabulary size (number of unique word types).

**General Add-k form (Lidstone smoothing):**

$$P_{Add\text{-}k}(w_i \mid w_{i-1}) = \frac{C(w_{i-1},\; w_i) + k}{C(w_{i-1}) + k \cdot V}$$

### Worked Example

```
Vocabulary V = 4  {cat, dog, sat, ate}
C(the, dog) = 0,  C(the) = 2

P_Laplace(dog | the) = (0 + 1) / (2 + 4) = 1/6 ≈ 0.167
```

Unseen bigrams now receive a non-zero probability.

### Laplace Code

```python
def laplace_bigram_prob(word, context, bigram_counts, unigram_counts, vocab, k=1):
    V   = len(vocab)
    num = bigram_counts.get((context, word), 0) + k
    den = unigram_counts.get((context,), 0)  + k * V
    return num / den
```

### ⚠️ Weakness of Laplace

Laplace assigns probability to **every possible bigram**, including obvious nonsense:

```
the airplane
the banana
the quantum
```

For a typical vocabulary of 50,000 words, the bigram table has **2.5 billion** entries. Adding 1 to all of them **steals too much probability mass** from observed events. Laplace over-smooths.

---

## 🟣 7. Kneser-Ney Smoothing

Kneser-Ney (KN) is the **gold-standard classical smoothing method**. It has two key innovations over Laplace.

### Innovation 1 — Absolute Discounting

Instead of adding counts, **subtract a fixed discount** $d \approx 0.75$ from every non-zero count:

$$\frac{\max(C(w_{i-1}, w_i) - d,\; 0)}{C(w_{i-1})}$$

Example: if `C(the, cat) = 5`, discounted count = `5 − 0.75 = 4.25`. The removed mass is redistributed to unseen bigrams.

### Innovation 2 — Continuation Probability

The lower-order distribution is not simply $P(w_i)$. Instead, it asks: **how many unique contexts does word $w_i$ appear in?**

$$P_{cont}(w_i) = \frac{|\{v : C(v, w_i) > 0\}|}{\sum_{w} |\{v : C(v, w) > 0\}|}$$

**Intuition:**

| Word | Contexts | Continuation Probability |
|------|---------|--------------------------|
| *Francisco* | only after *San* | **Low** — shouldn't appear standalone |
| *Tuesday* | after *last*, *next*, *this*, ... | **High** — versatile word |

This is more principled than raw frequency — a word that's very frequent but only in one context (like *Francisco*) gets a low standalone probability.

### Full KN Formula

$$P_{KN}(w_i \mid w_{i-1}) = \frac{\max(C(w_{i-1},w_i)-d,\;0)}{C(w_{i-1})} + \lambda(w_{i-1})\; P_{cont}(w_i)$$

Where the **back-off weight** $\lambda$ ensures probabilities sum to 1:

$$\lambda(w_{i-1}) = \frac{d \cdot |\{w : C(w_{i-1}, w) > 0\}|}{C(w_{i-1})}$$

### Step-by-Step Worked Example: $P_{KN}(\text{cat} \mid \text{the})$

```
Discount d            = 0.75
C(the, cat)           = 2
C(the)                = 3

Discounted term       = max(2 − 0.75, 0) / 3  = 1.25 / 3  = 0.4167

Types after 'the'     = 4   (cat, rat, dog, mat)
λ(the)                = 0.75 × 4 / 3           = 1.0

P_cont(cat)           = 2 unique left contexts / total = 0.1176

P_KN(cat | the)       = 0.4167 + 1.0 × 0.1176  = 0.5343
```

### Kneser-Ney Code

```python
def build_kneser_ney(token_sents, discount=0.75):
    bigram_c  = defaultdict(int)
    unigram_c = defaultdict(int)
    left_contexts = defaultdict(set)

    for sent in token_sents:
        for i in range(len(sent) - 1):
            w_prev, w_curr = sent[i], sent[i+1]
            bigram_c[(w_prev, w_curr)] += 1
            unigram_c[w_prev] += 1
            left_contexts[w_curr].add(w_prev)

    cont_count      = {w: len(ctxs) for w, ctxs in left_contexts.items()}
    total_cont      = sum(cont_count.values())

    def p_cont(word):
        return cont_count.get(word, 0) / total_cont if total_cont > 0 else 0

    def lambda_weight(context):
        n_types = sum(1 for (c, _) in bigram_c if c == context)
        denom   = unigram_c.get(context, 0)
        return (discount * n_types / denom) if denom > 0 else 0

    def kn_prob(word, context):
        c_ctx_word = bigram_c.get((context, word), 0)
        c_ctx      = unigram_c.get(context, 0)
        if c_ctx == 0:
            return p_cont(word)
        return max(c_ctx_word - discount, 0) / c_ctx + lambda_weight(context) * p_cont(word)

    return kn_prob
```

---

## 📐 8. Perplexity

**Perplexity (PP)** measures how *surprised* a model is by a test sentence. Lower = better.

$$PP(W) = P(w_1 w_2 \cdots w_N)^{-1/N} = \left(\prod_{i=1}^{N} P(w_i \mid w_{i-1})\right)^{-1/N}$$

**Log-space form (numerically stable):**

$$PP(W) = 2^{\,-\frac{1}{N}\sum_{i=1}^{N} \log_2 P(w_i \mid w_{i-1})}$$

### Intuition

| Perplexity | Meaning |
|-----------|---------|
| PP = 1 | Perfect predictor — always knows the next word |
| PP = 2 | As confused as picking between 2 equally likely words |
| PP = 100 | As confused as picking from 100 equally likely words |
| PP = V | Completely random — equivalent to a uniform distribution |

### Worked Example: `<s> the cat sat on the mat </s>`

```
Step    Context → Word        P(w|ctx)      log₂ P
────────────────────────────────────────────────────
  1     <s>     → the         0.800000      -0.3219
  2     the     → cat         0.533430      -0.9069
  3     cat     → sat         0.462500      -1.1133
  4     sat     → on          0.534375      -0.9044
  5     on      → the         0.750000      -0.4150
  6     the     → mat         0.180180      -2.4737
  7     mat     → </s>        0.534375      -0.9044

N = 7,   Σ log₂P = -7.031
Avg log₂P = -7.031 / 7 = -1.004

Perplexity = 2^(1.004) ≈ 2.006
```

### Perplexity on an Unseen Sentence

```
Sentence: <s> the dog sat on the mat </s>

MLE:        P(dog | the) = 0  →  PP = ∞      ❌ Fails completely
Laplace:    P(dog | the) = 1/6 → PP = finite  ✅ Works
Kneser-Ney: P(dog | the) > 0  → PP = lowest  ✅ Best
```

### Perplexity Code

```python
def sentence_perplexity(sentence_tokens, prob_fn):
    log_prob_sum, N = 0.0, 0
    for i in range(1, len(sentence_tokens)):
        p = max(prob_fn(sentence_tokens[i], sentence_tokens[i-1]), 1e-10)
        log_prob_sum += np.log2(p)
        N += 1
    return 2 ** (-log_prob_sum / N)
```

---

## 🛠️ 9. Reusable Classes

The notebook implements all three models as clean, reusable classes:

```python
class NGramLM:
    def __init__(self, n=2, smoothing="kneser-ney", discount=0.75, k=1): ...
    def fit(self, sentences): ...
    def prob(self, word, context): ...
    def perplexity(self, sentence): ...
```

**Usage:**

```python
sents = [
    "<s> the cat sat on the mat </s>",
    "<s> the cat ate the rat </s>",
    ...
]

model = NGramLM(smoothing="kneser-ney").fit(sents)
print(model.prob("cat", "the"))          # 0.5343
print(model.perplexity("<s> the cat sat on the mat </s>"))  # 2.006
```

Available smoothing options: `"mle"` · `"laplace"` · `"kneser-ney"`

---

## 📊 10. Model Comparison

| | **MLE** | **Laplace** | **Kneser-Ney** |
|---|---|---|---|
| **Formula** | $C(w,c)/C(c)$ | $(C+1)/(C+V)$ | discounting + continuation |
| **Zero probabilities?** | ✅ Yes | ❌ No | ❌ No |
| **Over-smoothing?** | N/A | ⚠️ Yes | ✅ Minimal |
| **Perplexity** | Worst / ∞ | Better | Best |
| **Handles unseen words?** | ❌ No | ✅ Yes | ✅ Yes |
| **Complexity** | O(N) | O(N) | O(N) |
| **Real-world use** | Rarely | Rarely | Industry standard |

---

## 🔄 Pipeline Summary

```
Raw Corpus
    ↓
Tokenisation  (<s> / </s> markers added)
    ↓
Bigram Counts  C(w_{i-1}, w_i)
    ↓
MLE  →  Zero Probability Problem  ← fails on unseen bigrams
    ↓
Laplace Smoothing  →  Over-smoothing  ← steals too much mass
    ↓
Kneser-Ney Smoothing  →  Continuation Probability + Absolute Discounting
    ↓
Probability Distribution Visualisation
    ↓
Perplexity Evaluation
    ↓
Best Language Model  ✅
```

---

## 🕰️ Historical Context

Before RNNs and Transformers, statistical n-gram models dominated NLP for decades.

```
N-Gram LMs  (1980s–1990s)
    ↓
Kneser-Ney  (1995)  ← gold standard of classical LMs
    ↓
RNN Language Models  (2010)
    ↓
LSTMs / GRUs  (2013–2015)
    ↓
Transformers  (2017)
    ↓
Large Language Models  (2018–present)
```

Kneser-Ney remained the dominant language modeling technique for **over 20 years** and is still used as a baseline and in hybrid systems today.

---

## 🚀 Getting Started

### Requirements

```bash
pip install numpy pandas matplotlib jupyter
```

### Run the Notebook

```bash
jupyter notebook ngram_lm_smoothing.ipynb
```

### Quick Start

```python
from collections import defaultdict
import numpy as np

# 1. Prepare corpus
corpus = [
    "<s> the cat sat on the mat </s>",
    "<s> the cat ate the rat </s>",
]

# 2. Fit model
model = NGramLM(smoothing="kneser-ney").fit(corpus)

# 3. Get probability
p = model.prob("cat", "the")       # P(cat | the)

# 4. Evaluate perplexity
pp = model.perplexity("<s> the cat sat on the mat </s>")
```

---

## 📐 Key Equations — Quick Reference

**MLE:**
$$P_{MLE}(w_i|w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$$

**Laplace:**
$$P_{Lap}(w_i|w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + V}$$

**Kneser-Ney:**
$$P_{KN}(w_i|w_{i-1}) = \frac{\max(C(w_{i-1},w_i)-d,\;0)}{C(w_{i-1})} + \underbrace{\frac{d \cdot |\text{types after }w_{i-1}|}{C(w_{i-1})}}_{\lambda(w_{i-1})} \cdot P_{cont}(w_i)$$

$$P_{cont}(w_i) = \frac{|\{v: C(v,w_i)>0\}|}{\sum_w|\{v:C(v,w)>0\}|}$$

**Perplexity:**
$$PP(W) = 2^{-\frac{1}{N}\sum_{i=1}^{N}\log_2 P(w_i|w_{i-1})}$$

---

*Built from scratch — no NLTK LM module used. All probability mechanics implemented manually for maximum transparency.*
