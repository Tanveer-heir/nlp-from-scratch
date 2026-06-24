# 🔤 Byte Pair Encoding (BPE) Tokenizer — From Scratch

> *Learn a vocabulary of subwords by greedily merging the most frequent neighboring symbols. Common words become single tokens; rare words become reusable subword pieces — no `[UNK]` ever needed.*

---

## What is BPE?

**Byte Pair Encoding** is a subword tokenization algorithm — originally invented for data compression, repurposed for NLP by Sennrich et al. (2016). It powers the tokenizers behind:

| Model | Vocabulary Size |
|-------|----------------|
| GPT-2 / GPT-3 / GPT-4 | 50,257 |
| LLaMA / LLaMA-2 | 32,000 |
| RoBERTa | 50,265 |

---

## Why Not Just Use Words or Characters?

### ❌ Word-level Tokenization

Every unique word is a separate token. Vocabulary explodes. Unseen words become `[UNK]`:

```
newness  →  [UNK]
```

Information is permanently lost.

### ❌ Character-level Tokenization

No unknown words, but sequences become impractically long:

```
internationalization  →  i n t e r n a t i o n a l i z a t i o n
```

### ✅ BPE — The Sweet Spot

Learns common subwords automatically. Handles unseen words gracefully:

```
lowering  →  low  er  i  n  g
```

No `[UNK]`. Compact sequences. Shared subword pieces across related words.

---

## Algorithm — Step by Step

### Step 1 — Character Initialization

Every word is split into individual characters. A special `</w>` marker is appended to distinguish word-final characters from interior ones.

```
low      →  l o w </w>
lower    →  l o w e r </w>
newest   →  n e w e s t </w>
widest   →  w i d e s t </w>
```

The `</w>` marker is critical: it ensures that `er` at the end of `lower` and `er` at the start of `era` remain distinct pairs during training.

---

### Step 2 — Count Adjacent Pairs

For every word in the corpus, slide a window of size 2 and count each pair, weighted by the word's frequency:

| Pair | Count |
|------|-------|
| (l, o) | 3 |
| (o, w) | 3 |
| (e, s) | 2 |
| (s, t) | 2 |
| (e, r) | 2 |

---

### Step 3 — Merge the Most Frequent Pair

The most frequent pair is merged into a single new symbol everywhere in the corpus:

```
(l, o)  →  lo

l o w </w>      becomes    lo w </w>
l o w e r </w>  becomes    lo w e r </w>
```

---

### Step 4 — Repeat

Each iteration finds the new most-frequent pair and merges it:

```
Merge 1:  (l, o)    →  lo
Merge 2:  (lo, w)   →  low
Merge 3:  (e, s)    →  es
Merge 4:  (es, t)   →  est
Merge 5:  (e, r)    →  er
...
```

After enough merges, the corpus looks like:

```
low </w>
low er </w>
new est </w>
w i d est </w>
```

---

### Step 5 — Encode New Words

Apply the learned merge table **in order** to any new word:

```
Input:  lowest

Start:  l  o  w  e  s  t  </w>
        ↓  apply merge (l,o) → lo
        lo  w  e  s  t  </w>
        ↓  apply merge (lo,w) → low
        low  e  s  t  </w>
        ↓  apply merge (e,s) → es
        low  es  t  </w>
        ↓  apply merge (es,t) → est
        low  est  </w>

Tokens: [ low, est, </w> ]
```

Order of merges is not optional — it defines the tokenizer.

---

## Training Flow

```
Raw corpus
    │
    ▼
Split each word into characters + </w>
    │
    ▼
Count all adjacent symbol pairs
    │
    ▼
Find most frequent pair
    │
    ▼
Merge that pair everywhere → new symbol
    │
    ▼
Store merge rule
    │
    ▼
Repeat N times (N = your hyperparameter)
    │
    ▼
Final vocabulary + ordered merge table
```

---

## Encoding Flow

```
New input word
    │
    ▼
Split into characters + </w>
    │
    ▼
Apply merge rule #1 (if pair exists)
    │
    ▼
Apply merge rule #2 (if pair exists)
    │
    ▼
... (apply all N rules in order)
    │
    ▼
Subword tokens
```

---

## Example: Unseen Words

The tokenizer was trained on `low`, `lower`, `new`, `newest`, `widest`. It never saw `lowering`. BPE still handles it:

```
lowering  →  l o w e r i n g
             ↓ merges applied
             low  er  i  n  g
```

No `[UNK]`. No information loss.

---

## Why the `</w>` Marker Matters

Without it, `er` in `lower` and `er` in `era` would be treated identically, merging indiscriminately. With `</w>`:

```
lower  →  l o w e r </w>      ← (r, </w>) is a pair
era    →  e r a </w>          ← (e, r) is a pair — different!
```

This gives BPE positional awareness within words.

---

## Why Order of Merges Matters

Merges are applied sequentially, not as a lookup table:

```
Merge #3:  (e, s) → es
Merge #7:  (es, t) → est

Correct:   e s t  → [apply #3] → es t  → [apply #7] → est  ✓
Wrong:     trying #7 first → no (es,t) pair exists yet        ✗
```

The merge table is an **ordered program**.

---

## Effect of `num_merges` Hyperparameter

```
num_merges=0    widest → [ w  i  d  e  s  t  </w> ]   7 tokens
num_merges=2    widest → [ w  i  d  es  t  </w> ]      6 tokens
num_merges=5    widest → [ w  i  d  est  </w> ]         5 tokens
num_merges=10   widest → [ w  i  d  est</w> ]           4 tokens
num_merges=15   widest → [ wid  est</w> ]               2 tokens
```

More merges = fewer, longer tokens = more compressed representation.

---

## Comparison with Other Tokenizers

| Method | Vocab Size | Handles OOV? | Used By |
|--------|-----------|--------------|---------|
| Word-level | 50k–500k | ❌ `[UNK]` | Classic NLP |
| Character-level | ~256 | ✅ | Some older models |
| **BPE** | 8k–50k | ✅ chars as fallback | GPT-2/3/4, LLaMA |
| WordPiece | 8k–30k | ✅ | BERT |
| SentencePiece | 8k–64k | ✅ | T5, XLNet |

---

## Project Structure

```
bpe_tokenizer_from_scratch.ipynb
│
├── Part 1 — Build Initial Vocabulary
│            get_vocab(): word → char sequence with </w>
│
├── Part 2 — Count Pair Frequencies
│            get_pair_stats(): adjacent symbol pair counts
│
├── Part 3 — Merge Step
│            merge_vocab(): apply one merge across full vocab
│
├── Part 4 — Full Training Loop
│            train_bpe(): run N merges, record merge table
│
├── Part 5 — Encoding New Words
│            encode_word(): apply merge table in order
│
├── Part 6 — Decoding
│            decode(): tokens → original string via </w>
│
├── Part 7 — BPETokenizer Class
│            .fit() / .encode() / .decode() / token2id / id2token
│
├── Part 8 — Effect of num_merges
│            same word, different compression at different merge counts
│
└── Part 9 — Larger Corpus Demo + Key Concepts Summary
```

---

## Quick Start

```python
from collections import defaultdict
import re

# 1. Prepare corpus
corpus = ['low'] * 5 + ['lower'] * 2 + ['newest'] * 3 + ['widest'] * 2

# 2. Train
tok = BPETokenizer(num_merges=15)
tok.fit(corpus)

# 3. Encode
tokens = tok.encode('lower newest')
# → ['low', 'er', '</w>', 'new', 'est', '</w>']

# 4. Get integer IDs
ids = tok.encode('lower newest', return_ids=True)
# → [4, 2, 0, 6, 3, 0]

# 5. Decode back
text = tok.decode(ids)
# → 'lower newest'
```

---

## Key Properties

- **No external libraries** — pure Python stdlib (`re`, `collections`)
- **No `[UNK]` token** — any word decomposes to characters at worst
- **Fully reproducible** — merge table completely defines the tokenizer
- **Generalizes to unseen words** — applies learned subword structure
- **Invertible** — `decode(encode(text)) == text` always holds

---

## References

- Sennrich, Rico et al. (2016). *Neural Machine Translation of Rare Words with Subword Units*. ACL 2016.
- Radford, Alec et al. (2019). *Language Models are Unsupervised Multitask Learners* (GPT-2).
- HuggingFace `tokenizers` library
- Karpathy, Andrej — *minBPE* (clean reference implementation)
