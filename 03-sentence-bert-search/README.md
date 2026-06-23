# Semantic Search from Scratch: TF-IDF → BM25 → SBERT → FAISS

A complete implementation and benchmark of modern information retrieval techniques, progressing from
traditional keyword search to transformer-based semantic retrieval and scalable vector search — all in a
single, runnable Jupyter notebook.

This project demonstrates:

* **TF-IDF** — sparse lexical retrieval
* **BM25** — probabilistic lexical ranking
* **SBERT (Sentence-BERT)** — dense semantic embeddings
* **FAISS** — scalable approximate nearest-neighbor search
* Pooling strategies and similarity metrics
* A hand-labeled evaluation set with **Precision@k**, **MRR**, and **latency** benchmarks
* The foundations of **RAG** (Retrieval-Augmented Generation)

📓 **Notebook:** [`sbert_semantic_search.ipynb`](./sbert_semantic_search.ipynb)

---

## Overview

Given a user query:

```text
I forgot my login credentials, how can I get back into my account?
```

a good search engine should retrieve:

```text
How do I reset my password if I forgot it?
Steps to recover your account after losing access to your email.
```

even though the query shares almost **no vocabulary** with either document. This is the central problem
this repo explores: lexical search methods (TF-IDF, BM25) only match exact words, while semantic methods
(SBERT) match *meaning*.

The notebook builds and compares four retrieval pipelines on the same 25-document corpus:

```text
TF-IDF
   ↓
BM25
   ↓
SBERT
   ↓
SBERT + FAISS
```

---

## Project Structure

The notebook (`sbert_semantic_search.ipynb`) is organized as:

```text
1.  Concept overview
    • Why raw BERT embeddings are bad at sentence similarity
    • Bi-encoders vs. cross-encoders
    • TF-IDF and BM25 fundamentals

2.  Setup
    • sentence-transformers, rank_bm25, scikit-learn, faiss-cpu

3.  Corpus construction
    • 25 documents across tech support, cooking, finance, health, and travel

4.  TF-IDF retrieval
    • Sparse vectors + cosine similarity

5.  BM25 retrieval
    • Tokenization + probabilistic ranking

6.  Sentence-BERT retrieval
    • all-MiniLM-L6-v2 embeddings
    • Mean pooling
    • Cosine similarity search

7.  Side-by-side query demos
    • Paraphrase queries designed to break lexical search

8.  Evaluation
    • Hand-labeled relevance judgments
    • Precision@k, MRR, latency — TF-IDF vs. BM25 vs. SBERT

9.  Scaling with FAISS
    • IndexFlatIP exact search
    • Notes on IVF / HNSW / Product Quantization for millions of vectors

10. Summary & hybrid search discussion
```

---

## TF-IDF

TF-IDF (Term Frequency–Inverse Document Frequency) converts documents into sparse vectors weighted by how
informative each word is across the corpus.

Pipeline:

```text
Documents
    ↓
Vocabulary
    ↓
TF-IDF Matrix
    ↓
Cosine Similarity
    ↓
Top-k Documents
```

Example:

```text
"forgot password"

      ↓

[0.12, 0.81, 0, 0, ...]
```

**Pros:** fast, simple, interpretable, no training required, excellent for exact keyword/code/name matches.
**Cons:** requires exact word overlap — cannot match synonyms or paraphrases.

---

## BM25

BM25 (Best Match 25) is the ranking function behind most production search engines (Elasticsearch, Lucene,
Solr by default). It improves on TF-IDF with **term-frequency saturation** (the 10th occurrence of a word
shouldn't matter as much as the 2nd) and **document-length normalization**.

Pipeline:

```text
Documents
    ↓
Tokenization
    ↓
BM25 Scoring
    ↓
Top-k Documents
```

Score:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\left(1-b+b\,\frac{|d|}{\text{avgdl}}\right)}$$

Typical values:

```text
k1 = 1.2 – 2.0
b  = 0.75
```

**Pros:** strong, battle-tested lexical baseline; better ranking than raw TF-IDF.
**Cons:** still relies entirely on lexical overlap — has no notion of meaning.

---

## Sentence-BERT (SBERT)

SBERT embeds entire sentences into a dense vector space where **cosine similarity reflects semantic
similarity** — something raw BERT was never trained to provide.

```text
"I forgot my password"

         ↓

[-0.21, 0.14, ...]   shape: (384,)
```

### Why not just use raw BERT?

BERT outputs one vector *per token*, not one per sentence. Naively averaging token vectors (or taking
`[CLS]`) produces embeddings that are often **worse than averaged GloVe vectors** on similarity benchmarks,
because BERT's pretraining (masked language modeling, next-sentence prediction) never optimizes the
geometry of whole-sentence representations — nothing pushes "similar meaning" toward "close vectors."

Using BERT directly for search (as a **cross-encoder**, scoring `[CLS] query [SEP] doc [SEP]` jointly) is
accurate but doesn't scale: comparing 1 query against 10,000 documents this way takes a full BERT forward
pass *per pair* — roughly **65 hours** on a V100 GPU, per the original SBERT paper.

### SBERT architecture

```text
Sentence
    ↓
Tokenizer
    ↓
Transformer Encoder (shared weights — siamese network)
    ↓
Token Embeddings   (N, 384)
    ↓
Pooling
    ↓
Sentence Embedding (384,)
```

Because the encoder weights are **shared** and each sentence is encoded **independently**, documents can
be embedded **once, offline**, and only the query needs encoding at search time — turning that 65-hour
comparison into about **5 seconds**.

### Pooling

BERT produces one vector per token:

```text
[CLS]  I  forgot  my  password  [SEP]
```

Output shape:

```python
(6, 384)
```

**Mean pooling** (used by SBERT) averages across the token dimension to get a fixed-length sentence vector:

$$v = \frac{1}{N}\sum_{i=1}^{N} h_i$$

```python
(6, 384)  →  (384,)
```

### Semantic similarity

$$\cos(x, y) = \frac{x \cdot y}{\lVert x\rVert \, \lVert y\rVert}$$

```text
"forgot password"  vs.  "reset credentials"   →  cos ≈ 0.7–0.9  (semantically related)
"forgot password"  vs.  "weather forecast"     →  cos ≈ 0.0–0.2  (unrelated)
```

### Why SBERT works

Lexical methods depend on word overlap. SBERT is fine-tuned (on NLI and STS sentence-pair datasets) so
that related concepts land near each other in vector space — without ever sharing a token:

```text
forgot      ≈  reset
credentials ≈  password
login       ≈  account access
```

### Training objective

SBERT is fine-tuned with **contrastive learning** on sentence pairs/triplets:

```text
Anchor:    "I forgot my password"
Positive:  "How do I reset my password?"
Negative:  "Best time to visit Japan"
```

**Goal:** increase $\cos(\text{anchor}, \text{positive})$, decrease $\cos(\text{anchor}, \text{negative})$.

Loss functions used in practice:

* **Triplet Loss**
* **Contrastive Loss**
* **MultipleNegativesRankingLoss** — the workhorse loss for most modern general-purpose embedding models

**Pros:** captures meaning, robust to paraphrase/synonymy, no manual feature engineering.
**Cons:** brute-force comparison is $O(n)$ per query; weaker than lexical search on exact identifiers,
codes, or rare proper nouns; embedding step requires a GPU-friendly forward pass per document.

---

## Why FAISS?

Suppose the corpus grows to:

```text
1,000,000 documents
```

SBERT still produces one vector per document:

```text
Doc1  → v1
Doc2  → v2
 ...
Doc1M → v1M
```

Naive retrieval compares the query against **every** vector:

```python
for every vector:
    cosine(query, vector)
```

Complexity: **O(N)** per query — too slow at scale.

### FAISS

FAISS (Facebook AI Similarity Search) builds an index over the embeddings to make nearest-neighbor search
sub-linear:

```text
Embeddings
    ↓
FAISS Index
```

```text
Query: "forgot login credentials"
    ↓
Query Embedding
    ↓
FAISS Search
    ↓
"Password reset guide"
"Account recovery"
"Two-factor authentication setup"
```

### FAISS workflow

**Offline (indexing, done once):**

```text
Documents → SBERT → Embeddings → FAISS Index
```

**Online (at query time):**

```text
User Query → SBERT → Query Embedding → FAISS Search → Top-k Documents
```

### Types of FAISS indexes

| Index | Type | Notes |
|---|---|---|
| `IndexFlatL2` | Exact | Euclidean distance, 100% accurate, brute-force |
| `IndexFlatIP` | Exact | Inner product — equivalent to cosine similarity on normalized SBERT embeddings. **Used in this notebook.** |
| `IndexIVFFlat` | Approximate | Clusters vectors with k-means, searches only relevant clusters — much faster |
| `IndexHNSWFlat` | Approximate | Graph-based search — state-of-the-art ANN speed/recall tradeoff |
| Product Quantization | Compressed | Compresses vectors for billion-scale corpora |

The notebook uses `IndexFlatIP` (exact search) since the demo corpus is tiny — at this scale it returns
identical results to plain `cosine_similarity`. The same code pattern swaps in `IndexHNSWFlat` or
`IndexIVFFlat` for production-scale corpora, trading a small amount of recall for large speedups.

---

## Similarity Metrics

| Metric | Formula | Notes |
|---|---|---|
| Euclidean distance | $d(x,y) = \lVert x-y\rVert$ | Used by `IndexFlatL2` |
| Inner product | $x^\top y$ | Used by `IndexFlatIP` |
| Cosine similarity | $\dfrac{x \cdot y}{\lVert x\rVert\lVert y\rVert}$, or $x^\top y$ if $\lVert x\rVert = \lVert y\rVert = 1$ | What SBERT search uses; the notebook normalizes embeddings so inner product **is** cosine similarity |

---

## Retrieval Pipeline Comparison

| | TF-IDF | BM25 | SBERT | SBERT + FAISS |
|---|---|---|---|---|
| **Pipeline** | Documents → Sparse Vectors → Cosine Similarity | Documents → BM25 Score → Top-k | Sentence → Transformer → Embedding → Cosine Similarity | Sentence → Embedding → FAISS → Nearest Neighbors |
| **Matches on** | Exact word overlap | Exact word overlap (TF saturation + length norm) | Semantic meaning | Semantic meaning |
| **Catches paraphrase/synonyms** | No | No | Yes | Yes |
| **Catches exact keywords/codes/names** | Yes | Yes | Not reliable | Not reliable |
| **Query-time complexity** | Sub-linear (inverted index) | Sub-linear (inverted index) | O(n) brute-force | Sub-linear (ANN index) |
| **Scales to millions of docs** | Yes | Yes | Slow at scale | Yes |
| **Interpretability** | High | High | Low | Low |

---

## Example Query

```text
Query: "I forgot my login credentials"

TF-IDF:           Weak match     — little vocabulary overlap with "reset password"
BM25:              Slightly better — same fundamental limitation
SBERT:             Strong semantic match
SBERT + FAISS:     Strong semantic match + sub-linear retrieval speed
```

The notebook runs this exact comparison live, plus two more paraphrase-style queries (about post-workout
recovery and investing), printing the top-k results from all three methods side by side.

---

## Evaluation

The notebook hand-labels **10 evaluation queries** against the 25-document corpus, deliberately split into:

* **Lexical-friendly queries** — share vocabulary with the relevant document (e.g. *"how to bake bread at
  home"*)
* **Semantic/paraphrase queries** — share little to no vocabulary (e.g. *"I forgot my login credentials,
  how can I get back into my account?"*)

### Metrics

| Metric | Definition |
|---|---|
| **Precision@k** | Fraction of the top-k retrieved documents that are actually relevant |
| **MRR** (Mean Reciprocal Rank) | Average of $1/\text{rank of first relevant result}$ across queries — rewards ranking relevant results near the top, not just anywhere in top-k |

(Recall@k and MAP are noted as natural extensions — see [Future Extensions](#future-extensions).)

### Expected pattern

* On **lexical-friendly** queries, TF-IDF and BM25 perform reasonably — there's word overlap to exploit.
* On **semantic/paraphrase** queries, TF-IDF and BM25 tend to score **near zero** (the relevant document
  often doesn't even appear in the top-k), while **SBERT stays consistently strong** across both query
  types, since it never depends on shared vocabulary.
* On **latency**, at this toy corpus size (25 docs), TF-IDF/BM25 are faster in absolute terms — they're
  pure array/dict lookups, while SBERT pays the cost of a neural network forward pass per query. This gap
  *narrows* in relative terms as corpora grow, since exact brute-force comparison must scan the whole
  corpus regardless of method, and SBERT's per-document cost is already paid upfront at indexing time.

The notebook outputs both an aggregate metrics table and a per-query reciprocal-rank breakdown so the
pattern is visible query-by-query, not hidden inside an average.

---

## Complete RAG Pipeline

The retrieval pattern built here is the first stage of a typical Retrieval-Augmented Generation system:

```text
PDFs / Documents
        ↓
   Chunking
        ↓
SBERT Embeddings
        ↓
  FAISS Index
=================================
   User Query
        ↓
SBERT Embedding
        ↓
 FAISS Search
        ↓
 Top-k Chunks
        ↓
     LLM
        ↓
 Final Answer
```

---

## Key Takeaway

> TF-IDF and BM25 rely on keyword overlap; SBERT learns semantic meaning by embedding sentences into a
> shared vector space where cosine similarity reflects how related two pieces of text actually are. FAISS
> makes that dense retrieval scalable, which is what enables modern semantic search systems and
> Retrieval-Augmented Generation (RAG) pipelines.

In practice, lexical and semantic search are usually **combined, not chosen between** — see
[Future Extensions](#future-extensions).

---

## Getting Started

```bash
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt   # or run the !pip install cell at the top of the notebook
jupyter notebook sbert_semantic_search.ipynb
```

### Requirements

```text
sentence-transformers
rank_bm25
scikit-learn
faiss-cpu
numpy
pandas
torch
```

Models used:

* **Embedding model:** [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  (384-dim, 6 layers — fast, strong general-purpose baseline). Swap in `all-mpnet-base-v2` for higher
  accuracy at a higher compute cost.

---

## Future Extensions

* **Hybrid Search** — combine BM25 + SBERT rankings (e.g. via Reciprocal Rank Fusion) to get lexical
  precision on exact terms *and* semantic recall on paraphrases
* **Cross-Encoder reranking** — rerank the SBERT/FAISS shortlist with a slower but more accurate
  `cross-encoder/ms-marco-MiniLM-L6-v2` model
* **ColBERT** — token-level late-interaction retrieval, a middle ground between bi-encoders and cross-encoders
* **FAISS IVF + Product Quantization** — for corpora in the hundreds of millions of vectors
* Managed vector databases — **ChromaDB**, **Qdrant**, **Pinecone**
* A full **RAG pipeline** with chunking and local LLM question answering over retrieved chunks
* **Recall@k** and **MAP** added to the evaluation suite alongside Precision@k and MRR

---

## References

* Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP.
* Robertson, S. & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.*
* Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-scale similarity search with GPUs* (FAISS).
