# Evaluation Metrics Deep Dive
## BLEU, ROUGE, and BERTScore from Scratch + Human Judgment Failures

---

## Overview

This notebook implements three core NLP evaluation metrics entirely from scratch -- no `sacrebleu`, `evaluate`, or `rouge_score` libraries are used. It then deliberately constructs cases where each metric disagrees with human judgment, which is the most important lesson in applied NLP evaluation.

The notebook covers:

- Full from-scratch implementations of BLEU, ROUGE-1/2/L/S, and BERTScore
- Visualisations of internal metric mechanics
- Seven deliberate failure cases with analysis
- Correlation analysis between metric scores and human scores
- IDF weighting for BERTScore
- Corpus-level BLEU
- A practical recommendation guide

---

## Why Evaluation Metrics Exist

Suppose you trained a translation model.

Reference:
```
J'aime l'apprentissage automatique.
```

Your model outputs:
```
J'adore l'apprentissage automatique.
```

A human immediately accepts this as correct. But a computer only sees two different strings. Evaluation metrics exist to bridge that gap -- to automatically compare generated text against a reference in a way that approximates human judgment.

The problem is that no metric approximates human judgment perfectly. This notebook shows exactly where and why each one breaks down.

---

## The Three Families

| Metric | Core idea | Orientation |
|--------|-----------|-------------|
| BLEU | n-gram precision | Did your output contain the right words? |
| ROUGE | n-gram / LCS recall | Did your output cover the reference? |
| BERTScore | Contextual embedding similarity | Do the tokens mean the same thing? |

---

## Part 1 -- BLEU

### Intuition

BLEU asks: how many words or phrases from the prediction appear in the reference?

This is a precision-oriented question. If you predict three words and all three appear in the reference, unigram precision is 100%, regardless of whether the reference had ten words you missed.

### Tokenization

Everything begins with lowercasing and tokenizing both strings into word lists:

```
"The cat sat on the mat"
-> ["the", "cat", "sat", "on", "the", "mat"]
```

### N-gram Precision

BLEU computes precision at four levels:

- P1: unigrams (individual words)
- P2: bigrams (two consecutive words)
- P3: trigrams
- P4: four-word phrases

Why multiple levels? Unigrams only check vocabulary. Bigrams check local word order. Trigrams and 4-grams check sentence flow and fluency. A model that shuffles words randomly will score well on P1 but near zero on P4.

### Modified Precision

Naive precision has a flaw. Consider:

Reference: `cat`
Prediction: `cat cat cat cat cat`

Naive precision: 5/5 = 100%. Obviously wrong.

BLEU fixes this by clipping each candidate n-gram count to its maximum count in any reference. Since "cat" appears once in the reference, the prediction gets credit for it exactly once:

Modified precision: 1/5.

### Brevity Penalty

Short predictions exploit precision. The single word "cat" achieves 100% unigram precision against any reference containing "cat". The brevity penalty discourages this:

```
BP = 1               if candidate length >= reference length
BP = exp(1 - r/c)    if candidate length < reference length
```

### Final Formula

```
BLEU = BP * exp( sum of w_n * log(P_n) )
```

where weights are uniform (1/N each) and the sum is a geometric mean in log space. The geometric mean is intentional: if any single n-gram level scores zero, the entire BLEU score becomes zero. Every level must contribute.

### Strengths and Weaknesses

Strengths: fast, reproducible, standard baseline for machine translation (WMT, etc.).

Weaknesses:
- Cannot understand meaning. "The kid is joyful" vs "The child is happy" scores near zero despite being identical in meaning.
- Word order insensitivity beyond the n-gram window. "The dog bit the man" and "The man bit the dog" share all unigrams and most bigrams.
- Single zero in any P_n kills the score -- requires smoothing at sentence level.

---

## Part 2 -- ROUGE

### Intuition

ROUGE flips the question. Instead of asking whether predicted words appear in the reference (precision), it asks whether reference content appears in the prediction (recall).

This makes ROUGE natural for summarization. A good summary should cover the key points of the reference document. Missing key content is penalised directly.

### ROUGE-N

Standard n-gram recall and precision, combined into F1.

```
Recall    = matching n-grams / total reference n-grams
Precision = matching n-grams / total hypothesis n-grams
F1        = 2 * P * R / (P + R)
```

ROUGE-1 uses unigrams. ROUGE-2 uses bigrams and is more sensitive to phrasing.

### ROUGE-L

The most structurally aware ROUGE variant. Instead of requiring n-grams to be contiguous, it finds the Longest Common Subsequence (LCS) between hypothesis and reference using dynamic programming.

Example:

Reference: `The cat sat on the mat`
Prediction: `The cat was sitting on mat`

Common subsequence: `The cat on mat` (length 4), even though the words are not adjacent.

This makes ROUGE-L more flexible than ROUGE-2 for paraphrased text.

### ROUGE-S

Skip-bigrams: ordered pairs of words with arbitrary gaps allowed. Captures word co-occurrence patterns even when intervening words differ.

### Strengths and Weaknesses

Strengths: works well for summarization; ROUGE-L captures structure better than ROUGE-N.

Weaknesses:
- Long outputs freely inflate recall without penalty -- always report P, R, F1 separately
- Surface-form only; synonyms are invisible
- Negation is completely transparent ("not safe" and "safe" share all content words)

---

## Part 3 -- BERTScore

### Intuition

Both BLEU and ROUGE compare surface tokens. BERTScore replaces token identity with contextual embedding similarity.

Reference: `The child is happy.`
Prediction: `The kid is joyful.`

BLEU and ROUGE score this near zero. But a pre-trained transformer knows that "child" and "kid" have nearly identical contextual embeddings, and "happy" and "joyful" are semantically close. BERTScore captures this.

### Embeddings

Each token is mapped to a dense vector by a transformer model. Similar meanings produce similar vectors:

```
child -> [0.28, -1.42, 0.91, ...]
kid   -> [0.31, -1.38, 0.89, ...]
```

### Cosine Similarity

```
cosine(A, B) = (A . B) / (||A|| * ||B||)
```

Range: 1 = identical direction, 0 = orthogonal, -1 = opposite.

### The Similarity Matrix

For a hypothesis of length m and reference of length n, a full (m x n) cosine similarity matrix is built. Every hypothesis token is compared to every reference token.

Precision: for each hypothesis token, take the maximum similarity to any reference token, then average across hypothesis tokens.

Recall: for each reference token, take the maximum similarity to any hypothesis token, then average across reference tokens.

F1: standard harmonic mean of precision and recall.

### Why Contextual Embeddings?

"Bank" in "river bank" has a different embedding from "bank" in "bank account". Unlike static word vectors, BERT-style models produce context-sensitive representations. This makes BERTScore robust to polysemy.

### IDF Weighting

Common tokens like "the", "is", "of" carry little meaning. IDF (Inverse Document Frequency) down-weights them:

```
IDF(token) = log((N + 1) / (df + 1)) + 1
```

Rare, content-bearing tokens receive higher weight, improving the signal quality of BERTScore.

### Strengths and Weaknesses

Strengths: handles synonyms and paraphrases well; correlates better with human judgment on most tasks than BLEU or ROUGE.

Weaknesses:
- Computationally expensive
- Biased toward fluent, in-domain text -- factually wrong but grammatically polished text can score high
- Does not detect negation, factual errors, or hallucinations
- Score scale is not 0-1 in a meaningful way without rescaling

---

## Part 4 -- Where All Three Metrics Fail

This is the central section of the notebook. Seven failure cases are constructed where metrics disagree with human judgment.

### Case 1 -- Synonym Blindness

Reference: "Scientists discovered a new planet outside our solar system."
Good output: "Researchers found an exoplanet beyond our solar system." (human: correct)
Bad output: same sentence with an appended hallucinated adverb (human: penalise)

BLEU and ROUGE reward the bad output because it shares more surface tokens. The good output uses synonyms ("researchers", "found", "exoplanet") that are invisible to n-gram metrics.

### Case 2 -- Word Order Insensitivity

Reference: "The dog bit the man."
Bad output: "The man bit the dog." -- opposite meaning
Good output: "A canine attacked a person." -- correct meaning, different words

"The man bit the dog" shares all unigrams and two bigrams ("the man", "the dog") with the reference. Its BLEU and ROUGE scores are high. The good output, using synonyms, scores lower on surface metrics despite being correct.

### Case 3 -- Repetition

Reference: "The economy grew last quarter."
Bad output: "The the the economy the economy the the the quarter."

Even with modified precision clipping, ROUGE-1 recall is partially gameable because repeated key words still get clipped recall credit proportional to reference frequency.

### Case 4 -- Negation Blindness

Reference: "The drug is safe for children under 12."
Bad output: "The drug is not safe for children under 12."

Adding "not" shares every content word with the reference. BLEU scores approximately 0.78, ROUGE-1 approximately 0.89, BERTScore approximately 0.91. This is not a minor quirk -- in medical or legal text it represents a safety-critical failure.

### Case 5 -- Padding and Length Inflation

Reference: "It rained heavily."
Bad output: "It rained heavily and there was also thunder and lightning and the streets flooded and people stayed indoors..."

ROUGE recall rises freely as the output length increases because more reference words get covered. BLEU has a brevity penalty for short outputs, but no penalty for long ones.

### Case 6 -- Factual Errors

Reference: "Marie Curie won the Nobel Prize in Physics in 1903."
Bad output: "Marie Curie won the Nobel Prize for Chemistry in 1903."

The wrong field ("Chemistry" vs "Physics") produces a fluent, topically relevant sentence. BERTScore for this pair is high because both sentences share nearly all semantic content. The factual error is one word, and BERTScore is not designed to detect it.

### Case 7 -- Multi-Reference Gaming

With five diverse references covering the same topic, shuffled word salad can borrow n-grams from each reference and achieve a surprisingly high BLEU score. A well-phrased single sentence may score lower.

---

## Part 5 -- Systematic Dashboard

All failure cases are plotted side by side. For each case, the metric scores of the "good" and "bad" candidate are compared. Cases where a metric ranks the bad output above the good output are marked as failures.

---

## Part 6 -- Correlation with Human Scores

A 19-pair corpus is scored by all three metrics and compared against human-assigned scores. Pearson correlation is computed. Typical findings from the literature:

```
BERTScore > ROUGE-L > ROUGE-1 > BLEU
```

However, the correlation reverses in adversarial cases (negation, factual error, fluency bias), reinforcing that no single metric is sufficient.

---

## Part 7 -- IDF Weighting in BERTScore

A small five-sentence corpus is used to compute IDF weights. Stop words ("the", "a", "on") receive low IDF values. Content words receive higher values. BERTScore computed with IDF weighting produces more meaningful similarity values for paraphrase pairs.

---

## Part 8 -- Sensitivity Heatmap

A summary matrix shows how well each metric handles ten common failure types:

- Synonym blindness
- Word order insensitivity
- Repetition exploit
- Negation blindness
- Length inflation
- Factual errors
- Multi-reference gaming
- Discourse coherence
- Hallucinations
- Cultural nuance

BERTScore handles synonyms well. All metrics fail on negation. Factual errors and hallucinations are beyond the scope of all three metrics.

---

## Part 9 -- Corpus-Level BLEU

Sentence-level BLEU has high variance because a single zero in any n-gram level collapses the score. Corpus-level BLEU accumulates clipped counts and total counts across all sentences, then computes a single precision value per n-gram level. This is the standard used by WMT and academic MT benchmarks.

---

## Practical Recommendations

**BLEU**
- Use for machine translation benchmarking with multiple references
- Always use corpus-level BLEU, not sentence-level, for reporting
- Use sentence-level BLEU only with smoothing enabled
- Do not use for summarization, dialog, or creative text generation

**ROUGE**
- Use for summarization tasks; report ROUGE-2 and ROUGE-L together
- Always report F1, not recall alone, to avoid length inflation bias
- Do not use for factuality evaluation or machine translation

**BERTScore**
- Use when synonym and paraphrase space is rich (QA, RAG, captioning)
- Apply IDF weighting for short or domain-specific references
- Do not rely on it for factual correctness, negation, or hallucination detection
- Rescale scores or report raw values -- the default range is not intuitive

**General**
- Never rely on a single metric
- Combine automated metrics with human evaluation or factuality checks
- For LLM evaluation, consider LLM-as-judge alongside at least one reference-based metric

---

## Key Takeaways

**Exact word overlap is not the same as semantic correctness.** BLEU and ROUGE rely on lexical overlap. BERTScore compares contextual meaning. Neither approach catches factual errors or negation.

**Different tasks require different metrics.** Translation commonly uses BLEU. Summarization commonly uses ROUGE. Modern LLM evaluation increasingly relies on semantic metrics or LLM-as-judge. Using BLEU to evaluate a summarisation system, or ROUGE to evaluate a translation system, produces misleading conclusions.

**No single metric is sufficient.** Every metric has blind spots. Research papers and production systems should report multiple metrics together and acknowledge what each one does not measure.

**Evaluation is as important as model design.** Two models with similar architectures can appear very different depending on which metric is used to compare them. Choosing the wrong evaluation metric is a real source of misleading conclusions in the literature.

**The negation problem is the most dangerous failure.** Adding "not" to a sentence shares all content words with the original. All three metrics score negated sentences highly against their positive counterparts. In safety-critical domains (medicine, law, policy), this is not a theoretical concern.

---

## Dependencies

```
numpy
matplotlib
sentence-transformers   # for BERTScore; falls back to hash-based embeddings if absent
```

No `sacrebleu`, `evaluate`, `rouge_score`, or `nltk` are used.

---

## File Structure

```
eval_metrics_deep_dive.ipynb    -- main notebook (9 parts)
README.md                       -- this file
```
