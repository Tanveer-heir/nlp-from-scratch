# nlp-from-scratch

15 mechanical NLP implementations in raw PyTorch — no HuggingFace Trainer, no shortcuts.
Each project is scoped to one concept, one paper, one key implementation challenge.

---

## Index

| # | Project | Paper(s) | Key Challenge |
|---|---------|----------|---------------|
| 01 | [Word2Vec](./01-word2vec/) | Mikolov et al. (2013) | Negative sampling, embedding update math |
| 02 | [BPE Tokenizer](./02-bpe-tokenizer/) | Sennrich et al. (2015) | Merge algorithm by hand, vocab/length tradeoff |
| 03 | [Sentence-BERT Search](./03-sbert-search/) | Reimers & Gurevych (2019) | Why raw BERT embeddings fail for cosine similarity |
| 04 | [N-gram LM + Smoothing](./04-ngram-lm/) | Jurafsky & Martin | Kneser-Ney smoothing, perplexity computation |
| 05 | [BiLSTM-CRF NER](./05-bilstm-crf/) | Lample et al. (2016) | CRF forward algorithm + Viterbi decode, no torchcrf |
| 06 | [Seq2Seq + Bahdanau Attention](./06-seq2seq-attention/) | Sutskever (2014), Bahdanau (2014) | Attention bottleneck fix, teacher forcing |
| 07 | [Multi-head Self-Attention](./07-attention-block/) | Vaswani et al. (2017) | Scaled dot-product, causal + padding masking |
| 08 | [Transformer Encoder-Decoder](./08-transformer/) | Vaswani et al. (2017) | Full model, train on toy seq2seq task |
| 09 | [Tiny GPT + KV-Cache](./09-tiny-gpt/) | Radford et al. (2018) | Causal LM loop, KV-caching for inference |
| 10 | [Positional Encoding Comparison](./10-positional-encoding/) | Shaw (2018), Su (2021), Press (2021) | Sinusoidal vs learned vs RoPE vs ALiBi |
| 11 | [BERT Fine-tuning](./11-bert-finetune/) | Devlin et al. (2018) | Subword-to-label alignment for NER, HF Trainer |
| 12 | [LoRA from Scratch](./12-lora/) | Hu et al. (2021) | Low-rank decomposition hooked into attention layers |
| 13 | [Knowledge Distillation](./13-distillation/) | Sanh et al. (2019) | Soft targets, temperature scaling, speed/accuracy tradeoff |
| 14 | [Decoding Strategies](./14-decoding/) | — | Greedy, beam, top-k, top-p, temperature — visualised |
| 15 | [BLEU / ROUGE / BERTScore](./15-eval-metrics/) | Papineni (2002), Lin (2004), Zhang (2019) | Implement from scratch, find human-judgment disagreements |

---

## Structure

Each project folder contains:
- `README.md` — what paper, what I implemented, key result or insight
- `notebook.ipynb` — main implementation (runs on Colab Pro)

---

## Curriculum context

These mini-projects are the mechanical foundation for 5 paper-level major projects:

- [`rag-retrieval-system`](https://github.com/Tanveer-heir/rag-retrieval-system)
- [`neural-search-ranking`](https://github.com/Tanveer-heir/neural-search-ranking)
- [`efficient-transformer`](https://github.com/Tanveer-heir/efficient-transformer)
- [`react-agent`](https://github.com/Tanveer-heir/react-agent)
- [`dpo-alignment`](https://github.com/Tanveer-heir/dpo-alignment)
