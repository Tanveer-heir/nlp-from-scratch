# BERT Fine-tuning vs. Raw PyTorch + CRF — NER on CoNLL-2003

Two approaches to the same task (Named Entity Recognition on CoNLL-2003), contrasted
end to end: one using Hugging Face's `Trainer` with a pretrained BERT encoder, the other
a hand-rolled PyTorch training loop around a BiLSTM-CRF trained from scratch.

## Files

| File | What it is |
|---|---|
| `bert_ner_trainer.ipynb` | BERT fine-tuned via HF `Trainer`, plain softmax token classification |
| `bilstm_crf.ipynb` | BiLSTM + CRF, raw PyTorch training loop, trained from scratch |

## Results

| | **HF Trainer (BERT + softmax)** | **Raw PyTorch (BiLSTM + CRF)** |
|---|---|---|
| Test F1 | **0.911** | 0.664 |
| Test accuracy | **0.983** | 0.917 |
| Best val F1 | 0.946 (epoch 3) | 0.765 (epoch 8) |
| Epochs trained | 3 | 15 |
| Params | ~110M (pretrained) | small, randomly initialized |
| Encoder | Pretrained BERT (`bert-base-cased`) | BiLSTM, trained from scratch |
| Decoding | Per-token argmax | Viterbi (CRF) |

### Per-entity F1 (test set, BiLSTM + CRF)

| Entity | F1 |
|---|---|
| LOC | 0.768 |
| MISC | 0.679 |
| PER | 0.670 |
| ORG | 0.564 |

*(BERT+softmax's per-entity breakdown wasn't logged separately in this run — the
overall F1/accuracy above are from the actual training output.)*

## Why the gap

BERT's score isn't mostly about the CRF-vs-softmax difference — it's about
**pretraining**. `bert-base-cased` has already seen a huge amount of English text and
learned general language structure before ever seeing a CoNLL-2003 example. Fine-tuning
just nudges that knowledge toward the NER task, so it converges in 3 epochs.

The BiLSTM, by contrast, starts from **randomly initialized embeddings and weights** —
it has to learn what English words even mean from this dataset alone (~14k sentences),
which is why it needs 5x more epochs and still tops out well below BERT.

This is the real lesson of the comparison: the architecture (softmax vs. CRF) matters
less here than whether the encoder is pretrained. A from-scratch BiLSTM+CRF versus a
pretrained-BERT+softmax isn't an apples-to-apples test of "does CRF help" — for that
you'd want BERT+CRF vs. BERT+softmax, holding the encoder constant. That comparison was
attempted first; see the note below for why it was dropped.

## Note on the original BERT + CRF attempt

The original plan was BERT (encoder) + CRF (decoder) vs. BERT + plain softmax — a
cleaner, single-variable comparison. That version trained without crashing but
plateaued at **F1 ≈ 0.31** no matter how learning rates were tuned, while every
individual component (CRF math, masking, label alignment, gradient flow into BERT)
checked out correctly under isolated testing.

The most likely explanation: BERT (~110M pretrained params) and a freshly-initialized
CRF transition matrix (~90 params) converge at very different speeds, and a 3-epoch
fine-tuning budget — tuned for the much simpler softmax case — wasn't enough for the
CRF half to escape its random initialization while sitting on top of a much larger,
more slowly-adapting encoder. This is a known practical difficulty with BERT+CRF; it
typically needs careful per-parameter-group learning rates and a larger epoch budget
than plain BERT fine-tuning to actually pay off.

Rather than keep tuning that specific combination, the raw-PyTorch side was switched
to BiLSTM+CRF — the classic, pre-BERT NER architecture the CRF layer was originally
designed for. It has no pretrained weights to clash with, so a single learning rate
and a standard training loop converge cleanly, producing the results shown above.

## How to run

Both notebooks download `tomaarsen/conll2003` (a script-free Hub mirror of the standard
CoNLL-2003 NER dataset) automatically — no manual data setup needed. Run each top to
bottom; GPU recommended for `bert_ner_trainer.ipynb`, optional but faster for
`bilstm_crf.ipynb`.

## Known overfitting note (BiLSTM + CRF)

Validation F1 plateaus around epoch 5-8 (~0.75-0.76) and gets noisy afterward, while
training loss keeps dropping toward 0 — classic overfitting on a small dataset trained
from scratch. The training loop already keeps the best validation checkpoint rather
than the final epoch, but stronger regularization (more dropout, weight decay) or
early stopping would likely close some of the val/test F1 gap (0.765 vs 0.664).
