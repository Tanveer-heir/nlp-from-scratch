# LoRA From Scratch 🔧

> Implement Low-Rank Adaptation (LoRA) from the ground up — no PEFT, no shortcuts. Build every component yourself, hook it into attention, fine-tune a small GPT-style model, and understand exactly what's happening under the hood.

---

## Table of Contents

- [What Is LoRA?](#what-is-lora)
- [The Math](#the-math)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Requirements](#requirements)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Key Concepts](#key-concepts)
- [Results](#results)
- [Hyperparameter Guide](#hyperparameter-guide)
- [Extensions: QLoRA](#extensions-qlora)
- [References](#references)

---

## What Is LoRA?

Fine-tuning large pre-trained models is expensive. GPT-3 (175B parameters) requires ~700 GB of optimizer states for a full fine-tune — far beyond what most hardware can handle.

**LoRA (Low-Rank Adaptation)** sidesteps this by observing that weight updates during fine-tuning have *low intrinsic rank*. Instead of updating the full weight matrix `W ∈ R^(d×k)`, LoRA learns a low-rank decomposition:

```
ΔW = B × A     where B ∈ R^(d×r),  A ∈ R^(r×k),  r ≪ min(d, k)
```

Only `A` and `B` are trained. The base model `W₀` is completely frozen.

**The savings are dramatic.** For a 768×768 weight matrix with rank 8:

| Approach | Parameters | Ratio |
|---|---|---|
| Full fine-tune | 589,824 | 1× |
| LoRA r=8 | 12,288 | **48× fewer** |
| LoRA r=4 | 6,144 | **96× fewer** |

---

## The Math

The full forward pass with LoRA becomes:

```
h = W₀x + ΔWx
  = W₀x + (α/r) · B · A · x
```

where:

- `W₀` — frozen pre-trained weights
- `A` — initialized with Kaiming uniform (random signal)
- `B` — initialized with **zeros** → so `ΔW = 0` at the start of training
- `α` — scaling hyperparameter (usually set equal to `r`, giving scale = 1.0)
- `r` — rank (the key hyperparameter, typically 4–64)

**Why B=0 at init?** So the model starts fine-tuning from the exact pre-trained state. No disruption to the base model's behavior at step zero.

**After training**, LoRA weights can be *merged* back into the base:

```
W_merged = W₀ + (α/r) · B · A
```

Inference then has **zero extra overhead** — the merged model is identical to a standard model.

---

## Project Structure

```
lora_from_scratch.ipynb     ← Main notebook (everything below lives here)
README.md                   ← This file
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install torch transformers datasets matplotlib numpy tqdm

# 2. Open the notebook
jupyter notebook lora_from_scratch.ipynb

# 3. Run all cells — works on CPU, no GPU required
```

No GPU needed. The model is intentionally small (4 layers, d=128). Full training runs in **2–5 minutes on CPU**.

---

## Requirements

| Package | Version | Notes |
|---|---|---|
| Python | ≥ 3.8 | |
| PyTorch | ≥ 2.0 | CPU-only install works fine |
| matplotlib | ≥ 3.5 | For visualizations |
| numpy | any | |
| tqdm | any | Progress bars |
| transformers | ≥ 4.30 | Optional (for real model experiments) |

Install all at once:

```bash
pip install torch matplotlib numpy tqdm transformers
```

---

## Notebook Walkthrough

### Section 1 — Setup

Imports, device detection (`cuda` if available, else `cpu`), and reproducibility seeds.

---

### Section 2 — Low-Rank Decomposition Intuition

Before writing any LoRA code, we visualize *why* low-rank approximation works.

- Create a synthetic weight matrix with a true rank-4 signal plus noise
- Compute its SVD and show that singular values drop sharply after index 4
- Plot reconstruction error vs approximation rank
- Show the parameter count tradeoff at different ranks

**Key takeaway:** Real fine-tuning updates tend to lie in a low-dimensional subspace — exactly what LoRA exploits.

---

### Section 3 — `LoRALayer`: The Core Building Block

```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16.0, dropout=0.0):
        ...
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))   # random init
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))  # zero init
        self.scale  = alpha / rank

    def forward(self, x):
        return self.scale * x @ self.lora_A.T @ self.lora_B.T

    def get_delta_weight(self):
        return self.scale * (self.lora_B @ self.lora_A)  # full ΔW matrix
```

Then `LoRALinear` wraps any existing `nn.Linear`:

```python
class LoRALinear(nn.Module):
    def forward(self, x):
        return self.linear(x) + self.lora(x)   # base (frozen) + adapter (trained)

    def merge_weights(self):
        # W_merged = W₀ + scale * B @ A
        merged.weight.data += self.lora.get_delta_weight()
        return merged
```

---

### Section 4 — Multi-Head Attention with LoRA

`MultiHeadAttentionWithLoRA` wraps Q, K, V, O projections with `LoRALinear`. You control which projections get adapted:

```python
mha = MultiHeadAttentionWithLoRA(
    d_model=256,
    n_heads=8,
    rank=8,
    alpha=16.0,
    lora_targets=('q', 'v')   # Standard: Q and V only
)
```

The rest of the attention math (scaled dot-product, masking, softmax) is unchanged.

---

### Section 5 — Injecting LoRA into an Existing Model

`inject_lora()` walks any model's module tree and replaces matching `nn.Linear` layers in-place:

```python
model = inject_lora(
    pretrained_model,
    target_modules=['q_proj', 'v_proj'],
    rank=8,
    alpha=16.0,
)
```

This is how real libraries like HuggingFace PEFT work internally — no need to modify the original model's source code.

---

### Section 6 — Fine-Tuning

Three models are trained on a character-level language modeling task:

| Model | Description |
|---|---|
| `model_full` | Full fine-tune — all parameters trainable |
| `model_lora` | LoRA rank=8, Q+V only |
| `model_lora_r4` | LoRA rank=4, Q+V only |

Training details:
- Optimizer: AdamW (`lr=3e-4`, `weight_decay=0.01`)
- Schedule: Cosine annealing
- Gradient clipping: max norm 1.0
- Epochs: 15

Only LoRA parameters are passed to the optimizer — the frozen base weights never receive gradient updates.

---

### Section 7 — Merging Weights

After training, `merge_lora_weights()` folds adapters back into the base:

```python
model_merged = merge_lora_weights(model_lora)
```

Verification: the merged model produces **numerically identical outputs** to the LoRA model (difference < 1e-4 from floating point only).

---

### Section 8 — Visualizing What LoRA Learns

Four panels:

1. **ΔW heatmaps** — the learned weight update per layer/projection
2. **Singular value spectrum of ΔW** — confirms updates are genuinely low-rank
3. **A and B matrix norms** — B grows from zero as training progresses
4. **Parameter pie chart** — frozen vs trainable breakdown

---

### Section 9 — Ablation Studies

**Rank ablation** (r = 1, 2, 4, 8, 16):
- Plots loss curve, final loss, and the efficiency frontier (loss vs param count)
- Shows diminishing returns beyond a certain rank for simple tasks

**Target module ablation** (Q / V / Q+V / Q+K+V / Q+K+V+O):
- Identifies which projections contribute most to adaptation
- Q+V is typically the sweet spot — matching the original paper's recommendation

---

### Section 10 — QLoRA Sketch & Summary

A `QuantizedLoRALinear` class demonstrates the QLoRA idea:

- Base weights stored as **int8** (1 byte/param vs 4 bytes for float32)
- LoRA adapters stay in **float32**
- Dequantize on-the-fly during forward pass

```
Full float32:        65,536 bytes
Int8 base + LoRA:    24,576 bytes  (2.7× smaller)
NF4 base + LoRA:     ~12,000 bytes (real QLoRA, ~5× smaller)
```

---

## Key Concepts

### Why Q and V?

The original LoRA paper found that adapting Q and V projections alone matches the performance of adapting all four (Q, K, V, O). This is the standard default. Adding K and O gives marginal improvement at higher parameter cost.

### The Scale Factor (α/r)

`alpha` controls how strongly the LoRA update influences the output. Common settings:

- `alpha = rank` → scale = 1.0 (neutral, most common)
- `alpha = 2 × rank` → scale = 2.0 (stronger adaptation signal)

Setting `alpha` separately from `rank` lets you change rank without retuning the effective learning rate.

### Initialization Matters

| Matrix | Init | Reason |
|---|---|---|
| A | Kaiming uniform | Provides a random starting signal |
| B | All zeros | Ensures ΔW = B·A = 0 at step 0 |

If both were random, the model would start fine-tuning from a disrupted state instead of the clean pretrained checkpoint.

### Merging for Inference

Unmerged LoRA adds one matrix multiply per adapted layer. After merging:

```
W_merged = W₀ + (α/r) · B · A
```

The merged model is a plain `nn.Linear` — identical inference cost to the original.

---

## Results

Results on the character-level LM task (15 epochs, d=128, 4 layers):

| Model | Trainable Params | Final Loss | vs Full |
|---|---|---|---|
| Full fine-tune | ~400K | baseline | — |
| LoRA r=8 | ~8K | comparable | **50× fewer params** |
| LoRA r=4 | ~4K | +0.02–0.05 | **100× fewer params** |

LoRA achieves near-identical loss to full fine-tuning at a fraction of the parameter cost — consistent with the original paper's findings on larger models.

---

## Hyperparameter Guide

| Hyperparameter | Typical Range | Notes |
|---|---|---|
| `rank` | 4–8 (simple tasks), 16–64 (complex) | Higher rank = more capacity, more params |
| `alpha` | Equal to `rank` | Controls effective learning rate of adapter |
| `dropout` | 0.0–0.1 | Applied to the down-projection path; helps regularize |
| `lora_targets` | `('q', 'v')` | Add `'o'` for more capacity; `'k'` rarely helps much |
| `lr` | 1e-4 – 5e-4 | Can be higher than full fine-tune since fewer params |
| optimizer | AdamW | With `weight_decay=0.01` |
| schedule | Cosine annealing | Linear warmup optional for larger models |

---

## Extensions: QLoRA

QLoRA combines LoRA with 4-bit quantization of the base model, enabling fine-tuning of very large models on consumer hardware:

| Model | Full Fine-tune VRAM | QLoRA VRAM |
|---|---|---|
| LLaMA 7B | ~112 GB | ~6 GB |
| LLaMA 13B | ~200 GB | ~10 GB |
| LLaMA 65B | ~780 GB | ~48 GB |

Key components not implemented here (require `bitsandbytes`):
- **NF4 (NormalFloat4)** — 4-bit dtype optimized for normally distributed weights
- **Double quantization** — quantize the quantization constants themselves
- **Paged optimizers** — handle GPU memory spikes during training

---

## References

- **LoRA paper:** Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
- **QLoRA paper:** Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)
- **HuggingFace PEFT** (production LoRA library): [github.com/huggingface/peft](https://github.com/huggingface/peft)
- **Intrinsic dimensionality:** Aghajanyan et al. (2020). *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.*

---

## License

MIT — use freely for learning, research, or production.
