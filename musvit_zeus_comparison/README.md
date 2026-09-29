# MuSViT vs. Zeus Benchmark Comparison

This module provides a reproducible, controlled experiment comparing:
- **Run 1 (Zeus Baseline):** CNN-BiLSTM Encoder + Zeus Bahdanau Attention LSTM Decoder (trained from scratch).
- **Run 2 (MuSViT + Zeus):** Pre-trained MuSViT Vision Transformer Feature Extractor + Zeus Bahdanau Attention LSTM Decoder.

Both runs share the **exact same decoder architecture, random seed, training/validation splits, learning rate schedule, optimizer, and loss function**.

---

## 1. Feature Pre-Extraction & Storage Optimization

### The Problem
Raw patch embeddings from ViT models on large images ($1024 \times 1024$) generate $4096$ patch tokens of dimension $768$:
- **Full uncompressed FP32:** $4096 \times 768 \times 4 \text{ bytes} \approx \mathbf{12.6\text{ MB}}$ per stave image.
- For $10,000$ staves: **$\approx 126\text{ GB}$**. This would quickly saturate disk space and cause heavy I/O bottlenecks.

### The Solution: Vertical 1D Pooling in FP16
Single musical staves are read **horizontally from left to right**. 
`extract_features.py` averages over the 64 vertical rows ($64 \times 64$ patch grid) and saves in `float16`:
- **Sequence shape:** `(64, 768)`
- **Storage per stave:** $64 \times 768 \times 2 \text{ bytes} = \mathbf{98.3\text{ KB}}$ (**128x smaller!**)
- **Storage for 420 staves (OmniOMR.Small):** **$\approx 41\text{ MB}$**
- **Storage for 10,000 staves:** **$\approx 980\text{ MB}$ (under 1 GB)**

*(Alternative: `--pool-mode vertical_4slice` preserves 4 vertical height bins $\to$ `[256, 768]`, taking ~393 KB/sample).*

---

## 2. Running on the GPU Cluster

### Step 1: Pre-Extract Features (Run once on GPU)

Extract and cache MuSViT embeddings for your dataset:

```bash
python -m musvit_zeus_comparison.extract_features \
    --dataset-dir OmniOMR.Small \
    --output-dir feature_cache \
    --pool-mode vertical_mean \
    --precision float16 \
    --device cuda
```

*Note: If interrupted, the script automatically resumes and skips already-extracted samples.*

---

### Step 2: Run Training and Benchmark Comparison

To train both **Run 1** and **Run 2** back-to-back under identical settings:

```bash
python -m musvit_zeus_comparison.train \
    --model compare \
    --dataset-dir OmniOMR.Small \
    --feature-cache-dir feature_cache \
    --output-dir experiment_results \
    --epochs 20 \
    --batch-size 16 \
    --lr 5e-4 \
    --device cuda
```

You can also run them individually using `--model zeus` or `--model musvit`.

---

### Step 3: Inspect the Generated Comparison Table

Upon completion, a formatted summary table is printed to stdout and saved to:
- `experiment_results/results_table.md` (Markdown format)
- `experiment_results/results_table.csv` (Spreadsheet format)

Example output table:

| Model Run | Encoder | Enc Params | Dec Params | Total Params | Val Loss (best) | Token Acc (%) | SER / NED (%) | Train Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Zeus Baseline (Run 1)** | CNN-BiLSTM | 851,200 | 1,185,420 | 2,036,620 | 0.8241 | 82.15% | 18.42% | 145.2s |
| **MuSViT + Zeus (Run 2)** | MuSViT + Adapter | 394,496 | 1,185,420 | 1,579,916 | 0.5120 | 89.64% | 11.20% | 42.1s |

---

## 3. Directory Layout

```
musvit_zeus_comparison/
├── __init__.py          # Package exports
├── models.py            # ZeusEncoder, MusvitEncoder, ZeusDecoder, CombinedOMRModel
├── dataset.py           # StaveOMRDataset (supports image + feature loading) & StaveCollate
├── extract_features.py  # Feature extractor with vertical pooling & FP16 compression
├── train.py             # Main trainer & evaluator with automated table generation
└── README.md            # Detailed documentation
```
