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

## 2. Tokenization & LMX Pre-Generation

### Why Generate `.lmx` Files?
By default, if only `transcription.musicxml` is found, the dataset loader uses a simple built-in MusicXML parser. However, generating official **Linearized MusicXML (`.lmx`)** files provides several significant advantages:

1. **100% Token Parity with Official Zeus:** Uses the exact same token grammar (`measure`, `clef:G-2`, `key:0`, `C4`, `quarter`, etc.) and vocabulary specification used by pre-trained Zeus checkpoints.
2. **Invisible Header Normalization:** Stave crops in MusiCorpus often omit a visible clef or key signature. The LMX compiler resolves and normalizes invisible G-clefs and transposes notes according to the Zeus/MusiCorpus specification.
3. **Faster Training & I/O:** Pre-tokenized single-line `.lmx` files avoid expensive XML DOM tree parsing during every epoch.
4. **Interoperability:** Enables direct comparison with official Zeus benchmarks and standard Symbol Error Rate (SER) tools.

### Step 1: Install `linearized-musicxml`
```bash
pip install "linearized-musicxml @ git+https://github.com/OMR-Research/lmx.git@87fca1c38bda83bc032596ab00af28412f92941d"
```

### Step 2: Generate `.lmx` Transcriptions
Generate `transcription.lmx` beside each `transcription.musicxml` in your dataset:
```bash
python -m musvit_zeus_comparison.generate_lmx
# Defaults to --dataset-dir UFAL.OmniOMR
```
*(Takes under 15 seconds for ~1,200 staves. Subsequent training automatically picks up these `.lmx` files).*

---

## 3. Running on the GPU Cluster

### Step 3: Pre-Extract Features (Run once on GPU)

Extract and cache MuSViT embeddings for your dataset:

```bash
python -m musvit_zeus_comparison.extract_features \
    --dataset-dir UFAL.OmniOMR \
    --output-dir feature_cache \
    --pool-mode vertical_mean \
    --precision float16 \
    --device cuda
```

*Note: If interrupted, the script automatically resumes and skips already-extracted samples.*

---

### Step 4: Run Training and Benchmark Comparison

In the official TensorFlow Zeus implementation (`docs/training-zeus.md`), models are trained from scratch for **400 to 500 epochs** (`--epochs 500`, `--learning-rate 1e-3`, `--lr-decay cos`, `--batch-size 32/64`).

#### Train from Scratch (Zeus Hyperparameters)
```bash
python -m musvit_zeus_comparison.train \
    --model compare \
    --dataset-dir UFAL.OmniOMR \
    --feature-cache-dir feature_cache \
    --output-dir experiment_results \
    --epochs 200 \
    --batch-size 32 \
    --lr 1e-3 \
    --optimizer adam \
    --eval-interval 20 \
    --device cuda
```

#### Running via SLURM on the Cluster
Submit as a background batch job (automatically allocates 1 GPU, 4 CPUs, 24G RAM on `-p gpu`):
```bash
sbatch run_comparison.slurm
```
To monitor progress in real-time:
```bash
tail -f logs/slurm-musvit-zeus-comp-*.out
```

Or run interactively inside your allocation:
```bash
srun -p gpu -G1 -c4 --mem=24G bash run_comparison.slurm
```

#### Fast Evaluation Only (Check Current Model Performance)
To evaluate the latest or best saved checkpoints without training:
```bash
python -m musvit_zeus_comparison.train \
    --model compare \
    --resume auto \
    --eval-only \
    --device cuda
```

---

### Step 5: Inspect the Generated Comparison Table

Upon completion, a formatted summary table is printed to stdout and saved to:
- `experiment_results/results_table.md` (Markdown format)
- `experiment_results/results_table.csv` (Spreadsheet format)

Example output table:

| Model Run | Encoder | Enc Params | Dec Params | Total Params | Val Loss (best) | Token Acc (%) | SER / NED (%) | Train Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Zeus Baseline (Run 1)** | CNN-BiLSTM | 851,200 | 1,185,420 | 2,036,620 | 0.8241 | 82.15% | 18.42% | 145.2s |
| **MuSViT + Zeus (Run 2)** | MuSViT + Adapter | 394,496 | 1,185,420 | 1,579,916 | 0.5120 | 89.64% | 11.20% | 42.1s |

---

## 4. Directory Layout

```
musvit_zeus_comparison/
├── __init__.py          # Package exports
├── models.py            # ZeusEncoder, MusvitEncoder, ZeusDecoder, CombinedOMRModel
├── dataset.py           # StaveOMRDataset (supports image + feature loading) & StaveCollate
├── generate_lmx.py      # MusicXML to official LMX transcription generator
├── extract_features.py  # Feature extractor with vertical pooling & FP16 compression
├── train.py             # Main trainer & evaluator with automated table generation
└── README.md            # Detailed documentation
```
