# MusViT Embedding Extraction

This module provides tools and scripts to extract embeddings from musical staves (such as those in `OmniOMR.Small`) using **MuSViT** (*Music Score Vision Transformer*), a foundation model pre-trained on sheet music by PRAIG (University of Alicante).

## Requirements

Ensure you have the dependencies installed:
```bash
pip install -r requirements.txt
pip install transformers huggingface_hub
```

## Hugging Face Authentication

Because `PRAIG/musvit` is hosted on Hugging Face, you may need an access token (from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)):

You can provide your token in any of the following 3 ways:

1. **Hugging Face CLI (recommended)**:
   ```bash
   huggingface-cli login
   ```
2. **Environment variable**:
   * PowerShell:
     ```powershell
     $env:HF_TOKEN = "hf_your_token_here"
     ```
   * Bash / Linux:
     ```bash
     export HF_TOKEN="hf_your_token_here"
     ```
3. **Command line argument**:
   ```bash
   python musvit/main.py --token "hf_your_token_here"
   ```

## Quick Start

### 1. Extract Embeddings from the First Stave in OmniOMR.Small
```bash
python musvit/main.py
```
Or directly:
```bash
python -m musvit.embed_stave
```

### 2. Specify a Different Stave Index
```bash
python musvit/main.py --stave-idx 5
```

### 3. Provide a Direct Stave Path
```bash
python musvit/main.py --stave-path "OmniOMR.Small/028ac720-8af7-4ecc-9884-edeaf6dce2ae_325f277f-4747-412b-9e64-7dbc8c4ffdb9/Staves/3/image.jpg"
```

### 4. Save Embeddings to Disk
```bash
python musvit/main.py --stave-idx 0 --save
```
By default, this will save the extracted tensor dictionary to `musvit/outputs/stave_0_musvit_embeddings.pt`. You can also specify `--output-file my_embeddings.pt`.

### 5. Use the Lightweight Model
To use the smaller ~39.4M parameter variant:
```bash
python musvit/main.py --model-name PRAIG/musvit-light
```

---

## Python API Usage

You can also use this directly in Python code:

```python
from musvit import load_musvit_model, extract_embeddings

# 1. Load model
model, device = load_musvit_model(model_name="PRAIG/musvit")

# 2. Extract embeddings from stave index or path
embeddings = extract_embeddings(
    stave_source=0,
    model=model,
    device=device,
    dataset_dir="OmniOMR.Small"
)

# Output shapes:
print(embeddings["last_hidden_state"].shape)  # torch.Size([1, 4097, 768])
print(embeddings["cls_token"].shape)          # torch.Size([1, 768]) - Global stave embedding
print(embeddings["patch_embeddings"].shape)    # torch.Size([1, 4096, 768]) - Spatial patch tokens
print(embeddings["spatial_grid"].shape)        # torch.Size([1, 64, 64, 768]) - 2D feature grid
```

## Embedding Structure

| Key | Shape | Description |
| --- | --- | --- |
| `last_hidden_state` | `(1, 4097, 768)` | All tokens (1 CLS token + 4096 patch tokens). |
| `cls_token` | `(1, 768)` | The `[CLS]` token embedding summarizing the whole stave. |
| `patch_embeddings` | `(1, 4096, 768)` | Tokens corresponding to 64x64 patches (16x16 pixels each at 1024x1024 resolution). |
| `spatial_grid` | `(1, 64, 64, 768)` | 2D reshaped spatial feature map (ideal for downstream detection or convolution heads). |
| `mean_patch_embedding` | `(1, 768)` | Global average pooled representation across all spatial patches. |
| `original_size` | `(width, height)` | Original pixel dimensions before resizing. |
| `image_path` | `Path` | Source file path of the stave image. |
