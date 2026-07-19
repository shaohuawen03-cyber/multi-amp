# Running MultiAMP on your machine (GTX 1650 / 4 GB)

This guide is written from a clean machine. It was validated on Linux (Debian 12,
Python 3.11) inside a sandbox, and the fixes in this repo (`crf_compat.py`,
`--fp16` flag, robust checkpoint loading) were confirmed there.

## 0. Environment note (IMPORTANT)

- Your `nvidia-smi` showed **Driver 576.02 → CUDA 12.9** and a **GTX 1650 (4 GB)**.
- **To RUN** the project you do **not** need to install the CUDA *toolkit* separately
  — the PyTorch wheel already bundles the CUDA runtime. You only need the NVIDIA
  **driver** (you have it) and a **CUDA-enabled PyTorch** build.
- **Recommended OS: Linux or WSL2 Ubuntu.** `torch-geometric` and `fair-esm` ship
  ready-made pip wheels for Linux. On **native Windows** those wheels are often
  unavailable and you'd have to compile — see the Windows note at the bottom.
  If you are on Windows, the smoothest path is **WSL2 + Ubuntu**.

## 1. Create the Python environment

```bash
# if you use conda (you have a conda 'base' env):
conda create -y -n amppre python=3.11
conda activate amppre

# or with venv:
python3 -m venv venv && source venv/bin/activate
```

## 2. Install a CUDA-enabled PyTorch (≤ your driver's CUDA 12.9)

Your driver supports up to CUDA 12.9, so install a **cu126** build (CUDA 12.6
runtime, forward-compatible with your driver). Do NOT just `pip install torch`
from PyPI — in the current release line the default PyPI wheel is **cu130**
(CUDA 13.0) which is NEWER than your driver and will fail with
`CUDA driver version is insufficient`.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Verify (must print a CUDA version ≤ 12.9 and `True`):

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# Example good output: 2.9.0+cu126 12.6 True
```

If `torch.version.cuda` shows `13.0` (you accidentally got the default PyPI wheel),
reinstall with the `--index-url .../cu126` command above.

## 3. Install the rest of the dependencies

```bash
pip install -r requirements.txt
```

> The repo already contains `crf_compat.py`, a drop-in compatibility shim for the
> `torchcrf` (TorchCRF) package. The project code originally called the CRF API
> with `batch_first=True`, `.decode(...)`, and `reduction='mean'`, none of which
> exist in TorchCRF 1.1.0. `crf_compat.py` bridges that gap, so no manual edits
> are needed.

## 4. Get the trained checkpoint + ESM-2 weights

The model needs the trained checkpoint from Hugging Face (`jiayi11/multi_amp`):

1. **The trained checkpoint** `best_model_overall.pth` (contains the full model weights, including the ESM backbone for the default config).

The repo now builds the default `esm2_t33_650M_UR50D` architecture locally and then
loads the weights from `best_model_overall.pth`, so **you do not need a second
2.6 GB ESM download** just to run prediction.

Download the checkpoint (install the HF CLI first if needed):

```bash
pip install -U huggingface_hub
mkdir -p checkpoints
hf download jiayi11/multi_amp checkpoints/best_model_overall.pth --local-dir .
# or, via git:
# git lfs install && git clone https://huggingface.co/jiayi11/multi_amp hf_tmp && cp hf_tmp/checkpoints/best_model_overall.pth checkpoints/
```

If `huggingface.co` is blocked on your network, download `best_model_overall.pth`
manually from https://huggingface.co/jiayi11/multi_amp in a browser and place it in
`checkpoints/`.

## 5. Run a prediction (FASTA mode — sequence only, no PDB needed)

This is the realistic "it works" run on a 4 GB GPU. Use **`--fp16`** and
**`--batch_size 1`** so the ~2.6 GB fp32 ESM-2 650M model fits as ~1.3 GB fp16:

```bash
python predict.py \
    --gpu 0 \
    --model_path ./checkpoints/best_model_overall.pth \
    --fasta_path ./examples/test_sequences.fasta \
    --output_path ./examples/test_sequences_predictions.csv \
    --batch_size 1 \
    --fp16
```

Expected output: a CSV (`examples/test_sequences_predictions.csv`) with columns
`id, sequence, length, probability, prediction` for each peptide in the FASTA file.
Known AMPs in that example file (magainin2, LL37, melittin, nisin) should score
high probabilities.

## 6. What about training / full evaluation?

- **Training is not feasible on a 4 GB GPU.** Fine-tuning ESM-2 650M needs ~16–24 GB
  VRAM (model + optimizer states + activations). Use a larger GPU (e.g. 24 GB+) or
  a smaller PLM (`config.PLM_NAME`) for any training.
- **Mode B** (`python predict.py --model_path ...` without `--fasta_path`) evaluates
  on the full validation set and **requires the full `data/` directory with PDB
  structures** (downloaded from the HF repo). It is much heavier and only needed for
  benchmarking, not for a first "run".
- **De novo / motif design** (`design.py`) also loads the full model; run it with
  `--fp16` and a small `--n_sequences` / `--iterations` on the 1650.

## Windows-native note

If you are NOT using WSL2, `pip install torch-geometric` may fail (no Windows wheel).
Use conda instead for that package:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
conda install -c pyg -c conda-forge pytorch-geometric
pip install fair-esm torchcrf biopython scikit-learn pandas tqdm
```

(Or, simplest: install WSL2 + Ubuntu and follow the Linux steps above.)

## Troubleshooting

- **`CUDA out of memory`** → add/keep `--fp16` and lower `--batch_size` to 1.
- **`RuntimeError: Error(s) in loading state_dict`** → the checkpoint loader now
  falls back to `strict=False` automatically. If predictions look random, the
  checkpoint likely excludes ESM-2 weights and only contains trained heads; in that
  case the ESM backbone is randomly initialized and you must obtain the full
  checkpoint.
- **`ModuleNotFoundError: No module named 'torchcrf'`** → you already have
  `crf_compat.py`; ensure you run the scripts from the repo root so
  `from crf_compat import CRF` resolves. (On case-insensitive filesystems the
  original `from torchcrf import CRF` would also resolve, but `crf_compat` is the
  safe cross-platform fix.)
- **Older checkout still tries to download ESM weights** → `git pull` first. In the
  current repo version, the default `esm2_t33_650M_UR50D` backbone is constructed
  locally and populated from `best_model_overall.pth`, so no extra Meta download is
  needed for prediction.
- **If you intentionally switch to a different PLM name** and fair-esm starts a remote
  download, a partial cache file can fail with
  `PytorchStreamReader failed reading zip archive` / `failed finding central directory`.
  The code now auto-deletes a corrupted cache file and retries once. If it still
  fails, delete `~/.cache/torch/hub/checkpoints/<model_name>*.pt` manually and rerun.
