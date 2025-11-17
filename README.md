# PLLD Release

This directory contains the cleaned release of our PLLD method. It includes:

- `code/`: the full training and evaluation pipeline (based on `PLL_PLUS`, stripped of logs, scripts, and checkpoints)
- `datasets/`: curated subsets for `amazon_book`, `amazon_movie`, and `ml_100k`

## Environment Setup

```bash
conda create -n plld python=3.8 -y
conda activate plld
pip install numpy scipy pandas tqdm matplotlib
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
# optional
pip install wandb
```

## Folder Layout

```
PLLD/
├── code/
├── datasets/
│   ├── amazon_book/
│   ├── amazon_movie/
│   └── ml_100k/
└── README.md
```

Each dataset folder provides `*.train.rating`, `*.valid.rating`, `*.test.rating`, `*.test.negative`, and a `stats.json` summary.

## Running PLLD

1. `cd code`
2. Example command (amazon_book):
   ```bash
   python main.py \
     --dataset amazon_book \
     --model GMF \
     --epochs 15 \
     --batch_size 1024 \
     --gpu 0 \
     --lr 0.001 \
     --factor_num 32 \
     --num_layers 3 \
     --num_ng 1 \
     --dropout 0.0 \
     --pll-alpha 0.5 \
     --init_noise_prob 0.3 \
     --min-pos-weight 0.2 \
     --pll_hidden_dim 128 \
     --tau-start 1.5 \
     --tau-end 0.5 \
     --transition-epochs 3 \
     --warmup-epochs 3 \
     --candidate-momentum 0.9 \
     --top_k 5 10 20 50 \
     --track_freq 50
   ```
   Replace `--dataset` with `amazon_movie` or `ml_100k` to run on other splits.
3. Enable the margin extension by adding, for example, `--noise-margin 0.1 --noise-margin-weight 0.1`.

## Notes
- PLLD looks for datasets under `../datasets/<name>` when you pass `--dataset amazon_book` (etc.). You can still point to custom paths if needed.
- All non-essential artifacts (logs, images, checkpoints) have been removed, so the repo is ready for sharing or publishing alongside your paper.
