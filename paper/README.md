# Reproducing ELK-Em (KG-NeSy 2026)

Self-contained code for the two experiments in the paper:

- **§5.1 — Ranking entailed atomic subsumptions** (Table 3, GALEN / GO / Anatomy)
- **§5.2 — Inductive protein-function prediction** (Table 4, CAFA5 human proteins)

Run everything from the repository root (`ELK-Em/`), not from `paper/`. All
paths below are relative to that root.

## Layout

```
paper/
├── requirements.txt
├── ranking/          §5.1 — ranking entailed atomic subsumptions
│   ├── ELK_Em.py         model
│   ├── geometry.py       box/role primitives
│   ├── load_data.py      normalised-axiom loader
│   ├── train.py          per-seed training entry point
│   ├── train.sh          SLURM script (one ontology, one seed per job)
│   ├── eval.py           ranking metrics + report generation
│   └── summarise.py      mean ± std across seeds
└── pfp/              §5.2 — protein-function prediction
    ├── ELKEm4pf.py       ELK-Em with ABox + lexical regularisation
    ├── geometry.py       box/role primitives (identical shape to ranking/)
    ├── load_data.py      protein splits + normalised TBox loader
    ├── load_esm.py       ESM-2 embedding loader (h5)
    ├── train.py          single-seed training entry point
    ├── train.sh          SLURM script for one run
    ├── train_multiseed.sh  worker script invoked by submit_multiseed.sh
    ├── submit_multiseed.sh six-job launcher (2 configs × 3 seeds)
    ├── eval.py           F-max / P / R / F1 with GO-hierarchy closure
    ├── aggregate_results.py  build results.md from the six eval sidecars
    └── knn/          ESM-2 kNN transfer baseline (last row of Table 4)
        ├── knn_baseline.py
        └── run_knn.sh
```

## Requirements

- Python ≥ 3.10, PyTorch with CUDA (an A100-class GPU is sufficient)
- `pip install -r paper/requirements.txt`
- Optional: a Weights & Biases account (`wandb login`). To disable, set
  `ELK_USE_WANDB=0` before submitting §5.2 jobs.

## Data

Both experiments read datasets shipped with this repository:

- **§5.1** uses the Box2EL benchmark under [data/](../data/):
  `data/{GALEN,GO,ANATOMY}/bins/*.npy` (normalised TBox) and
  `data/{GALEN,GO,ANATOMY}/inferences/{inferences,val}.json` (entailed
  atomic subsumptions computed with ELK).
- **§5.2** uses CAFA5 human protein–function annotations under
  [alt_data/GO/](../alt_data/GO/):
  `alt_data/GO/bins/*.npy` (normalised GO TBox),
  `alt_data/GO/DELEclosure/nf1.npy` (candidate-closure graph),
  `alt_data/GO/cafa-5/{human_proteins.fasta,esm2_480.h5}`, and
  `alt_data/GO/cafa-5/train_terms.tsv`, obtained by running
  `alt_data/GO/cafa-5/fetch_data.sh` (not redistributed here).
  `esm2_480.h5` contains 480-d ESM-2 (t12, 35M) embeddings mean-pooled over
  residues.

## §5.1 — Ranking entailed atomic subsumptions (Table 3)

Reported numbers are means (± std) over **five seeds (0–4)** per ontology.
`train.sh` runs one `(ontology, seed)` per SLURM job — submit fifteen jobs
in total.

```bash
for ont in GALEN GO ANATOMY; do
  for seed in 0 1 2 3 4; do
    ONT="$ont" SEED="$seed" sbatch paper/ranking/train.sh
  done
done
```

Each job writes `checkpoints/ranking/<ONT>/seed_<seed>/report.txt`. Per
ontology, the training config is identical except for `N_NEG_PER_POS`
(the number of corruptions per positive): 5 for GO, 3 for GALEN and
ANATOMY. The launcher sets this automatically from `$ONT`; override with
`N_NEG_PER_POS=<k>` if you need to.

Other hyperparameters (single configuration across the three ontologies,
matching the paper):

| flag                | value  |
|---------------------|--------|
| optimiser           | Adam   |
| learning rate       | 0.01   |
| LR schedule         | ×0.1 at epoch 5,000 (`MultiStepLR`) |
| margin γ            | 0.1    |
| batch size          | 1,024  |
| embedding dim       | 50     |
| epochs              | 10,000 |
| contrastive mode    | pearson |
| contrastive factor  | 3.0    |
| reg factor          | 1.0    |

After all 15 jobs have finished, average with:

```bash
python -m paper.ranking.summarise
```

## §5.2 — Inductive protein-function prediction (Table 4)

Three seeds × two configs = **six SLURM jobs**. `full` uses lexical
regularisation with weight 10.0; `no_lex` disables it (weight 0.0).

```bash
bash paper/pfp/submit_multiseed.sh
```

Each job writes `last_model.pt`, `last_model.efficiency.json` and
`last_model.eval.json` under
`checkpoints/pfp/{full,no_lex}/seed_{0,1,2}/`.

For a fast smoke test at reduced epochs:

```bash
ELK_EPOCHS=200 bash paper/pfp/submit_multiseed.sh
```

Full training config:

| symbol / field                    | value  |
|-----------------------------------|--------|
| margin γ                          | 0.05   |
| lexical weight (in front of ℒ_lex)| 10.0 (`full`) / 0.0 (`no_lex`) |
| lexical decay rate λ              | 1.0    |
| embedding dim                     | 100    |
| batch size (TBox / ABox / lex)    | 1024 / 1024 / 2000 |
| negatives per positive            | 2      |
| target score threshold τ          | 0.4    |
| ABox positive / negative weights  | 2.0 / 0.5 |
| box regularisation factor         | 0.001  |
| optimiser                         | Adam   |
| initial LR                        | 0.01   |
| total epochs                      | 15,000 |
| LR schedule                       | constant for the first 4,950 epochs, then cosine-anneal to lr × 0.01 |
| ESM-2 variant                     | `esm2_t12_35M_UR50D` (mean-pooled over residues, dim 480) |

The **ESM-2 kNN transfer baseline** (the row that outperforms ELK-Em) is a
separate one-off job:

```bash
sbatch paper/pfp/knn/run_knn.sh
```

It writes `paper/pfp/knn/results.json`. `k` and the decision threshold are
selected on the validation split from `k ∈ {1, 3, 5, 10, 20, 50}` and 71
threshold values in `[0.3, 1.0]`.

After all seven jobs (six ELK-Em + one kNN) have finished, aggregate with:

```bash
python -m paper.pfp.aggregate_results
```

which writes `paper/pfp/results.md` with the two blocks of Table 4.

## Citation

Naman Singh, Ernesto Jiménez-Ruiz, Tillman Weyde.
*ELK-Em: Closure-Aware Embeddings for ℰℒ+⊥ Ontologies.*
KG-NeSy 2026 (workshop at ISWC 2026).
