# ELK-Em
ELK-Em: Closure-Aware Embeddings for EL++ Ontologies

## Instructions to run the codes

### 1. Deductive-closure model - `model/train.sh`

Reported numbers are the mean ± std over **five seeds (0-4)** per ontology.

`model/train.sh` runs one ontology and one seed per SLURM submission. To reproduce all reported results, edit the final `srun` line in `model/train.sh` for each ontology (`GALEN`, `GO`, `ANATOMY`) and seed (`0`, `1`, `2`, `3`, `4`), then submit the script once for each combination.

Submit one job at a time with:

```bash
sbatch model/train.sh
```

Each job writes `checkpoints/<ONT>/seed_<seed>/report.txt`.

After all 15 jobs have finished, average the results with:

```bash
python -m model.summarise
```

### 2. Lexical-regularisation ablation - `pfp_new/train.sh`

This ablation uses two SLURM submissions that differ only in `MODEL_CONFIG['lex_weight']` in `pfp_new/train.py`.

For the lexical-regularisation run, set `lex_weight` to `10.0`, then submit:

```bash
sbatch pfp_new/train.sh
```

For the ablation without lexical regularisation, set `lex_weight` to `0.0`, then submit again:

```bash
sbatch pfp_new/train.sh
```

Each job trains the model, saves `checkpoints/pfp_new/last_model.pt`, and writes evaluation metrics to its log under `pfp_new/logs/`. Compare the two logs.

### Publications
Naman Singh, Ernesto Jiménez-Ruiz, and Tillman Weyde, **ELK-Em: Closure-Aware Embeddings for EL++ Ontologies (Technical Report)**. 2026. [[PDF](https://github.com/city-artificial-intelligence/elk-em/blob/main/elk-em-report-may-2026.pdf)]