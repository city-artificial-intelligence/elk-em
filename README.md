# ELK-Em
ELK-Em: Closure-Aware Embeddings for EL++ Ontologies

## Instructions to run the codes
### 1. Deductive-closure model — `model/train.sh`

Reported numbers are the mean ± std over **five seeds (0–4)** per ontology.
`model/train.sh` runs one ontology + one seed per submission, so edit the last line
and resubmit for each combination:

```bash
# in model/train.sh, edit the final srun line:
#   srun python -m model.train ANATOMY --device cuda --seed 0
# change ANATOMY -> {GALEN, GO, ANATOMY} and --seed -> {0,1,2,3,4}

sbatch model/train.sh        # submit once per (ontology, seed) = 15 jobs total
```

Each job writes `checkpoints/<ONT>/seed_<seed>/report.txt`.
Once all jobs finish, average across seeds:

```bash
python -m model.summarise
```

### 2. Lexical-regularisation ablation — `pfp_new/train.sh`

Two submissions differing only in `lex_weight`. Edit
`MODEL_CONFIG['lex_weight']` in `pfp_new/train.py`, then submit the script each time:

```bash
# Run A: set 'lex_weight': 10.0  in pfp_new/train.py
sbatch pfp_new/train.sh

# Run B: set 'lex_weight': 0.0   in pfp_new/train.py
sbatch pfp_new/train.sh
```

Each job trains, saves `checkpoints/pfp_new/last_model.pt`, and writes eval metrics to
its log under `pfp_new/logs/`. Compare the two logs.

### Publications
Naman Singh , Ernesto Jiménez-Ruiz , and Tillman Weyde, **ELK-Em: Closure-Aware Embeddings for EL++ Ontologies (Technical Report)**. 2026. [[PDF](https://github.com/city-artificial-intelligence/elk-em/blob/main/elk-em-report-may-2026.pdf)]