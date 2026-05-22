"""Print mean ± std of eval metrics across seeds for each ontology."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np

ONTOLOGIES = ['GALEN', 'GO', 'ANATOMY']
N_SEEDS    = 5
CKPT_DIR   = Path('checkpoints')

METRIC_RE = re.compile(
    r'eval\s+MR=([\d.]+)\s+MRR=([\d.]+)\s+Med=([\d.]+)\s+AUC=([\d.]+)'
    r'\s+H@1=([\d.]+)\s+H@10=([\d.]+)\s+H@100=([\d.]+)'
)
KEYS = ['MR', 'MRR', 'Med', 'AUC', 'H@1', 'H@10', 'H@100']

for ont in ONTOLOGIES:
    rows = []
    for seed in range(N_SEEDS):
        path = CKPT_DIR / ont / f'seed_{seed}' / 'report.txt'
        if not path.exists():
            print(f'[{ont} seed {seed}] missing')
            continue
        m = METRIC_RE.search(path.read_text())
        if m:
            rows.append([float(x) for x in m.groups()])

    print(f'\n{"="*55}  {ont}  (n={len(rows)})')
    if rows:
        arr = np.array(rows)
        for i, k in enumerate(KEYS):
            print(f'  {k:<6}  {arr[:,i].mean():.4f} ± {arr[:,i].std():.4f}')
    else:
        print('  no results found')