#!/usr/bin/env python
"""
Report per-ontology benchmark statistics as a summary table.

For each ontology (or those specified on the command line) reads the bins/
directory and counts how many axioms each class / role participates in,
then reports a single-row summary table.

Usage:
    python benchmark_stats.py                      # all ontologies
    python benchmark_stats.py GALEN GO ANATOMY     # selected ontologies
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ONTOLOGIES = ['ANATOMY', 'GALEN', 'GO', 'PPI']
BASE_DIR   = Path('data')


def load_npy(path: Path):
    if path.exists():
        arr = np.load(path)
        return arr if arr.ndim == 2 else None
    return None


def collect_stats(ontology: str) -> dict | None:
    bins = BASE_DIR / ontology / 'bins'
    if not bins.exists():
        return None

    with open(bins / 'classes.json')   as f: classes   = json.load(f)
    with open(bins / 'relations.json') as f: relations = json.load(f)

    class_counter: Counter = Counter()
    role_counter:  Counter = Counter()

    # NF1 / NF1_bot: class columns
    for fname, cols in [('nf1.npy', [(0,), (1,)]),
                        ('nf1_bot.npy', [(0,)]),
                        ('nf2.npy', [(0,), (1,), (2,)]),
                        ('nf2_bot.npy', [(0,), (1,)])]:
        arr = load_npy(bins / fname)
        if arr is None:
            continue
        for (col,) in cols:
            if arr.shape[1] > col:
                class_counter.update(arr[:, col].tolist())

    # NF3: class cols 0,2 — role col 1
    arr = load_npy(bins / 'nf3.npy')
    if arr is not None and arr.shape[1] >= 3:
        class_counter.update(arr[:, 0].tolist())
        class_counter.update(arr[:, 2].tolist())
        role_counter.update(arr[:, 1].tolist())

    # NF4: role col 0 — class cols 1,2
    arr = load_npy(bins / 'nf4.npy')
    if arr is not None and arr.shape[1] >= 3:
        role_counter.update(arr[:, 0].tolist())
        class_counter.update(arr[:, 1].tolist())
        class_counter.update(arr[:, 2].tolist())

    # NF5: role cols 0,1
    arr = load_npy(bins / 'nf5.npy')
    if arr is not None and arr.shape[1] >= 2:
        role_counter.update(arr[:, 0].tolist())
        role_counter.update(arr[:, 1].tolist())

    # NF6: role cols 0,1,2
    arr = load_npy(bins / 'nf6.npy')
    if arr is not None and arr.shape[1] >= 3:
        role_counter.update(arr[:, 0].tolist())
        role_counter.update(arr[:, 1].tolist())
        role_counter.update(arr[:, 2].tolist())

    def _stats(counter: Counter) -> dict:
        if not counter:
            return dict(n=0, mean=float('nan'), std=float('nan'),
                        med=float('nan'), min=float('nan'), max=float('nan'),
                        mode=float('nan'))
        v = np.array(list(counter.values()), dtype=float)
        freq_of_freq = Counter(int(x) for x in v)
        mode = max(freq_of_freq, key=freq_of_freq.get)
        return dict(n=len(v), mean=v.mean(), std=v.std(),
                    med=float(np.median(v)), min=int(v.min()), max=int(v.max()),
                    mode=mode)

    return dict(
        ontology  = ontology,
        n_classes = len(classes),
        n_roles   = len(relations),
        cls       = _stats(class_counter),
        role      = _stats(role_counter),
    )


def fmt(val, fmt_str='.1f') -> str:
    if isinstance(val, float) and (val != val):   # nan
        return '—'
    return format(val, fmt_str)


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else ONTOLOGIES

    rows = []
    for ont in targets:
        r = collect_stats(ont)
        if r is None:
            print(f'[skip] {ont}: bins/ not found at {BASE_DIR / ont / "bins"}')
        else:
            rows.append(r)

    if not rows:
        print('No data found.')
        return

    # ── Build flat rows: one per (ontology, type) ────────────────────────────
    flat = []
    for r in rows:
        flat.append((r['ontology'], 'Classes',   r['n_classes'], r['cls']))
        flat.append((r['ontology'], 'Relations', r['n_roles'],   r['role']))

    C      = ['Ontology', 'Type', 'Count', 'Mean', 'Std', 'Median', 'Min', 'Max', 'Mode']
    widths = [max(len(c), 8) for c in C]
    widths[0] = max(widths[0], max(len(r['ontology']) for r in rows))
    widths[1] = max(widths[1], len('Relations'))

    sep  = '  '.join('-' * w for w in widths)
    head = '  '.join(c.ljust(w) for c, w in zip(C, widths))

    print()
    print(head)
    print(sep)

    prev_ont = None
    for ont, typ, count, s in flat:
        if prev_ont is not None and ont != prev_ont:
            print(sep)
        prev_ont = ont
        vals = [
            ont.ljust(widths[0]),
            typ.ljust(widths[1]),
            str(count).rjust(widths[2]),
            fmt(s['mean']).rjust(widths[3]),
            fmt(s['std']).rjust(widths[4]),
            fmt(s['med']).rjust(widths[5]),
            fmt(s['min'], 'd').rjust(widths[6]),
            fmt(s['max'], 'd').rjust(widths[7]),
            fmt(s['mode'], 'd').rjust(widths[8]),
        ]
        print('  '.join(vals))

    print()


if __name__ == '__main__':
    main()