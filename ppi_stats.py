#!/usr/bin/env python
"""
Report statistics for the PPI datasets (human / yeast).

For each organism, reports:
  - number of GO classes, roles, proteins
  - protein-link (interaction) counts across train / val / test
  - degree distribution stats (how many interactions per protein)

Usage:
    python ppi_stats.py               # both organisms
    python ppi_stats.py human         # single organism
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

PPI_DIR   = Path('data/PPI')
ORGANISMS = ['human', 'yeast']


def load_links(path: Path) -> list[tuple[str, str]]:
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split()
            pairs.append((a, b))
    return pairs


def fmt(val, fmt_str='.1f') -> str:
    if isinstance(val, float) and val != val:
        return '—'
    return format(val, fmt_str)


def collect(organism: str) -> dict:
    base = PPI_DIR / organism

    import json
    with open(base / 'classes.json')   as f: classes   = json.load(f)
    with open(base / 'relations.json') as f: relations = json.load(f)
    with open(base / 'proteins.json')  as f: proteins  = json.load(f)

    splits      = ['train', 'val', 'test']
    split_counts = {}
    degree: Counter = Counter()

    for split in splits:
        pairs = load_links(base / split / 'protein_links.txt')
        split_counts[split] = len(pairs)
        for a, b in pairs:
            degree[a] += 1
            degree[b] += 1

    # degree values (each protein counted once per split combined)
    deg = np.array(list(degree.values()), dtype=float)

    return dict(
        organism    = organism,
        n_classes   = len(classes),
        n_roles     = len(relations),
        n_proteins  = len(proteins),
        train       = split_counts['train'],
        val         = split_counts['val'],
        test        = split_counts['test'],
        total_links = sum(split_counts.values()),
        deg_mean    = deg.mean(),
        deg_std     = deg.std(),
        deg_med     = float(np.median(deg)),
        deg_min     = int(deg.min()),
        deg_max     = int(deg.max()),
        deg_mode    = int(Counter(int(x) for x in deg).most_common(1)[0][0]),
    )


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else ORGANISMS

    rows = [collect(o) for o in targets]

    # ── Overview table ────────────────────────────────────────────────────────
    print('\n── Overview ──')
    C1 = ['Organism', 'GO Classes', 'Roles', 'Proteins',
          'Train links', 'Val links', 'Test links', 'Total links']
    w1 = [max(len(c), 10) for c in C1]
    w1[0] = max(w1[0], max(len(r['organism']) for r in rows))
    sep1  = '  '.join('-' * w for w in w1)
    print('  '.join(c.ljust(w) for c, w in zip(C1, w1)))
    print(sep1)
    for r in rows:
        vals = [
            r['organism'].ljust(w1[0]),
            str(r['n_classes']).rjust(w1[1]),
            str(r['n_roles']).rjust(w1[2]),
            str(r['n_proteins']).rjust(w1[3]),
            str(r['train']).rjust(w1[4]),
            str(r['val']).rjust(w1[5]),
            str(r['test']).rjust(w1[6]),
            str(r['total_links']).rjust(w1[7]),
        ]
        print('  '.join(vals))

    # ── Degree distribution table ─────────────────────────────────────────────
    print('\n── Protein degree distribution (interactions per protein, all splits) ──')
    C2 = ['Organism', 'Mean', 'Std', 'Median', 'Min', 'Max', 'Mode']
    w2 = [max(len(c), 8) for c in C2]
    w2[0] = w1[0]
    sep2  = '  '.join('-' * w for w in w2)
    print('  '.join(c.ljust(w) for c, w in zip(C2, w2)))
    print(sep2)
    for r in rows:
        vals = [
            r['organism'].ljust(w2[0]),
            fmt(r['deg_mean']).rjust(w2[1]),
            fmt(r['deg_std']).rjust(w2[2]),
            fmt(r['deg_med']).rjust(w2[3]),
            fmt(r['deg_min'], 'd').rjust(w2[4]),
            fmt(r['deg_max'], 'd').rjust(w2[5]),
            fmt(r['deg_mode'], 'd').rjust(w2[6]),
        ]
        print('  '.join(vals))

    print()


if __name__ == '__main__':
    main()
