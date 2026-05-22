
from __future__ import annotations

import json
import re
from pathlib import Path
import numpy as np
import torch

_IRI_RE = re.compile(r'<([^>]+)>')

OWL_NOTHING = 'owl:Nothing'
OWL_THING   = 'owl:Thing'


def _rhs_iri(text: str) -> str:
    m = _IRI_RE.match(text.strip())
    return m.group(1) if m else text.strip().split()[0]


def _existential_parts(inner: str) -> tuple[str | None, str | None, str]:

    depth, i = 1, 0
    while i < len(inner) and depth > 0:
        if inner[i] == '(':   depth += 1
        elif inner[i] == ')': depth -= 1
        i += 1
    iris = _IRI_RE.findall(inner[:i - 1])
    remainder = inner[i:].strip()
    if len(iris) < 2:
        return None, None, remainder
    return iris[0], iris[1], remainder


def _intersection_parts(inner: str) -> tuple[str | None, str | None, str]:

    depth, i = 1, 0
    while i < len(inner) and depth > 0:
        if inner[i] == '(':   depth += 1
        elif inner[i] == ')': depth -= 1
        i += 1
    iris = _IRI_RE.findall(inner[:i - 1])
    remainder = inner[i:].strip()
    if len(iris) < 2:
        return None, None, remainder
    return iris[0], iris[1], remainder


def _parse_norm_file(path: Path) -> tuple[dict, set, set]:
    raw: dict[str, list] = {
        'nf1': [], 'nf1_bot': [],
        'nf2': [], 'nf2_bot': [],
        'nf3': [],
        'nf4': [], 'nf4_bot': [],
        'nf5': [], 'nf6': [],
    }
    all_classes:   set[str] = set()
    all_relations: set[str] = set()

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('SubObjectPropertyOf('):
                inner = line[20:-1].strip()
                if inner.startswith('ObjectPropertyChain('):
                    iris = _IRI_RE.findall(inner)
                    if len(iris) >= 3:
                        r1, r2, s = iris[0], iris[1], iris[2]
                        all_relations.update([r1, r2, s])
                        raw['nf6'].append((r1, r2, s))
                else:
                    iris = _IRI_RE.findall(inner)
                    if len(iris) >= 2:
                        all_relations.update([iris[0], iris[1]])
                        raw['nf5'].append((iris[0], iris[1]))
                continue

            if not line.startswith('SubClassOf('):
                continue

            inner = line[11:-1].strip()

            if inner.startswith('ObjectIntersectionOf('):
                c, d, remainder = _intersection_parts(inner[21:])
                if c is None:
                    continue
                e = _rhs_iri(remainder)
                all_classes.update([c, d])
                if e == OWL_NOTHING:
                    raw['nf2_bot'].append((c, d))
                else:
                    all_classes.add(e)
                    raw['nf2'].append((c, d, e))

            elif inner.startswith('ObjectSomeValuesFrom('):
                r, c, remainder = _existential_parts(inner[21:])
                if r is None:
                    continue
                d = _rhs_iri(remainder)
                all_relations.add(r)
                all_classes.add(c)
                if d == OWL_NOTHING:
                    raw['nf4_bot'].append((r, c))
                else:
                    all_classes.add(d)
                    raw['nf4'].append((r, c, d))

            else:
                c_match = _IRI_RE.match(inner)
                if not c_match:
                    continue
                c   = c_match.group(1)
                rhs = inner[c_match.end():].strip()

                if rhs.startswith('ObjectSomeValuesFrom('):
                    r, d, _ = _existential_parts(rhs[21:])
                    if r is None:
                        continue
                    all_classes.update([c, d])
                    all_relations.add(r)
                    raw['nf3'].append((c, r, d))
                else:
                    d = _rhs_iri(rhs)
                    all_classes.add(c)
                    if d == OWL_NOTHING:
                        raw['nf1_bot'].append((c,))
                    else:
                        all_classes.add(d)
                        raw['nf1'].append((c, d))

    return raw, all_classes, all_relations


def _propagate_bot(raw: dict) -> None:

    bot_set = {c for (c,) in raw['nf1_bot']} | {OWL_NOTHING}
    changed = True
    while changed:
        changed = False
        for (sub, sup) in raw['nf1']:
            if sup in bot_set and sub not in bot_set:
                bot_set.add(sub)
                changed = True
    raw['nf1_bot'] = [(c,) for c in sorted(bot_set)]

    nf2_keep, nf2_bot_new = [], []
    for (c, d, e) in raw['nf2']:
        if e in bot_set:
            nf2_bot_new.append((c, d))
        else:
            nf2_keep.append((c, d, e))
    raw['nf2'] = nf2_keep
    raw['nf2_bot'].extend(nf2_bot_new)

    nf4_keep, nf4_bot_new = [], []
    for (r, c, d) in raw['nf4']:
        if d in bot_set:
            nf4_bot_new.append((r, c))
        else:
            nf4_keep.append((r, c, d))
    raw['nf4'] = nf4_keep
    raw['nf4_bot'].extend(nf4_bot_new)


def _build_dicts(
    all_classes: set[str],
    all_relations: set[str],
) -> tuple[dict[str, int], dict[str, int]]:
    named = sorted(all_classes - {OWL_NOTHING, OWL_THING})
    classes: dict[str, int] = {OWL_NOTHING: 0}
    for i, name in enumerate(named, start=1):
        classes[name] = i
    classes[OWL_THING] = len(classes)  # last index

    relations: dict[str, int] = {
        name: i for i, name in enumerate(sorted(all_relations))
    }
    return classes, relations


def _to_tensors(
    raw: dict,
    classes: dict[str, int],
    relations: dict[str, int],
) -> dict[str, torch.Tensor]:
    mappings = {
        'nf1':     (raw['nf1'],     lambda c, d:        (classes[c],    classes[d])),
        'nf1_bot': (raw['nf1_bot'], lambda c:           (classes[c],)),
        'nf2':     (raw['nf2'],     lambda c, d, e:     (classes[c],    classes[d],    classes[e])),
        'nf2_bot': (raw['nf2_bot'], lambda c, d:        (classes[c],    classes[d])),
        'nf3':     (raw['nf3'],     lambda c, r, d:     (classes[c],    relations[r],  classes[d])),
        'nf4':     (raw['nf4'],     lambda r, c, d:     (relations[r],  classes[c],    classes[d])),
        'nf4_bot': (raw['nf4_bot'], lambda r, c:        (relations[r],  classes[c])),
        'nf5':     (raw['nf5'],     lambda r, s:        (relations[r],  relations[s])),
        'nf6':     (raw['nf6'],     lambda r1, r2, s:   (relations[r1], relations[r2], relations[s])),
    }
    data: dict[str, torch.Tensor] = {}
    for key, (tuples, fn) in mappings.items():
        if tuples:
            data[key] = torch.tensor([fn(*t) for t in tuples], dtype=torch.long)
    return data


def build_bins(ontology: str, base_dir: str = 'data', norm_path: str | Path | None = None) -> tuple[dict, dict, dict]:
    """Parse the norm file, save bins + JSON dicts, return them."""
    if norm_path is not None:
        norm_file = Path(norm_path)
    else:
        norm_file = Path(base_dir) / ontology / f'{ontology}_norm_full.norm'
    if not norm_file.exists():
        raise FileNotFoundError(f'Norm file not found: {norm_file}')

    raw, all_classes, all_relations = _parse_norm_file(norm_file)
    _propagate_bot(raw)
    all_classes.add(OWL_THING)  # always present

    classes, relations = _build_dicts(all_classes, all_relations)
    data = _to_tensors(raw, classes, relations)

    out_dir = Path(base_dir) / ontology / 'bins'
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'classes.json', 'w') as f:
        json.dump(classes, f, indent=2)
    with open(out_dir / 'relations.json', 'w') as f:
        json.dump(relations, f, indent=2)
    for key, tensor in data.items():
        np.save(out_dir / f'{key}.npy', tensor.numpy())

    print(f'Saved bins to {out_dir}')
    print(f'  classes:   {len(classes)}  (owl:Nothing=0, owl:Thing={len(classes)-1})')
    print(f'  relations: {len(relations)}')
    for key, tensor in data.items():
        print(f'  {key}: {tensor.shape}')

    return data, classes, relations


def load_bins(ontology: str, base_dir: str = 'data') -> tuple[dict, dict, dict]:

    bins_dir = Path(base_dir) / ontology / 'bins'

    with open(bins_dir / 'classes.json') as f:
        classes: dict[str, int] = json.load(f)
    with open(bins_dir / 'relations.json') as f:
        relations: dict[str, int] = json.load(f)

    data: dict[str, torch.Tensor] = {}
    for npy_file in sorted(bins_dir.glob('*.npy')):
        data[npy_file.stem] = torch.from_numpy(np.load(npy_file))

    return data, classes, relations

def load_inferences(path, classes: dict) -> torch.Tensor:
    with open(path) as f:
        pairs = json.load(f)
    rows = []
    for a, b in pairs:
        c = _IRI_RE.match(a).group(1) if _IRI_RE.match(a) else a
        d = _IRI_RE.match(b).group(1) if _IRI_RE.match(b) else b
        if c in classes and d in classes:
            rows.append((classes[c], classes[d]))
    return torch.tensor(rows, dtype=torch.long)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build NF bins from a norm file.')
    parser.add_argument('ontology', help='Ontology name, e.g. GALEN')
    parser.add_argument('--base-dir', default='data', help='Base data directory (default: data)')
    parser.add_argument('--norm-path', default=None, help='Override norm file path')
    args = parser.parse_args()
    build_bins(args.ontology, base_dir=args.base_dir, norm_path=args.norm_path)

