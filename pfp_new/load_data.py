from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from model.load_data import load_bins as load_tbox_bins, load_inferences


_BOX3EL_ROOT = Path(__file__).resolve().parents[1]
_CAFA_ROOT = _BOX3EL_ROOT / 'alt_data' / 'GO' / 'cafa-5'
_IRI_RE = re.compile(r'<([^>]+)>')
_GO_IRI_PREFIX = 'http://purl.obolibrary.org/obo/'


@dataclass(frozen=True)
class ProteinFunctionSplits:
    train_pairs: torch.Tensor
    val_pairs: torch.Tensor
    test_pairs: torch.Tensor
    individual2id: dict[str, int]
    id2individual: list[str]
    train_protein_ids: torch.Tensor
    val_protein_ids: torch.Tensor
    test_protein_ids: torch.Tensor
    proteins_path: Path
    terms_path: Path
    skipped_annotations: int


def _resolve_path(path: str | Path | None, default_rel: str) -> Path:
    if path is not None:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = (_BOX3EL_ROOT / resolved).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f'File not found: {resolved}')
        return resolved

    candidate = _CAFA_ROOT / default_rel
    if not candidate.exists():
        raise FileNotFoundError(f'Could not find {default_rel}. Searched: {candidate}')
    return candidate


def _map_class_id(go_term: str, classes: dict[str, int]) -> int | None:
    term = go_term.strip()
    match = _IRI_RE.match(term)
    if match:
        term = match.group(1)

    if term.startswith(_GO_IRI_PREFIX):
        local = term.rsplit('/', 1)[-1]
        colon = local.replace('_', ':', 1) if local.startswith('GO_') else local
        candidates = (term, local, colon)
    elif term.startswith('GO:'):
        local = term.replace(':', '_', 1)
        candidates = (f'{_GO_IRI_PREFIX}{local}', local, term)
    elif term.startswith('GO_'):
        candidates = (f'{_GO_IRI_PREFIX}{term}', term, term.replace('_', ':', 1))
    else:
        candidates = (term,)

    for candidate in candidates:
        class_id = classes.get(candidate)
        if class_id is not None:
            return class_id
    return None


def load_protein_functions(
    classes: dict[str, int],
    proteins_path: str | Path | None = None,
    terms_path: str | Path | None = None,
    seed: int = 42,
    split: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> ProteinFunctionSplits:
    train_frac, val_frac, _ = split
    if min(split) < 0 or abs(sum(split) - 1.0) > 1e-6:
        raise ValueError(f'split must sum to 1.0, got {split!r}')

    resolved_proteins = _resolve_path(proteins_path, 'cafa5_proteins.txt')
    resolved_terms = _resolve_path(terms_path, 'Train/train_terms.tsv')

    with open(resolved_proteins) as handle:
        protein_names = sorted({line.strip() for line in handle if line.strip()})
    individual2id = {name: idx for idx, name in enumerate(protein_names)}

    all_pairs: list[tuple[int, int]] = []
    skipped = 0
    with open(resolved_terms) as handle:
        next(handle, None)
        for line in handle:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue

            protein_id = individual2id.get(parts[0])
            class_id = _map_class_id(parts[1], classes)
            if protein_id is None or class_id is None:
                skipped += 1
                continue
            all_pairs.append((protein_id, class_id))

    rng = random.Random(seed)
    shuffled = protein_names[:]
    rng.shuffle(shuffled)
    n_proteins = len(shuffled)
    n_train = int(train_frac * n_proteins)
    n_val = int(val_frac * n_proteins)

    train_ids = {individual2id[p] for p in shuffled[:n_train]}
    val_ids = {individual2id[p] for p in shuffled[n_train:n_train + n_val]}
    test_ids = {individual2id[p] for p in shuffled[n_train + n_val:]}

    def _split(ids: set[int]) -> list[tuple[int, int]]:
        return [(protein_id, class_id) for protein_id, class_id in all_pairs if protein_id in ids]

    train_raw = _split(train_ids)
    val_raw = _split(val_ids)
    test_raw = _split(test_ids)

    def _pair_tensor(rows: list[tuple[int, int]]) -> torch.Tensor:
        if rows:
            return torch.tensor(rows, dtype=torch.long)
        return torch.empty((0, 2), dtype=torch.long)

    def _id_tensor(ids: set[int]) -> torch.Tensor:
        if ids:
            return torch.tensor(sorted(ids), dtype=torch.long)
        return torch.empty((0,), dtype=torch.long)

    return ProteinFunctionSplits(
        train_pairs=_pair_tensor(train_raw),
        val_pairs=_pair_tensor(val_raw),
        test_pairs=_pair_tensor(test_raw),
        individual2id=individual2id,
        id2individual=protein_names,
        train_protein_ids=_id_tensor(train_ids),
        val_protein_ids=_id_tensor(val_ids),
        test_protein_ids=_id_tensor(test_ids),
        proteins_path=resolved_proteins,
        terms_path=resolved_terms,
        skipped_annotations=skipped,
    )


def load_data(
    ontology: str = 'GO',
    base_dir: str | Path = 'alt_data',
    proteins_path: str | Path | None = None,
    terms_path: str | Path | None = None,
    seed: int = 42,
    split: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> tuple[dict[str, object], dict[str, int], dict[str, int], ProteinFunctionSplits]:
    resolved_base = Path(base_dir)
    if not resolved_base.is_absolute():
        resolved_base = (_BOX3EL_ROOT / resolved_base).resolve()

    train_data, classes, relations = load_tbox_bins(ontology, str(resolved_base))

    pf_splits = load_protein_functions(
        classes=classes,
        proteins_path=proteins_path,
        terms_path=terms_path,
        seed=seed,
        split=split,
    )

    combined_data = dict(train_data)
    combined_data['abox'] = {'concept_assertions': pf_splits.train_pairs}
    return combined_data, classes, relations, pf_splits


__all__ = [
    'ProteinFunctionSplits',
    'load_data',
    'load_inferences',
    'load_protein_functions',
    'load_tbox_bins',
]