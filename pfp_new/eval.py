from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from pfp_new.load_data import ProteinFunctionSplits, load_data
from pfp_new.model import Box3Model

import torch.nn.functional as F
from torch import Tensor


CHECKPOINT_PATH = 'checkpoints/pfp_new/last_model.pt'

SEED = 42
SPLIT = (0.8, 0.1, 0.1)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ONTOLOGY = 'GO'
BASE_DIR = 'alt_data'
PROTEINS_PATH: str | None = None
TERMS_PATH: str | None = None

SCORE_BATCH_SIZE = 4096
NEG_SAMPLES = 50_000
FMAX_BATCH_SIZE = 256
FMAX_THRESHOLDS = 71


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_bot_ids(train_data: dict[str, object], classes: dict[str, int]) -> torch.Tensor:
    bot_set = {classes['owl:Nothing']}
    if 'nf1_bot' in train_data and len(train_data['nf1_bot']) > 0:
        bot_set.update(train_data['nf1_bot'][:, 0].tolist())
    return torch.tensor(sorted(bot_set), dtype=torch.long)


@torch.no_grad()
def score_pairs(
    model: Box3Model,
    pairs: torch.Tensor,
    batch_size: int = SCORE_BATCH_SIZE,
) -> torch.Tensor:
    model_was_training = model.training
    model.eval()

    scores: list[torch.Tensor] = []
    device = model.bot_ids.device

    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size].to(device)
        individuals = model.get_individual(batch[:, 0])
        class_boxes = model.get_class_box(batch[:, 1])

        loss = model.inclusion_loss(individuals, class_boxes, margin=0.0)
        scores.append(torch.exp(-loss).cpu())

    if model_was_training:
        model.train()

    if not scores:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(scores, dim=0)


def summarise_scores(scores: torch.Tensor) -> dict[str, float]:
    if len(scores) == 0:
        return {
            'mean': float('nan'),
            'std': float('nan'),
            'min': float('nan'),
            'max': float('nan'),
        }

    scores = scores.float()
    return {
        'mean': scores.mean().item(),
        'std': scores.std(unbiased=False).item(),
        'min': scores.min().item(),
        'max': scores.max().item(),
    }


def print_score_summary(name: str, scores: torch.Tensor) -> dict[str, float]:
    stats = summarise_scores(scores)
    print(
        f'{name:<16} '
        f'mean={stats["mean"]:.4f}  '
        f'std={stats["std"]:.4f}  '
        f'min={stats["min"]:.4f}  '
        f'max={stats["max"]:.4f}'
    )
    return stats


def sample_negative_pairs(
    protein_ids: torch.Tensor,
    num_classes: int,
    known_pairs: torch.Tensor,
    n_samples: int,
    seed: int,
) -> torch.Tensor:
    if len(protein_ids) == 0 or n_samples <= 0:
        return torch.empty((0, 2), dtype=torch.long)

    protein_ids = protein_ids.cpu()
    known_pairs = known_pairs.cpu()

    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)

    known_encoded = set(
        (known_pairs[:, 0] * num_classes + known_pairs[:, 1]).tolist()
    )
    sampled_encoded: set[int] = set()

    negatives: list[tuple[int, int]] = []
    proposal_batch = 8192

    while len(negatives) < n_samples:
        sampled_proteins = protein_ids[
            torch.randint(0, len(protein_ids), (proposal_batch,), generator=generator)
        ]
        sampled_classes = torch.randint(
            0, num_classes, (proposal_batch,), generator=generator
        )
        encoded = sampled_proteins * num_classes + sampled_classes

        for protein_id, class_id, code in zip(
            sampled_proteins.tolist(),
            sampled_classes.tolist(),
            encoded.tolist(),
        ):
            if code in known_encoded or code in sampled_encoded:
                continue
            sampled_encoded.add(code)
            negatives.append((protein_id, class_id))
            if len(negatives) >= n_samples:
                break

    return torch.tensor(negatives, dtype=torch.long)


def _load_closure_nf1(
    train_data: dict[str, object],
    ontology: str = ONTOLOGY,
    base_dir: str | Path = BASE_DIR,
) -> tuple[torch.Tensor, Path | None]:
    base_path = Path(base_dir)
    if not base_path.is_absolute():
        base_path = (Path(__file__).resolve().parents[1] / base_path).resolve()

    closure_path = base_path / ontology / 'DELEclosure' / 'nf1.npy'
    if closure_path.exists():
        return torch.from_numpy(np.load(closure_path)).long(), closure_path

    return train_data['nf1'].detach().cpu().long(), None


def _build_candidate_closure_sparse(
    num_classes: int,
    nf1_edges: torch.Tensor,
    classes: dict[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    excluded_ids = {classes['owl:Nothing']}
    thing_id = classes.get('owl:Thing')
    if thing_id is not None:
        excluded_ids.add(thing_id)

    candidate_ids = torch.tensor(
        sorted(set(range(num_classes)) - excluded_ids),
        dtype=torch.long,
        device=device,
    )

    class_to_candidate = torch.full(
        (num_classes,),
        -1,
        dtype=torch.long,
        device=device,
    )
    class_to_candidate[candidate_ids] = torch.arange(
        len(candidate_ids),
        dtype=torch.long,
        device=device,
    )

    edges = nf1_edges.to(device)
    sub_idx = class_to_candidate[edges[:, 0]]
    sup_idx = class_to_candidate[edges[:, 1]]
    keep = (sub_idx >= 0) & (sup_idx >= 0)

    eye = torch.arange(len(candidate_ids), device=device)
    rows = torch.cat([sub_idx[keep], eye], dim=0)
    cols = torch.cat([sup_idx[keep], eye], dim=0)
    values = torch.ones(rows.shape[0], dtype=torch.float32, device=device)

    closure = torch.sparse_coo_tensor(
        indices=torch.stack([rows, cols], dim=0),
        values=values,
        size=(len(candidate_ids), len(candidate_ids)),
        device=device,
    ).coalesce()

    return closure.transpose(0, 1).coalesce(), candidate_ids, class_to_candidate


def _close_annotation_matrix(
    raw_annotations: torch.Tensor,
    closure_t: torch.Tensor,
) -> torch.Tensor:
    closed = torch.sparse.mm(closure_t, raw_annotations.float().T).T
    return closed > 0


def _build_closed_truth_matrix(
    pairs: torch.Tensor,
    class_to_candidate: torch.Tensor,
    closure_t: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(pairs) == 0:
        empty_ids = torch.empty(0, dtype=torch.long, device=device)
        empty_truth = torch.empty(
            (0, closure_t.shape[0]),
            dtype=torch.bool,
            device=device,
        )
        empty_counts = torch.empty(0, dtype=torch.float32, device=device)
        return empty_ids, empty_truth, empty_counts

    pairs = pairs.to(device)
    protein_ids, protein_inverse = torch.unique(
        pairs[:, 0],
        sorted=True,
        return_inverse=True,
    )

    class_idx = class_to_candidate[pairs[:, 1]]
    keep = class_idx >= 0

    truth_raw = torch.zeros(
        (len(protein_ids), closure_t.shape[0]),
        dtype=torch.float32,
        device=device,
    )
    truth_raw[protein_inverse[keep], class_idx[keep]] = 1.0

    truth_closed = _close_annotation_matrix(truth_raw, closure_t)
    true_counts = truth_closed.sum(dim=1).float()
    return protein_ids, truth_closed, true_counts


@torch.no_grad()
def fmax_concept_assertions_closure(
    model: Box3Model,
    pairs: torch.Tensor,
    device: torch.device,
    closure_t: torch.Tensor,
    candidate_ids: torch.Tensor,
    class_to_candidate: torch.Tensor,
    n_thresholds: int = FMAX_THRESHOLDS,
    batch_size: int = FMAX_BATCH_SIZE,
    threshold_chunk_size: int = 8,
    t_min: float = 0.3,
    t_max: float = 1.0,
    fixed_threshold: float | None = None,
) -> dict[str, object]:
    model_was_training = model.training
    model.eval()

    protein_ids, truth_closed, true_counts = _build_closed_truth_matrix(
        pairs=pairs,
        class_to_candidate=class_to_candidate,
        closure_t=closure_t,
        device=device,
    )
    n_proteins = len(protein_ids)

    thresholds = torch.linspace(t_max, t_min, n_thresholds, device=device)

    if n_proteins == 0:
        zeros = torch.zeros(n_thresholds, device=device)
        thresholds_cpu = thresholds.cpu().numpy()
        zeros_cpu = zeros.cpu().numpy()
        if model_was_training:
            model.train()
        return {
            'fmax': 0.0,
            'threshold': float(thresholds_cpu[0]),
            'precision': 0.0,
            'recall': 0.0,
            'aupr': 0.0,
            'f1': zeros_cpu,
            'thresholds': thresholds_cpu,
            'avg_prec': zeros_cpu,
            'avg_rec': zeros_cpu,
        }

    precision_sum = torch.zeros(n_thresholds, device=device)
    recall_sum = torch.zeros(n_thresholds, device=device)
    proteins_with_predictions = torch.zeros(
        n_thresholds, dtype=torch.long, device=device
    )

    all_boxes = model.get_class_box(candidate_ids)
    all_lower = all_boxes.lower
    all_upper = all_boxes.upper

    for start in range(0, n_proteins, batch_size):
        batch_proteins = protein_ids[start:start + batch_size]
        batch_truth = truth_closed[start:start + batch_size]
        batch_true_counts = true_counts[start:start + batch_size]
        valid_true = batch_true_counts > 0

        protein_points = model.get_individual(batch_proteins).lower.unsqueeze(1)
        lower = all_lower.unsqueeze(0)
        upper = all_upper.unsqueeze(0)
        scores = torch.exp(
            -(F.relu(lower - protein_points) + F.relu(protein_points - upper)).mean(dim=-1)
        )

        for t_start in range(0, n_thresholds, threshold_chunk_size):
            t_end = min(t_start + threshold_chunk_size, n_thresholds)
            threshold_chunk = thresholds[t_start:t_end]

            raw_pred = scores.unsqueeze(1) >= threshold_chunk.view(1, -1, 1)
            flat_raw_pred = raw_pred.reshape(-1, raw_pred.shape[-1]).float()
            flat_closed_pred = _close_annotation_matrix(flat_raw_pred, closure_t)
            closed_pred = flat_closed_pred.reshape(
                scores.shape[0],
                threshold_chunk.shape[0],
                -1,
            )

            tp = (closed_pred & batch_truth.unsqueeze(1)).sum(dim=-1).float()
            pred_counts = closed_pred.sum(dim=-1).float()
            valid_pred = (pred_counts > 0) & valid_true.unsqueeze(1)

            precision_chunk = torch.where(
                valid_pred,
                tp / pred_counts.clamp_min(1.0),
                torch.zeros_like(tp),
            )
            recall_chunk = torch.where(
                valid_true.unsqueeze(1),
                tp / batch_true_counts.clamp_min(1.0).unsqueeze(1),
                torch.zeros_like(tp),
            )

            precision_sum[t_start:t_end] += precision_chunk.sum(dim=0)
            recall_sum[t_start:t_end] += recall_chunk.sum(dim=0)
            proteins_with_predictions[t_start:t_end] += valid_pred.sum(dim=0)

    avg_precision = torch.where(
        proteins_with_predictions > 0,
        precision_sum / proteins_with_predictions.float(),
        torch.zeros_like(precision_sum),
    )
    avg_recall = recall_sum / float(n_proteins)

    denom = avg_precision + avg_recall
    f1 = torch.where(
        denom > 0,
        2.0 * avg_precision * avg_recall / denom,
        torch.zeros_like(denom),
    )

    if fixed_threshold is not None:
        best_idx = int(torch.abs(thresholds - fixed_threshold).argmin().item())
    else:
        best_idx = int(f1.argmax().item())

    order = torch.argsort(avg_recall)
    aupr = float(torch.trapz(avg_precision[order], avg_recall[order]).item())

    thresholds_np = thresholds.detach().cpu().numpy()
    avg_precision_np = avg_precision.detach().cpu().numpy()
    avg_recall_np = avg_recall.detach().cpu().numpy()
    f1_np = f1.detach().cpu().numpy()

    if model_was_training:
        model.train()

    return {
        'fmax': float(f1[best_idx].item()),
        'threshold': float(thresholds[best_idx].item()),
        'precision': float(avg_precision[best_idx].item()),
        'recall': float(avg_recall[best_idx].item()),
        'aupr': aupr,
        'f1': f1_np,
        'thresholds': thresholds_np,
        'avg_prec': avg_precision_np,
        'avg_rec': avg_recall_np,
    }



@torch.no_grad()
def evaluate_model(
    model: Box3Model,
    train_data: dict[str, object],
    pf_splits: ProteinFunctionSplits,
    classes: dict[str, int],
    neg_samples: int = NEG_SAMPLES,
    score_batch_size: int = SCORE_BATCH_SIZE,
    seed: int = SEED,
) -> dict[str, dict[str, object]]:
    all_known_pairs = torch.cat(
        [pf_splits.train_pairs, pf_splits.val_pairs, pf_splits.test_pairs],
        dim=0,
    )

    split_specs = [
        ('val', pf_splits.val_pairs, pf_splits.val_protein_ids, seed + 2),
        ('test', pf_splits.test_pairs, pf_splits.test_protein_ids, seed + 3),
    ]

    results: dict[str, dict[str, object]] = {}
    nf1_edges, closure_path = _load_closure_nf1(train_data)
    closure_t, candidate_ids, class_to_candidate = _build_candidate_closure_sparse(
        num_classes=len(classes),
        nf1_edges=nf1_edges,
        classes=classes,
        device=model.bot_ids.device,
    )

    print('\nEvaluation summary\n')
    if closure_path is not None:
        print(f'Using closure graph: {closure_path}')
    else:
        print('Using train_data["nf1"] as closure graph fallback')

    for split_name, pos_pairs, protein_ids, split_seed in split_specs:
        pos_scores = score_pairs(
            model=model,
            pairs=pos_pairs,
            batch_size=score_batch_size,
        )
        pos_stats = print_score_summary(f'{split_name} positives', pos_scores)

        neg_pairs = sample_negative_pairs(
            protein_ids=protein_ids,
            num_classes=len(classes),
            known_pairs=all_known_pairs,
            n_samples=neg_samples,
            seed=split_seed,
        )
        neg_scores = score_pairs(
            model=model,
            pairs=neg_pairs,
            batch_size=score_batch_size,
        )
        neg_stats = print_score_summary(f'{split_name} negatives', neg_scores)

        results[split_name] = {
            'num_proteins': int(len(protein_ids)),
            'num_positive_pairs': int(len(pos_pairs)),
            'num_negative_pairs': int(len(neg_pairs)),
            'positives': pos_stats,
            'negatives': neg_stats,
        }

    val_fmax = fmax_concept_assertions_closure(
        model=model,
        pairs=pf_splits.val_pairs,
        device=model.bot_ids.device,
        closure_t=closure_t,
        candidate_ids=candidate_ids,
        class_to_candidate=class_to_candidate,
    )
    selected_threshold = val_fmax['threshold']

    test_at_val_threshold = fmax_concept_assertions_closure(
        model=model,
        pairs=pf_splits.test_pairs,
        device=model.bot_ids.device,
        closure_t=closure_t,
        candidate_ids=candidate_ids,
        class_to_candidate=class_to_candidate,
        fixed_threshold=selected_threshold,
    )

    print('Validation-selected threshold summary')
    print(
        f'val   fmax={val_fmax["fmax"]:.4f}  '
        f'precision={val_fmax["precision"]:.4f}  '
        f'recall={val_fmax["recall"]:.4f}  '
        f'threshold={val_fmax["threshold"]:.3f}'
    )
    print(
        f'test@val-threshold  f1={test_at_val_threshold["fmax"]:.4f}  '
        f'precision={test_at_val_threshold["precision"]:.4f}  '
        f'recall={test_at_val_threshold["recall"]:.4f}  '
        f'threshold={selected_threshold:.3f}'
    )

    results['val']['fmax'] = {
        'fmax': val_fmax['fmax'],
        'precision': val_fmax['precision'],
        'recall': val_fmax['recall'],
        'aupr': val_fmax['aupr'],
        'threshold': val_fmax['threshold'],
    }
    results['test']['fixed_threshold'] = {
        'f1': test_at_val_threshold['fmax'],
        'precision': test_at_val_threshold['precision'],
        'recall': test_at_val_threshold['recall'],
        'aupr': test_at_val_threshold['aupr'],
        'threshold': selected_threshold,
    }
    print('')
    return results


def flatten_eval_results(
    results: dict[str, dict[str, object]],
    prefix: str = 'eval',
) -> dict[str, float]:
    flat: dict[str, float] = {}

    for split_name, split_metrics in results.items():
        flat[f'{prefix}_{split_name}_num_proteins'] = float(split_metrics['num_proteins'])
        flat[f'{prefix}_{split_name}_num_positive_pairs'] = float(split_metrics['num_positive_pairs'])
        flat[f'{prefix}_{split_name}_num_negative_pairs'] = float(split_metrics['num_negative_pairs'])

        for label in ('positives', 'negatives'):
            stats = split_metrics[label]
            for stat_name, stat_value in stats.items():
                flat[f'{prefix}_{split_name}_{label}_{stat_name}'] = float(stat_value)

        if 'fmax' in split_metrics:
            for stat_name, stat_value in split_metrics['fmax'].items():
                flat[f'{prefix}_{split_name}_fmax_{stat_name}'] = float(stat_value)

        if 'fixed_threshold' in split_metrics:
            for stat_name, stat_value in split_metrics['fixed_threshold'].items():
                flat[f'{prefix}_{split_name}_fixed_threshold_{stat_name}'] = float(stat_value)

    return flat


def main() -> None:
    set_seed(SEED)

    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path.resolve()}')

    train_data, classes, relations, pf_splits = load_data(
        ontology=ONTOLOGY,
        base_dir=BASE_DIR,
        proteins_path=PROTEINS_PATH,
        terms_path=TERMS_PATH,
        seed=SEED,
        split=SPLIT,
    )

    bot_ids = build_bot_ids(train_data, classes).to(DEVICE)

    model = Box3Model.load(
        path=str(checkpoint_path),
        device=DEVICE,
        bot_ids=bot_ids,
        num_classes=len(classes),
        num_roles=len(relations),
        num_individuals=len(pf_splits.individual2id),
    ).to(DEVICE)

    model.eval()

    print('\nLoaded checkpoint.\n')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Device:     {DEVICE}')
    print(f'Concepts:   {len(classes):,}')
    print(f'Relations:  {len(relations):,}')
    print(f'Proteins:   {len(pf_splits.individual2id):,}')

    evaluate_model(
        model=model,
        train_data=train_data,
        pf_splits=pf_splits,
        classes=classes,
        neg_samples=NEG_SAMPLES,
        score_batch_size=SCORE_BATCH_SIZE,
        seed=SEED,
    )


if __name__ == '__main__':
    main()