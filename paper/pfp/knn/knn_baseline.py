"""ESM-2 kNN transfer baseline for CAFA5 protein-function prediction.

Shares the evaluation path with ELK-Em:
    - same splits (via paper.pfp.load_data.load_data)
    - same ESM-2 embeddings (via paper.pfp.load_esm.load_esm_embeddings)
    - same candidate GO term set (all classes minus owl:Nothing, owl:Thing)
    - same GO-hierarchy propagation (closure imported from paper.pfp.eval)
    - same 71-threshold sweep in [0.3, 1.0]
    - same protein-centric F-max metric (sweep duplicated verbatim here so
      eval.py stays untouched)
    - same val -> test threshold transfer

The kNN score is defined as
    s(G | p) = sum_{q in N_k(p)} sim(p,q) * 1[G in ann(q)]
             / sum_{q in N_k(p)} sim(p,q)
using **propagated** training annotations for ann(q).

Both k and the decision threshold are selected on the validation split only.

Usage:
    python -m paper.pfp.knn.knn_baseline
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from paper.pfp.load_data import load_data
from paper.pfp.load_esm import load_esm_embeddings

from paper.pfp.eval import (
    _build_candidate_closure_sparse,
    _build_closed_truth_matrix,
    _close_annotation_matrix,
    _load_closure_nf1,
)


SEED = 42
SPLIT = (0.8, 0.1, 0.1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ESM_PATH = "alt_data/GO/cafa-5/esm2_480.h5"

K_GRID = (1, 3, 5, 10, 20, 50)

FMAX_THRESHOLDS = 71
FMAX_BATCH_SIZE = 256
THRESHOLD_CHUNK_SIZE = 8
T_MIN = 0.3
T_MAX = 1.0

QUERY_CHUNK = 256

OUT_JSON = "paper/pfp/knn/results.json"
OUT_MD = "paper/pfp/knn/results.md"


def build_propagated_annotations(
    pairs: torch.Tensor,
    protein_ids: torch.Tensor,
    class_to_candidate: torch.Tensor,
    closure_t: torch.Tensor,
    n_candidates: int,
    device: torch.device,
) -> torch.Tensor:
    """(n_proteins, n_candidates) propagated annotations for a split.

    Rows are indexed by position in `protein_ids` (sorted 1-D tensor of the
    protein ids). Only candidate GO terms are kept; the closure matrix
    propagates annotations up the GO hierarchy via `_close_annotation_matrix`
    (same helper as ELK-Em).
    """
    row_lookup = torch.full(
        (int(protein_ids.max().item()) + 1,),
        -1,
        dtype=torch.long,
        device=device,
    )
    row_lookup[protein_ids.to(device)] = torch.arange(
        len(protein_ids), dtype=torch.long, device=device
    )

    pairs = pairs.to(device)
    class_idx = class_to_candidate[pairs[:, 1]]
    row_idx = row_lookup[pairs[:, 0]]
    keep = (class_idx >= 0) & (row_idx >= 0)

    raw = torch.zeros(
        (len(protein_ids), n_candidates), dtype=torch.float32, device=device
    )
    raw[row_idx[keep], class_idx[keep]] = 1.0

    return _close_annotation_matrix(raw, closure_t).float()


def compute_knn_scores(
    query_esm_n: torch.Tensor,
    train_esm_n: torch.Tensor,
    train_ann_closed: torch.Tensor,
    k: int,
    chunk: int = QUERY_CHUNK,
) -> torch.Tensor:
    """Similarity-weighted top-k transfer.

    query_esm_n:      (Q, D) L2-normalised ESM embeddings
    train_esm_n:      (T, D) L2-normalised ESM embeddings
    train_ann_closed: (T, C) propagated training annotations (float, 0/1)
    Returns: (Q, C) scores.
    """
    device = query_esm_n.device
    Q = query_esm_n.shape[0]
    C = train_ann_closed.shape[1]
    scores = torch.empty((Q, C), device=device, dtype=torch.float32)

    for start in range(0, Q, chunk):
        end = min(start + chunk, Q)
        sim = query_esm_n[start:end] @ train_esm_n.T  # (b, T)
        topk_sim, topk_idx = sim.topk(k, dim=1)

        # Sparse-weight matmul: only the top-k neighbours contribute.
        M = torch.zeros_like(sim)
        M.scatter_(1, topk_idx, topk_sim)

        weighted = M @ train_ann_closed  # (b, C)
        weights_sum = topk_sim.sum(dim=1, keepdim=True).clamp_min(1e-12)
        scores[start:end] = weighted / weights_sum

    return scores


@torch.no_grad()
def fmax_from_score_matrix(
    pairs: torch.Tensor,
    score_matrix: torch.Tensor,
    protein_row_lookup: torch.Tensor,
    closure_t: torch.Tensor,
    class_to_candidate: torch.Tensor,
    device: torch.device,
    n_thresholds: int = FMAX_THRESHOLDS,
    batch_size: int = FMAX_BATCH_SIZE,
    threshold_chunk_size: int = THRESHOLD_CHUNK_SIZE,
    t_min: float = T_MIN,
    t_max: float = T_MAX,
    fixed_threshold: float | None = None,
) -> dict[str, object]:
    """Threshold sweep + closure of predictions + protein-centric P/R/F1.

    DUPLICATED verbatim from
    paper.pfp.eval.fmax_concept_assertions_closure so eval.py stays
    untouched. The only differences: scores come from a precomputed
    `score_matrix` instead of the model, and batch protein ids are mapped
    through `protein_row_lookup` to score-matrix rows. Everything else --
    closure propagation of predictions, protein-centric averaging, val->test
    threshold transfer -- is identical.
    """
    protein_ids, truth_closed, true_counts = _build_closed_truth_matrix(
        pairs=pairs,
        class_to_candidate=class_to_candidate,
        closure_t=closure_t,
        device=device,
    )
    n_proteins = len(protein_ids)
    thresholds = torch.linspace(t_max, t_min, n_thresholds, device=device)

    if n_proteins == 0:
        zeros = torch.zeros(n_thresholds).numpy()
        return {
            "fmax": 0.0,
            "threshold": float(thresholds[0].item()),
            "precision": 0.0,
            "recall": 0.0,
            "aupr": 0.0,
            "f1": zeros,
            "thresholds": thresholds.cpu().numpy(),
            "avg_prec": zeros,
            "avg_rec": zeros,
        }

    precision_sum = torch.zeros(n_thresholds, device=device)
    recall_sum = torch.zeros(n_thresholds, device=device)
    proteins_with_predictions = torch.zeros(
        n_thresholds, dtype=torch.long, device=device
    )

    for start in range(0, n_proteins, batch_size):
        batch_proteins = protein_ids[start:start + batch_size]
        batch_truth = truth_closed[start:start + batch_size]
        batch_true_counts = true_counts[start:start + batch_size]
        valid_true = batch_true_counts > 0

        rows = protein_row_lookup[batch_proteins]
        scores = score_matrix[rows]

        for t_start in range(0, n_thresholds, threshold_chunk_size):
            t_end = min(t_start + threshold_chunk_size, n_thresholds)
            threshold_chunk = thresholds[t_start:t_end]

            raw_pred = scores.unsqueeze(1) >= threshold_chunk.view(1, -1, 1)
            flat_raw_pred = raw_pred.reshape(-1, raw_pred.shape[-1]).float()
            flat_closed_pred = _close_annotation_matrix(flat_raw_pred, closure_t)
            closed_pred = flat_closed_pred.reshape(
                scores.shape[0], threshold_chunk.shape[0], -1
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

    return {
        "fmax": float(f1[best_idx].item()),
        "threshold": float(thresholds[best_idx].item()),
        "precision": float(avg_precision[best_idx].item()),
        "recall": float(avg_recall[best_idx].item()),
        "aupr": aupr,
        "f1": f1.detach().cpu().numpy(),
        "thresholds": thresholds.detach().cpu().numpy(),
        "avg_prec": avg_precision.detach().cpu().numpy(),
        "avg_rec": avg_recall.detach().cpu().numpy(),
    }


def make_row_lookup(
    protein_ids: torch.Tensor,
    num_proteins_total: int,
    device: torch.device,
) -> torch.Tensor:
    lookup = torch.full((num_proteins_total,), -1, dtype=torch.long, device=device)
    lookup[protein_ids.to(device)] = torch.arange(
        len(protein_ids), dtype=torch.long, device=device
    )
    return lookup


def main() -> None:
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")

    print("Loading splits and TBox...")
    train_data, classes, relations, pf_splits = load_data(seed=SEED, split=SPLIT)
    print(
        f"  proteins={len(pf_splits.individual2id):,}  "
        f"train={len(pf_splits.train_protein_ids):,}  "
        f"val={len(pf_splits.val_protein_ids):,}  "
        f"test={len(pf_splits.test_protein_ids):,}"
    )
    print(f"  classes={len(classes):,}")

    print("Loading ESM-2 embeddings...")
    esm_embs, prot_to_esm = load_esm_embeddings(
        ESM_PATH, pf_splits.individual2id, device=DEVICE
    )
    print(f"  esm shape={tuple(esm_embs.shape)}  dtype={esm_embs.dtype}")

    print("Building candidate closure...")
    nf1_edges, closure_path = _load_closure_nf1(train_data)
    if closure_path is not None:
        print(f"  using DELE closure: {closure_path}")
    else:
        print("  using train_data['nf1'] as closure fallback")
    closure_t, candidate_ids, class_to_candidate = _build_candidate_closure_sparse(
        num_classes=len(classes),
        nf1_edges=nf1_edges,
        classes=classes,
        device=DEVICE,
    )
    n_candidates = len(candidate_ids)
    print(f"  n_candidates={n_candidates:,}")

    print("Building propagated training annotations...")
    train_pids = pf_splits.train_protein_ids.to(DEVICE)
    val_pids = pf_splits.val_protein_ids.to(DEVICE)
    test_pids = pf_splits.test_protein_ids.to(DEVICE)

    t0 = time.perf_counter()
    train_ann_closed = build_propagated_annotations(
        pairs=pf_splits.train_pairs,
        protein_ids=train_pids,
        class_to_candidate=class_to_candidate,
        closure_t=closure_t,
        n_candidates=n_candidates,
        device=DEVICE,
    )
    print(
        f"  train_ann_closed shape={tuple(train_ann_closed.shape)}  "
        f"nnz={int(train_ann_closed.sum().item()):,}  "
        f"({time.perf_counter()-t0:.1f}s)"
    )

    print("Preparing ESM matrices...")
    esm_train = esm_embs[prot_to_esm[train_pids]].float()
    esm_val = esm_embs[prot_to_esm[val_pids]].float()
    esm_test = esm_embs[prot_to_esm[test_pids]].float()

    esm_train_n = F.normalize(esm_train, dim=-1)
    esm_val_n = F.normalize(esm_val, dim=-1)
    esm_test_n = F.normalize(esm_test, dim=-1)

    num_proteins_total = len(pf_splits.individual2id)
    val_row_lookup = make_row_lookup(val_pids, num_proteins_total, DEVICE)
    test_row_lookup = make_row_lookup(test_pids, num_proteins_total, DEVICE)

    print("\n== Validation sweep over k ==")
    val_results: dict[int, dict[str, object]] = {}
    for k in K_GRID:
        t0 = time.perf_counter()
        val_scores = compute_knn_scores(esm_val_n, esm_train_n, train_ann_closed, k=k)
        res = fmax_from_score_matrix(
            pairs=pf_splits.val_pairs,
            score_matrix=val_scores,
            protein_row_lookup=val_row_lookup,
            closure_t=closure_t,
            class_to_candidate=class_to_candidate,
            device=DEVICE,
        )
        val_results[k] = res
        print(
            f"  k={k:<3d}  fmax={res['fmax']:.4f}  "
            f"P={res['precision']:.4f}  R={res['recall']:.4f}  "
            f"thr={res['threshold']:.3f}  "
            f"({time.perf_counter()-t0:.1f}s)"
        )
        del val_scores
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    best_k = max(val_results, key=lambda k: val_results[k]["fmax"])
    best_val = val_results[best_k]
    print(
        f"\nSelected on val:  k={best_k}  threshold={best_val['threshold']:.3f}  "
        f"val_fmax={best_val['fmax']:.4f}"
    )

    print("\n== Test with (best_k, best_threshold) fixed from val ==")
    t0 = time.perf_counter()
    test_scores = compute_knn_scores(
        esm_test_n, esm_train_n, train_ann_closed, k=best_k
    )
    test_res = fmax_from_score_matrix(
        pairs=pf_splits.test_pairs,
        score_matrix=test_scores,
        protein_row_lookup=test_row_lookup,
        closure_t=closure_t,
        class_to_candidate=class_to_candidate,
        device=DEVICE,
        fixed_threshold=best_val["threshold"],
    )
    print(
        f"  test  F1={test_res['fmax']:.4f}  "
        f"P={test_res['precision']:.4f}  R={test_res['recall']:.4f}  "
        f"thr={test_res['threshold']:.3f}  "
        f"({time.perf_counter()-t0:.1f}s)"
    )

    payload = {
        "seed": SEED,
        "split": list(SPLIT),
        "esm_path": ESM_PATH,
        "esm_dim": int(esm_embs.shape[1]),
        "n_train": int(len(train_pids)),
        "n_val": int(len(val_pids)),
        "n_test": int(len(test_pids)),
        "n_candidates": int(n_candidates),
        "k_grid": list(K_GRID),
        "n_thresholds": FMAX_THRESHOLDS,
        "t_min": T_MIN,
        "t_max": T_MAX,
        "val_by_k": {
            int(k): {
                "fmax": float(res["fmax"]),
                "precision": float(res["precision"]),
                "recall": float(res["recall"]),
                "threshold": float(res["threshold"]),
                "aupr": float(res["aupr"]),
            }
            for k, res in val_results.items()
        },
        "selected_k": int(best_k),
        "selected_threshold": float(best_val["threshold"]),
        "val_at_selected": {
            "fmax": float(best_val["fmax"]),
            "precision": float(best_val["precision"]),
            "recall": float(best_val["recall"]),
        },
        "test": {
            "f1": float(test_res["fmax"]),
            "precision": float(test_res["precision"]),
            "recall": float(test_res["recall"]),
            "threshold": float(test_res["threshold"]),
            "aupr": float(test_res["aupr"]),
        },
    }

    out_json = Path(OUT_JSON)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_json}")

    md_lines = [
        "# Task 1 - ESM-2 kNN transfer baseline",
        "",
        f"Seed={SEED}, split={SPLIT}, ESM-2 dim={esm_embs.shape[1]}, "
        f"n_candidates={n_candidates:,}",
        "",
        "## Validation F-max by k",
        "",
        "| k | val_fmax | val_P | val_R | val_thr |",
        "|---|----------|-------|-------|---------|",
    ]
    for k in K_GRID:
        r = val_results[k]
        md_lines.append(
            f"| {k} | {r['fmax']:.4f} | {r['precision']:.4f} | "
            f"{r['recall']:.4f} | {r['threshold']:.3f} |"
        )
    md_lines += [
        "",
        f"Selected on validation: k\\*={best_k}, "
        f"threshold\\*={best_val['threshold']:.3f}",
        "",
        "## Test with (k\\*, threshold\\*) fixed from val",
        "",
        "| split | F1 | precision | recall | threshold |",
        "|-------|----|-----------|--------|-----------|",
        f"| test  | {test_res['fmax']:.4f} | {test_res['precision']:.4f} | "
        f"{test_res['recall']:.4f} | {test_res['threshold']:.3f} |",
        "",
    ]
    out_md = Path(OUT_MD)
    out_md.write_text("\n".join(md_lines))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
