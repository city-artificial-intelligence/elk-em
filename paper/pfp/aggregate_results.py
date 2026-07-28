"""Aggregate the two §5.2 experiments into a single markdown file.

Reads JSON outputs from:
  - the ESM-2 kNN transfer baseline (`paper.pfp.knn.knn_baseline`)
  - the six multi-seed ELK-Em runs launched by `paper/pfp/submit_multiseed.sh`

Emits a markdown report with the numbers that appear in Table 4 of the paper.
Missing artefacts are noted with a placeholder so you can tell what still
needs to run.

Usage:
    python -m paper.pfp.aggregate_results [--out paper/pfp/results.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


KNN_RESULTS = "paper/pfp/knn/results.json"
CKPT_ROOT = "checkpoints/pfp"


def load_json(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        return {"__error__": f"parse failed: {e}"}


def fmt_missing(msg: str) -> str:
    return f"_(not available yet — {msg})_"


def section_knn() -> str:
    d = load_json(KNN_RESULTS)
    lines = ["## ESM-2 kNN transfer baseline\n"]
    if d is None:
        lines.append(fmt_missing("run `bash paper/pfp/knn/run_knn.sh`"))
        return "\n".join(lines) + "\n"
    lines += [
        f"Seed={d['seed']}, split={tuple(d['split'])}, ESM-2 dim={d['esm_dim']}, "
        f"n_candidates={d['n_candidates']:,}, k ∈ {d['k_grid']}, "
        f"thresholds = {d['n_thresholds']} values in [{d['t_min']}, {d['t_max']}]",
        "",
        "### Validation F-max by k",
        "",
        "| k | val_fmax | val_P | val_R | val_thr |",
        "|---|----------|-------|-------|---------|",
    ]
    for k in d["k_grid"]:
        r = d["val_by_k"][str(k)]
        lines.append(f"| {k} | {r['fmax']:.4f} | {r['precision']:.4f} | "
                     f"{r['recall']:.4f} | {r['threshold']:.3f} |")
    lines += [
        "",
        f"Selected on val: k\\*={d['selected_k']}, "
        f"threshold\\*={d['selected_threshold']:.3f} (val F-max {d['val_at_selected']['fmax']:.4f})",
        "",
        "### Test with (k\\*, threshold\\*) fixed from val",
        "",
        "| split | F1 | precision | recall | threshold |",
        "|-------|----|-----------|--------|-----------|",
        f"| test  | {d['test']['f1']:.4f} | {d['test']['precision']:.4f} | "
        f"{d['test']['recall']:.4f} | {d['test']['threshold']:.3f} |",
        "",
    ]
    return "\n".join(lines) + "\n"


def _eval_json_dir(cfg: str, seed: int) -> tuple[dict | None, dict | None]:
    base = f"{CKPT_ROOT}/{cfg}/seed_{seed}"
    return (
        load_json(f"{base}/last_model.eval.json"),
        load_json(f"{base}/last_model.efficiency.json"),
    )


def section_multiseed() -> str:
    lines = ["## ELK-Em multi-seed (Table 4, first two rows)\n"]
    configs = ["full", "no_lex"]
    seeds = [0, 1, 2]

    any_present = False
    per_cfg: dict[str, dict[str, list[float]]] = {}
    per_cfg_eff: dict[str, dict[str, list[float]]] = {}

    for cfg in configs:
        per_cfg[cfg] = {"F1": [], "P": [], "R": [], "val_fmax": []}
        per_cfg_eff[cfg] = {"wall_s": [], "peak_mem_gb": []}
        for s in seeds:
            ev, eff = _eval_json_dir(cfg, s)
            if ev is not None:
                any_present = True
                if "eval_test_fixed_threshold_f1" in ev:
                    per_cfg[cfg]["F1"].append(ev["eval_test_fixed_threshold_f1"])
                    per_cfg[cfg]["P"].append(ev["eval_test_fixed_threshold_precision"])
                    per_cfg[cfg]["R"].append(ev["eval_test_fixed_threshold_recall"])
                if "eval_val_fmax_fmax" in ev:
                    per_cfg[cfg]["val_fmax"].append(ev["eval_val_fmax_fmax"])
            if eff is not None:
                per_cfg_eff[cfg]["wall_s"].append(eff.get("train_wallclock_s", float("nan")))
                per_cfg_eff[cfg]["peak_mem_gb"].append(
                    eff.get("peak_mem_bytes", 0) / (1024 ** 3)
                )

    if not any_present:
        lines.append(fmt_missing(
            "run `bash paper/pfp/submit_multiseed.sh` and wait for the 6 jobs"))
        return "\n".join(lines) + "\n"

    def _ms(vals):
        if not vals:
            return "—"
        if len(vals) == 1:
            return f"{vals[0]:.4f}"
        return f"{mean(vals):.4f} ± {stdev(vals):.4f}"

    lines += [
        "### Test-set metrics (mean ± std over seeds {0, 1, 2})",
        "",
        "| config | seeds present | test F1 | test P | test R | val F-max |",
        "|--------|---------------|---------|--------|--------|-----------|",
    ]
    for cfg in configs:
        n = len(per_cfg[cfg]["F1"])
        lines.append(
            f"| {cfg} | {n} | {_ms(per_cfg[cfg]['F1'])} | "
            f"{_ms(per_cfg[cfg]['P'])} | {_ms(per_cfg[cfg]['R'])} | "
            f"{_ms(per_cfg[cfg]['val_fmax'])} |"
        )
    lines += [
        "",
        "### Efficiency (mean ± std)",
        "",
        "| config | wallclock (s) | peak memory (GB) |",
        "|--------|---------------|------------------|",
    ]
    for cfg in configs:
        w = per_cfg_eff[cfg]["wall_s"]
        m = per_cfg_eff[cfg]["peak_mem_gb"]
        lines.append(f"| {cfg} | {_ms(w) if w else '—'} | {_ms(m) if m else '—'} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/pfp/results.md")
    args = ap.parse_args()

    header = [
        "# §5.2 protein-function results",
        "",
        "Auto-generated by `python -m paper.pfp.aggregate_results`. Missing "
        "sections indicate the corresponding job has not yet produced its "
        "JSON artefact.",
        "",
    ]
    body = "\n".join(header) + "\n" + "".join([
        section_knn(),
        section_multiseed(),
    ])
    Path(args.out).write_text(body)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
