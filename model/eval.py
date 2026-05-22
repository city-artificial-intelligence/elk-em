from __future__ import annotations

from pathlib import Path

import csv
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor

def nf1_scores(model, data: Tensor) -> Tensor:
    saved = model.margin; model.margin = 0.0
    s = torch.exp(-model.inclusion_loss(
        model.get_class_box(data[:, 0]),
        model.get_class_box(data[:, 1])))
    model.margin = saved
    return s

def nf2_scores(model, data: Tensor) -> Tensor:
    saved = model.margin; model.margin = 0.0
    c = model.get_class_box(data[:, 0])
    d = model.get_class_box(data[:, 1])
    e = model.get_class_box(data[:, 2])
    s = torch.exp(-model.inclusion_loss(c.intersect(d), e))
    model.margin = saved
    return s

def nf3_scores(model, data: Tensor) -> Tensor:
    saved = model.margin; model.margin = 0.0
    c = model.get_class_box(data[:, 0])
    r = model.get_role(data[:, 1])
    d = model.get_class_box(data[:, 2])
    s = torch.exp(-model.inclusion_loss(c, r.existential(d)))
    model.margin = saved
    return s

def nf4_scores(model, data: Tensor) -> Tensor:
    saved = model.margin; model.margin = 0.0
    r = model.get_role(data[:, 0])
    c = model.get_class_box(data[:, 1])
    d = model.get_class_box(data[:, 2])
    s = torch.exp(-model.inclusion_loss(r.existential(c), d))
    model.margin = saved
    return s

def nf5_scores(model, data: Tensor) -> Tensor:
    saved = model.margin; model.margin = 0.0
    r = model.get_role(data[:, 0])
    s_ = model.get_role(data[:, 1])
    s = torch.exp(-model.role_inclusion_loss(r, s_))
    model.margin = saved
    return s

def nf6_scores(model, data: Tensor) -> Tensor:
    saved = model.margin; model.margin = 0.0
    r1 = model.get_role(data[:, 0])
    r2 = model.get_role(data[:, 1])
    t  = model.get_role(data[:, 2])
    s = torch.exp(-model.role_inclusion_loss(r1.compose(r2), t))
    model.margin = saved
    return s

def nf1_bot_scores(model, data: Tensor) -> Tensor:
    saved = model.margin; model.margin = 0.0
    c = model.get_class_box(data[:, 0])
    s = torch.exp(-F.relu(c.lower - c.upper).mean(dim=-1))
    model.margin = saved
    return s

def nf2_bot_scores(model, data: Tensor) -> Tensor:
    saved = model.margin; model.margin = 0.0
    c = model.get_class_box(data[:, 0])
    d = model.get_class_box(data[:, 1])
    s = torch.exp(-F.relu(-c.separation(d).max(dim=-1).values))
    model.margin = saved
    return s

def nf4_bot_scores(model, data: Tensor) -> Tensor:
    saved = model.margin; model.margin = 0.0
    r = model.get_role(data[:, 0])
    c = model.get_class_box(data[:, 1])
    s = torch.exp(-F.relu(-r.range.separation(c).max(dim=-1).values))
    model.margin = saved
    return s

_SCORE_FNS = {
    'nf1':     nf1_scores,     'nf2':     nf2_scores,
    'nf3':     nf3_scores,     'nf4':     nf4_scores,
    'nf5':     nf5_scores,     'nf6':     nf6_scores,
    'nf1_bot': nf1_bot_scores, 'nf2_bot': nf2_bot_scores,
    'nf4_bot': nf4_bot_scores,
}

@torch.no_grad()
def evaluate_scores(model, key: str, data: Tensor) -> tuple[float, float, float]:
    scores = _SCORE_FNS[key](model, data)
    return scores.mean().item(), scores.min().item(), scores.max().item()


@torch.no_grad()
def evaluate_auc(model, pairs: Tensor, num_classes: int, owl_thing_idx: int,
                 device, chunk_size: int = 64) -> float:
    all_ids   = torch.arange(num_classes, device=device)
    all_boxes = model.get_class_box(all_ids)
    ca_all    = all_boxes.center   # [M, dim]
    oa_all    = all_boxes.offset   # [M, dim]

    total_auc = 0.0
    n_pairs   = len(pairs)

    for start in range(0, n_pairs, chunk_size):
        chunk = pairs[start : start + chunk_size]   # [Q, 2]
        c_ids = chunk[:, 0]
        d_ids = chunk[:, 1]
        Q     = len(chunk)

        c_boxes = model.get_class_box(c_ids)        # [Q, dim]
        # scores[q, m] = exp(-relu(|cq - cm| + oq - om).mean(-1))
        diff   = torch.abs(c_boxes.center.unsqueeze(1) - ca_all.unsqueeze(0))   # [Q, M, dim]
        scores = torch.exp(-F.relu(diff + c_boxes.offset.unsqueeze(1) - oa_all.unsqueeze(0)).mean(-1))  # [Q, M]

        q_idx = torch.arange(Q, device=device)
        scores[q_idx, c_ids]     = -1.0
        scores[:, owl_thing_idx] = -1.0

        true_scores = scores[q_idx, d_ids]                          # [Q]
        n_cands     = (scores >= 0).sum(dim=1).float()              # [Q]
        ranks       = (scores >= true_scores.unsqueeze(1)).sum(dim=1).float()  # [Q]
        auc_chunk   = ((n_cands - ranks) / torch.clamp(n_cands - 1, min=1)).sum()
        total_auc  += auc_chunk.item()

    return total_auc / n_pairs


@torch.no_grad()
def evaluate_ranking_metrics(
    model, pairs: Tensor, num_classes: int, owl_thing_idx: int,
    device, ks: tuple = (1, 10, 100), chunk_size: int = 64,
) -> dict:
    all_ids   = torch.arange(num_classes, device=device)
    all_boxes = model.get_class_box(all_ids)
    ca_all    = all_boxes.center   # [M, dim]
    oa_all    = all_boxes.offset   # [M, dim]

    all_ranks:  list[Tensor] = []
    all_ncands: list[Tensor] = []

    for start in range(0, len(pairs), chunk_size):
        chunk = pairs[start : start + chunk_size]   # [Q, 2]
        c_ids = chunk[:, 0]
        d_ids = chunk[:, 1]
        Q     = len(chunk)

        c_boxes = model.get_class_box(c_ids)        # [Q, dim]
        diff   = torch.abs(c_boxes.center.unsqueeze(1) - ca_all.unsqueeze(0))   # [Q, M, dim]
        scores = torch.exp(-F.relu(diff + c_boxes.offset.unsqueeze(1) - oa_all.unsqueeze(0)).mean(-1))  # [Q, M]

        q_idx = torch.arange(Q, device=device)
        scores[q_idx, c_ids]     = -1.0
        scores[:, owl_thing_idx] = -1.0

        true_scores = scores[q_idx, d_ids]                                          # [Q]
        n_cands     = (scores >= 0).sum(dim=1)                                      # [Q]
        ranks       = (scores >= true_scores.unsqueeze(1)).sum(dim=1)               # [Q]
        all_ranks.append(ranks.cpu())
        all_ncands.append(n_cands.cpu())

    rt = torch.cat(all_ranks).double()
    nt = torch.cat(all_ncands).double()
    metrics: dict = {
        'mr':  float(rt.mean()),
        'mrr': float((1.0 / rt).mean()),
        'med': float(rt.median()),
        'auc': float(((nt - rt) / torch.clamp(nt - 1, min=1)).mean()),
    }
    for k in ks:
        metrics[f'hits@{k}'] = float((rt <= k).double().mean())
    return metrics


def generate_report(model, data: dict, classes: dict, val_pairs: Tensor,
                    eval_pairs: Tensor, ckpt_dir, device):
    ckpt_dir      = Path(ckpt_dir)
    num_classes   = len(classes)
    owl_thing_idx = classes['owl:Thing']
    model.eval()

    # loss plot
    epochs, totals, nf1_6s, negs, contrasts, regs = [], [], [], [], [], []
    with open(ckpt_dir / 'loss_log.csv') as f:
        for row in csv.DictReader(f):
            epochs.append(int(row['epoch']))
            totals.append(float(row['total']))
            nf1_6s.append(float(row['nf1_6']))
            negs.append(float(row['neg']))
            contrasts.append(float(row['contrast']))
            regs.append(float(row['reg']))

    fig, ax = plt.subplots()
    ax.plot(epochs, totals,     label='total')
    ax.plot(epochs, nf1_6s,    label='nf1_6')
    ax.plot(epochs, negs,      label='neg')
    ax.plot(epochs, contrasts,  label='contrast')
    ax.plot(epochs, regs,      label='reg')
    ax.set_xlabel('epoch')
    ax.legend()
    fig.savefig(ckpt_dir / 'loss_plot.png', dpi=150)
    plt.close(fig)

    lines = []

    # per-NF score stats on training data
    for key in ['nf1','nf2','nf3','nf4','nf5','nf6','nf1_bot','nf2_bot','nf4_bot']:
        if key in data and len(data[key]) > 0:
            mean, mn, mx = evaluate_scores(model, key, data[key])
            lines.append(f'{key:<10}  mean={mean:.4f}  min={mn:.4f}  max={mx:.4f}')

    # nf1 neg sample
    if 'nf1' in data:
        neg = model._sample_nf1_negatives(data['nf1'][:model.batch_size])
        s   = nf1_scores(model, neg)
        lines.append(f'{"nf1_neg":<10}  mean={s.mean():.4f}  '
                     f'min={s.min():.4f}  max={s.max():.4f}')

    def _fmt(tag: str, m: dict) -> str:
        return (f'{tag:<18} MR={m["mr"]:.1f}  MRR={m["mrr"]:.4f}  Med={m["med"]:.0f}  '
                f'AUC={m["auc"]:.4f}  H@1={m["hits@1"]:.4f}  '
                f'H@10={m["hits@10"]:.4f}  H@100={m["hits@100"]:.4f}')

    # ranking metrics on nf1 train
    m_tr = evaluate_ranking_metrics(model, data['nf1'], num_classes, owl_thing_idx, device)
    lines.append('')
    lines.append(_fmt('nf1 train', m_tr))

    # val
    mean_v, mn_v, mx_v = evaluate_scores(model, 'nf1', val_pairs)
    m_v = evaluate_ranking_metrics(model, val_pairs, num_classes, owl_thing_idx, device)
    lines.append(f'val scores         mean={mean_v:.4f}  min={mn_v:.4f}  max={mx_v:.4f}')
    lines.append(_fmt('val', m_v))

    # eval
    mean_e, mn_e, mx_e = evaluate_scores(model, 'nf1', eval_pairs)
    m_e = evaluate_ranking_metrics(model, eval_pairs, num_classes, owl_thing_idx, device)
    lines.append(f'eval scores        mean={mean_e:.4f}  min={mn_e:.4f}  max={mx_e:.4f}')
    lines.append(_fmt('eval', m_e))

    report = '\n'.join(lines)
    print(report)
    (ckpt_dir / 'report.txt').write_text(report)
    model.train()

