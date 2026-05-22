from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import wandb

from model.load_data import load_bins, load_inferences
from model.Box3EL import Box3Model
from model.eval import evaluate_auc, generate_report

def main():
    parser = argparse.ArgumentParser(description='Train Box3Model on a normalised EL ontology.')
    parser.add_argument('ontology',              help='Ontology name, e.g. GALEN')
    parser.add_argument('--base-dir',            default='data')
    parser.add_argument('--embedding-dim',       type=int,   default=50)
    parser.add_argument('--epochs',              type=int,   default=10000)
    parser.add_argument('--lr',                  type=float, default=0.01)
    parser.add_argument('--lr-drop-epoch',       type=int,   default=5000)
    parser.add_argument('--margin',              type=float, default=0.1)
    parser.add_argument('--batch-size',          type=int,   default=1024)
    parser.add_argument('--reg-factor',          type=float, default=1.0)
    parser.add_argument('--n-neg-per-pos',       type=int,   default=5)
    parser.add_argument('--contrastive-mode',    default='pearson', choices=['none', 'pearson'])
    parser.add_argument('--contrastive-factor',  type=float, default=3.0)
    parser.add_argument('--val-every',           type=int,   default=100)
    parser.add_argument('--ckpt-dir',            default='checkpoints')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device',              default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--wandb-project',       default='box3el')
    args = parser.parse_args()

    device  = torch.device(args.device)
    torch.manual_seed(args.seed)
    ontology = args.ontology

    bins_dir = Path(args.base_dir) / ontology / 'bins'
    if not bins_dir.exists():
        raise FileNotFoundError(
            f'Bins not found at {bins_dir}. Run: python -m volmodel.load_data {ontology}')

    data, classes, relations = load_bins(ontology, args.base_dir)

    bot_set = {classes['owl:Nothing']}
    if 'nf1_bot' in data and len(data['nf1_bot']) > 0:
        bot_set.update(data['nf1_bot'][:, 0].tolist())
    bot_ids = torch.tensor(sorted(bot_set), dtype=torch.long)

    inf_dir    = Path(args.base_dir) / ontology / 'inferences'
    val_pairs  = load_inferences(inf_dir / 'val.json',         classes)
    eval_pairs = load_inferences(inf_dir / 'inferences.json',  classes)

    data       = {k: v.to(device) for k, v in data.items()}
    val_pairs  = val_pairs.to(device)
    eval_pairs = eval_pairs.to(device)
    bot_ids    = bot_ids.to(device)

    num_classes   = len(classes)
    owl_thing_idx = classes['owl:Thing']

    wandb.init(project=args.wandb_project, config=vars(args))
    wandb.define_metric('epoch')
    wandb.define_metric('loss/*', step_metric='epoch')
    wandb.define_metric('auc/*', step_metric='epoch')

    model = Box3Model(
        device=device,
        embedding_dim=args.embedding_dim,
        num_classes=num_classes,
        num_roles=len(relations),
        bot_ids=bot_ids,
        margin=args.margin,
        batch_size=args.batch_size,
        reg_factor=args.reg_factor,
        n_neg_per_pos=args.n_neg_per_pos,
        contrastive_mode=args.contrastive_mode,
        contrastive_factor=args.contrastive_factor,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[args.lr_drop_epoch], gamma=0.1)

    ckpt_dir = Path(args.ckpt_dir) / ontology / f'seed_{args.seed}'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_path = ckpt_dir / 'loss_log.csv'
    with open(log_path, 'w') as f:
        f.write('epoch,total,nf1_6,neg,contrast,reg,auc_unfilt\n')

    best_auc = float('-inf')
    best_state = None
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        total_loss, breakdown = model(data)
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        log_payload = {
            'epoch':          epoch,
            'loss/total':     breakdown['total'],
            'loss/nf1':       breakdown['nf1'],
            'loss/nf1_bot':   breakdown['nf1_bot'],
            'loss/nf2':       breakdown['nf2'],
            'loss/nf2_bot':   breakdown['nf2_bot'],
            'loss/nf3':       breakdown['nf3'],
            'loss/nf4':       breakdown['nf4'],
            'loss/nf4_bot':   breakdown['nf4_bot'],
            'loss/nf5':       breakdown['nf5'],
            'loss/nf6':       breakdown['nf6'],
            'loss/nf1_6':     breakdown['nf1_6'],
            'loss/neg':       breakdown['neg'],
            'loss/contrast':  breakdown['contrast'],
            'loss/reg':       breakdown['reg'],
        }

        if epoch % args.val_every == 0 or epoch == args.epochs:
            auc_u = evaluate_auc(model, val_pairs, num_classes, owl_thing_idx, device)
            print(f'[{epoch:>6}]  loss={total_loss.item():.4f}  '
                  f'AUC(unfilt)={auc_u:.4f}')
            with open(log_path, 'a') as f:
                f.write(f"{epoch},{total_loss.item():.6f},"
                        f"{breakdown['nf1_6']:.6f},{breakdown['neg']:.6f},"
                        f"{breakdown['contrast']:.6f},{breakdown['reg']:.6f},"
                        f"{auc_u:.6f}\n")
            log_payload['auc/unfilt'] = auc_u
            model.save(str(ckpt_dir / 'latest.pt'))
            if auc_u > best_auc:
                best_auc = auc_u
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                model.save(str(ckpt_dir / 'best_model.pt'))

        wandb.log(log_payload, step=epoch)

    model.save(str(ckpt_dir / 'final_model.pt'))

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f'Generating report from best validation checkpoint at epoch {best_epoch} '
              f'(AUC={best_auc:.4f})...')
    else:
        print('Generating report from final checkpoint...')
    generate_report(model, data, classes, val_pairs, eval_pairs, ckpt_dir, device)


if __name__ == '__main__':
    main()
