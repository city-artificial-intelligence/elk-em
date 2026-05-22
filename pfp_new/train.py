from __future__ import annotations

import os
os.environ["WANDB_SILENT"] = "true"

import random

import numpy as np
import torch
import wandb
from pathlib import Path



from pfp_new.load_data import load_data
from pfp_new.load_esm import load_esm_embeddings
from pfp_new.model import Box3Model
from pfp_new.eval import evaluate_model, flatten_eval_results


SEED = 42
SPLIT = (0.8, 0.1, 0.1)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ESM_PATH: str | None = 'alt_data/GO/cafa-5/esm2_480.h5'

MODEL_CONFIG = {
    'embedding_dim': 100,
    'margin': 0.05,
    'batch_size': 1024,
    'reg_factor': 0.001,
    'ca_weight': 2.0,
    'neg_weight': 0.5,
    'neg_k': 2,
    'neg_score': 0.4,
    'lex_weight': 10.0,
    'lex_batch_size': 2000,
    'lex_gamma': 1.0,
}

TRAIN_CONFIG = {
    'lr': 0.01,
    'epochs': 15000,
    'lr_warmup_frac': 0.33,
    'log_every': 100,
}

CHECKPOINT_PATH = 'checkpoints/pfp_new/last_model.pt'
RUN_POST_EVAL = True

EVAL_CONFIG = {
    'neg_samples': 50_000,
    'score_batch_size': 4096,
}

USE_WANDB = True
WANDB_PROJECT = 'box3el-pfp-new'
WANDB_RUN_NAME = 'init-model-smoke'


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


def move_train_data_to_device(
    train_data: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    model_data: dict[str, object] = {}

    for key, value in train_data.items():
        if key == 'abox':
            concept_assertions = value['concept_assertions'].to(device)
            model_data['abox'] = {'concept_assertions': concept_assertions}
        else:
            model_data[key] = value.to(device)

    return model_data


def print_data_summary(
    train_data: dict[str, object],
    classes: dict[str, int],
    relations: dict[str, int],
    pf_splits,
) -> None:
    nf_counts = []
    for key in ('nf1', 'nf2', 'nf3', 'nf4', 'nf5', 'nf6'):
        value = train_data.get(key)
        count = len(value) if isinstance(value, torch.Tensor) else 0
        nf_counts.append(f'{key}={count:,}')

    print('\nLoaded data successfully.')
    print(
        f'Concepts={len(classes):,}  '
        f'Relations={len(relations):,}  '
        f'Proteins={len(pf_splits.individual2id):,}'
    )
    print(f'Normal forms: {", ".join(nf_counts)}')
    print(
        f'Protein splits: '
        f'train={len(pf_splits.train_protein_ids):,}, '
        f'val={len(pf_splits.val_protein_ids):,}, '
        f'test={len(pf_splits.test_protein_ids):,}'
    )
    print(
        f'Term counts: '
        f'train={len(pf_splits.train_pairs):,}, '
        f'val={len(pf_splits.val_pairs):,}, '
        f'test={len(pf_splits.test_pairs):,}\n'
    )


def print_model_summary(
    model: Box3Model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    init_loss: torch.Tensor,
    breakdown: dict[str, float],
) -> None:
    print('\nModel initialised.\n')
    print(f'Device:     {model.device}')
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'LR:         {optimizer.param_groups[0]["lr"]:.6f}')
    print(f'Scheduler:  {scheduler.__class__.__name__}')
    print(f'Init loss:  {init_loss.item():.4f}')

    print('\nInitial breakdown')
    for key in ('tbox', 'ca_pos', 'ca_neg', 'lex', 'reg', 'total'):
        if key in breakdown:
            print(f'{key}:        {breakdown[key]:.4f}')

def main() -> None:
    set_seed(SEED)

    run = None
    if USE_WANDB:
        run = wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config={
                'seed': SEED,
                'split': SPLIT,
                'device': str(DEVICE),
                'esm_path': ESM_PATH,
                **{f'model_{k}': v for k, v in MODEL_CONFIG.items()},
                **{f'train_{k}': v for k, v in TRAIN_CONFIG.items()},
            },
        )

    try:
        train_data, classes, relations, pf_splits = load_data(
            seed=SEED,
            split=SPLIT,
        )
        print_data_summary(train_data, classes, relations, pf_splits)

        bot_ids = build_bot_ids(train_data, classes).to(DEVICE)
        model_data = move_train_data_to_device(train_data, DEVICE)

        model = Box3Model(
            device=DEVICE,
            embedding_dim=MODEL_CONFIG['embedding_dim'],
            num_classes=len(classes),
            num_roles=len(relations),
            bot_ids=bot_ids,
            num_individuals=len(pf_splits.individual2id),
            margin=MODEL_CONFIG['margin'],
            batch_size=MODEL_CONFIG['batch_size'],
            reg_factor=MODEL_CONFIG['reg_factor'],
            ca_weight=MODEL_CONFIG['ca_weight'],
            neg_weight=MODEL_CONFIG['neg_weight'],
            neg_k=MODEL_CONFIG['neg_k'],
            neg_score=MODEL_CONFIG['neg_score'],
            lex_weight=MODEL_CONFIG['lex_weight'],
            lex_batch_size=MODEL_CONFIG['lex_batch_size'],
            lex_gamma=MODEL_CONFIG['lex_gamma'],
        ).to(DEVICE)

        model.register_positives(model_data['abox']['concept_assertions'])

        if ESM_PATH is not None and MODEL_CONFIG['lex_weight'] > 0.0:
            esm_embs, prot_to_esm = load_esm_embeddings(
                ESM_PATH,
                pf_splits.individual2id,
                device=DEVICE,
            )
            model.register_esm(esm_embs, prot_to_esm)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=TRAIN_CONFIG['lr'],
        )

        warmup_epochs = int(TRAIN_CONFIG['epochs'] * TRAIN_CONFIG['lr_warmup_frac'])
        remaining_epochs = TRAIN_CONFIG['epochs'] - warmup_epochs

        if remaining_epochs <= 0:
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=1.0,
                total_iters=max(1, TRAIN_CONFIG['epochs']),
            )
        elif warmup_epochs <= 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=remaining_epochs,
                eta_min=TRAIN_CONFIG['lr'] * 0.1,
            )
        else:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.ConstantLR(
                        optimizer,
                        factor=1.0,
                        total_iters=warmup_epochs,
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=remaining_epochs,
                        eta_min=TRAIN_CONFIG['lr'] * 0.01,
                    ),
                ],
                milestones=[warmup_epochs],
            )
        model.eval()
        with torch.no_grad():
            init_loss, init_breakdown = model(model_data)

        print_model_summary(model, optimizer, scheduler, init_loss, init_breakdown)

        if run is not None:
            wandb.log(
                {
                    'init_loss': init_loss.item(),
                    **{f'init_{k}': v for k, v in init_breakdown.items()},
                },
                step=0,
            )

        for epoch in range(1, TRAIN_CONFIG['epochs'] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            loss, breakdown = model(model_data)
            loss.backward()
            optimizer.step()
            scheduler.step()

            lr = optimizer.param_groups[0]['lr']

            if epoch == 1 or epoch % TRAIN_CONFIG['log_every'] == 0 or epoch == TRAIN_CONFIG['epochs']:
                print(
                    f'Epoch {epoch:4d}/{TRAIN_CONFIG["epochs"]} '
                    f'lr={lr:.6f} '
                    f'total={breakdown["total"]:.4f} '
                    f'tbox={breakdown["tbox"]:.4f} '
                    f'ca_pos={breakdown["ca_pos"]:.4f} '
                    f'ca_neg={breakdown["ca_neg"]:.4f} '
                    f'lex={breakdown["lex"]:.4f} '
                    f'reg={breakdown["reg"]:.4f}'
                )

            if run is not None:
                wandb.log(
                    {
                        'epoch': epoch,
                        'lr': lr,
                        'train_total': breakdown['total'],
                        'train_tbox': breakdown['tbox'],
                        'train_ca_pos': breakdown['ca_pos'],
                        'train_ca_neg': breakdown['ca_neg'],
                        'train_lex': breakdown['lex'],
                        'train_reg': breakdown['reg'],
                    },
                    step=epoch,
                )

        checkpoint_path = Path(CHECKPOINT_PATH)
        model.save(str(checkpoint_path))
        print(f'\nSaved checkpoint: {checkpoint_path}\n')

        eval_results = None
        if RUN_POST_EVAL:
            eval_results = evaluate_model(
                model=model,
                train_data=train_data,
                pf_splits=pf_splits,
                classes=classes,
                neg_samples=EVAL_CONFIG['neg_samples'],
                score_batch_size=EVAL_CONFIG['score_batch_size'],
                seed=SEED,
            )

        if run is not None:
            log_payload = {
                'num_concepts': len(classes),
                'num_relations': len(relations),
                'num_proteins': len(pf_splits.individual2id),
                'num_parameters': sum(p.numel() for p in model.parameters()),
                'train_proteins': len(pf_splits.train_protein_ids),
                'val_proteins': len(pf_splits.val_protein_ids),
                'test_proteins': len(pf_splits.test_protein_ids),
                'train_terms': len(pf_splits.train_pairs),
                'val_terms': len(pf_splits.val_pairs),
                'test_terms': len(pf_splits.test_pairs),
                'skipped_annotations': pf_splits.skipped_annotations,
                'checkpoint_path': str(checkpoint_path),
                'init_loss': init_loss.item(),
                **{f'init_{k}': v for k, v in init_breakdown.items()},
            }

            if eval_results is not None:
                log_payload.update(flatten_eval_results(eval_results))

            wandb.log(log_payload)
    finally:
        if run is not None:
            wandb.finish()


if __name__ == '__main__':
    main()