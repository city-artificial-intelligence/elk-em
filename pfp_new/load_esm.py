
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor


def load_esm_embeddings(
    h5_path: str | Path,
    individual2id: dict[str, int],
    device: str | torch.device = 'cpu',
) -> tuple[Tensor, Tensor]:
    with h5py.File(h5_path, 'r') as f:
        keys = sorted(f.keys())
        embs = np.stack([f[k][:] for k in keys]).astype(np.float32)  # (N, D)

    key_to_row = {k: i for i, k in enumerate(keys)}

    prot_to_esm = torch.zeros(len(individual2id), dtype=torch.long)
    for name, model_idx in individual2id.items():
        if name not in key_to_row:
            raise KeyError(f'Protein {name!r} not found in ESM-2 HDF5 file {h5_path}')
        prot_to_esm[model_idx] = key_to_row[name]

    return torch.tensor(embs, device=device), prot_to_esm.to(device)
