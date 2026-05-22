from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pfp_new.geometry import Box, Role, Transform

class Box3Model(nn.Module):
    def __init__(self,
                device,
                embedding_dim: int,
                num_classes: int,
                num_roles: int,
                bot_ids: Tensor,
                num_individuals: int = 0,
                margin: float = 0.05,
                batch_size: int = 512,
                reg_factor: float = 1.0,
                ca_weight: float = 0.5,
                neg_weight: float = 1.0,
                neg_k: int = 5,
                neg_score: float = 0.1,
                lex_weight: float = 0.0,
                lex_batch_size: int = 1000,
                lex_gamma: float = 6.0,
         ):
        super().__init__()

        self.device          = device
        self.embedding_dim   = embedding_dim
        self.num_classes     = num_classes
        self.num_roles       = num_roles
        self.num_individuals = num_individuals
        self.margin          = margin
        self.batch_size      = batch_size
        self.reg_factor      = reg_factor
        self.ca_weight       = ca_weight
        self.neg_weight      = neg_weight
        self.neg_k           = neg_k
        self.neg_score       = neg_score
        self.lex_weight      = lex_weight
        self.lex_batch_size  = lex_batch_size
        self.lex_gamma = lex_gamma
        self.epsilon = 1e-5

        # ESM-2 embeddings — set via register_esm() before training
        self.register_buffer('_esm_embs', None, persistent=False)
        self.register_buffer('_prot_to_esm', None, persistent=False)

        self.register_buffer('bot_ids', bot_ids)

        # Class box: centre + raw half-width (abs applied at lookup time)
        self.class_center     = self._init_emb(num_classes, -0.5,  0.5)
        self.class_offset_raw = self._init_emb(num_classes, -0.1,  0.1)

        # Role parameters
        self.role_range_center     = self._init_emb(num_roles,  -0.5,  0.5)
        self.role_range_offset_raw = self._init_emb(num_roles,  -0.5,  0.5)
        self.role_error_center     = self._init_emb(num_roles,  0.00, 0.00)
        self.role_error_offset_raw = self._init_emb(num_roles, -0.01, 0.01)
        self.role_scale_raw        = self._init_emb(num_roles,  0.9,  1.1)
        self.role_shift            = self._init_emb(num_roles,  -0.1,  0.1)

        # Individual (protein) embeddings — points, not boxes
        if num_individuals > 0:
            self.individual_emb = self._init_emb(num_individuals, -0.5, 0.5)


    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, individual: Box, class_box: Box) -> Tensor:
        sep = individual.separation(class_box)
        return torch.exp(-F.relu(sep + 2 * individual.offset).mean(dim=-1))

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _init_emb(self, num_embeddings: int, low: float, high: float) -> nn.Embedding:
        emb = nn.Embedding(num_embeddings, self.embedding_dim)
        if low == high:
            nn.init.constant_(emb.weight, low)
            emb.weight.requires_grad_(False)
        else:
            nn.init.uniform_(emb.weight, low, high)
        return emb

    # ------------------------------------------------------------------
    # Geometric lookups
    # ------------------------------------------------------------------

    def get_class_box(self, ids: Tensor) -> Box:
        c   = self.class_center(ids)
        raw = self.class_offset_raw(ids)
        is_bot = torch.isin(ids, self.bot_ids).unsqueeze(-1)
        half = torch.where(is_bot, raw, torch.abs(raw) + self.epsilon)
        return Box(lower=c - half, upper=c + half)

    def get_role(self, ids: Tensor) -> Role:
        range_c    = self.role_range_center(ids)
        range_half = self.role_range_offset_raw(ids).abs()
        error_c    = self.role_error_center(ids)
        error_half = self.role_error_offset_raw(ids).abs()
        return Role(
            range=Box(lower=range_c - range_half, upper=range_c + range_half),
            error=Box(lower=error_c - error_half, upper=error_c + error_half),
            transform=Transform(
                scale=self.role_scale_raw(ids).abs() + self.epsilon,
                shift=self.role_shift(ids),
            ),
        )

    def get_individual(self, ids: Tensor) -> Box:
        e = self.individual_emb(ids)
        return Box(lower=e, upper=e)

    # ------------------------------------------------------------------
    # Core geometric losses
    # ------------------------------------------------------------------

    def inclusion_loss(self, inner: Box, outer: Box, margin: float | None = None) -> Tensor:
        """0 when inner ⊆ outer. Uses separation + inner width."""
        if margin is None:
            margin = self.margin
        sep = inner.separation(outer)
        return F.relu(sep + 2 * inner.offset + margin).mean(dim=-1)

    def overlap_loss(self, box1: Box, box2: Box, margin: float | None = None) -> Tensor:
        """0 when boxes overlap."""
        if margin is None:
            margin = self.margin
        return F.relu(box1.separation(box2) + margin).mean(dim=-1)

    def disjoint_loss(self, box1: Box, box2: Box, margin: float | None = None) -> Tensor:
        """0 when boxes are separated in at least one dimension."""
        if margin is None:
            margin = self.margin
        return F.relu(margin - box1.separation(box2).max(dim=-1).values)

    # ------------------------------------------------------------------
    # TBox losses (NF1–NF6)
    # ------------------------------------------------------------------

    def nf1_loss(self, data: Tensor) -> Tensor:
        """C ⊑ D  →  box(C) ⊆ box(D)"""
        return self.inclusion_loss(self.get_class_box(data[:, 0]),
                                   self.get_class_box(data[:, 1]))

    def nf1_bot_loss(self, data: Tensor) -> Tensor:
        """C ⊑ ⊥  →  raw offset should be negative (box inverted/empty)"""
        c_boxes = self.get_class_box(data[:, 0])
        return F.relu(c_boxes.offset + self.margin).mean(dim=-1)

    def nf2_loss(self, data: Tensor) -> Tensor:
        """C ⊓ D ⊑ E  →  box(C) ∩ box(D) ⊆ box(E), and C,D overlap"""
        c = self.get_class_box(data[:, 0])
        d = self.get_class_box(data[:, 1])
        e = self.get_class_box(data[:, 2])
        intersection = c.intersect(d)
        return self.inclusion_loss(intersection, e) + self.overlap_loss(c, d)

    def nf2_bot_loss(self, data: Tensor) -> Tensor:
        """C ⊓ D ⊑ ⊥  →  box(C) and box(D) are disjoint"""
        return self.disjoint_loss(self.get_class_box(data[:, 0]),
                                  self.get_class_box(data[:, 1]))

    def nf3_loss(self, data: Tensor) -> Tensor:
        """C ⊑ ∃r.D  →  box(C) ⊆ existential(r, box(D))"""
        c = self.get_class_box(data[:, 0])
        r = self.get_role(data[:, 1])
        d = self.get_class_box(data[:, 2])
        existential = r.existential(d)
        return 2 * self.inclusion_loss(c, existential) + self.overlap_loss(r.range, d)

    def nf4_loss(self, data: Tensor) -> Tensor:
        """∃r.C ⊑ D  →  existential(r, box(C)) ⊆ box(D), and range(r) overlaps box(C)"""
        r = self.get_role(data[:, 0])
        c = self.get_class_box(data[:, 1])
        d = self.get_class_box(data[:, 2])
        existential = r.existential(c)
        return 2 * self.inclusion_loss(existential, d) + self.overlap_loss(r.range, c)

    def nf4_bot_loss(self, data: Tensor) -> Tensor:
        """∃r.C ⊑ ⊥  →  range(r) and box(C) are disjoint"""
        r = self.get_role(data[:, 0])
        c = self.get_class_box(data[:, 1])
        return self.disjoint_loss(r.range, c)

    def _role_inclusion_loss(self, r: Role, s: Role) -> Tensor:
        """r ⊑ s as roles: range inclusion + transform compatibility at corners."""
        r_lo = Box(lower=r.range.lower, upper=r.range.lower)
        r_up = Box(lower=r.range.upper, upper=r.range.upper)
        return (
            self.inclusion_loss(r.range, s.range)
            + self.inclusion_loss(r.transform(r_lo).minkowski_sum(r.error),
                                  s.transform(r_lo).minkowski_sum(s.error))
            + self.inclusion_loss(r.transform(r_up).minkowski_sum(r.error),
                                  s.transform(r_up).minkowski_sum(s.error))
        )

    def nf5_loss(self, data: Tensor) -> Tensor:
        """r ⊑ s"""
        return self._role_inclusion_loss(self.get_role(data[:, 0]),
                                         self.get_role(data[:, 1]))

    def nf6_loss(self, data: Tensor) -> Tensor:
        """r ∘ s ⊑ t"""
        r = self.get_role(data[:, 0])
        s = self.get_role(data[:, 1])
        t = self.get_role(data[:, 2])
        composed = r.compose(s)
        admissible = s.transform.inverse()(r.range.minkowski_difference(s.error))
        return (
            self._role_inclusion_loss(composed, t)
            + self.overlap_loss(s.range, admissible)
        )

    # ------------------------------------------------------------------
    # Regularisation
    # ------------------------------------------------------------------

    def box_regularisation_loss(self) -> Tensor:
        device = self.bot_ids.device

        bot_mask = torch.zeros(self.num_classes, dtype=torch.bool, device=device)
        bot_mask[self.bot_ids] = True

        # Bot classes should stay empty/inverted.
        bot_validity = F.relu(
            self.class_offset_raw.weight[bot_mask] + self.epsilon
        ).mean()

        # Role-error offsets should stay <= 0.01 in magnitude.
        size_penalty = F.relu(
            self.role_error_offset_raw.weight.abs() - 0.01
        ).mean()

        # centers = torch.cat(
        #     [self.class_center.weight, self.role_range_center.weight, self.individual_emb.weight], dim=0
        # )
        # center_penalty = F.relu(centers.norm(p=1, dim=-1) - 1.0).mean()

        return bot_validity + size_penalty #+  + center_penalty

    # ------------------------------------------------------------------
    # Batch sampling helper
    # ------------------------------------------------------------------

    def get_batch(self, data: Tensor, size: int) -> Tensor:
        if len(data) <= size:
            return data
        idx = torch.randint(len(data), (size,), device=data.device)
        return data[idx]

    # ------------------------------------------------------------------
    # ABox losses
    # ------------------------------------------------------------------

    def register_positives(self, train_pairs: Tensor) -> None:
        device = self.bot_ids.device
        pairs = train_pairs.to(device)
        encoded = pairs[:, 0].long() * self.num_classes + pairs[:, 1].long()
        self.register_buffer('_pos_encoded', encoded)
        self.register_buffer('_scoreable_cids', torch.arange(self.num_classes, device=device))

    def sample_negatives(self, pos_batch: Tensor, k: int = 1) -> Tensor:

        device      = pos_batch.device
        n           = len(pos_batch)
        scoreable   = self._scoreable_cids.to(device)
        prot_ids    = pos_batch[:, 0].repeat_interleave(k)
        idx         = torch.randint(0, len(scoreable), (n * k,), device=device)
        cls_ids     = scoreable[idx]

        encoded = prot_ids * self.num_classes + cls_ids
        mask    = ~torch.isin(encoded, self._pos_encoded.to(device))

        return torch.stack([prot_ids[mask], cls_ids[mask]], dim=1)

    def ca_pos_loss(self, pairs: Tensor) -> Tensor:
        """inclusion_loss: 0 when individual inside class box, gradient everywhere."""
        individuals = self.get_individual(pairs[:, 0])
        class_boxes = self.get_class_box(pairs[:, 1])
        loss = self.inclusion_loss(individuals, class_boxes, margin=self.margin)
        return  loss

    def ca_neg_loss(self, pairs: Tensor) -> Tensor:
        individuals = self.get_individual(pairs[:, 0])
        class_boxes = self.get_class_box(pairs[:, 1])
        loss = self.inclusion_loss(individuals, class_boxes, margin=0.0)
        return F.relu(-math.log(self.neg_score) - loss)

    # ------------------------------------------------------------------
    # Lexical regularisation
    # ------------------------------------------------------------------

    def register_esm(self, esm_embs: Tensor, prot_to_esm: Tensor) -> None:
        # ESM-2 embeddings — set via register_esm() before training
        self._esm_embs = esm_embs.to(self.bot_ids.device)
        self._prot_to_esm = prot_to_esm.to(self.bot_ids.device)
        print(f'Registered ESM-2 embeddings: {esm_embs.shape}')

    def lex_loss(self) -> Tensor:
        if self._esm_embs is None or self.lex_weight == 0.0:
            return torch.zeros(1, device=self.bot_ids.device).squeeze()

        device = self.bot_ids.device
        ids = torch.randint(0, self.num_individuals, (self.lex_batch_size,), device=device)

        # protein point embeddings
        points = self.get_individual(ids).lower           # (B, dim)
        esm    = self._esm_embs[self._prot_to_esm[ids]]   # (B, D)

        sq_dists = torch.cdist(points, points).pow(2)     # (B, B)
        geo_sim = torch.exp(-self.lex_gamma * (sq_dists / self.embedding_dim))

        esm_sim = F.normalize(esm.float(), dim=-1) @ F.normalize(esm.float(), dim=-1).T
        esm_sim = (esm_sim + 1.0) / 2.0

        # upper triangle only — avoid diagonal and double-counting
        rows, cols = torch.triu_indices(len(ids), len(ids), offset=1, device=device)
        return F.mse_loss(geo_sim[rows, cols], esm_sim[rows, cols])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, train_data: dict) -> tuple[Tensor, dict]:
        device = self.bot_ids.device

        # --- TBox ---
        tbox_keys = [
            ('nf1',     self.nf1_loss),
            ('nf1_bot', self.nf1_bot_loss),
            ('nf2',     self.nf2_loss),
            ('nf2_bot', self.nf2_bot_loss),
            ('nf3',     self.nf3_loss),
            ('nf4',     self.nf4_loss),
            ('nf4_bot', self.nf4_bot_loss),
            ('nf5',     self.nf5_loss),
            ('nf6',     self.nf6_loss),
        ]
        parts: dict[str, Tensor] = {}
        for key, fn in tbox_keys:
            if key in train_data and len(train_data[key]) > 0:
                batch = self.get_batch(train_data[key], self.batch_size)
                parts[key] = fn(batch).pow(2).mean()
            else:
                parts[key] = torch.zeros(1, device=device).squeeze()

        tbox_loss = torch.stack(list(parts.values())).pow(2).mean()

        # --- Concept assertions ---
        ca_pos_l = torch.zeros(1, device=device).squeeze()
        ca_neg_l = torch.zeros(1, device=device).squeeze()
        if 'abox' in train_data:
            ca_pairs = train_data['abox']['concept_assertions']
            if len(ca_pairs) > 0:
                pos_batch = self.get_batch(ca_pairs, self.batch_size)
                ca_pos_l = self.ca_pos_loss(pos_batch).pow(2).mean()

                neg_batch = self.sample_negatives(pos_batch, k=self.neg_k)
                if len(neg_batch) > 0:
                    ca_neg_l = self.ca_neg_loss(neg_batch).pow(2).mean()

        # --- Lexical regularisation ---
        lex_l = self.lex_loss()

        # --- Regularisation ---
        reg_loss = self.box_regularisation_loss()

        total = 1000*(tbox_loss
                + self.ca_weight  * ca_pos_l
                + self.neg_weight * ca_neg_l
                + self.lex_weight * lex_l 
                + self.reg_factor * reg_loss)

        breakdown = {k: 1000 * v.item() for k, v in parts.items()}
        breakdown['tbox']   = 1000 * tbox_loss.item()
        breakdown['ca_pos'] = 1000 * self.ca_weight  * ca_pos_l.item()
        breakdown['ca_neg'] = 1000 * self.neg_weight * ca_neg_l.item()
        breakdown['lex']    = 1000 * self.lex_weight * lex_l.item()
        breakdown['reg']    = 1000 * self.reg_factor * reg_loss.item()
        breakdown['total']  = total.item()
        return total, breakdown

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'embedding_dim':        self.embedding_dim,
            'class_center':         self.class_center.weight.detach(),
            'class_offset_raw':     self.class_offset_raw.weight.detach(),
            'role_range_center':        self.role_range_center.weight.detach(),
            'role_range_offset_raw':    self.role_range_offset_raw.weight.detach(),
            'role_error_center':        self.role_error_center.weight.detach(),
            'role_error_offset_raw':    self.role_error_offset_raw.weight.detach(),
            'role_scale_raw':           self.role_scale_raw.weight.detach(),
            'role_shift':               self.role_shift.weight.detach(),
        }
        if self.num_individuals > 0:
            state['individual_emb'] = self.individual_emb.weight.detach()
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, device: torch.device, bot_ids: Tensor, **kwargs) -> Box3Model:
        state = torch.load(path, map_location=device)
        dim = state['embedding_dim']
        model = cls(embedding_dim=dim, bot_ids=bot_ids, **kwargs)
        model.class_center.weight.data.copy_(state['class_center'])
        model.class_offset_raw.weight.data.copy_(state['class_offset_raw'])
        model.role_range_center.weight.data.copy_(state['role_range_center'])
        model.role_range_offset_raw.weight.data.copy_(state['role_range_offset_raw'])
        model.role_error_center.weight.data.copy_(state['role_error_center'])
        model.role_error_offset_raw.weight.data.copy_(state['role_error_offset_raw'])
        model.role_scale_raw.weight.data.copy_(state['role_scale_raw'])
        model.role_shift.weight.data.copy_(state['role_shift'])
        if 'individual_emb' in state:
            model.individual_emb.weight.data.copy_(state['individual_emb'])
        return model.to(device)
