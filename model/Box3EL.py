from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.geometry import Box, Transform, Role, exist

class Box3Model(nn.Module):
    def __init__(self,
                device, 
                embedding_dim, 
                num_classes, 
                num_roles,
                bot_ids,
                num_individuals=0,
                margin = 0.1,
                batch_size=512,
                reg_factor=1.0,
                n_neg_per_pos=5,
                contrastive_mode   = 'none',   # 'none' | 'pearson'
                contrastive_factor = 1.0
                ):
        super().__init__()

        self.device = device
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.num_roles = num_roles
        self.num_individuals = num_individuals
        self.batch_size = batch_size
        self.reg_factor = reg_factor
        self.margin = margin
        self.n_neg_per_pos = n_neg_per_pos
        self.contrastive_mode = contrastive_mode
        self.contrastive_factor = contrastive_factor

        self.epsilon = 1e-5
        self.bot_ids = bot_ids.to(device)

        if num_individuals > 0:
            self.individual_emb = self.init_embeddings(num_individuals, low=-0.5, high=0.5)

        self.class_center              = self.init_embeddings(num_classes, low=-0.5,  high=0.5)
        self.class_raw_offset          = self.init_embeddings(num_classes, low=-0.1,  high=0.1)

        self.relation_range_center     = self.init_embeddings(num_roles,   low=-0.5,  high=0.5)
        self.relation_range_raw_offset = self.init_embeddings(num_roles,   low=-0.1,  high=0.1)

        self.relation_error_center     = self.init_embeddings(num_roles,   low=0.0, high=0.0)
        self.relation_error_raw_offset = self.init_embeddings(num_roles,   low=-0.1,  high=0.1)

        self.relation_scale_raw = self.init_embeddings(num_roles, low=0.9,  high=1.1)
        self.relation_shift     = self.init_embeddings(num_roles, low=-0.1, high=0.1)

    def init_embeddings(self, num_embeddings: int, low: float, high: float) -> nn.Embedding:
        emb = nn.Embedding(num_embeddings, self.embedding_dim)
        if low == high:
            nn.init.constant_(emb.weight, low)
            emb.weight.requires_grad_(False)
        else:
            nn.init.uniform_(emb.weight, low, high)
        return emb

    def get_class_box(self, ids) -> Box:
        c    = self.class_center(ids)
        raw  = self.class_raw_offset(ids)
        is_bot = torch.isin(ids, self.bot_ids).unsqueeze(-1)  # [batch, 1]
        half = torch.where(is_bot, raw, torch.abs(raw)+self.epsilon)  # [batch, dim]
        return Box(lower=c - half, upper=c + half)

    def get_role(self, ids) -> Role:
        range_c    = self.relation_range_center(ids)
        range_half = torch.abs(self.relation_range_raw_offset(ids))
        error_c    = self.relation_error_center(ids)
        error_half = torch.abs(self.relation_error_raw_offset(ids))
        return Role(
            range=Box(lower=range_c - range_half, upper=range_c + range_half),
            error=Box(lower=error_c - error_half, upper=error_c + error_half),
            transform=Transform(
                scale=torch.abs(self.relation_scale_raw(ids))+self.epsilon,
                shift=self.relation_shift(ids),
            ),
        )

    def get_individual(self, ids) -> Box:
        e = self.individual_emb(ids)
        return Box(lower=e, upper=e)

    def inclusion_loss(self, box1: Box, box2: Box) -> Tensor:
        sep = box1.separation(box2)
        return F.relu(sep + 2 * box1.offset + self.margin).mean(dim=-1)

    def overlap_loss(self, box1: Box, box2: Box) -> Tensor:
        d = box1.separation(box2)
        return F.relu(d + self.margin).mean(dim=-1)

    def disjoint_loss(self, box1: Box, box2: Box) -> Tensor:
        d = box1.separation(box2)
        return F.relu(self.margin - d.max(dim=-1).values)

    def nf1_loss(self, data) -> Tensor:
        c_boxes = self.get_class_box(data[:,0])
        d_boxes = self.get_class_box(data[:,1])
        return self.inclusion_loss(c_boxes, d_boxes)
    
    def nf1_pearson_loss(self, pos_data: Tensor, neg_data: Tensor) -> Tensor:
        saved = self.margin
        self.margin = 0.0
        try:
            pos_scores = torch.exp(-self.inclusion_loss(
                self.get_class_box(pos_data[:, 0]),
                self.get_class_box(pos_data[:, 1])))
            neg_scores = torch.exp(-self.inclusion_loss(
                self.get_class_box(neg_data[:, 0]),
                self.get_class_box(neg_data[:, 1])))
        finally:
            self.margin = saved
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        s = scores - scores.mean()
        y = labels - labels.mean()
        r = (s * y).sum() / (s.norm() * y.norm() + 1e-8)
        return 1 - r

    def nf1_bot_loss(self, data) -> Tensor:
        c_boxes = self.get_class_box(data[:,0])
        return F.relu(c_boxes.offset + self.margin).mean(dim=-1)

    def nf2_loss(self, data) -> Tensor:
        c_boxes = self.get_class_box(data[:,0])
        d_boxes = self.get_class_box(data[:,1])
        e_boxes = self.get_class_box(data[:,2])
        intersection = c_boxes.intersect(d_boxes)
        return self.inclusion_loss(intersection, e_boxes) + self.overlap_loss(c_boxes, d_boxes)

    def nf2_bot_loss(self, data) -> Tensor:
        c_boxes = self.get_class_box(data[:,0])
        d_boxes = self.get_class_box(data[:,1])
        return self.disjoint_loss(c_boxes, d_boxes)

    def nf3_loss(self, data) -> Tensor:
        c_boxes = self.get_class_box(data[:,0])
        relation = self.get_role(data[:,1])
        d_boxes = self.get_class_box(data[:,2])
        existential = relation.existential(d_boxes)
        return 2 * self.inclusion_loss(c_boxes, existential) + self.overlap_loss(relation.range, d_boxes)

    def nf4_loss(self, data) -> Tensor:
        relation = self.get_role(data[:,0])
        c_boxes = self.get_class_box(data[:,1])
        d_boxes = self.get_class_box(data[:,2])
        existential = relation.existential(c_boxes)
        return 2 * self.inclusion_loss(existential, d_boxes) + self.overlap_loss(relation.range, c_boxes)

    def nf4_bot_loss(self, data) -> Tensor:
        relation = self.get_role(data[:,0])
        c_boxes = self.get_class_box(data[:,1])
        return self.disjoint_loss(relation.range, c_boxes)

    def role_inclusion_loss(self, r: Role, s: Role) -> Tensor:
        r_lower_pt = Box(lower=r.range.lower, upper=r.range.lower)
        r_upper_pt = Box(lower=r.range.upper, upper=r.range.upper)

        r_lower_img = r.transform(r_lower_pt).minkowski_sum(r.error)
        s_lower_img = s.transform(r_lower_pt).minkowski_sum(s.error)

        r_upper_img = r.transform(r_upper_pt).minkowski_sum(r.error)
        s_upper_img = s.transform(r_upper_pt).minkowski_sum(s.error)

        return (
            self.inclusion_loss(r.range, s.range)
            + self.inclusion_loss(r_lower_img, s_lower_img)
            + self.inclusion_loss(r_upper_img, s_upper_img)
        )

    def nf5_loss(self, data) -> Tensor:
        r = self.get_role(data[:, 0])
        s = self.get_role(data[:, 1])
        return self.role_inclusion_loss(r, s)

    def nf6_loss(self, data) -> Tensor:
        r1 = self.get_role(data[:, 0])
        r2 = self.get_role(data[:, 1])
        s  = self.get_role(data[:, 2])

        # OWL order: r1 ; r2, r1 first, r2 second
        composed = r1.compose(r2)
        admissible_preimage = r2.transform.inverse()(
            r1.range.minkowski_difference(r2.error)
        )

        return (
            self.role_inclusion_loss(composed, s)
            + self.overlap_loss(r2.range, admissible_preimage)
        )

    def concept_assertion_loss(self, data) -> Tensor:
        individuals = self.get_individual(data[:,0])
        classes = self.get_class_box(data[:,1])
        return 1 - torch.exp(-self.inclusion_loss(individuals, classes))

    def ca_neg_hinge_loss(self, data) -> Tensor:
        """Hinge: relu(ln2 - inclusion_loss), so negatives are pushed to exp(-incl) <= 0.5."""
        individuals = self.get_individual(data[:,0])
        classes = self.get_class_box(data[:,1])
        return F.relu(math.log(2) - self.inclusion_loss(individuals, classes))

    def ca_pearson_loss(self, pos_data: Tensor, neg_data: Tensor) -> Tensor:
        saved = self.margin
        self.margin = 0.0
        try:
            pos_scores = torch.exp(-self.inclusion_loss(
                self.get_individual(pos_data[:, 0]),
                self.get_class_box(pos_data[:, 1])))
            neg_scores = torch.exp(-self.inclusion_loss(
                self.get_individual(neg_data[:, 0]),
                self.get_class_box(neg_data[:, 1])))
        finally:
            self.margin = saved
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        s = scores - scores.mean()
        y = labels - labels.mean()
        r = (s * y).sum() / (s.norm() * y.norm() + 1e-8)
        return 1 - r

    def box_regularisation_loss(self) -> Tensor:
        bot_mask = torch.zeros(self.num_classes, dtype=torch.bool, device=self.device)
        bot_mask[self.bot_ids] = True

        bot_validity = F.relu(self.class_raw_offset.weight[bot_mask] + self.epsilon).mean()

        # bounds = (
        #       F.relu(self.class_center.weight.norm(dim=-1) - 10.0).pow(2).mean()
        #       + F.relu(self.relation_range_center.weight.norm(dim=-1) - 10.0).pow(2).mean()
        #       + F.relu(self.relation_error_center.weight.norm(dim=-1) - 2.5).pow(2).mean()
        #  )

        size_penalty = (
                F.relu(torch.abs(self.class_raw_offset.weight[~bot_mask]) - 0.25).pow(2).mean()
                + F.relu(torch.abs(self.relation_range_raw_offset.weight) - 0.25).pow(2).mean()
                + F.relu(torch.abs(self.relation_error_raw_offset.weight) - 0.05).pow(2).mean()
        )

        return self.reg_factor * bot_validity  + size_penalty

    def get_data_batch(self, train_data, key, batch_size):
        data = train_data[key]
        if len(data) <= batch_size:
            return data
        idx = torch.randint(len(data), (batch_size,), device=data.device)
        return data[idx]

    def _sample_ca_negatives(
        self,
        pos_batch: Tensor,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        n = len(pos_batch)
        k = self.n_neg_per_pos
        prot_col = pos_batch[:, 0].repeat_interleave(k)
        pos_cls  = pos_batch[:, 1].repeat_interleave(k)
        neg_cls  = torch.randint(
            0,
            self.num_classes,
            (n * k,),
            device=self.device,
            generator=generator,
        )
        collision = neg_cls == pos_cls
        while collision.any():
            neg_cls[collision] = torch.randint(
                0,
                self.num_classes,
                (collision.sum().item(),),
                device=self.device,
                generator=generator,
            )
            collision = neg_cls == pos_cls
        return torch.stack([prot_col, neg_cls], dim=1)

    def _sample_nf1_negatives(
        self,
        pos_batch: Tensor,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        n = len(pos_batch)
        k = self.n_neg_per_pos
        c_col = pos_batch[:, 0].repeat_interleave(k)
        d_col = pos_batch[:, 1].repeat_interleave(k)
        neg_d = torch.randint(
            1,
            self.num_classes - 1,
            (n * k,),
            device=self.device,
            generator=generator,
        )
        collision = (neg_d == d_col) | (neg_d == c_col)
        while collision.any():
            neg_d[collision] = torch.randint(
                1,
                self.num_classes - 1,
                (collision.sum().item(),),
                device=self.device,
                generator=generator,
            )
            collision = (neg_d == d_col) | (neg_d == c_col)
        return torch.stack([c_col, neg_d], dim=1)

    def _sample_nf3_negatives(self, pos_batch, generator=None):
        """Corrupt either C or D with equal probability, keeping r fixed."""
        n = len(pos_batch)
        k = self.n_neg_per_pos
        c_col = pos_batch[:, 0].repeat_interleave(k)
        r_col = pos_batch[:, 1].repeat_interleave(k)
        d_col = pos_batch[:, 2].repeat_interleave(k)

        corrupt_d = torch.rand(n * k, device=self.device, generator=generator) < 0.5

        rand_concepts = torch.randint(1, self.num_classes - 1, (n * k,), device=self.device, generator=generator)
        collision = torch.where(corrupt_d, rand_concepts == d_col, rand_concepts == c_col)
        while collision.any():
            rand_concepts[collision] = torch.randint(
                1, self.num_classes - 1, (collision.sum().item(),), device=self.device, generator=generator)
            collision = torch.where(corrupt_d, rand_concepts == d_col, rand_concepts == c_col)

        neg_c = torch.where(corrupt_d, c_col, rand_concepts)
        neg_d = torch.where(corrupt_d, rand_concepts, d_col)
        return torch.stack([neg_c, r_col, neg_d], dim=1)

    def nf3_pearson_loss(self, pos_data, neg_data):
        saved = self.margin
        self.margin = 0.0
        try:
            def _score(data):
                c = self.get_class_box(data[:, 0])
                r = self.get_role(data[:, 1])
                d = self.get_class_box(data[:, 2])
                return torch.exp(-(self.inclusion_loss(c, r.existential(d)) + self.overlap_loss(r.range, d)))
            pos_scores = _score(pos_data)
            neg_scores = _score(neg_data)
        finally:
            self.margin = saved
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        s = scores - scores.mean()
        y = labels - labels.mean()
        r = (s * y).sum() / (s.norm() * y.norm() + 1e-8)
        return 1 - r

    def forward(self, train_data):
        parts = {
            'nf1': torch.tensor(0.0, device=self.device),
            'nf1_bot': torch.tensor(0.0, device=self.device),
            'nf2': torch.tensor(0.0, device=self.device),
            'nf2_bot': torch.tensor(0.0, device=self.device),
            'nf3': torch.tensor(0.0, device=self.device),
            'nf4': torch.tensor(0.0, device=self.device),
            'nf4_bot': torch.tensor(0.0, device=self.device),
            'nf5': torch.tensor(0.0, device=self.device),
            'nf6': torch.tensor(0.0, device=self.device),
        }
        neg_loss      = torch.tensor(0.0, device=self.device)
        contrast_loss = torch.tensor(0.0, device=self.device)

        if 'nf1' in train_data and len(train_data['nf1']) > 0:
            pos_batch = self.get_data_batch(train_data, 'nf1', self.batch_size)
            parts['nf1'] = self.nf1_loss(pos_batch).pow(2).mean()
            if self.n_neg_per_pos > 0:
                neg_batch = self._sample_nf1_negatives(pos_batch)
                if self.contrastive_mode == 'pearson':
                    contrast_loss = self.contrastive_factor * self.nf1_pearson_loss(pos_batch, neg_batch)

        if ('nf3' in train_data and len(train_data['nf3']) > 0
            and self.n_neg_per_pos > 0 and self.contrastive_mode == 'pearson'):
            nf3_pos = self.get_data_batch(train_data, 'nf3', self.batch_size)
            nf3_neg = self._sample_nf3_negatives(nf3_pos)
            contrast_loss = contrast_loss + self.contrastive_factor * self.nf3_pearson_loss(nf3_pos, nf3_neg)

        for key, loss_fn in [
            ('nf1_bot',  self.nf1_bot_loss),
            ('nf2',      self.nf2_loss),
            ('nf2_bot',  self.nf2_bot_loss),
            ('nf3',      self.nf3_loss),
            ('nf4',      self.nf4_loss),
            ('nf4_bot',  self.nf4_bot_loss),
            ('nf5',      self.nf5_loss),
            ('nf6',      self.nf6_loss),
        ]:
            if key in train_data and len(train_data[key]) > 0:
                batch = self.get_data_batch(train_data, key, self.batch_size)
                parts[key] = loss_fn(batch).pow(2).mean()

        nf1_6_loss = torch.stack(tuple(parts.values())).sum()

        abox_loss = torch.tensor(0.0, device=self.device)
        if 'abox' in train_data:
            ca = train_data['abox']['concept_assertions']
            if len(ca) > 0:
                pos_batch = self.get_data_batch({'ca': ca}, 'ca', self.batch_size)
                neg_batch = self._sample_ca_negatives(pos_batch)
                abox_loss = (
                    self.concept_assertion_loss(pos_batch).mean()
                    + self.ca_neg_hinge_loss(neg_batch).mean()
                    + self.ca_pearson_loss(pos_batch, neg_batch)
                )

    
        reg_loss = self.box_regularisation_loss()
        total = 1000 * (nf1_6_loss + contrast_loss + abox_loss + reg_loss)

        breakdown = {key: 1000 * value.item() for key, value in parts.items()}
        breakdown.update({
            'nf1_6':    1000 * nf1_6_loss.item(),
            'neg':      1000 * neg_loss.item(),
            'contrast': 1000 * contrast_loss.item(),
            'abox':     1000 * abox_loss.item(),
            'reg':      1000 * reg_loss.item(),
            'total':    total.item(),
        })
        return total, breakdown

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'embedding_dim':                self.embedding_dim,
            'class_center':                 self.class_center.weight.detach(),
            'class_raw_offset':             self.class_raw_offset.weight.detach(),
            'relation_range_center':        self.relation_range_center.weight.detach(),
            'relation_range_raw_offset':    self.relation_range_raw_offset.weight.detach(),
            'relation_error_center':        self.relation_error_center.weight.detach(),
            'relation_error_raw_offset':    self.relation_error_raw_offset.weight.detach(),
            'relation_scale_raw':           self.relation_scale_raw.weight.detach(),
            'relation_shift':               self.relation_shift.weight.detach(),
        }
        if self.num_individuals > 0:
            checkpoint['individual_emb'] = self.individual_emb.weight.detach()
        torch.save(checkpoint, path)

class Box3ModelLoaded:
    """Lightweight inference-only wrapper loaded from a checkpoint."""

    epsilon: float = 1e-5

    def __init__(
        self,
        class_center: torch.Tensor,
        class_raw_offset: torch.Tensor,
        relation_range_center: torch.Tensor,
        relation_range_raw_offset: torch.Tensor,
        relation_error_center: torch.Tensor,
        relation_error_raw_offset: torch.Tensor,
        relation_scale_raw: torch.Tensor,
        relation_shift: torch.Tensor,
        embedding_dim: int,
        bot_ids: torch.Tensor,
    ):
        self.class_center              = class_center
        self.class_raw_offset          = class_raw_offset
        self.relation_range_center     = relation_range_center
        self.relation_range_raw_offset = relation_range_raw_offset
        self.relation_error_center     = relation_error_center
        self.relation_error_raw_offset = relation_error_raw_offset
        self.relation_scale_raw        = relation_scale_raw
        self.relation_shift            = relation_shift
        self.embedding_dim             = embedding_dim
        self.bot_ids                   = bot_ids
        self.individual_emb: torch.Tensor | None = None

    def get_class_box(self, ids) -> Box:
        c      = self.class_center[ids]
        raw    = self.class_raw_offset[ids]
        is_bot = torch.isin(ids, self.bot_ids).unsqueeze(-1)
        half   = torch.where(is_bot, raw, torch.abs(raw) + self.epsilon)
        return Box(lower=c - half, upper=c + half)

    def get_individual(self, ids) -> Box:
        e = self.individual_emb[ids]
        return Box(lower=e, upper=e)

    def get_role(self, ids) -> Role:
        range_c    = self.relation_range_center[ids]
        range_half = torch.abs(self.relation_range_raw_offset[ids])
        error_c    = self.relation_error_center[ids]
        error_half = torch.abs(self.relation_error_raw_offset[ids])
        return Role(
            range=Box(lower=range_c - range_half, upper=range_c + range_half),
            error=Box(lower=error_c - error_half, upper=error_c + error_half),
            transform=Transform(
                scale=torch.abs(self.relation_scale_raw[ids]) + self.epsilon,
                shift=self.relation_shift[ids],
            ),
        )

    @staticmethod
    def load(path: str, device, bot_ids: torch.Tensor) -> Box3ModelLoaded:
        checkpoint = torch.load(path, map_location=device)
        loaded = Box3ModelLoaded(
            class_center=checkpoint['class_center'].to(device),
            class_raw_offset=checkpoint['class_raw_offset'].to(device),
            relation_range_center=checkpoint['relation_range_center'].to(device),
            relation_range_raw_offset=checkpoint['relation_range_raw_offset'].to(device),
            relation_error_center=checkpoint['relation_error_center'].to(device),
            relation_error_raw_offset=checkpoint['relation_error_raw_offset'].to(device),
            relation_scale_raw=checkpoint['relation_scale_raw'].to(device),
            relation_shift=checkpoint['relation_shift'].to(device),
            embedding_dim=checkpoint['embedding_dim'],
            bot_ids=bot_ids.to(device),
        )
        if 'individual_emb' in checkpoint:
            loaded.individual_emb = checkpoint['individual_emb'].to(device)
        return loaded

