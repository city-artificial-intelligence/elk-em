from __future__ import annotations

from dataclasses import dataclass
from torch import Tensor 
import torch.nn.functional as F
import torch

@dataclass
class Box: 
    lower: Tensor
    upper: Tensor

    def intersect(self, other: Box) -> Box:
        lower = torch.maximum(self.lower, other.lower)
        upper = torch.minimum(self.upper, other.upper)
        return Box(lower, upper)

    def minkowski_sum(self, other: Box) -> Box:
        return Box(
            lower=self.lower + other.lower,
            upper=self.upper + other.upper,
        )

    def minkowski_difference(self, other: Box) -> Box:
        return Box(
            lower=self.lower - other.upper,
            upper=self.upper - other.lower,
        )

    def delta(self) -> Tensor:
        return self.upper - self.lower

    @property
    def center(self) -> Tensor:
        return (self.lower + self.upper) / 2

    @property
    def offset(self) -> Tensor:
        return (self.upper - self.lower) / 2

    def separation(self, other: Box) -> Tensor:
        return torch.abs(self.center - other.center) - self.offset - other.offset


@dataclass
class Transform:
    scale: Tensor
    shift: Tensor

    def __call__(self, box: Box) -> Box:
        return Box(
            lower=box.lower * self.scale + self.shift,
            upper=box.upper * self.scale + self.shift,
        )

    def inverse(self) -> Transform:
        return Transform(
            scale=1 / self.scale,
            shift=-self.shift / self.scale,
        )

    def compose(self, other: Transform) -> Transform:
        return Transform(
            scale=self.scale * other.scale,
            shift=self.scale * other.shift + self.shift,
        )

@dataclass
class Role:
    range: Box
    error: Box
    transform: Transform

    @property
    def domain(self) -> Box:
        return self.transform(self.range).minkowski_sum(self.error)

    def existential(self, concept_box: Box) -> Box:
        return self.transform(self.range.intersect(concept_box)).minkowski_sum(self.error)

    def compose(self, other: Role) -> Role:
        composed_range = other.range.intersect(
           other.transform.inverse()(self.range.minkowski_difference(other.error)))
        composed_error = self.transform(other.error).minkowski_sum(self.error)
        composed_transform = self.transform.compose(other.transform)
        return Role(composed_range, composed_error, composed_transform)

def exist(role: Role, concept_box: Box) -> Box:
    return role.existential(concept_box)