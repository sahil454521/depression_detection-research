"""Fairness-aware regularizer with adversarial debiasing and fairness metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.autograd import Function


@dataclass
class FairnessConfig:
    num_genders: int = 2
    num_age_bins: int = 5
    num_languages: int = 8
    num_cultures: int = 8
    grl_scale: float = 1.0
    adv_weight: float = 0.1
    eod_weight: float = 0.05
    dp_weight: float = 0.05


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


def gradient_reversal(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, scale)


class SensitiveAttributeDiscriminator(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def equal_opportunity_difference(
    binary_logits: torch.Tensor,
    labels: torch.Tensor,
    sensitive_attr: torch.Tensor,
    num_groups: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable proxy for EOD: max TPR gap across sensitive groups (positive class only)."""
    probs = torch.softmax(binary_logits, dim=-1)[:, 1]
    labels = labels.float()
    positive_mask = labels > 0.5
    if positive_mask.sum() < 1:
        return torch.zeros((), device=binary_logits.device, dtype=binary_logits.dtype)

    tprs = []
    for group in range(num_groups):
        group_mask = (sensitive_attr == group) & positive_mask
        if group_mask.sum() < 1:
            continue
        tpr = (probs * group_mask.float()).sum() / group_mask.float().sum().clamp(min=eps)
        tprs.append(tpr)

    if len(tprs) < 2:
        return torch.zeros((), device=binary_logits.device, dtype=binary_logits.dtype)

    tpr_stack = torch.stack(tprs)
    return tpr_stack.max() - tpr_stack.min()


def demographic_parity_difference(
    binary_logits: torch.Tensor,
    sensitive_attr: torch.Tensor,
    num_groups: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable proxy for DP: max gap in positive prediction rate across groups."""
    probs = torch.softmax(binary_logits, dim=-1)[:, 1]
    rates = []
    for group in range(num_groups):
        group_mask = sensitive_attr == group
        if group_mask.sum() < 1:
            continue
        rate = (probs * group_mask.float()).sum() / group_mask.float().sum().clamp(min=eps)
        rates.append(rate)

    if len(rates) < 2:
        return torch.zeros((), device=binary_logits.device, dtype=binary_logits.dtype)

    rate_stack = torch.stack(rates)
    return rate_stack.max() - rate_stack.min()


class FairnessAwareRegularizer(nn.Module):
    """Projects fused representations and applies adversarial debiasing + fairness penalties."""

    SENSITIVE_ATTR_CONFIG = (
        ("gender", "num_genders"),
        ("age_bin", "num_age_bins"),
        ("language", "num_languages"),
        ("culture", "num_cultures"),
    )

    def __init__(self, hidden_dim: int, config: Optional[FairnessConfig] = None):
        super().__init__()
        self.config = config or FairnessConfig()
        self.projector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.discriminators = nn.ModuleDict(
            {
                attr_name: SensitiveAttributeDiscriminator(
                    hidden_dim, getattr(self.config, num_classes_key)
                )
                for attr_name, num_classes_key in self.SENSITIVE_ATTR_CONFIG
            }
        )

    def forward(
        self,
        representation: torch.Tensor,
        sensitive_attrs: Optional[Dict[str, torch.Tensor]] = None,
        binary_logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        debiased = self.projector(representation)
        device = representation.device
        fairness_loss = torch.zeros((), device=device, dtype=representation.dtype)
        aux: Dict[str, torch.Tensor] = {}

        if not training or sensitive_attrs is None:
            return debiased, fairness_loss, aux

        adv_loss = torch.zeros((), device=device, dtype=representation.dtype)
        reversed_repr = gradient_reversal(debiased, self.config.grl_scale)

        for attr_name, num_classes_key in self.SENSITIVE_ATTR_CONFIG:
            if attr_name not in sensitive_attrs:
                continue
            attr_labels = sensitive_attrs[attr_name].long()
            adv_logits = self.discriminators[attr_name](reversed_repr)
            attr_loss = nn.functional.cross_entropy(adv_logits, attr_labels)
            adv_loss = adv_loss + attr_loss
            aux[f"adv_{attr_name}"] = attr_loss.detach()

        eod_loss = torch.zeros((), device=device, dtype=representation.dtype)
        dp_loss = torch.zeros((), device=device, dtype=representation.dtype)

        if binary_logits is not None and labels is not None:
            flat_labels = labels.view(-1)
            for attr_name, num_classes_key in self.SENSITIVE_ATTR_CONFIG:
                if attr_name not in sensitive_attrs:
                    continue
                num_groups = getattr(self.config, num_classes_key)
                attr_values = sensitive_attrs[attr_name].view(-1)
                eod_loss = eod_loss + equal_opportunity_difference(
                    binary_logits, flat_labels, attr_values, num_groups
                )
                dp_loss = dp_loss + demographic_parity_difference(
                    binary_logits, attr_values, num_groups
                )

        fairness_loss = (
            self.config.adv_weight * adv_loss
            + self.config.eod_weight * eod_loss
            + self.config.dp_weight * dp_loss
        )
        aux["eod"] = eod_loss.detach()
        aux["dp"] = dp_loss.detach()
        aux["adv_total"] = adv_loss.detach()

        return debiased, fairness_loss, aux
