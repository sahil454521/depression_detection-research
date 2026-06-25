"""Multi-task prediction layer: binary classification, severity regression, symptom recognition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class PredictionConfig:
    num_symptoms: int = 20
    binary_weight: float = 1.0
    severity_weight: float = 1.0
    symptom_weight: float = 0.5


class BinaryClassifier(nn.Module):
    """Depressed vs controlled — 2-class softmax output at inference."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class SeverityRegressor(nn.Module):
    """PHQ-9 score estimation with MSE loss."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 1)),
            nn.ReLU(),
            nn.Linear(max(hidden_dim // 2, 1), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class SymptomRecognizer(nn.Module):
    """Multi-label DSM-5 per-symptom probability via sigmoid."""

    def __init__(self, hidden_dim: int, num_symptoms: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_symptoms),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class PredictionLayer(nn.Module):
    def __init__(self, hidden_dim: int, config: Optional[PredictionConfig] = None):
        super().__init__()
        self.config = config or PredictionConfig()
        self.binary_classifier = BinaryClassifier(hidden_dim)
        self.severity_regressor = SeverityRegressor(hidden_dim)
        self.symptom_recognizer = SymptomRecognizer(hidden_dim, self.config.num_symptoms)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        binary_logits = self.binary_classifier(x)
        severity = self.severity_regressor(x)
        symptom_logits = self.symptom_recognizer(x)
        return {
            "binary_logits": binary_logits,
            "binary_probs": F.softmax(binary_logits, dim=-1),
            "severity": severity,
            "symptom_logits": symptom_logits,
            "symptom_probs": torch.sigmoid(symptom_logits),
        }


def compute_prediction_losses(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    config: Optional[PredictionConfig] = None,
) -> torch.Tensor:
    cfg = config or PredictionConfig()
    binary_loss = F.cross_entropy(
        outputs["binary_logits"],
        targets["label"].long().view(-1),
    )
    severity_loss = F.mse_loss(
        outputs["severity"].view(-1),
        targets["phq9_score"].view(-1).float(),
    )
    symptom_loss = F.binary_cross_entropy_with_logits(
        outputs["symptom_logits"],
        targets["symptom_labels"].float(),
    )
    return (
        cfg.binary_weight * binary_loss
        + cfg.severity_weight * severity_loss
        + cfg.symptom_weight * symptom_loss
    )
