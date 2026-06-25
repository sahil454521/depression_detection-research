"""Validation layer: LODO cross-dataset validation, metric suite, and bias audit.

Components
----------
1. MetricSuite          -- F1, AUC-ROC, sensitivity, specificity, MCC
2. BiasAudit            -- per-demographic fairness metrics
3. LODOValidator        -- leave-one-dataset-out cross-dataset evaluation
4. ValidationLayer      -- orchestrates all three + retrain decision
5. format_validation_report() -- ASCII report formatter

All metrics are computed from scratch (no sklearn dependency).

Pipeline position
-----------------
... -> FL -> [VALIDATION LAYER] --retrain loop--> back to Model / FL
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from prediction import PredictionConfig, compute_prediction_losses


# ---------------------------------------------------------------------------
# MetricResults
# ---------------------------------------------------------------------------

@dataclass
class MetricResults:
    """All scalar metrics for one evaluation split."""

    accuracy: float     # (TP+TN) / total
    f1: float           # 2*TP / (2*TP + FP + FN)
    auc_roc: float      # Area under ROC curve (trapezoidal rule)
    sensitivity: float  # TP / (TP+FN)  -- recall for positive class
    specificity: float  # TN / (TN+FP)  -- recall for negative class
    mcc: float          # Matthews Correlation Coefficient
    support: int        # total samples evaluated

    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy":    round(self.accuracy, 4),
            "f1":          round(self.f1, 4),
            "auc_roc":     round(self.auc_roc, 4),
            "sensitivity": round(self.sensitivity, 4),
            "specificity": round(self.specificity, 4),
            "mcc":         round(self.mcc, 4),
            "support":     self.support,
        }


# ---------------------------------------------------------------------------
# BiasAuditResult
# ---------------------------------------------------------------------------

@dataclass
class BiasAuditResult:
    """Fairness metrics for a single demographic subgroup."""

    subgroup_name: str        # e.g. "gender=male", "source=reddit"
    attribute: str            # e.g. "gender", "source"
    subgroup_value: int       # integer code

    metrics: MetricResults

    # --- Fairness gap measures ---
    equalized_odds_gap: float
    """Absolute gap between this group's TPR and the reference group's TPR."""

    demographic_parity_gap: float
    """Absolute gap between this group's P(y_hat=1) and reference group's."""

    reference_group: str
    """The group used as the fairness reference baseline."""


# ---------------------------------------------------------------------------
# LODOResult
# ---------------------------------------------------------------------------

@dataclass
class LODOResult:
    """Results for one LODO (Leave-One-Dataset-Out) fold."""

    fold_index: int
    held_out_dataset: str          # e.g. "reddit"
    train_datasets: List[str]
    train_size: int
    test_size: int
    metrics: MetricResults
    bias_audit: List[BiasAuditResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Full output of the ValidationLayer for one model checkpoint."""

    model_round: int
    overall_metrics: MetricResults
    lodo_results: List[LODOResult] = field(default_factory=list)
    bias_audit_results: List[BiasAuditResult] = field(default_factory=list)

    # Derived convenience fields
    lodo_avg_f1: float = 0.0
    lodo_avg_auc: float = 0.0
    retrain_needed: bool = False


# ---------------------------------------------------------------------------
# MetricSuite
# ---------------------------------------------------------------------------

class MetricSuite:
    """Compute classification metrics without external dependencies.

    All metrics derived from confusion matrix entries and probability scores.
    AUC-ROC uses the trapezoidal rule on the sorted score sequence.
    """

    @staticmethod
    def _confusion_matrix(
        y_true: List[int], y_pred: List[int]
    ) -> Tuple[int, int, int, int]:
        """Return (TP, FP, FN, TN)."""
        tp = fp = fn = tn = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 1 and yp == 1:
                tp += 1
            elif yt == 0 and yp == 1:
                fp += 1
            elif yt == 1 and yp == 0:
                fn += 1
            else:
                tn += 1
        return tp, fp, fn, tn

    @staticmethod
    def _auc_roc(y_true: List[int], y_prob: List[float]) -> float:
        """Trapezoidal AUC-ROC."""
        pos = sum(y_true)
        neg = len(y_true) - pos
        if pos == 0 or neg == 0:
            return 0.5

        # Sort by descending predicted probability
        pairs = sorted(zip(y_prob, y_true), key=lambda x: -x[0])
        tpr_list = [0.0]
        fpr_list = [0.0]
        tp = fp = 0
        for prob, label in pairs:
            if label == 1:
                tp += 1
            else:
                fp += 1
            tpr_list.append(tp / pos)
            fpr_list.append(fp / neg)
        tpr_list.append(1.0)
        fpr_list.append(1.0)

        # Trapezoidal rule
        auc = sum(
            (fpr_list[i + 1] - fpr_list[i]) * (tpr_list[i + 1] + tpr_list[i]) / 2.0
            for i in range(len(fpr_list) - 1)
        )
        return max(0.0, min(1.0, auc))

    def compute(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_prob: List[float],
    ) -> MetricResults:
        """Compute all metrics.

        Parameters
        ----------
        y_true : list of int (0/1)
        y_pred : list of int (0/1), predicted class
        y_prob : list of float, predicted probability of class 1
        """
        n = len(y_true)
        if n == 0:
            return MetricResults(0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0)

        tp, fp, fn, tn = self._confusion_matrix(y_true, y_pred)

        eps = 1e-9
        accuracy    = (tp + tn) / n
        precision   = tp / max(tp + fp, eps)
        recall      = tp / max(tp + fn, eps)
        f1          = 2 * precision * recall / max(precision + recall, eps)
        sensitivity = recall
        specificity = tn / max(tn + fp, eps)
        mcc_denom   = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), eps))
        mcc         = (tp * tn - fp * fn) / mcc_denom
        auc_roc     = self._auc_roc(y_true, y_prob)

        return MetricResults(
            accuracy=round(accuracy, 6),
            f1=round(f1, 6),
            auc_roc=round(auc_roc, 6),
            sensitivity=round(sensitivity, 6),
            specificity=round(specificity, 6),
            mcc=round(mcc, 6),
            support=n,
        )


# ---------------------------------------------------------------------------
# BiasAudit
# ---------------------------------------------------------------------------

class BiasAudit:
    """Per-demographic-subgroup fairness evaluation.

    Evaluates fairness with respect to two attributes:
      - gender  (0=unknown, 1=male, 2=female)  -- from WU3D
      - source  (0=wu3d, 1=reddit)             -- dataset of origin

    Computes per-subgroup MetricResults and two fairness gap metrics:
      - Equalized Odds Gap   = |TPR_group - TPR_reference|
      - Demographic Parity Gap = |P(y_hat=1|group) - P(y_hat=1|reference)|

    The reference group is the largest subgroup for each attribute.
    """

    ATTRIBUTE_MAPS = {
        "gender": {0: "unknown", 1: "male", 2: "female"},
        "source": {0: "wu3d", 1: "reddit"},
    }

    def __init__(self):
        self._metric_suite = MetricSuite()

    def audit(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_prob: List[float],
        sensitive_attrs: Dict[str, List[int]],
    ) -> List[BiasAuditResult]:
        """Compute fairness metrics per subgroup.

        Parameters
        ----------
        y_true, y_pred, y_prob : lists aligned by sample index.
        sensitive_attrs : {'gender': [int, ...], 'source': [int, ...]}
            Integer codes per sample for each demographic attribute.
        """
        results: List[BiasAuditResult] = []
        n = len(y_true)

        for attr, value_map in self.ATTRIBUTE_MAPS.items():
            attr_values = sensitive_attrs.get(attr, [])
            if len(attr_values) != n:
                continue

            # Find all subgroups present
            present = sorted(set(attr_values))
            if len(present) < 2:
                continue

            # Compute metrics per subgroup
            subgroup_metrics: Dict[int, MetricResults] = {}
            subgroup_sizes: Dict[int, int] = {}
            for val in present:
                idx = [i for i, v in enumerate(attr_values) if v == val]
                if not idx:
                    continue
                sub_true = [y_true[i] for i in idx]
                sub_pred = [y_pred[i] for i in idx]
                sub_prob = [y_prob[i] for i in idx]
                m = self._metric_suite.compute(sub_true, sub_pred, sub_prob)
                subgroup_metrics[val] = m
                subgroup_sizes[val]   = len(idx)

            if len(subgroup_metrics) < 2:
                continue

            # Reference = largest subgroup
            ref_val = max(subgroup_sizes, key=subgroup_sizes.get)
            ref_m   = subgroup_metrics[ref_val]
            ref_name = value_map.get(ref_val, str(ref_val))
            ref_tpr  = ref_m.sensitivity
            ref_pos_rate = sum(
                1 for i, v in enumerate(attr_values) if v == ref_val and y_pred[i] == 1
            ) / max(subgroup_sizes[ref_val], 1)

            for val, m in subgroup_metrics.items():
                if val == ref_val:
                    continue
                val_name = value_map.get(val, str(val))
                pos_rate = sum(
                    1 for i, v in enumerate(attr_values) if v == val and y_pred[i] == 1
                ) / max(subgroup_sizes.get(val, 1), 1)

                results.append(BiasAuditResult(
                    subgroup_name="{}={}".format(attr, val_name),
                    attribute=attr,
                    subgroup_value=val,
                    metrics=m,
                    equalized_odds_gap=round(abs(m.sensitivity - ref_tpr), 6),
                    demographic_parity_gap=round(abs(pos_rate - ref_pos_rate), 6),
                    reference_group="{}={}".format(attr, ref_name),
                ))

        return results


# ---------------------------------------------------------------------------
# LODOValidator
# ---------------------------------------------------------------------------

class LODOValidator:
    """Leave-One-Dataset-Out cross-dataset validation.

    Simulates real-world clinical deployment where a model trained on one
    institution's data is tested on a completely different population/dataset.

    Folds
    -----
    - For each unique 'source' in the dataset:
        train on all other sources -> evaluate on held-out source
    - Typically: WU3D <-> Reddit (two folds)
    """

    def __init__(self, batch_size: int = 32):
        self.batch_size   = batch_size
        self._metric_suite = MetricSuite()
        self._bias_audit   = BiasAudit()

    def _collect_predictions(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> Tuple[List[int], List[int], List[float], Dict[str, List[int]]]:
        """Run model on loader, return (y_true, y_pred, y_prob, sensitive_attrs)."""
        model.eval()
        y_true, y_pred, y_prob = [], [], []
        sens_gender, sens_source = [], []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(
                    text_input=batch["text_emb"],
                    eeg=batch["eeg"],
                    wearable=batch["wearable"],
                    audio=batch["audio"],
                    video=batch["video"],
                    clinical=batch["clinical"],
                    mfcc=batch.get("mfcc"),
                )
                probs = out["binary_probs"].cpu()    # [B, 2]
                preds = probs.argmax(dim=-1).tolist()
                trues = batch["label"].view(-1).cpu().tolist()
                prob1 = probs[:, 1].tolist()          # P(depressed)

                y_true.extend(int(t) for t in trues)
                y_pred.extend(int(p) for p in preds)
                y_prob.extend(float(p) for p in prob1)
                sens_gender.extend(batch["gender"].view(-1).cpu().tolist())
                sens_source.extend(batch["language"].view(-1).cpu().tolist())

        sensitive = {"gender": sens_gender, "source": sens_source}
        return y_true, y_pred, y_prob, sensitive

    def _retrain_on_subset(
        self,
        model_factory: Callable[[], nn.Module],
        train_dataset,
        prediction_config: PredictionConfig,
        collate_fn: Callable,
        device: torch.device,
        epochs: int = 3,
        lr: float = 1e-3,
        batch_size: int = 16,
    ) -> nn.Module:
        """Train a fresh model on the provided training subset."""
        model = model_factory().to(device)
        model.train()
        loader = DataLoader(
            train_dataset,
            batch_size=max(1, batch_size),
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=False,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                out = model(
                    text_input=batch["text_emb"],
                    eeg=batch["eeg"],
                    wearable=batch["wearable"],
                    audio=batch["audio"],
                    video=batch["video"],
                    clinical=batch["clinical"],
                    mfcc=batch.get("mfcc"),
                    labels=batch["label"],
                )
                loss = compute_prediction_losses(
                    out,
                    {"label": batch["label"],
                     "phq9_score": batch["phq9_score"],
                     "symptom_labels": batch["symptom_labels"]},
                    prediction_config,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item()
            avg = total_loss / max(len(loader), 1)
            print("    LODO retrain epoch {}/{}: loss={:.4f}".format(epoch + 1, epochs, avg))

        return model

    def validate(
        self,
        model_factory: Callable[[], nn.Module],
        full_dataset,
        prediction_config: PredictionConfig,
        collate_fn: Callable,
        device: torch.device,
        lodo_train_epochs: int = 3,
        lodo_lr: float = 1e-3,
        batch_size: int = 16,
    ) -> List[LODOResult]:
        """Run full LODO validation.

        Parameters
        ----------
        model_factory : callable
            Returns a fresh, untrained DepressionDetectionModel.
        full_dataset : RealDepressionDataset
            Full dataset with .records attribute (each has .source field).
        prediction_config : PredictionConfig
        collate_fn : callable
        device : torch.device
        lodo_train_epochs : int
            Epochs to train the LODO fold model.
        lodo_lr : float
            Learning rate for LODO fold training.
        batch_size : int
        """
        records = full_dataset.records
        sources = sorted(set(r.source for r in records))
        print("\n=== LODO Validation ({} folds: {}) ===".format(len(sources), sources))

        results: List[LODOResult] = []
        for fold_idx, held_out in enumerate(sources):
            train_idx = [i for i, r in enumerate(records) if r.source != held_out]
            test_idx  = [i for i, r in enumerate(records) if r.source == held_out]

            if not train_idx or not test_idx:
                print("  Fold {}: skipped (empty split).".format(fold_idx))
                continue

            train_sources = sorted(set(records[i].source for i in train_idx))
            print(
                "\n  Fold {}: held-out='{}', train={}  ({} train / {} test)".format(
                    fold_idx, held_out, train_sources, len(train_idx), len(test_idx)
                )
            )

            train_subset = Subset(full_dataset, train_idx)
            test_subset  = Subset(full_dataset, test_idx)

            # Train fresh model on fold's training split
            fold_model = self._retrain_on_subset(
                model_factory=model_factory,
                train_dataset=train_subset,
                prediction_config=prediction_config,
                collate_fn=collate_fn,
                device=device,
                epochs=lodo_train_epochs,
                lr=lodo_lr,
                batch_size=batch_size,
            )

            # Evaluate on held-out split
            test_loader = DataLoader(
                test_subset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
            y_true, y_pred, y_prob, sensitive = self._collect_predictions(
                fold_model, test_loader, device
            )
            metrics = self._metric_suite.compute(y_true, y_pred, y_prob)
            bias    = self._bias_audit.audit(y_true, y_pred, y_prob, sensitive)

            print(
                "  Fold {} results: F1={:.4f} AUC={:.4f} "
                "sensitivity={:.4f} specificity={:.4f} MCC={:.4f}".format(
                    fold_idx,
                    metrics.f1, metrics.auc_roc,
                    metrics.sensitivity, metrics.specificity, metrics.mcc,
                )
            )

            results.append(LODOResult(
                fold_index=fold_idx,
                held_out_dataset=held_out,
                train_datasets=train_sources,
                train_size=len(train_idx),
                test_size=len(test_idx),
                metrics=metrics,
                bias_audit=bias,
            ))

        return results


# ---------------------------------------------------------------------------
# ValidationLayer
# ---------------------------------------------------------------------------

@dataclass
class ValidationConfig:
    """Thresholds and options for the ValidationLayer."""

    target_f1: float = 0.65
    """Minimum acceptable F1 score on the overall validation set."""

    target_auc_roc: float = 0.70
    """Minimum acceptable AUC-ROC."""

    target_mcc: float = 0.20
    """Minimum acceptable MCC (above-chance threshold)."""

    max_equalized_odds_gap: float = 0.15
    """Maximum tolerated equalized-odds gap for fairness."""

    run_lodo: bool = True
    """Whether to run full LODO validation (expensive)."""

    lodo_train_epochs: int = 3
    """Epochs for each LODO fold model."""

    lodo_lr: float = 1e-3


class ValidationLayer:
    """Orchestrates overall, LODO, and bias validation for a model checkpoint.

    Sits at the end of the pipeline and drives the retrain loop:
        fl_layer.run(model) -> val_layer.validate(model) -> retrain_needed?

    Usage
    -----
    val_layer = ValidationLayer(val_cfg, lodo_validator, metric_suite, bias_audit)
    report = val_layer.validate(model, val_loader, full_dataset, model_factory, ...)
    if report.retrain_needed:
        fl_layer.run(model)  # retrain
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.cfg           = config or ValidationConfig()
        self._metric_suite = MetricSuite()
        self._bias_audit   = BiasAudit()
        self._lodo         = LODOValidator()

    # ------------------------------------------------------------------

    def _collect_predictions(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> Tuple[List[int], List[int], List[float], Dict[str, List[int]]]:
        model.eval()
        y_true, y_pred, y_prob = [], [], []
        sens_gender, sens_source = [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(
                    text_input=batch["text_emb"],
                    eeg=batch["eeg"],
                    wearable=batch["wearable"],
                    audio=batch["audio"],
                    video=batch["video"],
                    clinical=batch["clinical"],
                    mfcc=batch.get("mfcc"),
                )
                probs = out["binary_probs"].cpu()
                preds = probs.argmax(dim=-1).tolist()
                trues = batch["label"].view(-1).cpu().tolist()
                prob1 = probs[:, 1].tolist()
                y_true.extend(int(t) for t in trues)
                y_pred.extend(int(p) for p in preds)
                y_prob.extend(float(p) for p in prob1)
                sens_gender.extend(batch["gender"].view(-1).cpu().tolist())
                sens_source.extend(batch["language"].view(-1).cpu().tolist())
        return y_true, y_pred, y_prob, {"gender": sens_gender, "source": sens_source}

    # ------------------------------------------------------------------

    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        model_round: int = 0,
        full_dataset=None,
        model_factory: Optional[Callable[[], nn.Module]] = None,
        prediction_config: Optional[PredictionConfig] = None,
        collate_fn: Optional[Callable] = None,
    ) -> ValidationReport:
        """Full validation of a model checkpoint.

        Parameters
        ----------
        model : nn.Module
            Trained model to evaluate.
        val_loader : DataLoader
            Standard held-out validation set loader.
        device : torch.device
        model_round : int
            Current retrain round index (for logging).
        full_dataset : RealDepressionDataset, optional
            Required for LODO validation (provides .records with .source).
        model_factory : callable, optional
            Returns a fresh DepressionDetectionModel.  Required for LODO.
        prediction_config : PredictionConfig, optional
        collate_fn : callable, optional
        """
        print("\n=== Validation (round {}) ===".format(model_round))

        # 1. Overall metrics on standard val split
        y_true, y_pred, y_prob, sensitive = self._collect_predictions(
            model, val_loader, device
        )
        overall_metrics = self._metric_suite.compute(y_true, y_pred, y_prob)
        print(
            "  Overall: F1={:.4f}  AUC={:.4f}  sensitivity={:.4f}  "
            "specificity={:.4f}  MCC={:.4f}  (n={})".format(
                overall_metrics.f1, overall_metrics.auc_roc,
                overall_metrics.sensitivity, overall_metrics.specificity,
                overall_metrics.mcc, overall_metrics.support,
            )
        )

        # 2. Bias audit on overall val split
        bias_results = self._bias_audit.audit(y_true, y_pred, y_prob, sensitive)
        for b in bias_results:
            print(
                "  Bias [{} vs {}]: EO-gap={:.4f}  DP-gap={:.4f}".format(
                    b.subgroup_name, b.reference_group,
                    b.equalized_odds_gap, b.demographic_parity_gap,
                )
            )

        # 3. LODO validation (optional, expensive)
        lodo_results: List[LODOResult] = []
        if (self.cfg.run_lodo
                and full_dataset is not None
                and model_factory is not None
                and prediction_config is not None
                and collate_fn is not None):
            lodo_results = self._lodo.validate(
                model_factory=model_factory,
                full_dataset=full_dataset,
                prediction_config=prediction_config,
                collate_fn=collate_fn,
                device=device,
                lodo_train_epochs=self.cfg.lodo_train_epochs,
                lodo_lr=self.cfg.lodo_lr,
            )
        else:
            print("  (LODO skipped: missing model_factory or full_dataset)")

        # 4. Aggregate LODO metrics
        lodo_avg_f1  = (sum(r.metrics.f1  for r in lodo_results) / len(lodo_results)
                        if lodo_results else 0.0)
        lodo_avg_auc = (sum(r.metrics.auc_roc for r in lodo_results) / len(lodo_results)
                        if lodo_results else 0.0)
        if lodo_results:
            print(
                "  LODO avg: F1={:.4f}  AUC={:.4f}".format(lodo_avg_f1, lodo_avg_auc)
            )

        # 5. Decide if retraining is needed
        retrain = (
            overall_metrics.f1      < self.cfg.target_f1
            or overall_metrics.auc_roc < self.cfg.target_auc_roc
            or overall_metrics.mcc  < self.cfg.target_mcc
        )
        # Also check bias
        if not retrain:
            for b in bias_results:
                if b.equalized_odds_gap > self.cfg.max_equalized_odds_gap:
                    retrain = True
                    print(
                        "  Retrain triggered by bias: {} EO-gap={:.4f} > {:.4f}".format(
                            b.subgroup_name, b.equalized_odds_gap,
                            self.cfg.max_equalized_odds_gap,
                        )
                    )
                    break

        if retrain:
            print("  => Retrain NEEDED (metrics below thresholds)")
        else:
            print("  => Metrics SATISFACTORY -- no retrain required")

        return ValidationReport(
            model_round=model_round,
            overall_metrics=overall_metrics,
            lodo_results=lodo_results,
            bias_audit_results=bias_results,
            lodo_avg_f1=round(lodo_avg_f1, 4),
            lodo_avg_auc=round(lodo_avg_auc, 4),
            retrain_needed=retrain,
        )

    # ------------------------------------------------------------------

    def metrics_satisfactory(self, report: ValidationReport) -> bool:
        """Return True if the model meets all configured quality thresholds."""
        return not report.retrain_needed


# ---------------------------------------------------------------------------
# ASCII Report Formatter
# ---------------------------------------------------------------------------

def format_validation_report(report: ValidationReport, width: int = 68) -> str:
    """Return a full ASCII validation report string."""

    def _bar(v, mx=1.0, w=25):
        f = int(round(max(v, 0.0) / max(mx, 1e-9) * w))
        return "#" * f + "-" * (w - f)

    lines = [
        "+" + "=" * width + "+",
        "| VALIDATION REPORT  (round {})".format(report.model_round).ljust(width) + " |",
        "+" + "-" * width + "+",
        "| OVERALL METRICS".ljust(width) + " |",
    ]

    m = report.overall_metrics
    for name, val in [
        ("Accuracy",    m.accuracy),
        ("F1 Score",    m.f1),
        ("AUC-ROC",     m.auc_roc),
        ("Sensitivity", m.sensitivity),
        ("Specificity", m.specificity),
        ("MCC",         m.mcc),
    ]:
        bar = _bar(val)
        lines.append("|   {:<14} {:6.4f}  {}".format(name, val, bar).ljust(width) + " |")
    lines.append("|   Support: {:>6}".format(m.support).ljust(width) + " |")

    if report.bias_audit_results:
        lines += [
            "+" + "-" * width + "+",
            "| BIAS AUDIT".ljust(width) + " |",
        ]
        for b in report.bias_audit_results:
            lines.append(
                "|   {:>24} vs {:>12}  EO-gap={:.4f}  DP-gap={:.4f}".format(
                    b.subgroup_name, b.reference_group,
                    b.equalized_odds_gap, b.demographic_parity_gap,
                ).ljust(width) + " |"
            )

    if report.lodo_results:
        lines += [
            "+" + "-" * width + "+",
            "| LODO (Leave-One-Dataset-Out) CROSS-DATASET RESULTS".ljust(width) + " |",
        ]
        for lr in report.lodo_results:
            lines.append(
                "|   Fold {}: held-out={:<8}  F1={:.4f}  AUC={:.4f}  MCC={:.4f}".format(
                    lr.fold_index, lr.held_out_dataset,
                    lr.metrics.f1, lr.metrics.auc_roc, lr.metrics.mcc,
                ).ljust(width) + " |"
            )
        lines.append(
            "|   LODO avg ->  F1={:.4f}  AUC={:.4f}".format(
                report.lodo_avg_f1, report.lodo_avg_auc,
            ).ljust(width) + " |"
        )

    lines += [
        "+" + "-" * width + "+",
        "| RETRAIN NEEDED: {}".format("YES" if report.retrain_needed else "NO").ljust(width) + " |",
        "+" + "=" * width + "+",
    ]
    return "\n".join(lines)
