"""Output layer: Patient Report and Clinical Dashboard.

Sits immediately after the XAI module.  Transforms raw model outputs
(binary prediction, PHQ-9 severity, DSM-5 symptom probabilities) and XAI
explanations into two structured artefacts:

  1. PatientReport -- per-sample clinical summary
  2. ClinicalDashboard -- aggregated view across a batch / clinical session

Both are JSON-serialisable and can be printed as ASCII reports.

Pipeline position
-----------------
Data -> Model -> XAI -> [OUTPUT LAYER] -> FL -> Validation
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# DSM-5 symptom names (must match prediction.py PredictionConfig.num_symptoms)
# ---------------------------------------------------------------------------

DSM5_SYMPTOM_NAMES: List[str] = [
    "Depressed Mood", "Hopelessness", "Worthlessness", "Suicidal Ideation",
    "Anhedonia", "Fatigue", "Insomnia", "Hypersomnia", "Appetite Change",
    "Concentration Deficit", "Excessive Guilt", "Psychomotor Disturbance",
    "Irritability", "Anxiety", "Panic Attacks", "Tearfulness",
    "Loneliness", "Social Withdrawal", "Self-Harm Ideation", "Loss of Motivation",
]


# ---------------------------------------------------------------------------
# PatientReport
# ---------------------------------------------------------------------------

@dataclass
class PatientReport:
    """Per-sample clinical summary produced by the Output Layer."""

    patient_id: str
    timestamp: str

    # --- Prediction -------------------------------------------------------
    prediction_label: str       # "Depressed" | "Normal"
    prediction_class: int       # 1 | 0
    confidence: float           # probability of predicted class (0-1)
    risk_level: str             # "High" | "Moderate" | "Low"

    # --- Severity ---------------------------------------------------------
    severity_score: float       # PHQ-9 proxy (0-27 clipped)
    severity_category: str      # "Minimal" | "Mild" | "Moderate" | "Moderately Severe" | "Severe"

    # --- Symptoms (DSM-5) -------------------------------------------------
    active_symptoms: List[str]  # symptom names above threshold
    symptom_scores: Dict[str, float]   # {symptom_name: probability}

    # --- XAI --------------------------------------------------------------
    top_modality: str           # modality with highest IG attribution
    top_features: List[Tuple[str, float]]   # [(feature_name, importance)]
    named_feature_importances: Dict[str, float]  # sleep/alpha/sentiment
    counterfactual_narrative: str
    fusion_gate_weights: Dict[str, float]   # {modality: attention weight}

    # --- Clinical ---------------------------------------------------------
    recommended_action: str
    follow_up_priority: str     # "Urgent" | "Soon" | "Routine"
    clinical_notes: str
    data_source: str            # "wu3d" | "reddit" | "unknown"


# ---------------------------------------------------------------------------
# ClinicalDashboard
# ---------------------------------------------------------------------------

@dataclass
class ClinicalDashboard:
    """Aggregated view across a set of PatientReports (a clinical session)."""

    generated_at: str
    total_patients: int
    depressed_count: int
    normal_count: int
    high_risk_count: int
    moderate_risk_count: int
    low_risk_count: int

    avg_severity_score: float
    avg_confidence: float

    severity_distribution: Dict[str, int]   # category -> count
    symptom_prevalence: Dict[str, float]    # symptom -> fraction of patients
    modality_importances: Dict[str, float]  # aggregated across all patients
    named_feature_avg: Dict[str, float]     # sleep/alpha/sentiment averages
    source_breakdown: Dict[str, int]        # wu3d / reddit -> count

    reports: List[PatientReport] = field(default_factory=list)


# ---------------------------------------------------------------------------
# OutputLayer
# ---------------------------------------------------------------------------

class OutputLayer:
    """Converts model + XAI outputs into clinically meaningful reports.

    Parameters
    ----------
    symptom_threshold : float
        Minimum predicted probability to mark a DSM-5 symptom as active.
    severity_scale : float
        Maximum value of the model's severity output (used for scaling to
        the 0-27 PHQ-9 range).
    """

    # PHQ-9 category boundaries
    _PHQ9_CATEGORIES = [
        (4,  "Minimal"),
        (9,  "Mild"),
        (14, "Moderate"),
        (19, "Moderately Severe"),
        (27, "Severe"),
    ]

    # Modality order used in the stack (matches model internals)
    _MODALITY_NAMES = ["text", "eeg", "wearable", "av", "clinical", "mfcc"]

    def __init__(
        self,
        symptom_threshold: float = 0.5,
        severity_scale: float = 10.0,
    ):
        self.symptom_threshold = symptom_threshold
        self.severity_scale = severity_scale

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _phq9_category(self, score: float) -> str:
        for upper, label in self._PHQ9_CATEGORIES:
            if score <= upper:
                return label
        return "Severe"

    def _risk_level(self, prob_depressed: float, severity: float, prediction_label: str) -> Tuple[str, str]:
        """Combine probability and severity into a 3-tier risk level, gated by model prediction."""
        if prob_depressed >= 0.80 or severity >= 20:
            raw_risk = "High"
        elif prob_depressed >= 0.50 or severity >= 10:
            raw_risk = "Moderate"
        else:
            raw_risk = "Low"

        # Gate on model output to avoid contradiction
        if prediction_label == "Normal":
            # If the model predicts Normal, we gate the risk to at most "Moderate"
            if raw_risk == "High":
                gated_risk = "Moderate"
            else:
                gated_risk = raw_risk
        else:
            # If the model predicts Depressed, we gate the risk to at least "Moderate"
            if raw_risk == "Low":
                gated_risk = "Moderate"
            else:
                gated_risk = raw_risk

        return gated_risk, raw_risk

    def _recommended_action(
        self,
        risk: str,
        prediction_label: str,
        active_symptoms: List[str],
        raw_risk: Optional[str] = None,
        severity_score: Optional[float] = None,
    ) -> str:
        if "Suicidal Ideation" in active_symptoms or "Self-Harm Ideation" in active_symptoms:
            return (
                "IMMEDIATE psychiatric evaluation required. "
                "Suicidal/self-harm ideation detected. Contact crisis services."
            )
        
        # Check for gating discrepancy where clinical severity is High (Severe) but model predicts Normal
        if raw_risk == "High" and risk == "Moderate" and prediction_label == "Normal":
            category = self._phq9_category(severity_score) if severity_score is not None else "Severe"
            return f"Model predicts Normal; clinical severity indicators are {category}. Clinician review recommended before de-escalation."

        if risk == "High" and prediction_label == "Depressed":
            return "Urgent psychiatric consultation (within 24-48 hours). Consider immediate support services."
        if risk == "Moderate" and prediction_label == "Depressed":
            return "Schedule psychiatric consultation within 1 week. Monitor symptom progression."
        if risk == "Low" and prediction_label == "Depressed":
            return "Follow-up appointment within 2-4 weeks. Self-help resources and psychoeducation recommended."
        if risk == "High" and prediction_label == "Normal":
            return "Preventive counselling session within 2 weeks. Monitor risk factors."
        return "Routine wellness monitoring. Next scheduled check-up as planned."

    def _follow_up_priority(self, risk: str, active_symptoms: List[str], raw_risk: Optional[str] = None) -> str:
        critical = {"Suicidal Ideation", "Self-Harm Ideation"}
        if critical.intersection(active_symptoms):
            return "Urgent"
        
        # Take the maximum of gated and raw risk to determine priority
        risk_hierarchy = {"High": 3, "Moderate": 2, "Low": 1}
        gated_val = risk_hierarchy.get(risk, 0)
        raw_val = risk_hierarchy.get(raw_risk, 0) if raw_risk is not None else 0
        effective_risk = risk if gated_val >= raw_val else raw_risk

        if effective_risk == "High":
            return "Urgent"
        if effective_risk == "Moderate":
            return "Soon"
        return "Routine"

    def _clinical_notes(
        self,
        report: "PatientReport",
        raw_risk: str,
    ) -> str:
        sym_list = ", ".join(report.active_symptoms[:5]) if report.active_symptoms else "None"
        base_notes = (
            f"AI-assisted screening result [{report.timestamp}]. "
            f"Prediction: {report.prediction_label} (confidence {report.confidence:.1%}). "
            f"PHQ-9 proxy: {report.severity_score:.1f} ({report.severity_category}). "
            f"Active DSM-5 symptoms: {sym_list}. "
            f"Primary modality driving prediction: {report.top_modality}. "
            f"Data source: {report.data_source}. "
        )
        if raw_risk != report.risk_level:
            base_notes += f"Note: Model predicted '{report.prediction_label}' but rule-based severity suggests '{raw_risk}' risk; gated final risk level to '{report.risk_level}'. "
        
        base_notes += "NOTE: This is an AI decision-support tool; clinical judgement must override."
        return base_notes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_report(
        self,
        model_output: Dict[str, torch.Tensor],
        patient_id: str,
        sample_index: int = 0,
        xai_explanation: Optional[Any] = None,   # ExplanationOutput | None
        data_source: str = "unknown",
    ) -> PatientReport:
        """Build a PatientReport from model outputs for one sample.

        Parameters
        ----------
        model_output : dict
            Output of ``DepressionDetectionModel.forward()``.
            Keys: binary_probs [B,2], severity [B,1], symptom_probs [B,20],
                  fusion_weights [B,M], and optionally 'explanations'.
        patient_id : str
            Unique identifier for the patient / sample.
        sample_index : int
            Which sample in the batch to extract (default 0).
        xai_explanation : ExplanationOutput or None
            XAI module output.  If None, XAI fields will be empty.
        data_source : str
            'wu3d' | 'reddit' | 'unknown'
        """
        i = sample_index

        # --- Binary prediction ---
        binary_probs = model_output["binary_probs"][i].detach().cpu()
        pred_class   = int(binary_probs.argmax().item())
        confidence   = float(binary_probs[pred_class].item())
        pred_label   = "Depressed" if pred_class == 1 else "Normal"

        # --- Severity ---
        raw_sev   = float(model_output["severity"][i].detach().cpu().item())
        # Map from model output range to PHQ-9 [0, 27]
        sev_score = max(0.0, min(27.0, abs(raw_sev) * self.severity_scale))

        # --- Symptoms ---
        sym_probs = model_output["symptom_probs"][i].detach().cpu().tolist()
        # Pad/truncate to match DSM5_SYMPTOM_NAMES
        n = min(len(sym_probs), len(DSM5_SYMPTOM_NAMES))
        symptom_scores = {DSM5_SYMPTOM_NAMES[k]: round(sym_probs[k], 4) for k in range(n)}
        active_symptoms = [
            name for name, prob in symptom_scores.items()
            if prob >= self.symptom_threshold
        ]

        # --- Fusion gate weights ---
        fusion_w = model_output["fusion_weights"][i].detach().cpu().tolist()
        fusion_gate_weights: Dict[str, float] = {}
        for k, w in enumerate(fusion_w):
            name = self._MODALITY_NAMES[k] if k < len(self._MODALITY_NAMES) else f"mod_{k}"
            fusion_gate_weights[name] = round(float(w), 4)

        # --- XAI fields ---
        if xai_explanation is not None:
            top_mod = max(xai_explanation.modality_importances, key=xai_explanation.modality_importances.get)
            named_fi = xai_explanation.named_feature_importances
            cf_text  = xai_explanation.cf_explanation_text
            top_feats: List[Tuple[str, float]] = sorted(
                xai_explanation.modality_importances.items(), key=lambda x: -x[1]
            )[:5]
        else:
            top_mod  = max(fusion_gate_weights, key=fusion_gate_weights.get, default="unknown")
            named_fi = {"sleep_duration": 0.0, "alpha_band": 0.0, "negative_sentiment": 0.0}
            cf_text  = "XAI module not run for this sample."
            top_feats = list(fusion_gate_weights.items())[:5]

        # --- Risk + actions ---
        risk, raw_risk = self._risk_level(float(binary_probs[1].item()), sev_score, pred_label)

        # Partial report to pass into _clinical_notes
        partial = PatientReport(
            patient_id=patient_id,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            prediction_label=pred_label,
            prediction_class=pred_class,
            confidence=round(confidence, 4),
            risk_level=risk,
            severity_score=round(sev_score, 2),
            severity_category=self._phq9_category(sev_score),
            active_symptoms=active_symptoms,
            symptom_scores=symptom_scores,
            top_modality=top_mod,
            top_features=top_feats,
            named_feature_importances=named_fi,
            counterfactual_narrative=cf_text,
            fusion_gate_weights=fusion_gate_weights,
            recommended_action="",
            follow_up_priority="",
            clinical_notes="",
            data_source=data_source,
        )
        partial.recommended_action = self._recommended_action(risk, pred_label, active_symptoms, raw_risk, sev_score)
        partial.follow_up_priority = self._follow_up_priority(risk, active_symptoms, raw_risk)
        partial.clinical_notes     = self._clinical_notes(partial, raw_risk)
        return partial

    # ------------------------------------------------------------------

    def generate_dashboard(self, reports: List[PatientReport]) -> ClinicalDashboard:
        """Aggregate a list of PatientReports into a ClinicalDashboard."""
        if not reports:
            return ClinicalDashboard(
                generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                total_patients=0,
                depressed_count=0, normal_count=0,
                high_risk_count=0, moderate_risk_count=0, low_risk_count=0,
                avg_severity_score=0.0, avg_confidence=0.0,
                severity_distribution={}, symptom_prevalence={},
                modality_importances={}, named_feature_avg={},
                source_breakdown={},
                reports=[],
            )

        n = len(reports)
        dep = sum(1 for r in reports if r.prediction_class == 1)

        risk_counts = {"High": 0, "Moderate": 0, "Low": 0}
        sev_dist: Dict[str, int] = {}
        sym_counts: Dict[str, int] = {s: 0 for s in DSM5_SYMPTOM_NAMES}
        mod_imp: Dict[str, float] = {}
        named_avg: Dict[str, float] = {"sleep_duration": 0.0, "alpha_band": 0.0, "negative_sentiment": 0.0}
        source_cnt: Dict[str, int] = {}

        for r in reports:
            risk_counts[r.risk_level] = risk_counts.get(r.risk_level, 0) + 1
            sev_dist[r.severity_category] = sev_dist.get(r.severity_category, 0) + 1
            for sym in r.active_symptoms:
                if sym in sym_counts:
                    sym_counts[sym] += 1
            for mod, imp in r.top_features:
                mod_imp[mod] = mod_imp.get(mod, 0.0) + imp
            for feat, val in r.named_feature_importances.items():
                named_avg[feat] = named_avg.get(feat, 0.0) + val
            source_cnt[r.data_source] = source_cnt.get(r.data_source, 0) + 1

        avg_sev  = sum(r.severity_score for r in reports) / n
        avg_conf = sum(r.confidence for r in reports) / n

        symptom_prevalence = {s: round(c / n, 4) for s, c in sym_counts.items() if c > 0}
        modality_importances = {k: round(v / n, 6) for k, v in mod_imp.items()}
        named_feature_avg = {k: round(v / n, 6) for k, v in named_avg.items()}

        return ClinicalDashboard(
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            total_patients=n,
            depressed_count=dep,
            normal_count=n - dep,
            high_risk_count=risk_counts.get("High", 0),
            moderate_risk_count=risk_counts.get("Moderate", 0),
            low_risk_count=risk_counts.get("Low", 0),
            avg_severity_score=round(avg_sev, 3),
            avg_confidence=round(avg_conf, 4),
            severity_distribution=sev_dist,
            symptom_prevalence=symptom_prevalence,
            modality_importances=modality_importances,
            named_feature_avg=named_feature_avg,
            source_breakdown=source_cnt,
            reports=reports,
        )

    # ------------------------------------------------------------------
    # Formatters
    # ------------------------------------------------------------------

    def format_report(self, r: PatientReport, width: int = 68) -> str:
        """Return a formatted ASCII clinical report string."""
        lines = [
            "+" + "=" * width + "+",
            "| PATIENT REPORT".ljust(width) + " |",
            "+" + "-" * width + "+",
            "| Patient ID  : {}".format(r.patient_id).ljust(width) + " |",
            "| Timestamp   : {}".format(r.timestamp).ljust(width) + " |",
            "| Data Source : {}".format(r.data_source).ljust(width) + " |",
            "+" + "-" * width + "+",
            "| PREDICTION  : {:>10}  (confidence {:.1%})".format(
                r.prediction_label, r.confidence).ljust(width) + " |",
            "| RISK LEVEL  : {:>10}".format(r.risk_level).ljust(width) + " |",
            "| SEVERITY    : {:5.1f} / 27  [{}]".format(
                r.severity_score, r.severity_category).ljust(width) + " |",
            "+" + "-" * width + "+",
            "| DSM-5 ACTIVE SYMPTOMS ({})".format(len(r.active_symptoms)).ljust(width) + " |",
        ]
        for sym in r.active_symptoms or ["(none above threshold)"]:
            score = r.symptom_scores.get(sym, 0.0)
            lines.append("|   - {:<38} {:5.1%}".format(sym, score).ljust(width) + " |")

        lines += [
            "+" + "-" * width + "+",
            "| KEY MODALITY IMPORTANCES (SHAP/IG)".ljust(width) + " |",
        ]
        for feat, imp in r.top_features:
            lines.append("|   {:>12}: {:6.4f}".format(feat, imp).ljust(width) + " |")

        lines += [
            "+" + "-" * width + "+",
            "| NAMED FEATURE IMPORTANCES".ljust(width) + " |",
        ]
        for feat, val in r.named_feature_importances.items():
            lines.append("|   {:>22}: {:6.4f}".format(feat, val).ljust(width) + " |")

        lines += [
            "+" + "-" * width + "+",
            "| COUNTERFACTUAL".ljust(width) + " |",
        ]
        words, line_ = r.counterfactual_narrative.split(), "|   "
        for w in words:
            if len(line_) + len(w) + 1 > width:
                lines.append(line_.rstrip().ljust(width) + " |")
                line_ = "|   " + w + " "
            else:
                line_ += w + " "
        if line_.strip("|").strip():
            lines.append(line_.rstrip().ljust(width) + " |")

        action_prefix = "| Action   : "
        words = r.recommended_action.split()
        current_line = action_prefix
        action_lines = []
        for w in words:
            if len(current_line) + len(w) + 1 > width:
                action_lines.append(current_line.rstrip().ljust(width) + " |")
                current_line = "|            " + w + " "
            else:
                current_line += w + " "
        if current_line.strip("|").strip():
            action_lines.append(current_line.rstrip().ljust(width) + " |")

        lines += [
            "+" + "-" * width + "+",
            "| CLINICAL GUIDANCE".ljust(width) + " |",
        ]
        lines.extend(action_lines)
        lines += [
            "| Priority : {}".format(r.follow_up_priority).ljust(width) + " |",
            "+" + "=" * width + "+",
        ]
        return "\n".join(lines)

    def format_dashboard(self, d: ClinicalDashboard, width: int = 68) -> str:
        """Return a formatted ASCII clinical dashboard string."""

        def _bar(v, mx, w=30):
            f = int(round(v / max(mx, 1e-9) * w))
            return "#" * f + "-" * (w - f)

        lines = [
            "+" + "=" * width + "+",
            "| CLINICAL DASHBOARD  {}".format(d.generated_at).ljust(width) + " |",
            "+" + "-" * width + "+",
            "| Total Patients   : {:>6}".format(d.total_patients).ljust(width) + " |",
            "| Depressed        : {:>6}  ({:.1%})".format(
                d.depressed_count,
                d.depressed_count / max(d.total_patients, 1)).ljust(width) + " |",
            "| Normal           : {:>6}  ({:.1%})".format(
                d.normal_count,
                d.normal_count / max(d.total_patients, 1)).ljust(width) + " |",
            "| Avg Severity     : {:6.2f} / 27".format(d.avg_severity_score).ljust(width) + " |",
            "| Avg Confidence   : {:6.1%}".format(d.avg_confidence).ljust(width) + " |",
            "+" + "-" * width + "+",
            "| RISK DISTRIBUTION".ljust(width) + " |",
            "|   High     : {:>4}  {}".format(
                d.high_risk_count,
                _bar(d.high_risk_count, d.total_patients)).ljust(width) + " |",
            "|   Moderate : {:>4}  {}".format(
                d.moderate_risk_count,
                _bar(d.moderate_risk_count, d.total_patients)).ljust(width) + " |",
            "|   Low      : {:>4}  {}".format(
                d.low_risk_count,
                _bar(d.low_risk_count, d.total_patients)).ljust(width) + " |",
            "+" + "-" * width + "+",
            "| TOP SYMPTOMS (prevalence)".ljust(width) + " |",
        ]
        top_sym = sorted(d.symptom_prevalence.items(), key=lambda x: -x[1])[:8]
        max_prev = top_sym[0][1] if top_sym else 1.0
        for sym, prev in top_sym:
            lines.append("|   {:<30} {:5.1%}  {}".format(
                sym[:30], prev, _bar(prev, max_prev, 15)).ljust(width) + " |")

        lines += [
            "+" + "-" * width + "+",
            "| NAMED FEATURE AVERAGES".ljust(width) + " |",
        ]
        for feat, avg in d.named_feature_avg.items():
            lines.append("|   {:>22}: {:6.4f}".format(feat, avg).ljust(width) + " |")

        lines += [
            "+" + "-" * width + "+",
            "| DATA SOURCES".ljust(width) + " |",
        ]
        for src, cnt in d.source_breakdown.items():
            lines.append("|   {:>10}: {:>5}".format(src, cnt).ljust(width) + " |")

        lines.append("+" + "=" * width + "+")
        return "\n".join(lines)

    # ------------------------------------------------------------------

    def save_reports_json(
        self,
        reports: List[PatientReport],
        path: str,
        dashboard: Optional[ClinicalDashboard] = None,
    ) -> None:
        """Serialise reports (and optionally dashboard) to JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload: Dict[str, Any] = {
            "reports": [
                {
                    **{k: v for k, v in asdict(r).items()
                       if not isinstance(v, list) or k != "top_features"},
                    "top_features": [list(t) for t in r.top_features],
                }
                for r in reports
            ]
        }
        if dashboard is not None:
            d = asdict(dashboard)
            d.pop("reports", None)  # avoid duplication
            payload["dashboard"] = d
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print("Saved {} report(s) to: {}".format(len(reports), path))
