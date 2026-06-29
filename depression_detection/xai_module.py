"""Explainable AI (XAI) module for multimodal depression detection.

Three complementary explanation methods, all gradient-based and fully
differentiable ? no external SHAP library required.

Methods
-------
1. Integrated Gradients (IG)
   SHAP approximation over modality embeddings (Sundararajan et al., 2017).
   Yields per-modality attribution vectors and three named-feature scores:
     - sleep_duration   (wearable channel 0, proxy for rest/activity patterns)
     - alpha_band       (EEG channels 8-12, proxy for 8-13 Hz alpha rhythm)
     - negative_sentiment (first N dims of the GoogleNews text embedding)

2. Attention Heatmaps
   Gradient ? Input attribution:
     - EEG channels  ? [B, C] importance map
     - Text embedding dims ? [B, D] importance map
   Plus the fusion gating weights [B, num_modalities] from the cross-attention.

3. Counterfactual Explanations
   Finds the minimal perturbation ? in the modality-embedding space such that
   the predicted class flips.  Uses manual Adam with autograd.grad() so model
   parameters are never polluted.

Gradient Safety
---------------
All three methods use ``torch.autograd.grad()`` targeting only leaf tensors
(interpolated embeddings or delta buffers), leaving model parameter gradients
completely untouched.  Safe to call during or after training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


# ?????????????????????????????????????????????????????????????????????????????
# Configuration
# ?????????????????????????????????????????????????????????????????????????????

@dataclass
class XAIConfig:
    """Hyper-parameters and domain mappings for the XAI module."""

    # Integrated Gradients
    ig_steps: int = 50
    """Number of interpolation steps from baseline (zeros) to actual embedding."""

    # Counterfactual
    cf_steps: int = 150
    """Maximum optimisation steps for counterfactual search."""

    cf_lr: float = 0.05
    """Learning rate for manual Adam in counterfactual search."""

    cf_lambda_proximity: float = 0.5
    """Weight for L2 proximity penalty: keeps ? close to zero."""

    cf_lambda_sparsity: float = 0.1
    """Weight for L1 sparsity penalty: encourages minimal feature changes."""

    # Named feature domain mappings
    sleep_duration_wearable_ch: int = 0
    """Wearable tensor channel index treated as the sleep/rest duration proxy."""

    alpha_band_eeg_channels: List[int] = field(
        default_factory=lambda: list(range(8, 13))
    )
    """EEG channel indices corresponding to the alpha-band (8?13 Hz proxy)."""

    negative_sentiment_text_dims: int = 50
    """First N dimensions of the GoogleNews text embedding used as a
    negative-sentiment proxy (high-frequency negative words cluster here)."""


# ?????????????????????????????????????????????????????????????????????????????
# Output container
# ?????????????????????????????????????????????????????????????????????????????

@dataclass
class ExplanationOutput:
    """All explanation artifacts for one forward pass."""

    # SHAP / Integrated Gradients
    modality_importances: Dict[str, float]
    """Per-modality global importance score (L2 norm of IG attribution, batch-avg)."""

    named_feature_importances: Dict[str, float]
    """Importance scores for: sleep_duration, alpha_band, negative_sentiment."""

    ig_attributions: Dict[str, torch.Tensor]
    """Raw IG attribution tensors {modality: [B, hidden_dim]}."""

    # Attention Heatmaps
    eeg_channel_importances: torch.Tensor
    """Gradient ? Input importance per EEG channel: [B, eeg_channels]."""

    text_span_importances: torch.Tensor
    """Gradient ? Input importance per text-embedding dimension: [B, text_dim]."""

    fusion_attention_weights: torch.Tensor
    """Dynamic gating weights from MultimodalFusion: [B, num_modalities]."""

    # Counterfactual
    cf_found: bool
    """Whether the optimiser successfully flipped the prediction."""

    cf_delta_norm: float
    """Total perturbation magnitude ?_m ??_m??."""

    cf_predicted_label: int
    """Model's predicted label for the counterfactual input."""

    cf_modality_perturbations: Dict[str, float]
    """Per-modality perturbation norm ??_m?? (sorted descending = most changed)."""

    cf_explanation_text: str
    """Human-readable counterfactual narrative."""


# ?????????????????????????????????????????????????????????????????????????????
# Integrated Gradients
# ?????????????????????????????????????????????????????????????????????????????

class IntegratedGradientsExplainer:
    """SHAP approximation via Integrated Gradients over modality embeddings.

    The baseline is the zero vector (absence of information).
    Attribution for modality m:

        IG_m = (e_m - 0) ? (1/(N+1)) ?_{k=0}^{N} ?f/?e_m^(k)

    where e_m^(k) = (k/N) ? e_m  is the k-th interpolated embedding.

    ``torch.autograd.grad()`` is used so model parameters receive no gradients.
    """

    def __init__(self, steps: int = 50):
        self.steps = steps

    def attribute(
        self,
        embeddings: Dict[str, torch.Tensor],    # {name: [B, D]}
        logit_fn: Callable[[torch.Tensor], torch.Tensor],  # [B, M, D] ? [B, 2]
        target_class: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Return {modality: attribution [B, D]}."""
        names = list(embeddings.keys())
        accum: Dict[str, torch.Tensor] = {k: torch.zeros_like(embeddings[k]) for k in names}

        for step in range(self.steps + 1):
            alpha = step / self.steps

            # Interpolated embeddings ? treated as leaf variables for grad
            interp = [
                (alpha * embeddings[k].detach()).requires_grad_(True)
                for k in names
            ]

            stacked = torch.stack(interp, dim=1)   # [B, M, D]
            with torch.backends.cudnn.flags(enabled=False):
                logits  = logit_fn(stacked)            # [B, 2]
            score   = logits[:, target_class].sum()

            grads = torch.autograd.grad(
                outputs=score,
                inputs=interp,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )

            for k, g in zip(names, grads):
                if g is not None:
                    accum[k] = accum[k] + g.detach()

        # IG = input ? average gradient
        return {
            k: embeddings[k].detach() * (accum[k] / (self.steps + 1))
            for k in names
        }


# ?????????????????????????????????????????????????????????????????????????????
# EEG channel attribution
# ?????????????????????????????????????????????????????????????????????????????

class EEGChannelAttribution:
    """Per-channel importance via Gradient ? Input, averaged across time.

    Returns [B, C] where C = number of EEG channels.
    Highlights which EEG channels (e.g. frontal alpha, temporal beta)
    drive the depression prediction most.
    """

    def attribute(
        self,
        eeg: torch.Tensor,                          # [B, T, C]
        score_fn: Callable[[torch.Tensor], torch.Tensor],  # eeg ? [B] or scalar
    ) -> torch.Tensor:                              # [B, C]
        eeg_leaf = eeg.detach().requires_grad_(True)
        with torch.backends.cudnn.flags(enabled=False):
            score = score_fn(eeg_leaf)
        if score.dim() > 0:
            score = score.sum()
        grads = torch.autograd.grad(
            outputs=score,
            inputs=eeg_leaf,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )[0]
        if grads is None:
            return torch.zeros(eeg.shape[0], eeg.shape[-1], device=eeg.device)
        # |grad ? input|, mean over time ? [B, C]
        return (grads * eeg_leaf).abs().mean(dim=1).detach()


# ?????????????????????????????????????????????????????????????????????????????
# Text span attribution
# ?????????????????????????????????????????????????????????????????????????????

class TextSpanAttribution:
    """Per-dimension importance for the text embedding via Gradient ? Input.

    For a 300-dim GoogleNews embedding each dimension captures a semantic
    cluster.  High-attribution dims indicate which semantic clusters (e.g.
    negative valence, sleep-related terms) drive the prediction.

    Returns [B, D] ? can be visualised as a bar chart over embedding dims or
    aggregated into named semantic groups (see named_feature_importances).
    """

    def attribute(
        self,
        text_emb: torch.Tensor,                             # [B, D]
        score_fn: Callable[[torch.Tensor], torch.Tensor],   # text_emb ? [B] or scalar
    ) -> torch.Tensor:                                      # [B, D]
        text_leaf = text_emb.detach().requires_grad_(True)
        with torch.backends.cudnn.flags(enabled=False):
            score = score_fn(text_leaf)
        if score.dim() > 0:
            score = score.sum()
        grads = torch.autograd.grad(
            outputs=score,
            inputs=text_leaf,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )[0]
        if grads is None:
            return torch.zeros_like(text_emb)
        return (grads * text_leaf).abs().detach()


# ?????????????????????????????????????????????????????????????????????????????
# Counterfactual explainer
# ?????????????????????????????????????????????????????????????????????????????

class CounterfactualExplainer:
    """Gradient-based counterfactual generation in modality-embedding space.

    Optimises a set of per-modality perturbations ?_m such that:
        argmax f([e_m + ?_m]) ? original_label   (prediction flips)

    Objective:
        min  CE(f([e_m + ?_m]), target_class)
           + ?_prox   ?_m ??_m??       (L2 proximity)
           + ?_sparse ?_m ??_m??       (L1 sparsity)

    Uses manual Adam with ``torch.autograd.grad()`` so model parameters are
    never accumulated into ? completely safe to call without zeroing grads.

    Only the **first sample** of the batch is explained (index 0) for
    computational tractability.
    """

    def __init__(
        self,
        steps: int = 150,
        lr: float = 0.05,
        lambda_proximity: float = 0.5,
        lambda_sparsity: float = 0.1,
    ):
        self.steps = steps
        self.lr = lr
        self.lambda_proximity = lambda_proximity
        self.lambda_sparsity = lambda_sparsity

    def generate(
        self,
        embeddings: Dict[str, torch.Tensor],              # {name: [B, D]}
        logit_fn: Callable[[torch.Tensor], torch.Tensor], # stacked [1, M, D] ? [1, 2]
        original_label: int,
        target_class: int,
    ) -> Tuple[bool, float, int, Dict[str, float], Dict[str, torch.Tensor]]:
        """
        Returns
        -------
        cf_found, total_delta_norm, cf_pred_label,
        per_modality_delta_norms, cf_embeddings
        """
        names = list(embeddings.keys())
        device = next(iter(embeddings.values())).device

        # Work on a single sample
        single = {k: v[:1].detach().clone() for k, v in embeddings.items()}
        target = torch.tensor([target_class], dtype=torch.long, device=device)

        # Perturbation buffers (not nn.Parameters ? we update .data manually)
        deltas: Dict[str, torch.Tensor] = {k: torch.zeros_like(v) for k, v in single.items()}

        # Manual Adam state
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        m_buf = {k: torch.zeros_like(v) for k, v in single.items()}
        v_buf = {k: torch.zeros_like(v) for k, v in single.items()}

        cf_found = False
        for step in range(self.steps):
            t = step + 1

            # Build leaf tensors for autograd
            delta_leaves = {k: deltas[k].requires_grad_(True) for k in names}
            perturbed = {k: single[k] + delta_leaves[k] for k in names}
            stacked = torch.stack([perturbed[k] for k in names], dim=1)   # [1, M, D]

            with torch.backends.cudnn.flags(enabled=False):
                logits = logit_fn(stacked)                                     # [1, 2]
            ce     = F.cross_entropy(logits, target)
            prox   = sum(d.pow(2).mean() for d in delta_leaves.values())
            sparse = sum(d.abs().mean()  for d in delta_leaves.values())
            loss   = ce + self.lambda_proximity * prox + self.lambda_sparsity * sparse

            grads = torch.autograd.grad(
                outputs=loss,
                inputs=[delta_leaves[k] for k in names],
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )

            # Manual Adam update for each modality delta
            with torch.no_grad():
                for k, g in zip(names, grads):
                    if g is None:
                        continue
                    m_buf[k] = beta1 * m_buf[k] + (1 - beta1) * g
                    v_buf[k] = beta2 * v_buf[k] + (1 - beta2) * g ** 2
                    m_hat = m_buf[k] / (1 - beta1 ** t)
                    v_hat = v_buf[k] / (1 - beta2 ** t)
                    deltas[k] = deltas[k] - self.lr * m_hat / (v_hat.sqrt() + eps_adam)

            # Check if prediction flipped
            with torch.no_grad():
                stacked_check = torch.stack(
                    [(single[k] + deltas[k]) for k in names], dim=1
                )
                check_pred = logit_fn(stacked_check).argmax(dim=-1).item()
                if check_pred == target_class:
                    cf_found = True
                    break

        with torch.no_grad():
            delta_norms = {k: deltas[k].norm().item() for k in names}
            total_norm  = sum(delta_norms.values())
            cf_embs = {k: (single[k] + deltas[k]).detach() for k in names}
            stacked_cf = torch.stack([cf_embs[k] for k in names], dim=1)
            cf_pred = logit_fn(stacked_cf).argmax(dim=-1).item()

        return cf_found, total_norm, cf_pred, delta_norms, cf_embs


# ?????????????????????????????????????????????????????????????????????????????
# XAI Module
# ?????????????????????????????????????????????????????????????????????????????

class XAIModule(nn.Module):
    """Orchestrates all three XAI explanation methods.

    This is registered as ``model.xai`` inside ``DepressionDetectionModel``.
    Call ``model.xai.explain(...)`` to generate explanations ? it is **not**
    invoked automatically during ``model.forward()`` unless ``explain=True``.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of modality embeddings (must match the main model).
    cfg : XAIConfig, optional
        Hyper-parameters and domain channel mappings.
    """

    def __init__(self, hidden_dim: int, cfg: Optional[XAIConfig] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cfg = cfg or XAIConfig()
        self.ig     = IntegratedGradientsExplainer(steps=self.cfg.ig_steps)
        self.eeg_attr  = EEGChannelAttribution()
        self.text_attr = TextSpanAttribution()
        self.cf     = CounterfactualExplainer(
            steps=self.cfg.cf_steps,
            lr=self.cfg.cf_lr,
            lambda_proximity=self.cfg.cf_lambda_proximity,
            lambda_sparsity=self.cfg.cf_lambda_sparsity,
        )

    # Named feature importances

    def _named_feature_importances(
        self,
        ig_attrs: Dict[str, torch.Tensor],
        eeg_channel_attr: torch.Tensor,     # [B, C]
        raw_text: torch.Tensor,             # [B, D]
        raw_wearable: torch.Tensor,         # [B, T, W]
    ) -> Dict[str, float]:
        cfg = self.cfg

        # Sleep duration wearable channel 0 (rest/sleep activity proxy)
        if (raw_wearable is not None
                and raw_wearable.shape[-1] > cfg.sleep_duration_wearable_ch):
            ch = cfg.sleep_duration_wearable_ch
            raw_signal      = raw_wearable[:, :, ch].abs().mean().item()
            emb_importance  = ig_attrs.get("wearable", torch.zeros(1)).abs().mean().item()
            sleep = raw_signal * (1.0 + emb_importance)
        else:
            sleep = ig_attrs.get("wearable", torch.zeros(1)).abs().mean().item()

        # Alpha band EEG channels 8-12 (8?13 Hz alpha-rhythm proxy)
        alpha_chs = [c for c in cfg.alpha_band_eeg_channels if c < eeg_channel_attr.shape[-1]]
        if alpha_chs:
            alpha = eeg_channel_attr[:, alpha_chs].mean().item()
        else:
            alpha = eeg_channel_attr.mean().item()

        # Negative sentiment first N dims of GoogleNews embedding
        if raw_text is not None and raw_text.shape[-1] >= cfg.negative_sentiment_text_dims:
            n = cfg.negative_sentiment_text_dims
            raw_neg        = raw_text[:, :n].abs().mean().item()
            emb_importance = ig_attrs.get("text", torch.zeros(1)).abs().mean().item()
            neg_sent = raw_neg * (1.0 + emb_importance)
        else:
            neg_sent = ig_attrs.get("text", torch.zeros(1)).abs().mean().item()

        return {
            "sleep_duration":      float(sleep),
            "alpha_band":          float(alpha),
            "negative_sentiment":  float(neg_sent),
        }

    # Counterfactual narrative

    @staticmethod
    def _cf_narrative(
        original_label: int,
        cf_found: bool,
        cf_pred: int,
        delta_norms: Dict[str, float],
        label_map: Dict[int, str],
    ) -> str:
        if not cf_found:
            return (
                "No counterfactual found within the optimisation budget. "
                "The prediction is highly robust -- no small perturbation "
                "in the modality-embedding space was sufficient to flip it."
            )
        orig_name = label_map.get(original_label, str(original_label))
        cf_name   = label_map.get(cf_pred,        str(cf_pred))
        top_mod   = max(delta_norms, key=delta_norms.get)
        top_norm  = delta_norms[top_mod]
        return (
            f"The model predicted '{orig_name}'. "
            f"A minimal perturbation (||delta|| total = {sum(delta_norms.values()):.3f}) "
            f"in the '{top_mod}' modality embedding (d={top_norm:.3f}) "
            f"is sufficient to shift the prediction to '{cf_name}'. "
            f"This suggests that the '{top_mod}' features are the most decisive "
            f"for this individual's classification."
        )

    # Main explain method

    @torch.enable_grad()
    def explain(
        self,
        *,
        raw_inputs: Dict[str, torch.Tensor],
        modality_embeddings: Dict[str, torch.Tensor],
        fusion_weights: torch.Tensor,
        binary_logits: torch.Tensor,
        embedding_to_logits_fn: Callable[[torch.Tensor], torch.Tensor],
        full_encode_fn: Callable[..., Dict[str, torch.Tensor]],
        label_map: Optional[Dict[int, str]] = None,
    ) -> ExplanationOutput:
        """Compute all explanations for the current batch.

        Parameters
        ----------
        raw_inputs : dict
            Keys: ``text_emb``, ``eeg``, ``wearable``, ``audio``, ``video``,
            ``clinical``, optionally ``mfcc``.
        modality_embeddings : dict
            Keys: ``text``, ``eeg``, ``wearable``, ``av``, ``clinical``,
            optionally ``mfcc``.  Each value is ``[B, hidden_dim]``.
        fusion_weights : Tensor [B, num_modalities]
            Dynamic gating weights from ``MultimodalFusion``.
        binary_logits : Tensor [B, 2]
            Output of the prediction head.
        embedding_to_logits_fn : callable
            ``stacked_embs [B, M, D] ? binary_logits [B, 2]``.
            Wraps fusion + prediction (no fairness).
        full_encode_fn : callable
            ``(text_emb, eeg, wearable, audio, video, clinical, mfcc=None)
            ? dict`` with key ``binary_logits``.
            Runs full model sans fairness ? used by EEG/text attributors.
        label_map : dict, optional
            Maps int label ? human string, e.g. ``{0: 'Normal', 1: 'Depressed'}``.
        """
        if label_map is None:
            label_map = {0: "Normal", 1: "Depressed"}

        target_cls = binary_logits.argmax(dim=-1)[0].item()

        # ???????????????????????????????????????????????????????????????????
        # 1. Integrated Gradients (SHAP-style feature importance)
        # ???????????????????????????????????????????????????????????????????
        ig_attrs = self.ig.attribute(
            embeddings=modality_embeddings,
            logit_fn=embedding_to_logits_fn,
            target_class=target_cls,
        )
        modality_importances = {
            k: float(v.norm(dim=-1).mean().item())
            for k, v in ig_attrs.items()
        }

        # ???????????????????????????????????????????????????????????????????
        # 2a. EEG Channel Attention Heatmap
        # ???????????????????????????????????????????????????????????????????
        def _eeg_score(eeg_t: torch.Tensor) -> torch.Tensor:
            out = full_encode_fn(
                text_emb=raw_inputs["text_emb"].detach(),
                eeg=eeg_t,
                wearable=raw_inputs["wearable"].detach(),
                audio=raw_inputs["audio"].detach(),
                video=raw_inputs["video"].detach(),
                clinical=raw_inputs["clinical"].detach(),
                mfcc=raw_inputs.get("mfcc"),
            )
            return out["binary_logits"][:, target_cls]

        eeg_channel_attr = self.eeg_attr.attribute(
            eeg=raw_inputs["eeg"],
            score_fn=_eeg_score,
        )

        # ???????????????????????????????????????????????????????????????????
        # 2b. Text Span Attention Heatmap
        # ???????????????????????????????????????????????????????????????????
        def _text_score(text_t: torch.Tensor) -> torch.Tensor:
            out = full_encode_fn(
                text_emb=text_t,
                eeg=raw_inputs["eeg"].detach(),
                wearable=raw_inputs["wearable"].detach(),
                audio=raw_inputs["audio"].detach(),
                video=raw_inputs["video"].detach(),
                clinical=raw_inputs["clinical"].detach(),
                mfcc=raw_inputs.get("mfcc"),
            )
            return out["binary_logits"][:, target_cls]

        text_span_attr = self.text_attr.attribute(
            text_emb=raw_inputs["text_emb"],
            score_fn=_text_score,
        )

        # ???????????????????????????????????????????????????????????????????
        # 2c. Named Feature Importances
        # ???????????????????????????????????????????????????????????????????
        named_importances = self._named_feature_importances(
            ig_attrs=ig_attrs,
            eeg_channel_attr=eeg_channel_attr,
            raw_text=raw_inputs.get("text_emb"),
            raw_wearable=raw_inputs.get("wearable"),
        )

        # ???????????????????????????????????????????????????????????????????
        # 3. Counterfactual Explanation
        # ???????????????????????????????????????????????????????????????????
        cf_target = 1 - target_cls
        cf_found, total_norm, cf_pred, delta_norms, _ = self.cf.generate(
            embeddings=modality_embeddings,
            logit_fn=embedding_to_logits_fn,
            original_label=target_cls,
            target_class=cf_target,
        )
        cf_narrative = self._cf_narrative(
            original_label=target_cls,
            cf_found=cf_found,
            cf_pred=cf_pred,
            delta_norms=delta_norms,
            label_map=label_map,
        )

        return ExplanationOutput(
            modality_importances=modality_importances,
            named_feature_importances=named_importances,
            ig_attributions={k: v.detach() for k, v in ig_attrs.items()},
            eeg_channel_importances=eeg_channel_attr,
            text_span_importances=text_span_attr,
            fusion_attention_weights=fusion_weights.detach(),
            cf_found=cf_found,
            cf_delta_norm=total_norm,
            cf_predicted_label=cf_pred,
            cf_modality_perturbations=delta_norms,
            cf_explanation_text=cf_narrative,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Identity pass-through.  Explanations are obtained via ``self.explain()``."""
        return x


# ?????????????????????????????????????????????????????????????????????????????
# Pretty-printer
# ?????????????????????????????????????????????????????????????????????????????

def format_explanations(
    exp: ExplanationOutput,
    label_map=None,
):
    """Return a human-readable ASCII report of all XAI outputs."""
    if label_map is None:
        label_map = {0: "Normal", 1: "Depressed"}

    W = 64
    lines = []

    def _bar(score, max_score, width=28):
        filled = int(round(score / max(max_score, 1e-9) * width))
        return "#" * filled + "-" * (width - filled)

    lines += [
        "+" + "=" * W + "+",
        "|" + "  XAI Explanation Report".center(W) + "|",
        "+" + "=" * W + "+",
        "",
    ]

    # 1. SHAP / IG Feature Importance
    lines += ["[SHAP Feature Importance (Integrated Gradients)]", "-" * W]
    sorted_mi = sorted(exp.modality_importances.items(), key=lambda x: -x[1])
    max_mi = max(v for v in exp.modality_importances.values()) or 1.0
    for rank, (mod, score) in enumerate(sorted_mi, 1):
        bar = _bar(score, max_mi)
        lines.append("  [{}] {:<12} {:6.4f}  {}".format(rank, mod, score, bar))
    lines.append("")

    # 1b. Named features
    lines += ["[Named Feature Importances]", "-" * W]
    max_nf = max(exp.named_feature_importances.values()) or 1.0
    nf_labels = {
        "sleep_duration":     "Sleep Duration     (wearable ch-0)",
        "alpha_band":         "Alpha Band         (EEG ch 8-12)",
        "negative_sentiment": "Negative Sentiment (text emb dim 0-49)",
    }
    for feat, score in exp.named_feature_importances.items():
        bar = _bar(score, max_nf)
        lbl = nf_labels.get(feat, feat)
        lines.append("  {:<38}  {:6.4f}  {}".format(lbl, score, bar))
    lines.append("")

    # 2. Attention Heatmaps
    lines += ["[Attention Heatmaps]", "-" * W]
    lines.append("  Fusion gate weights  (modality attention):")
    faw = exp.fusion_attention_weights[0].tolist()
    modality_order = ["text", "eeg", "wearable", "av", "clinical", "mfcc"]
    max_faw = max(faw) or 1.0
    for i, w in enumerate(faw):
        name = modality_order[i] if i < len(modality_order) else "mod_{}".format(i)
        bar  = _bar(w, max_faw, width=20)
        lines.append("    {:<12}  {:.3f}  {}".format(name, w, bar))
    lines.append("")

    lines.append("  Top-10 EEG channels  (gradient x input attribution):")
    ch_scores = exp.eeg_channel_importances[0].tolist()
    top_chs   = sorted(enumerate(ch_scores), key=lambda x: -x[1])[:10]
    max_ch    = top_chs[0][1] if top_chs else 1.0
    for ch, score in top_chs:
        if ch < 4:
            band = "delta(0-4Hz)"
        elif ch < 8:
            band = "theta(4-8Hz)"
        elif ch < 13:
            band = "alpha(8-13Hz)"
        elif ch < 30:
            band = "beta(13-30Hz)"
        else:
            band = "gamma(30+Hz)"
        bar = _bar(score, max_ch, width=16)
        lines.append("    CH-{:02d} [{:<12}]  {:.4f}  {}".format(ch, band, score, bar))
    lines.append("")

    lines.append("  Top-10 text embedding dims  (gradient x input):")
    txt_scores = exp.text_span_importances[0].tolist()
    top_dims   = sorted(enumerate(txt_scores), key=lambda x: -x[1])[:10]
    max_td     = top_dims[0][1] if top_dims else 1.0
    for dim, score in top_dims:
        if dim < 50:
            region = "neg-sentiment"
        elif dim < 100:
            region = "pos-sentiment"
        else:
            region = "neutral"
        bar = _bar(score, max_td, width=16)
        lines.append("    dim-{:03d} [{:<13}]  {:.4f}  {}".format(dim, region, score, bar))
    lines.append("")

    # 3. Counterfactual
    lines += ["[Counterfactual Explanation]", "-" * W]
    cf_lbl = label_map.get(exp.cf_predicted_label, str(exp.cf_predicted_label))
    lines.append("  Found          : {}".format("YES" if exp.cf_found else "NO"))
    lines.append("  Flipped to     : {}".format(cf_lbl))
    lines.append("  Total ||delta||: {:.4f}".format(exp.cf_delta_norm))
    lines.append("  Per-modality delta (most changed first):")
    sorted_cf = sorted(exp.cf_modality_perturbations.items(), key=lambda x: -x[1])
    max_cf    = sorted_cf[0][1] if sorted_cf else 1.0
    for mod, norm in sorted_cf:
        bar = _bar(norm, max_cf, width=20)
        lines.append("    {:<12}  d={:.4f}  {}".format(mod, norm, bar))
    lines.append("")
    lines.append("  Narrative:")
    words  = exp.cf_explanation_text.split()
    line_  = "    "
    for w in words:
        if len(line_) + len(w) + 1 > W - 2:
            lines.append(line_.rstrip())
            line_ = "    " + w + " "
        else:
            line_ += w + " "
    if line_.strip():
        lines.append(line_.rstrip())

    lines += [
        "",
        "+" + "-" * W + "+",
    ]
    return "\n".join(lines)

