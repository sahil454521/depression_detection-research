"""Multimodal model for five modalities with simple encoders and adaptive fusion.

Modalities:
- text (token ids + attention mask)
- eeg (multichannel time series)
- wearable (tabular time series)
- audio/video (acoustic + visual features)
- clinical (tabular)

Dataset:
- WU3D: depressed.json + normal.json  (tweet_content field)
- Reddit: reddit_depression_dataset.csv  (title + body columns)
- Text embeddings: GoogleNews-vectors-negative300.bin (word2vec, 300-dim)

Explainability (XAI):
- Integrated Gradients (SHAP approximation) per modality + named features
  (sleep duration, alpha-band EEG, negative sentiment)
- Attention heatmaps: EEG channel importance, text span importance, fusion gates
- Counterfactual explanations: minimal embedding perturbation to flip prediction
Enable via model.forward(..., explain=True) or run --explain in CLI.
"""

from __future__ import annotations

import csv
import json
import os
import re
import unicodedata
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from encoding_layers import TextEncoderWithBackbone, TemporalEncoderBlock
from fairness import FairnessAwareRegularizer, FairnessConfig
from prediction import PredictionConfig, PredictionLayer, compute_prediction_losses
from xai_module import (
    ExplanationOutput,
    XAIConfig,
    XAIModule,
    format_explanations,
)


# ---------------------------------------------------------------------------
# DSM-5 symptom keywords (20 symptoms matching PredictionConfig.num_symptoms)
# ---------------------------------------------------------------------------
DSM5_KEYWORDS: List[str] = [
    "depressed", "hopeless", "worthless", "suicidal", "anhedonia",
    "fatigue", "insomnia", "hypersomnia", "appetite", "concentration",
    "guilt", "psychomotor", "irritable", "anxiety", "panic",
    "tearful", "lonely", "withdrawn", "self-harm", "no motivation",
]
_DSM5_PATTERNS = {k: re.compile(re.escape(k) + r"\w*", re.IGNORECASE) for k in DSM5_KEYWORDS}


def _dsm5_vector(text: str) -> np.ndarray:
    """Return a 20-dim float32 array of DSM-5 keyword match counts (clipped to [0,1])."""
    lowered = text.lower()
    counts = np.array(
        [min(1.0, len(_DSM5_PATTERNS[k].findall(lowered))) for k in DSM5_KEYWORDS],
        dtype=np.float32,
    )
    return counts


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?(?:'[a-z0-9]+)?", _normalize_text(text))


# ---------------------------------------------------------------------------
# GoogleNews word2vec loader (streams the binary file, loads only needed words)
# ---------------------------------------------------------------------------

def _load_googlenews_subset(
    vector_path: str,
    needed_words: set,
) -> Tuple[Dict[str, np.ndarray], int]:
    """Return {word: vector} for words in *needed_words* and the vector dimension."""
    path = Path(vector_path)
    if not path.exists():
        raise FileNotFoundError(f"GoogleNews vectors not found: {vector_path}")

    word_vectors: Dict[str, np.ndarray] = {}
    with path.open("rb") as f:
        header = f.readline().decode("utf-8", errors="replace").strip().split()
        if len(header) != 2:
            raise RuntimeError("Invalid GoogleNews vectors file header.")
        vocab_size, vector_size = int(header[0]), int(header[1])
        for _ in range(vocab_size):
            word_bytes = bytearray()
            while True:
                ch = f.read(1)
                if not ch:
                    break
                if ch == b" ":
                    break
                if ch != b"\n":
                    word_bytes.extend(ch)
            word = word_bytes.decode("utf-8", errors="ignore")
            vec = np.fromfile(f, dtype=np.float32, count=vector_size)
            f.read(1)  # trailing newline
            if word in needed_words:
                word_vectors[word] = vec
    return word_vectors, vector_size


def _text_to_googlenews_embedding(
    tokens: List[str],
    word_vectors: Dict[str, np.ndarray],
    vector_size: int,
) -> np.ndarray:
    """Average GoogleNews vectors for all tokens that have entries."""
    vecs = []
    for tok in tokens:
        for variant in (tok, tok.lower(), tok.capitalize(), tok.upper()):
            if variant in word_vectors:
                vecs.append(word_vectors[variant])
                break
    if vecs:
        return np.mean(vecs, axis=0).astype(np.float32)
    return np.zeros(vector_size, dtype=np.float32)


# ---------------------------------------------------------------------------
# Data loading: WU3D (JSON) + Reddit (CSV)
# ---------------------------------------------------------------------------

@dataclass
class SampleRecord:
    text: str
    label: int          # 0 = normal, 1 = depressed
    gender: int         # 0 = unknown/other, 1 = male, 2 = female
    source: str         # 'wu3d' | 'reddit'


def _load_wu3d(depressed_path: str, normal_path: str) -> List[SampleRecord]:
    """Load WU3D dataset. One record per user (all tweets concatenated)."""
    records: List[SampleRecord] = []

    for label_int, path in [(1, depressed_path), (0, normal_path)]:
        if not os.path.exists(path):
            warnings.warn(f"WU3D file not found: {path}", UserWarning)
            continue
        print(f"  Loading WU3D {'depressed' if label_int else 'normal'}: {path}")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)

        for user in data:
            # Concatenate all tweet_content fields for the user
            tweets = user.get("tweets", [])
            text_parts = [t.get("tweet_content", "") for t in tweets if t.get("tweet_content", "").strip()]
            if not text_parts:
                continue
            text = " ".join(text_parts)

            # Gender: map Chinese labels
            gender_raw = user.get("gender", "")
            if "男" in gender_raw:
                gender = 1   # male
            elif "女" in gender_raw:
                gender = 2   # female
            else:
                gender = 0

            records.append(SampleRecord(text=text, label=label_int, gender=gender, source="wu3d"))

    return records


def _load_reddit(csv_path: str) -> List[SampleRecord]:
    """Load Reddit depression dataset. One record per post."""
    records: List[SampleRecord] = []
    if not os.path.exists(csv_path):
        warnings.warn(f"Reddit CSV not found: {csv_path}", UserWarning)
        return records

    print(f"  Loading Reddit dataset: {csv_path}")
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("title", "") or ""
            body = row.get("body", "") or ""
            text = (title + " " + body).strip()
            if not text:
                continue
            try:
                label_int = int(row.get("label", 0))
            except ValueError:
                label_int = 0
            records.append(SampleRecord(text=text, label=label_int, gender=0, source="reddit"))

    return records


# ---------------------------------------------------------------------------
# Real multimodal dataset
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    depressed_json:    str = "depressed.json"
    normal_json:       str = "normal.json"
    reddit_csv:        str = "reddit_depression_dataset.csv"
    google_news_bin:   str = "GoogleNews-vectors-negative300.bin"
    use_wu3d:          bool = True
    use_reddit:        bool = True
    # DAIC-WOZ: set use_daic_woz=True and point to the raw folder + labels CSV
    use_daic_woz:      bool = False
    daic_woz_raw_dir:  str  = "daic-woz"
    daic_woz_labels:   str  = "daic-woz/labels.csv"
    max_samples:       int = 0              # 0 = use all
    text_embed_dim:    int = 300            # GoogleNews output dim
    # Synthetic modality shapes (these datasets have no EEG/wearable/audio/video)
    eeg_time:          int = 512
    eeg_channels:      int = 32
    wearable_time:     int = 100
    wearable_dim:      int = 8
    mfcc_time:         int = 64
    mfcc_dim:          int = 13
    audio_time:        int = 200
    audio_dim:         int = 40
    video_time:        int = 200
    video_dim:         int = 64
    clinical_dim:      int = 10
    num_symptoms:      int = 20
    seed:              int = 42


class RealDepressionDataset(torch.utils.data.Dataset):
    """Dataset loading WU3D JSON + Reddit CSV with GoogleNews text embeddings.

    Non-text modalities (EEG, wearable, audio, video, clinical) are synthesised
    using label-conditioned Gaussian noise because these datasets are text-only.
    The text modality uses averaged GoogleNews word2vec vectors (300-dim).
    """

    def __init__(self, records: List[SampleRecord], embeddings: np.ndarray, cfg: DatasetConfig):
        assert len(records) == len(embeddings), "Mismatch between records and embeddings."
        self.records = records
        self.embeddings = embeddings  # shape [N, text_embed_dim]
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

    # ------------------------------------------------------------------
    @classmethod
    def build(cls, cfg: DatasetConfig) -> "RealDepressionDataset":
        """Factory: load data sources and compute GoogleNews embeddings."""
        all_records: List[SampleRecord] = []

        if cfg.use_wu3d:
            wu3d = _load_wu3d(cfg.depressed_json, cfg.normal_json)
            print(f"  WU3D records loaded: {len(wu3d)}")
            all_records.extend(wu3d)

        if cfg.use_reddit:
            reddit = _load_reddit(cfg.reddit_csv)
            print(f"  Reddit records loaded: {len(reddit)}")
            all_records.extend(reddit)

        if getattr(cfg, "use_daic_woz", False):
            from daic_woz_dataset import DAICWOZDataset, DAICConfig
            _daic_cfg = DAICConfig(
                raw_dir=cfg.daic_woz_raw_dir,
                labels_csv=cfg.daic_woz_labels,
                split="train",
                seed=cfg.seed,
            )
            _daic_ds = DAICWOZDataset(_daic_cfg)
            for _i in range(len(_daic_ds)):
                _row = _daic_ds.records[_i]
                all_records.append(
                    SampleRecord(
                        text="[daic]",
                        label=int(float(_row.get("label", 0))),
                        gender=int(float(_row.get("gender", 0))),
                        source="daic",
                    )
                )
            print(f"  DAIC-WOZ records loaded: {len(_daic_ds)}")

        if not all_records:
            raise RuntimeError("No records loaded. Check dataset paths.")


        # Shuffle deterministically and optionally cap
        rng = np.random.default_rng(cfg.seed)
        idx = rng.permutation(len(all_records)).tolist()
        all_records = [all_records[i] for i in idx]
        if cfg.max_samples > 0:
            all_records = all_records[: cfg.max_samples]

        print(f"\nTotal samples: {len(all_records)}")
        dep = sum(r.label for r in all_records)
        print(f"  Depressed: {dep}  |  Normal: {len(all_records) - dep}")

        # ---- Build GoogleNews embeddings ----
        print("\nBuilding GoogleNews embeddings (streaming binary file)...")
        tokenized: List[List[str]] = [_tokenize(r.text) for r in all_records]
        needed_words: set = set()
        for toks in tokenized:
            for tok in toks:
                needed_words.update([tok, tok.lower(), tok.capitalize(), tok.upper()])

        print(f"  Unique token variants needed: {len(needed_words)}")
        word_vectors, vector_size = _load_googlenews_subset(cfg.google_news_bin, needed_words)
        print(f"  Loaded {len(word_vectors)} word vectors (dim={vector_size})")

        embeddings = np.stack(
            [_text_to_googlenews_embedding(toks, word_vectors, vector_size) for toks in tokenized],
            axis=0,
        ).astype(np.float32)

        return cls(all_records, embeddings, cfg)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        cfg = self.cfg

        # ---- Text embedding (300-dim GoogleNews) ----
        text_emb = torch.from_numpy(self.embeddings[idx])   # [300]

        # ---- DSM-5 symptom labels from text keywords ----
        symptom_labels = torch.from_numpy(_dsm5_vector(rec.text))  # [20]

        # ---- PHQ-9 severity proxy ----
        # Depressed: 10–27, Normal: 0–9  (crude label-based proxy)
        rng_state = np.random.default_rng(idx + cfg.seed * 10000)
        if rec.label == 1:
            phq9_score = float(rng_state.integers(10, 28))
        else:
            phq9_score = float(rng_state.integers(0, 10))

        # ---- Synthetic non-text modalities (label-conditioned noise) ----
        # Depressed samples have slightly elevated mean to give the model a
        # signal even from random modalities during integration testing.
        noise_mean = 0.3 * rec.label
        eeg       = torch.randn(cfg.eeg_time, cfg.eeg_channels).add_(noise_mean)
        wearable  = torch.randn(cfg.wearable_time, cfg.wearable_dim).add_(noise_mean)
        mfcc      = torch.randn(cfg.mfcc_time, cfg.mfcc_dim).add_(noise_mean)
        audio     = torch.randn(cfg.audio_time, cfg.audio_dim).add_(noise_mean)
        video     = torch.randn(cfg.video_time, cfg.video_dim).add_(noise_mean)
        clinical  = torch.randn(cfg.clinical_dim).add_(noise_mean)

        # ---- Sensitive attributes ----
        gender   = torch.tensor([rec.gender], dtype=torch.long)
        age_bin  = torch.tensor([0], dtype=torch.long)   # not available in these datasets
        language = torch.tensor([0 if rec.source == "wu3d" else 1], dtype=torch.long)
        culture  = torch.tensor([0], dtype=torch.long)

        return {
            "text_emb":      text_emb,
            "eeg":           eeg,
            "wearable":      wearable,
            "mfcc":          mfcc,
            "audio":         audio,
            "video":         video,
            "clinical":      clinical,
            "label":         torch.tensor([rec.label], dtype=torch.long),
            "phq9_score":    torch.tensor([phq9_score], dtype=torch.float32),
            "symptom_labels": symptom_labels,
            "gender":        gender,
            "age_bin":       age_bin,
            "language":      language,
            "culture":       culture,
        }


def collate_batch(batch: list) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}


# ---------------------------------------------------------------------------
# Text projection head (replaces token-based TextEncoder for real data)
# ---------------------------------------------------------------------------

class TextEmbeddingProjector(nn.Module):
    """Projects pre-computed GoogleNews 300-dim embeddings to hidden_dim."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Encoder modules (unchanged from original architecture)
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """Text encoder using BERT/RoBERTa backbone with CNN+BiLSTM+Transformer."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        backbone_name: Optional[str] = "bert-base-uncased",
        transformer_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = TextEncoderWithBackbone(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            backbone_name=backbone_name,
            transformer_layers=transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(token_ids, attention_mask)


class EegEncoder(nn.Module):
    """EEG encoder using 1D-CNN + BiLSTM temporal encoding."""

    def __init__(self, channels: int, hidden_dim: int):
        super().__init__()
        self.encoder = TemporalEncoderBlock(
            input_dim=channels,
            hidden_dim=hidden_dim,
            use_transformer=False,
            dropout=0.1,
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.encoder(eeg)


class WearableEncoder(nn.Module):
    """Wearable encoder using temporal CNN+BiLSTM."""

    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = TemporalEncoderBlock(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            use_transformer=False,
            dropout=0.1,
        )

    def forward(self, wearable: torch.Tensor) -> torch.Tensor:
        return self.encoder(wearable)


class MfccEncoder(nn.Module):
    """MFCC encoder using temporal CNN+BiLSTM."""

    def __init__(self, mfcc_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = TemporalEncoderBlock(
            input_dim=mfcc_dim,
            hidden_dim=hidden_dim,
            use_transformer=False,
            dropout=0.1,
        )

    def forward(self, mfcc: torch.Tensor) -> torch.Tensor:
        return self.encoder(mfcc)


class AudioVideoEncoder(nn.Module):
    def __init__(self, audio_dim: int, video_dim: int, hidden_dim: int):
        super().__init__()
        self.audio_gru = nn.GRU(audio_dim, hidden_dim, batch_first=True)
        self.video_gru = nn.GRU(video_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        a, _ = self.audio_gru(audio)
        v, _ = self.video_gru(video)
        fused = torch.cat([a[:, -1], v[:, -1]], dim=-1)
        return self.proj(fused)


class ClinicalEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, clinical: torch.Tensor) -> torch.Tensor:
        return self.mlp(clinical)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

class DynamicGatingNetwork(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.weight_head = nn.Linear(hidden_dim, 1)
        self.reliability_head = nn.Linear(hidden_dim, 1)

    def forward(self, modality_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(modality_embeddings)
        weight_logits = self.weight_head(features).squeeze(-1)
        reliability = torch.sigmoid(self.reliability_head(features).squeeze(-1))
        combined_logits = weight_logits + torch.log(reliability.clamp(min=1e-6))
        weights = torch.softmax(combined_logits, dim=1)
        return weights, reliability


class MultimodalFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gating = DynamicGatingNetwork(hidden_dim, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, modality_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # modality_embeddings: [batch, modalities, hidden_dim]
        modality_weights, reliability = self.gating(modality_embeddings)
        weighted_modalities = modality_embeddings * modality_weights.unsqueeze(-1)

        attn_out, attn_weights = self.cross_attn(
            weighted_modalities,
            weighted_modalities,
            weighted_modalities,
        )
        attended = self.norm1(weighted_modalities + attn_out)
        fused = self.norm2(attended + self.ffn(attended))
        pooled = torch.sum(fused * modality_weights.unsqueeze(-1), dim=1)
        return pooled, modality_weights, reliability


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DepressionDetectionModel(nn.Module):
    """Multimodal depression detection model.

    For the real-data pipeline the text modality is supplied as a pre-computed
    GoogleNews embedding (300-dim) and projected with TextEmbeddingProjector.
    Set ``use_text_projector=True`` to enable this path (default for real data).
    """

    def __init__(
        self,
        vocab_size: int,
        text_embed_dim: int,
        hidden_dim: int,
        eeg_channels: int,
        wearable_dim: int,
        mfcc_dim: int,
        audio_dim: int,
        video_dim: int,
        clinical_dim: int,
        text_backbone_name: Optional[str] = None,
        fairness_config: Optional[FairnessConfig] = None,
        prediction_config: Optional[PredictionConfig] = None,
        use_text_projector: bool = True,
        use_xai: bool = True,
        xai_config: Optional[XAIConfig] = None,
    ):
        super().__init__()
        self.prediction_config = prediction_config or PredictionConfig()
        self.use_text_projector = use_text_projector

        if use_text_projector:
            # Pre-computed GoogleNews embeddings → MLP projection
            self.text_projector = TextEmbeddingProjector(text_embed_dim, hidden_dim)
        else:
            # Token-based encoder (BERT backbone or learned embedding)
            self.text_encoder = TextEncoder(
                vocab_size, text_embed_dim, hidden_dim,
                backbone_name=text_backbone_name,
            )

        self.eeg_encoder      = EegEncoder(eeg_channels, hidden_dim)
        self.wearable_encoder = WearableEncoder(wearable_dim, hidden_dim)
        self.mfcc_encoder     = MfccEncoder(mfcc_dim, hidden_dim)
        self.av_encoder       = AudioVideoEncoder(audio_dim, video_dim, hidden_dim)
        self.clinical_encoder = ClinicalEncoder(clinical_dim, hidden_dim)
        self.fusion           = MultimodalFusion(hidden_dim)
        self.fairness         = FairnessAwareRegularizer(hidden_dim, fairness_config)
        self.prediction       = PredictionLayer(hidden_dim, self.prediction_config)

        # ── XAI module (registered as a sub-module so it's saved with state_dict) ──
        self.xai: Optional[XAIModule] = XAIModule(hidden_dim, xai_config) if use_xai else None

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers exposed to XAIModule (no recursive self() call)
    # ──────────────────────────────────────────────────────────────────────

    def _embedding_to_logits(self, stacked_embs: torch.Tensor) -> torch.Tensor:
        """Map stacked modality embeddings [B, M, D] → binary logits [B, 2].

        Used by IntegratedGradientsExplainer and CounterfactualExplainer.
        Skips fairness debiasing (no sensitive attributes available during XAI).
        Gradients flow through fusion + prediction — model param grads are NOT
        accumulated (callers use ``torch.autograd.grad()`` on leaf tensors).
        """
        pooled, _, _ = self.fusion(stacked_embs)
        debiased, _, _ = self.fairness(pooled, None, None, None, training=False)
        return self.prediction(debiased)["binary_logits"]

    def _encode_all_modalities(
        self,
        text_emb: torch.Tensor,
        eeg: torch.Tensor,
        wearable: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
        clinical: torch.Tensor,
        mfcc: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode raw modality inputs and return prediction outputs.

        Used by EEGChannelAttribution and TextSpanAttribution so gradients
        can flow through encoders → fusion → prediction.  Skips fairness.
        """
        t  = self.text_projector(text_emb) if self.use_text_projector else self.text_encoder(text_emb)
        e  = self.eeg_encoder(eeg)
        w  = self.wearable_encoder(wearable)
        m  = self.mfcc_encoder(mfcc) if mfcc is not None else None
        av = self.av_encoder(audio, video)
        c  = self.clinical_encoder(clinical)

        embs = [t, e, w, av, c]
        if m is not None:
            embs.append(m)
        stacked = torch.stack(embs, dim=1)
        pooled, _, _ = self.fusion(stacked)
        debiased, _, _ = self.fairness(pooled, None, None, None, training=False)
        return self.prediction(debiased)

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        # text: either pre-computed embedding or token ids
        text_input: torch.Tensor,
        eeg: torch.Tensor,
        wearable: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
        clinical: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mfcc: Optional[torch.Tensor] = None,
        sensitive_attrs: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
        explain: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        text_input : Tensor
            Pre-computed GoogleNews embedding [B, 300] when ``use_text_projector``
            is True, otherwise token id tensor [B, seq_len].
        explain : bool, default False
            When True, run the XAI module after prediction and attach an
            ``ExplanationOutput`` under the key ``"explanations"`` in the
            returned dict.  Should NOT be set during training (adds overhead).
        """
        # ── Encode text ────────────────────────────────────────────────────
        if self.use_text_projector:
            text_emb = self.text_projector(text_input)
        else:
            text_emb = self.text_encoder(text_input, attention_mask)

        # ── Encode all other modalities ────────────────────────────────────
        eeg_emb  = self.eeg_encoder(eeg)
        wear_emb = self.wearable_encoder(wearable)
        mfcc_emb = self.mfcc_encoder(mfcc) if mfcc is not None else None
        av_emb   = self.av_encoder(audio, video)
        clin_emb = self.clinical_encoder(clinical)

        modality_list = [text_emb, eeg_emb, wear_emb, av_emb, clin_emb]
        if mfcc_emb is not None:
            modality_list.append(mfcc_emb)
        modality_stack = torch.stack(modality_list, dim=1)

        # ── Fusion ─────────────────────────────────────────────────────────
        fused, modality_weights, reliability = self.fusion(modality_stack)

        # ── Fairness debiasing ─────────────────────────────────────────────
        if self.training and sensitive_attrs is not None:
            pre_outputs = self.prediction(fused)
            debiased, fairness_loss, fairness_aux = self.fairness(
                fused,
                sensitive_attrs=sensitive_attrs,
                binary_logits=pre_outputs["binary_logits"],
                labels=labels,
                training=True,
            )
            outputs = self.prediction(debiased)
        else:
            debiased, fairness_loss, fairness_aux = self.fairness(
                fused,
                sensitive_attrs=None,
                binary_logits=None,
                labels=None,
                training=False,
            )
            outputs = self.prediction(debiased)

        # ── Prediction outputs ─────────────────────────────────────────────
        result: Dict[str, torch.Tensor] = {
            **outputs,
            "debiased":           debiased,
            "fused":              fused,
            "fairness_loss":      fairness_loss,
            "fairness_aux":       fairness_aux,
            "fusion_weights":     modality_weights,
            "fusion_reliability": reliability,
        }

        # ── XAI explanations (optional, on-demand) ─────────────────────────
        if explain and self.xai is not None:
            # Build named modality-embedding dict (must match logit_fn order)
            modality_embeddings: Dict[str, torch.Tensor] = {
                "text":     text_emb.detach(),
                "eeg":      eeg_emb.detach(),
                "wearable": wear_emb.detach(),
                "av":       av_emb.detach(),
                "clinical": clin_emb.detach(),
            }
            if mfcc_emb is not None:
                modality_embeddings["mfcc"] = mfcc_emb.detach()

            raw_inputs: Dict[str, torch.Tensor] = {
                "text_emb": text_input,
                "eeg":      eeg,
                "wearable": wearable,
                "audio":    audio,
                "video":    video,
                "clinical": clinical,
            }
            if mfcc is not None:
                raw_inputs["mfcc"] = mfcc

            result["explanations"] = self.xai.explain(
                raw_inputs=raw_inputs,
                modality_embeddings=modality_embeddings,
                fusion_weights=modality_weights,
                binary_logits=outputs["binary_logits"],
                embedding_to_logits_fn=self._embedding_to_logits,
                full_encode_fn=self._encode_all_modalities,
            )

        return result


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def _sensitive_attrs_from_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        "gender":   batch["gender"].view(-1),
        "age_bin":  batch["age_bin"].view(-1),
        "language": batch["language"].view(-1),
        "culture":  batch["culture"].view(-1),
    }


def _forward_batch(model: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return model(
        text_input=batch["text_emb"],
        eeg=batch["eeg"],
        wearable=batch["wearable"],
        audio=batch["audio"],
        video=batch["video"],
        clinical=batch["clinical"],
        attention_mask=None,
        mfcc=batch["mfcc"],
        sensitive_attrs=_sensitive_attrs_from_batch(batch),
        labels=batch["label"],
    )


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    prediction_config: PredictionConfig,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        out = _forward_batch(model, batch)
        task_loss = compute_prediction_losses(
            out,
            {
                "label":          batch["label"],
                "phq9_score":     batch["phq9_score"],
                "symptom_labels": batch["symptom_labels"],
            },
            prediction_config,
        )
        loss = task_loss + out["fairness_loss"]
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    prediction_config: PredictionConfig,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = _forward_batch(model, batch)
        task_loss = compute_prediction_losses(
            out,
            {
                "label":          batch["label"],
                "phq9_score":     batch["phq9_score"],
                "symptom_labels": batch["symptom_labels"],
            },
            prediction_config,
        )
        loss = task_loss + out["fairness_loss"]
        running_loss += loss.item()
        preds = out["binary_probs"].argmax(dim=-1)
        correct += (preds == batch["label"].view(-1)).sum().item()
        total += preds.numel()
    avg_loss = running_loss / max(len(loader), 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train multimodal depression detection model.")
    parser.add_argument("--depressed_json",  default="depressed.json")
    parser.add_argument("--normal_json",     default="normal.json")
    parser.add_argument("--reddit_csv",      default="reddit_depression_dataset.csv")
    parser.add_argument("--google_news_bin", default="GoogleNews-vectors-negative300.bin")
    parser.add_argument("--max_samples",     type=int, default=2000,
                        help="Cap total samples (0=all). Default 2000 for faster prototyping.")
    parser.add_argument("--epochs",          type=int, default=3)
    parser.add_argument("--batch_size",      type=int, default=16)
    parser.add_argument("--hidden_dim",      type=int, default=256)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--val_split",       type=float, default=0.15,
                        help="Fraction of data to use for validation.")
    parser.add_argument("--no_wu3d",         action="store_true", help="Skip WU3D dataset.")
    parser.add_argument("--no_reddit",       action="store_true", help="Skip Reddit dataset.")
    parser.add_argument("--explain",         action="store_true",
                        help="Run XAI explanations on two sample records after training.")
    parser.add_argument("--ig_steps",        type=int, default=50,
                        help="Integrated-Gradients interpolation steps (default 50).")
    parser.add_argument("--cf_steps",        type=int, default=150,
                        help="Counterfactual optimisation steps (default 150).")
    args = parser.parse_args()

    # ---- Dataset ----
    ds_cfg = DatasetConfig(
        depressed_json=args.depressed_json,
        normal_json=args.normal_json,
        reddit_csv=args.reddit_csv,
        google_news_bin=args.google_news_bin,
        use_wu3d=not args.no_wu3d,
        use_reddit=not args.no_reddit,
        max_samples=args.max_samples,
        text_embed_dim=300,
    )

    print("=== Loading datasets ===")
    full_dataset = RealDepressionDataset.build(ds_cfg)

    val_size  = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    train_data, val_data = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(ds_cfg.seed),
    )
    print(f"\nTrain: {len(train_data)}  |  Val: {len(val_data)}")

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch
    )

    # ---- Model ----
    prediction_config = PredictionConfig(num_symptoms=ds_cfg.num_symptoms)
    xai_cfg = XAIConfig(ig_steps=args.ig_steps, cf_steps=args.cf_steps)
    model = DepressionDetectionModel(
        vocab_size=30522,          # not used when use_text_projector=True
        text_embed_dim=ds_cfg.text_embed_dim,
        hidden_dim=args.hidden_dim,
        eeg_channels=ds_cfg.eeg_channels,
        wearable_dim=ds_cfg.wearable_dim,
        mfcc_dim=ds_cfg.mfcc_dim,
        audio_dim=ds_cfg.audio_dim,
        video_dim=ds_cfg.video_dim,
        clinical_dim=ds_cfg.clinical_dim,
        text_backbone_name=None,
        prediction_config=prediction_config,
        use_text_projector=True,   # use pre-computed GoogleNews embeddings
        use_xai=True,
        xai_config=xai_cfg,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"\nDevice: {device}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ---- Training loop ----
    print("\n=== Training ===")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, prediction_config, device)
        val_loss, val_acc = evaluate(model, val_loader, prediction_config, device)
        print(
            f"epoch={epoch + 1:02d}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}"
        )

    # ---- Quick inference check ----
    print("\n=== Sample inference ===")
    sample_batch = collate_batch([full_dataset[0], full_dataset[1]])
    sample_batch = {k: v.to(device) for k, v in sample_batch.items()}
    model.eval()
    with torch.no_grad():
        sample_out = _forward_batch(model, sample_batch)
    print("binary_logits   :", sample_out["binary_logits"].shape)
    print("severity        :", sample_out["severity"].shape)
    print("symptom_logits  :", sample_out["symptom_logits"].shape)
    print("debiased        :", sample_out["debiased"].shape)
    print("fusion_weights  :", sample_out["fusion_weights"].shape)

    # ---- XAI explanations (optional) ----
    if args.explain:
        print("\n=== XAI Explanations ===")
        print("Running Integrated Gradients, attention heatmaps, and counterfactual")
        print(f"search (IG steps={args.ig_steps}, CF steps={args.cf_steps}) ...\n")

        xai_batch = collate_batch([full_dataset[0], full_dataset[1]])
        xai_batch = {k: v.to(device) for k, v in xai_batch.items()}

        model.eval()
        # explain=True triggers the XAI module; gradients are managed internally
        xai_out = model(
            text_input=xai_batch["text_emb"],
            eeg=xai_batch["eeg"],
            wearable=xai_batch["wearable"],
            audio=xai_batch["audio"],
            video=xai_batch["video"],
            clinical=xai_batch["clinical"],
            mfcc=xai_batch["mfcc"],
            explain=True,
        )

        exp: ExplanationOutput = xai_out["explanations"]
        print(format_explanations(exp))

        # Tensor shape summary
        print("\nTensor shapes:")
        print(f"  eeg_channel_importances : {exp.eeg_channel_importances.shape}")
        print(f"  text_span_importances   : {exp.text_span_importances.shape}")
        print(f"  fusion_attention_weights: {exp.fusion_attention_weights.shape}")
        for k, v in exp.ig_attributions.items():
            print(f"  ig_attributions[{k:<10}]: {v.shape}")
