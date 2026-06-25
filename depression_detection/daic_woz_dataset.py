"""DAIC-WOZ dataset loader — real audio, facial, and text modalities.

Produces the **exact same batch-dict keys** as RealDepressionDataset so it can be
passed directly to collate_batch, _forward_batch, and the full pipeline with
zero model changes.

Real modalities loaded from disk
---------------------------------
audio   <- {pid}_COVAREP.csv      74 COVAREP acoustic features, no header row
            shape after padding/trimming: [MAX_SEQ_LEN, COVAREP_DIM]
video   <- {pid}_CLNF_AUs.txt     comma-separated with header row; 24 total columns:
            first 4 = frame/timestamp/confidence/success (skipped);
            columns 4-23 = AU features (20 columns used)
            shape after padding/trimming: [MAX_SEQ_LEN, AU_DIM]
text_emb <- {pid}_TRANSCRIPT.csv  tab-separated with header; 'Participant' rows only;
            deterministic hash-BOW embedding -> [GOOGLENEWS_DIM]
            (or GoogleNews word2vec if cfg.google_news_bin is set)

Synthetic (label-conditioned Gaussian noise, same as RealDepressionDataset)
--------------------------------------------------------------------
eeg, wearable, mfcc, clinical

Folder layout expected
----------------------
daic_woz_raw_dir/
    {pid}_P/
        {pid}_COVAREP.csv
        {pid}_CLNF_AUs.txt
        {pid}_TRANSCRIPT.csv

labels_csv required columns: participant_id, label, phq_score, gender, age_bin, split
    split values: 'train' | 'dev' | 'test'

Usage
-----
    cfg = DAICConfig(raw_dir="daic-woz", labels_csv="daic-woz/labels.csv", split="train")
    ds  = DAICWOZDataset(cfg)
    sample = ds[0]
"""

from __future__ import annotations

import csv
import math
import os
import re
import unicodedata
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Dimension constants
# ---------------------------------------------------------------------------

COVAREP_DIM    = 74   # number of acoustic feature cols in COVAREP.csv (no header)
AU_DIM         = 20   # AU feature columns in CLNF_AUs.txt (skip first 4 metadata cols)
AU_SKIP_COLS   = 4    # frame, timestamp, confidence, success
MAX_SEQ_LEN    = 300  # temporal frames kept (trim if longer, zero-pad if shorter)
GOOGLENEWS_DIM = 300  # must match DatasetConfig.text_embed_dim


# ---------------------------------------------------------------------------
# DSM-5 symptom keywords (identical to multimodal_model.py)
# ---------------------------------------------------------------------------

DSM5_KEYWORDS: List[str] = [
    "depressed", "hopeless", "worthless", "suicidal", "anhedonia",
    "fatigue", "insomnia", "hypersomnia", "appetite", "concentration",
    "guilt", "psychomotor", "irritable", "anxiety", "panic",
    "tearful", "lonely", "withdrawn", "self-harm", "no motivation",
]
_DSM5_PAT = {k: re.compile(re.escape(k) + r"\w*", re.IGNORECASE) for k in DSM5_KEYWORDS}


def _dsm5_vector(text: str) -> torch.Tensor:
    """Return a 20-dim float32 tensor of DSM-5 keyword hit counts clipped to [0, 1]."""
    lowered = text.lower()
    counts = [min(1.0, float(len(_DSM5_PAT[k].findall(lowered)))) for k in DSM5_KEYWORDS]
    return torch.tensor(counts, dtype=torch.float32)


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w\s'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DAICConfig:
    """Configuration for DAICWOZDataset."""

    raw_dir:    str = "daic-woz"
    """Root directory containing {pid}_P/ participant sub-folders."""

    labels_csv: str = "daic-woz/labels.csv"
    """Columns required: participant_id, label, phq_score, gender, age_bin, split"""

    split:      str = "train"
    """'train' | 'dev' | 'test'"""

    seed:       int = 42

    # Synthetic modality dims (same defaults as DatasetConfig)
    eeg_time:      int = 512
    eeg_channels:  int = 32
    wearable_time: int = 100
    wearable_dim:  int = 8
    mfcc_time:     int = 64
    mfcc_dim:      int = 13
    clinical_dim:  int = 10
    num_symptoms:  int = 20

    # Optional — set to GoogleNews-vectors-negative300.bin path for real text embeddings
    google_news_bin: str = ""


# ---------------------------------------------------------------------------
# GoogleNews helpers (pure Python — no numpy, no pandas)
# ---------------------------------------------------------------------------

def _load_googlenews_subset_local(
    bin_path: str,
    needed_words: set,
) -> Tuple[Dict[str, List[float]], int]:
    """Stream GoogleNews binary and return {word: list[float]} for needed words."""
    if not os.path.exists(bin_path):
        return {}, GOOGLENEWS_DIM
    word_vectors: Dict[str, List[float]] = {}
    with open(bin_path, "rb") as fh:
        header = fh.readline().decode("utf-8", errors="replace").strip().split()
        vocab_size, vec_size = int(header[0]), int(header[1])
        import struct
        float_size = 4
        for _ in range(vocab_size):
            word_bytes = bytearray()
            while True:
                ch = fh.read(1)
                if not ch or ch == b" ":
                    break
                if ch != b"\n":
                    word_bytes.extend(ch)
            word = word_bytes.decode("utf-8", errors="ignore")
            raw = fh.read(vec_size * float_size)
            fh.read(1)  # trailing newline
            if word in needed_words:
                vec = list(struct.unpack("{}f".format(vec_size), raw))
                word_vectors[word] = vec
    return word_vectors, vec_size


def _hash_embedding(tokens: List[str], dim: int = GOOGLENEWS_DIM) -> torch.Tensor:
    """Deterministic bag-of-words hash embedding — no external files needed."""
    emb = [0.0] * dim
    for tok in tokens:
        emb[hash(tok) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in emb))
    if norm > 0:
        emb = [x / norm for x in emb]
    return torch.tensor(emb, dtype=torch.float32)


def _googlenews_embed(
    tokens: List[str],
    word_vectors: Dict[str, List[float]],
    vec_size: int,
) -> torch.Tensor:
    """Average word2vec vectors for tokens present in word_vectors."""
    vecs: List[List[float]] = []
    for tok in tokens:
        for variant in (tok, tok.lower(), tok.capitalize(), tok.upper()):
            if variant in word_vectors:
                vecs.append(word_vectors[variant])
                break
    if not vecs:
        return _hash_embedding(tokens, vec_size)
    avg = [sum(vecs[j][i] for j in range(len(vecs))) / len(vecs) for i in range(vec_size)]
    return torch.tensor(avg, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Pure-Python / torch tensor helpers (no numpy)
# ---------------------------------------------------------------------------

def _rows_to_tensor(
    rows: List[List[float]],
    max_len: int,
    dim: int,
) -> torch.Tensor:
    """Convert list-of-rows to a zero-padded [max_len, dim] tensor.

    Rows longer than dim are truncated; rows shorter are right-padded with 0.
    The sequence is trimmed to max_len or zero-padded below max_len.
    """
    if not rows:
        return torch.zeros(max_len, dim, dtype=torch.float32)

    # Build tensor row by row
    seq_len = min(len(rows), max_len)
    out = torch.zeros(max_len, dim, dtype=torch.float32)
    for i in range(seq_len):
        row = rows[i]
        usable = min(len(row), dim)
        for j in range(usable):
            v = row[j]
            # sanitise NaN/Inf
            out[i, j] = float(v) if math.isfinite(v) else 0.0
    return out


def _parse_float(s: str) -> Optional[float]:
    """Return float or None if parsing fails."""
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# DAICWOZDataset
# ---------------------------------------------------------------------------

class DAICWOZDataset(Dataset):
    """DAIC-WOZ interview dataset returning real audio+facial+text modalities.

    Compatible with ``collate_batch`` and all downstream layers (FL, XAI,
    Output Layer, Validation) without any model changes.

    Parameters
    ----------
    cfg : DAICConfig
    google_news_vectors : dict, optional
        Pre-loaded {word: list[float]} GoogleNews lookup. Avoids reloading
        across multiple splits.
    """

    def __init__(
        self,
        cfg: DAICConfig,
        google_news_vectors: Optional[Dict[str, List[float]]] = None,
    ):
        self.cfg = cfg
        self._gn_vectors: Dict[str, List[float]] = google_news_vectors or {}
        self._gn_vec_size: int = GOOGLENEWS_DIM
        self._gn_loaded: bool = bool(self._gn_vectors)

        self.records: List[Dict[str, str]] = self._load_labels(cfg.labels_csv, cfg.split)
        if not self.records:
            warnings.warn(
                "No records for split='{}' in {}".format(cfg.split, cfg.labels_csv),
                UserWarning,
            )
        print("DAICWOZDataset [split={}]: {:,} participants.".format(
            cfg.split, len(self.records)))

    # ------------------------------------------------------------------
    @staticmethod
    def _load_labels(csv_path: str, split: str) -> List[Dict[str, str]]:
        if not os.path.exists(csv_path):
            warnings.warn("Labels CSV not found: {}".format(csv_path), UserWarning)
            return []
        result = []
        with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split", "").strip() == split:
                    result.append(dict(row))
        return result

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row    = self.records[idx]
        pid    = str(int(float(row["participant_id"])))
        label  = int(float(row.get("label", 0)))
        phq    = float(row.get("phq_score", 10 * label))
        folder = os.path.join(self.cfg.raw_dir, "{}_P".format(pid))

        # ── Real modalities ───────────────────────────────────────────────────
        audio = self._load_covarep(
            os.path.join(folder, "{}_COVAREP.csv".format(pid))
        )
        video = self._load_clnf_aus(
            os.path.join(folder, "{}_CLNF_AUs.txt".format(pid))
        )
        text_emb, transcript_text = self._load_transcript(
            os.path.join(folder, "{}_TRANSCRIPT.csv".format(pid))
        )
        symptom_labels = _dsm5_vector(transcript_text)

        # ── Synthetic modalities (label-conditioned Gaussian noise) ───────────
        noise_mean = 0.3 * label
        # Use a deterministic seed per sample (matches RealDepressionDataset pattern)
        gen = torch.Generator().manual_seed(idx + self.cfg.seed * 10_000)
        eeg      = torch.randn(self.cfg.eeg_time,      self.cfg.eeg_channels,  generator=gen) + noise_mean
        wearable = torch.randn(self.cfg.wearable_time, self.cfg.wearable_dim,  generator=gen) + noise_mean
        mfcc     = torch.randn(self.cfg.mfcc_time,     self.cfg.mfcc_dim,      generator=gen) + noise_mean
        clinical = torch.randn(self.cfg.clinical_dim,                           generator=gen) + noise_mean

        return {
            "text_emb":       text_emb,                                  # [300]
            "eeg":            eeg,                                        # [eeg_time, eeg_channels]
            "wearable":       wearable,                                   # [wearable_time, wearable_dim]
            "mfcc":           mfcc,                                       # [mfcc_time, mfcc_dim]
            "audio":          audio,                                      # [MAX_SEQ_LEN, COVAREP_DIM]
            "video":          video,                                      # [MAX_SEQ_LEN, AU_DIM]
            "clinical":       clinical,                                   # [clinical_dim]
            "label":          torch.tensor([label], dtype=torch.long),
            "phq9_score":     torch.tensor([phq], dtype=torch.float32),
            "symptom_labels": symptom_labels,                             # [20]
            "gender":         torch.tensor([int(float(row.get("gender", 0)))], dtype=torch.long),
            "age_bin":        torch.tensor([int(float(row.get("age_bin", 0)))], dtype=torch.long),
            "language":       torch.tensor([1], dtype=torch.long),       # English = 1
            "culture":        torch.tensor([1], dtype=torch.long),
        }

    # ------------------------------------------------------------------
    # Modality loaders
    # ------------------------------------------------------------------

    def _load_covarep(self, path: str) -> torch.Tensor:
        """COVAREP.csv: no header, 74 float columns per row.
        Reads at most MAX_SEQ_LEN rows (files can be 36MB+).
        Returns [MAX_SEQ_LEN, COVAREP_DIM].
        """
        if not os.path.exists(path):
            return torch.zeros(MAX_SEQ_LEN, COVAREP_DIM)
        try:
            rows: List[List[float]] = []
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(rows) >= MAX_SEQ_LEN:   # early exit — we only need 300 frames
                        break
                    if not row:
                        continue
                    vals = [_parse_float(c) for c in row[:COVAREP_DIM]]
                    if all(v is None for v in vals):
                        continue   # skip unparseable / header-like rows
                    cleaned = [v if v is not None else 0.0 for v in vals]
                    while len(cleaned) < COVAREP_DIM:
                        cleaned.append(0.0)
                    rows.append(cleaned)
            return _rows_to_tensor(rows, MAX_SEQ_LEN, COVAREP_DIM)
        except Exception as exc:
            warnings.warn("Failed loading COVAREP {}: {}".format(path, exc), UserWarning)
            return torch.zeros(MAX_SEQ_LEN, COVAREP_DIM)

    def _load_clnf_aus(self, path: str) -> torch.Tensor:
        """CLNF_AUs.txt: comma-separated, header row, 24 columns total.
        Skips first AU_SKIP_COLS=4 metadata columns (frame, timestamp, confidence, success).
        Takes the next AU_DIM=20 columns as AU features.
        Returns [MAX_SEQ_LEN, AU_DIM].
        """
        if not os.path.exists(path):
            return torch.zeros(MAX_SEQ_LEN, AU_DIM)
        try:
            rows: List[List[float]] = []
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                header_skipped = False
                for row in reader:
                    if not header_skipped:
                        header_skipped = True
                        continue  # skip header line
                    row = [c.strip() for c in row]
                    if not row:
                        continue
                    au_cells = row[AU_SKIP_COLS: AU_SKIP_COLS + AU_DIM]
                    vals = [_parse_float(c) for c in au_cells]
                    cleaned = [v if v is not None else 0.0 for v in vals]
                    while len(cleaned) < AU_DIM:
                        cleaned.append(0.0)
                    rows.append(cleaned)
            return _rows_to_tensor(rows, MAX_SEQ_LEN, AU_DIM)
        except Exception as exc:
            warnings.warn("Failed loading CLNF_AUs {}: {}".format(path, exc), UserWarning)
            return torch.zeros(MAX_SEQ_LEN, AU_DIM)

    def _load_transcript(self, path: str) -> Tuple[torch.Tensor, str]:
        """TRANSCRIPT.csv: tab-separated, header row.
        Columns: start_time, stop_time, speaker, value.
        Only 'Participant' speaker rows are kept.
        Returns (text_emb [GOOGLENEWS_DIM], raw_text str).
        """
        empty = torch.zeros(GOOGLENEWS_DIM)
        if not os.path.exists(path):
            return empty, ""
        try:
            participant_lines: List[str] = []
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f, delimiter="\t")
                header_skipped = False
                for row in reader:
                    if not header_skipped:
                        header_skipped = True
                        continue
                    if len(row) >= 4 and row[2].strip() == "Participant":
                        participant_lines.append(row[3].strip())

            text = " ".join(participant_lines).strip()
            if not text:
                return empty, ""

            tokens = _normalize_text(text).split()
            emb = self._embed_tokens(tokens)
            return emb, text

        except Exception as exc:
            warnings.warn("Failed loading transcript {}: {}".format(path, exc), UserWarning)
            return empty, ""

    def _embed_tokens(self, tokens: List[str]) -> torch.Tensor:
        """Embed a token list. Uses GoogleNews if available, else hash-BOW."""
        # Lazy-load GoogleNews once per dataset instance
        if not self._gn_loaded and self.cfg.google_news_bin:
            if os.path.exists(self.cfg.google_news_bin):
                print("  [DAIC] Loading GoogleNews (lazy)...")
                needed: set = set()
                for tok in tokens:
                    needed.update([tok, tok.lower(), tok.capitalize(), tok.upper()])
                vecs, sz = _load_googlenews_subset_local(self.cfg.google_news_bin, needed)
                self._gn_vectors   = vecs
                self._gn_vec_size  = sz
            self._gn_loaded = True   # don't retry regardless

        if self._gn_vectors:
            return _googlenews_embed(tokens, self._gn_vectors, self._gn_vec_size)
        return _hash_embedding(tokens, GOOGLENEWS_DIM)


# ---------------------------------------------------------------------------
# make_labels_csv helper
# ---------------------------------------------------------------------------

def make_labels_csv(
    train_split_csv: str,
    dev_split_csv:   str,
    output_csv:      str,
    test_split_csv:  str = "",
) -> None:
    """Build labels.csv from the official DAIC-WOZ AVEC2017 split CSV files.

    Expected columns in each split file (any extras are ignored):
        Participant_ID, PHQ_Binary, PHQ_Score, Gender

    Parameters
    ----------
    train_split_csv : str  Path to train_split_Depression_AVEC2017.csv
    dev_split_csv   : str  Path to dev_split_Depression_AVEC2017.csv
    output_csv      : str  Where to write the combined labels.csv
    test_split_csv  : str  Optional test split (labels may be absent)
    """
    RENAME_MAP = {
        "Participant_ID": "participant_id",
        "PHQ_Binary":     "label",
        "PHQ_Score":      "phq_score",
        "Gender":         "gender",
    }

    def _read_split(path: str, split_name: str) -> List[Dict]:
        if not path or not os.path.exists(path):
            warnings.warn("Split file not found: {}".format(path), UserWarning)
            return []
        result = []
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out: Dict[str, str] = {"split": split_name}
                for old_k, new_k in RENAME_MAP.items():
                    out[new_k] = row.get(old_k, row.get(new_k, "")).strip()
                # normalise gender: M/F/1/2 -> 1/2, else 0
                g = out.get("gender", "")
                if g.upper() in ("M", "MALE", "1"):
                    out["gender"] = "1"
                elif g.upper() in ("F", "FEMALE", "2"):
                    out["gender"] = "2"
                else:
                    out["gender"] = "0"
                out["age_bin"] = "0"
                result.append(out)
        return result

    all_rows: List[Dict] = []
    all_rows.extend(_read_split(train_split_csv, "train"))
    all_rows.extend(_read_split(dev_split_csv,   "dev"))
    if test_split_csv:
        all_rows.extend(_read_split(test_split_csv, "test"))

    if not all_rows:
        raise RuntimeError("No rows read. Check CSV paths and column names.")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    fieldnames = ["participant_id", "label", "phq_score", "gender", "age_bin", "split"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print("labels.csv created: {} rows -> {}".format(len(all_rows), output_csv))


# ---------------------------------------------------------------------------
# CLI: python daic_woz_dataset.py make_labels / verify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DAIC-WOZ dataset utilities")
    sub = parser.add_subparsers(dest="command")

    ml = sub.add_parser("make_labels", help="Build labels.csv from AVEC2017 split CSVs")
    ml.add_argument("--train", required=True)
    ml.add_argument("--dev",   required=True)
    ml.add_argument("--test",  default="")
    ml.add_argument("--out",   required=True)

    vf = sub.add_parser("verify", help="Verify participant folder readability")
    vf.add_argument("--raw_dir",    required=True)
    vf.add_argument("--labels_csv", required=True)
    vf.add_argument("--split",      default="train")

    args = parser.parse_args()

    if args.command == "make_labels":
        make_labels_csv(args.train, args.dev, args.out, args.test)

    elif args.command == "verify":
        cfg = DAICConfig(raw_dir=args.raw_dir, labels_csv=args.labels_csv, split=args.split)
        ds = DAICWOZDataset(cfg)
        ok = fail = 0
        for i in range(len(ds)):
            try:
                s = ds[i]
                assert s["audio"].shape == (MAX_SEQ_LEN, COVAREP_DIM)
                assert s["video"].shape == (MAX_SEQ_LEN, AU_DIM)
                assert s["text_emb"].shape == (GOOGLENEWS_DIM,)
                ok += 1
            except Exception as e:
                print("  FAIL pid={}: {}".format(ds.records[i].get("participant_id"), e))
                fail += 1
        print("Verify done: {} OK, {} failed".format(ok, fail))

    else:
        parser.print_help()
