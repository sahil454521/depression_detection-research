"""Extract text features: GoogleNews/BERT embeddings, sentiment, LDA topics, DSM-5 keyword counts.

Default input is reddit_depression_dataset.csv with columns: title, body.
Outputs:
- features.csv (sentiment, LDA topic probs, DSM-5 keyword counts)
- embeddings.npz (text embeddings keyed by row index)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from dotenv import load_dotenv

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
except Exception:  # pragma: no cover - optional dependency
    SentimentIntensityAnalyzer = None


DSM5_KEYWORDS = [
    "depressed",
    "hopeless",
    "worthless",
    "suicidal",
    "anhedonia",
    "fatigue",
    "insomnia",
    "hypersomnia",
    "appetite",
    "concentration",
    "guilt",
    "psychomotor",
    "irritable",
    "anxiety",
    "panic",
    "tearful",
    "lonely",
    "withdrawn",
    "self-harm",
    "no motivation",
]

def _make_dsm5_pattern(keyword: str) -> re.Pattern:
    escaped = re.escape(keyword)
    return re.compile(rf"\b{escaped}\w*", re.IGNORECASE)

DSM5_PATTERNS = {k: _make_dsm5_pattern(k) for k in DSM5_KEYWORDS}

def compute_dsm5_keyword_counts(texts: List[str]) -> Dict[str, List[int]]:
    counts: Dict[str, List[int]] = {f"dsm5_{k}": [] for k in DSM5_KEYWORDS}
    for text in texts:
        lowered = normalize_text(text)
        for k in DSM5_KEYWORDS:
            counts[f"dsm5_{k}"].append(len(DSM5_PATTERNS[k].findall(lowered)))
    return counts

@dataclass
class ExtractConfig:
    input_path: str
    output_dir: str
    text_columns: Tuple[str, ...]
    embedding_source: str
    bert_model: str
    google_news_path: str
    max_length: int
    batch_size: int
    lda_topics: int


def read_texts_csv(
    path: str,
    text_columns: Tuple[str, ...],
) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read texts from CSV file (Reddit dataset)."""
    texts: List[str] = []
    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parts = [row.get(col, "") for col in text_columns]
            text = " ".join(p for p in parts if p).strip()
            texts.append(text)
            rows.append(row)
    return texts, rows


def read_texts_json(
    depressed_path: str,
    normal_path: str,
    text_key: str = "text",
) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read texts from WU3D JSON files (depressed.json, normal.json)."""
    texts: List[str] = []
    rows: List[Dict[str, str]] = []

    # Keys tried in order at the TOP level of each item; first non-empty wins.
    # Deduplicated so text_key doesn't appear twice when it equals a fallback.
    _seen: set = set()
    _FALLBACK_KEYS: tuple = tuple(
        k for k in (text_key, "text", "content", "post", "body",
                    "tweet_content", "message", "sentence", "description")
        if not (_seen.add(k) or k in _seen - {k})  # preserve order, no dupes
    )

    # Keys that may hold a *list of tweet/post dicts* one level down.
    _NESTED_LIST_KEYS = ("tweets", "posts", "weibo", "messages", "entries", "items")
    # Text keys tried inside each nested dict.
    _NESTED_TEXT_KEYS = ("tweet_content", "text", "content", "body", "post", "message")

    for label, path in [("depressed", depressed_path), ("normal", normal_path)]:
        if not os.path.exists(path):
            warnings.warn(f"File not found: {path}", UserWarning)
            continue

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)

        def _extract_text(item) -> str:
            """Return the concatenated text for one item (user / post).

            Strategy:
              1. Try each top-level text key directly.
              2. Scan every value that is a list: if its elements are plain
                 strings, join them; if they are dicts, recurse with
                 _NESTED_TEXT_KEYS (handles WU3D user→tweets→tweet_content).
              3. Return "" if nothing is found.
            """
            if not isinstance(item, dict):
                return str(item)

            # ── pass 1: direct text keys ────────────────────────────────
            for k in _FALLBACK_KEYS:
                val = item.get(k)
                if val is None:
                    continue
                if isinstance(val, str) and val.strip():
                    return val
                if isinstance(val, list):
                    joined = " ".join(str(v) for v in val if v)
                    if joined.strip():
                        return joined

            # ── pass 2: nested list of dicts (e.g. WU3D tweets list) ───
            # Try known list-key names first, then fall back to any list value.
            candidate_lists = []
            for k in _NESTED_LIST_KEYS:
                val = item.get(k)
                if isinstance(val, list) and val:
                    candidate_lists.append(val)
            # Also include any other list-of-dicts values not already covered.
            for k, val in item.items():
                if k not in _NESTED_LIST_KEYS and isinstance(val, list) and val:
                    candidate_lists.append(val)

            for lst in candidate_lists:
                parts = []
                for element in lst:
                    if isinstance(element, str) and element.strip():
                        parts.append(element)
                    elif isinstance(element, dict):
                        for tk in _NESTED_TEXT_KEYS:
                            nested_val = element.get(tk, "")
                            if isinstance(nested_val, str) and nested_val.strip():
                                parts.append(nested_val)
                                break
                if parts:
                    return " ".join(parts)

            return ""

        if isinstance(data, list):
            for item in data:
                text = _extract_text(item)
                if text.strip():
                    texts.append(text)
                    rows.append({"text": text, "label": label})
        elif isinstance(data, dict):
            for key, item in data.items():
                text = _extract_text(item)
                if text.strip():
                    texts.append(text)
                    rows.append({"text": text, "label": label, "id": key})

        if not texts:
            # Show ALL keys from the first item so the user can identify the
            # right one without having to re-run just to see more keys.
            sample_keys: list = []
            nested_preview: dict = {}
            if isinstance(data, list) and data and isinstance(data[0], dict):
                sample_keys = list(data[0].keys())
                # Show keys inside the first list-of-dicts child too.
                for k, v in data[0].items():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        nested_preview[k] = list(v[0].keys())
            elif isinstance(data, dict):
                first_val = next(iter(data.values()), None)
                if isinstance(first_val, dict):
                    sample_keys = list(first_val.keys())
            hint = f"Actual top-level keys in first item: {sample_keys}."
            if nested_preview:
                hint += f" Nested keys: {nested_preview}."
            warnings.warn(
                f"No texts extracted from '{path}' with text_key={text_key!r}. "
                f"Tried top-level fallbacks {list(_FALLBACK_KEYS)} and nested-list "
                f"keys {list(_NESTED_LIST_KEYS)} → {list(_NESTED_TEXT_KEYS)}. "
                f"{hint} Pass the correct key via --text_key.",
                UserWarning,
            )

    return texts, rows


def read_texts(
    path: str,
    text_columns: Tuple[str, ...] = ("text",),
    dataset_type: str = "csv",
    depressed_path: str = None,
    normal_path: str = None,
    text_key: str = "text",
) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read texts from either CSV or JSON format.

    Args:
        path: Path to CSV file or base directory for JSON files
        text_columns: Columns to use from CSV (ignored for JSON)
        dataset_type: 'csv' for Reddit dataset, 'json' for WU3D dataset
        depressed_path: Path to depressed.json (for JSON datasets)
        normal_path: Path to normal.json (for JSON datasets)
        text_key: JSON key containing text (default: 'text')
    """
    if dataset_type.lower() == "json":
        if depressed_path is None or normal_path is None:
            raise ValueError(
                "For JSON datasets, both depressed_path and normal_path must be provided."
            )
        return read_texts_json(depressed_path, normal_path, text_key=text_key)
    else:
        return read_texts_csv(path, text_columns)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"[^\w\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def tokenize_text(text: str) -> List[str]:
    normalized = normalize_text(text)
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?(?:'[a-z0-9]+)?", normalized)


def compute_bert_embeddings(
    texts: List[str],
    model_name: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("transformers is not installed. Install: pip install transformers torch") from exc

    # Load HF token from .env for faster downloads
    load_dotenv()
    hf_token = os.getenv("hf_token")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModel.from_pretrained(model_name, token=hf_token)
    model.to(device)
    model.eval()

    all_embeddings: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand_as(hidden).float()
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts
        all_embeddings.append(pooled.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


def load_google_news_vectors(vector_path: str):
    path = Path(vector_path)
    if not path.exists():
        raise FileNotFoundError(f"GoogleNews vectors not found: {vector_path}")
    return path


def compute_google_news_embeddings(texts: List[str], vector_path: str) -> np.ndarray:
    path = load_google_news_vectors(vector_path)

    needed_words = set()
    tokenized_texts: List[List[str]] = []
    for text in texts:
        tokens = tokenize_text(text)
        tokenized_texts.append(tokens)
        for token in tokens:
            needed_words.add(token)
            needed_words.add(token.lower())
            needed_words.add(token.capitalize())
            needed_words.add(token.upper())

    word_vectors = _load_google_news_vectors_subset(path, needed_words)
    vector_size = 300
    if word_vectors:
        vector_size = next(iter(word_vectors.values())).shape[0]

    embeddings: List[np.ndarray] = []
    for tokens in tokenized_texts:
        token_vectors = []
        for token in tokens:
            vector = None
            for variant in (token, token.lower(), token.capitalize(), token.upper()):
                if variant in word_vectors:
                    vector = word_vectors[variant]
                    break
            if vector is not None:
                token_vectors.append(vector)
        if token_vectors:
            embeddings.append(np.mean(token_vectors, axis=0))
        else:
            embeddings.append(np.zeros(vector_size, dtype=np.float32))
    return np.asarray(embeddings, dtype=np.float32)


def _load_google_news_vectors_subset(vector_path: Path, needed_words: set[str]) -> Dict[str, np.ndarray]:
    word_vectors: Dict[str, np.ndarray] = {}
    with vector_path.open("rb") as f:
        header = f.readline().decode("utf-8", errors="replace")
        parts = header.strip().split()
        if len(parts) != 2:
            raise RuntimeError("Invalid GoogleNews vectors header")
        vocab_size, vector_size = int(parts[0]), int(parts[1])

        for _ in range(vocab_size):
            word_bytes = bytearray()
            while True:
                ch = f.read(1)
                if not ch:
                    raise RuntimeError("Unexpected end of GoogleNews vectors file")
                if ch == b" ":
                    break
                if ch != b"\n":
                    word_bytes.extend(ch)

            word = word_bytes.decode("utf-8", errors="ignore")
            vector = np.fromfile(f, dtype=np.float32, count=vector_size)
            f.read(1)

            if word in needed_words:
                word_vectors[word] = vector

    return word_vectors


def compute_text_embeddings(cfg: ExtractConfig, texts: List[str], device: torch.device) -> np.ndarray:
    embedding_source = cfg.embedding_source.lower()
    if embedding_source == "googlenews":
        return compute_google_news_embeddings(texts, cfg.google_news_path)
    if embedding_source == "bert":
        return compute_bert_embeddings(
            texts,
            model_name=cfg.bert_model,
            max_length=cfg.max_length,
            batch_size=cfg.batch_size,
            device=device,
        )
    if embedding_source == "both":
        google_news_embeddings = compute_google_news_embeddings(texts, cfg.google_news_path)
        try:
            bert_embeddings = compute_bert_embeddings(
                texts,
                model_name=cfg.bert_model,
                max_length=cfg.max_length,
                batch_size=cfg.batch_size,
                device=device,
            )
        except Exception as exc:
            warnings.warn(
                f"BERT embeddings unavailable ({exc}); using GoogleNews embeddings only for this run.",
                RuntimeWarning,
            )
            return google_news_embeddings
        return np.concatenate([bert_embeddings, google_news_embeddings], axis=1)
    raise ValueError("--embedding_source must be one of: googlenews, bert, both")


def compute_sentiment(texts: List[str]) -> Dict[str, List[float]]:
    if SentimentIntensityAnalyzer is None:
        raise RuntimeError("nltk is not installed. Install: pip install nltk and run nltk.download('vader_lexicon')")
    sia = SentimentIntensityAnalyzer()
    scores = [sia.polarity_scores(t) for t in texts]
    return {
        "sent_pos": [s["pos"] for s in scores],
        "sent_neg": [s["neg"] for s in scores],
        "sent_neu": [s["neu"] for s in scores],
        "sent_compound": [s["compound"] for s in scores],
    }


def compute_lda_topics(texts: List[str], num_topics: int) -> np.ndarray:
    """Fit LDA and return topic-probability matrix [n_texts, num_topics].

    Handles edge cases:
    - Empty texts list → zero matrix of shape (0, num_topics)
    - Texts that reduce to stop-words only → zero matrix with a warning
    - Corpus smaller than num_topics → caps n_components to n_docs - 1
    - Small corpus → lowers min_df so the vocabulary is non-empty
    """
    if not texts:
        return np.zeros((0, num_topics), dtype=np.float64)

    normalized_texts = [normalize_text(text) for text in texts]

    # min_df must be ≤ number of documents; scale down for small corpora.
    n_docs = len(normalized_texts)
    min_df = min(5, max(1, n_docs // 10))

    # LDA requires num_topics ≤ n_docs.
    n_components = min(num_topics, max(1, n_docs - 1))
    if n_components < num_topics:
        warnings.warn(
            f"Only {n_docs} document(s) available; reducing LDA n_components "
            f"from {num_topics} to {n_components}.",
            UserWarning,
        )

    vectorizer = CountVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
    )
    try:
        counts = vectorizer.fit_transform(normalized_texts)
    except ValueError as exc:
        if "empty vocabulary" in str(exc):
            warnings.warn(
                f"LDA skipped: vocabulary is empty after stop-word filtering "
                f"({n_docs} text(s), min_df={min_df}). "
                f"This usually means your texts are too short, all stop-words, "
                f"or the wrong JSON key was used to load text. "
                f"Returning a zero topic matrix.",
                UserWarning,
            )
            return np.zeros((n_docs, num_topics), dtype=np.float64)
        raise

    lda = LatentDirichletAllocation(n_components=n_components, random_state=42)
    topic_probs = lda.fit_transform(counts)          # shape: (n_docs, n_components)

    # Pad columns to always return shape (n_docs, num_topics).
    if n_components < num_topics:
        padding = np.zeros((n_docs, num_topics - n_components), dtype=np.float64)
        topic_probs = np.concatenate([topic_probs, padding], axis=1)

    return topic_probs


def compute_dsm5_keyword_counts(texts: List[str]) -> Dict[str, List[int]]:
    counts: Dict[str, List[int]] = {f"dsm5_{k}": [] for k in DSM5_KEYWORDS}
    for text in texts:
        lowered = normalize_text(text)
        for k in DSM5_KEYWORDS:
            counts[f"dsm5_{k}"].append(lowered.count(k))
    return counts


def write_features_csv(
    output_path: str,
    row_ids: Iterable[int],
    sentiment: Dict[str, List[float]],
    lda: np.ndarray,
    dsm5: Dict[str, List[int]],
    labels: Optional[List[int]] = None,
) -> None:
    fieldnames = ["row_id"]
    if labels is not None:
        fieldnames.append("label")
    fieldnames += list(sentiment.keys())
    fieldnames += [f"lda_topic_{i}" for i in range(lda.shape[1])]
    fieldnames += list(dsm5.keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row_id in enumerate(row_ids):
            row: Dict[str, float] = {"row_id": row_id}
            if labels is not None:
                row["label"] = labels[i]
            for k, v in sentiment.items():
                row[k] = v[i]
            for j in range(lda.shape[1]):
                row[f"lda_topic_{j}"] = float(lda[i, j])
            for k, v in dsm5.items():
                row[k] = v[i]
            writer.writerow(row)


def write_embeddings_npz(output_path: str, embeddings: np.ndarray) -> None:
    np.savez_compressed(output_path, embeddings=embeddings)


def parse_args() -> Tuple[ExtractConfig, str, str, str]:
    parser = argparse.ArgumentParser(description="Extract text features from depression datasets.")
    parser.add_argument(
        "--dataset",
        choices=["reddit", "wu3d"],
        default="reddit",
        help="Dataset type: 'reddit' for reddit_depression_dataset.csv, 'wu3d' for depressed.json + normal.json",
    )
    parser.add_argument("--input", default="reddit_depression_dataset.csv", help="Path to CSV file (reddit) or base directory (wu3d)")
    parser.add_argument("--depressed_path", default="depressed.json", help="Path to depressed.json (WU3D dataset)")
    parser.add_argument("--normal_path", default="normal.json", help="Path to normal.json (WU3D dataset)")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--text_columns", default="title,body", help="CSV columns to use (comma-separated, ignored for JSON)")
    parser.add_argument("--text_key", default="text", help="JSON key containing text (WU3D dataset)")
    parser.add_argument(
        "--embedding_source",
        default="googlenews",
        choices=["googlenews", "bert", "both"],
        help="Embedding source: googlenews (default), bert, or both",
    )
    parser.add_argument("--bert_model", default="bert-base-uncased")
    parser.add_argument("--google_news_path", default="GoogleNews-vectors-negative300.bin", help="Path to GoogleNews vectors")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lda_topics", type=int, default=20)
    args = parser.parse_args()
    text_columns = tuple(c.strip() for c in args.text_columns.split(",") if c.strip())
    return (
        ExtractConfig(
            input_path=args.input,
            output_dir=args.output_dir,
            text_columns=text_columns,
            embedding_source=args.embedding_source,
            bert_model=args.bert_model,
            google_news_path=args.google_news_path,
            max_length=args.max_length,
            batch_size=args.batch_size,
            lda_topics=args.lda_topics,
        ),
        args.dataset,
        args.depressed_path,
        args.normal_path,
    )


def main() -> None:
    cfg, dataset_type, depressed_path, normal_path = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"Loading {dataset_type.upper()} dataset...")
    if dataset_type.lower() == "wu3d":
        texts, rows = read_texts(
            path=cfg.input_path,
            text_columns=cfg.text_columns,
            dataset_type="json",
            depressed_path=depressed_path,
            normal_path=normal_path,
            text_key="text",
        )
        print(f"Loaded {len(texts)} texts from WU3D dataset")
    else:
        texts, rows = read_texts(
            path=cfg.input_path,
            text_columns=cfg.text_columns,
            dataset_type="csv",
        )
        print(f"Loaded {len(texts)} texts from Reddit dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Computing text embeddings...")
    embeddings = compute_text_embeddings(cfg, texts, device)
    print("Computing sentiment scores...")
    sentiment = compute_sentiment(texts)
    print("Computing LDA topics...")
    lda = compute_lda_topics(texts, num_topics=cfg.lda_topics)
    print("Computing DSM-5 keyword counts...")
    dsm5 = compute_dsm5_keyword_counts(texts)

    embeddings_path = os.path.join(cfg.output_dir, f"{dataset_type}_embeddings.npz")
    features_path = os.path.join(cfg.output_dir, f"{dataset_type}_features.csv")

    print(f"Saving embeddings to {embeddings_path}...")
    write_embeddings_npz(embeddings_path, embeddings)
    print(f"Saving features to {features_path}...")
    labels = []
    for r in rows:
        if dataset_type.lower() == "wu3d":
            val = 1 if r.get("label") == "depressed" else 0
        else:
            try:
                val = int(r.get("label", 0))
            except (ValueError, TypeError):
                val = 0
        labels.append(val)

    write_features_csv(
        features_path,
        row_ids=range(len(texts)),
        sentiment=sentiment,
        lda=lda,
        dsm5=dsm5,
        labels=labels,
    )
    print("Done!")


if __name__ == "__main__":
    main()
