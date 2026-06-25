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
from typing import Dict, Iterable, List, Tuple

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

    for label, path in [("depressed", depressed_path), ("normal", normal_path)]:
        if not os.path.exists(path):
            warnings.warn(f"File not found: {path}", UserWarning)
            continue

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        text = item.get(text_key, "")
                    else:
                        text = str(item)
                    if text.strip():
                        texts.append(text)
                        rows.append({"text": text, "label": label})
            elif isinstance(data, dict):
                for key, item in data.items():
                    if isinstance(item, dict):
                        text = item.get(text_key, "")
                    else:
                        text = str(item)
                    if text.strip():
                        texts.append(text)
                        rows.append({"text": text, "label": label, "id": key})

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
    normalized_texts = [normalize_text(text) for text in texts]
    vectorizer = CountVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
    )
    counts = vectorizer.fit_transform(normalized_texts)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    return lda.fit_transform(counts)


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
) -> None:
    fieldnames = ["row_id"] + list(sentiment.keys())
    fieldnames += [f"lda_topic_{i}" for i in range(lda.shape[1])]
    fieldnames += list(dsm5.keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row_id in enumerate(row_ids):
            row: Dict[str, float] = {"row_id": row_id}
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
    write_features_csv(
        features_path,
        row_ids=range(len(texts)),
        sentiment=sentiment,
        lda=lda,
        dsm5=dsm5,
    )
    print("Done!")


if __name__ == "__main__":
    main()
