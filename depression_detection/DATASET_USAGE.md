# Dataset Usage Guide

The `text_feature_extraction.py` script now supports both Reddit depression dataset (CSV) and WU3D dataset (JSON).

## Reddit Dataset (Default)

Extract features using the Reddit depression dataset with GoogleNews vectors:

```bash
python text_feature_extraction.py \
  --dataset reddit \
  --input reddit_depression_dataset.csv \
  --text_columns title,body \
  --embedding_source googlenews \
  --google_news_path GoogleNews-vectors-negative300.bin \
  --output_dir outputs
```

Output files:
- `outputs/reddit_embeddings.npz` - text embeddings
- `outputs/reddit_features.csv` - sentiment, LDA topics, DSM-5 keyword counts

## WU3D Dataset (Depressed + Normal)

Extract features from WU3D dataset with JSON format (depressed.json + normal.json):

```bash
python text_feature_extraction.py \
  --dataset wu3d \
  --depressed_path depressed.json \
  --normal_path normal.json \
  --text_key text \
  --embedding_source googlenews \
  --google_news_path GoogleNews-vectors-negative300.bin \
  --output_dir outputs
```

Output files:
- `outputs/wu3d_embeddings.npz` - text embeddings
- `outputs/wu3d_features.csv` - sentiment, LDA topics, DSM-5 keyword counts (includes label column)

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `reddit` | Dataset type: `reddit` (CSV) or `wu3d` (JSON) |
| `--input` | `reddit_depression_dataset.csv` | Path to CSV file (reddit) |
| `--depressed_path` | `depressed.json` | Path to depressed.json (wu3d) |
| `--normal_path` | `normal.json` | Path to normal.json (wu3d) |
| `--text_columns` | `title,body` | CSV columns to extract (reddit only) |
| `--text_key` | `text` | JSON key containing text (wu3d only) |
| `--embedding_source` | `googlenews` | Embedding: `googlenews`, `bert`, or `both` |
| `--google_news_path` | `GoogleNews-vectors-negative300.bin` | Path to GoogleNews vectors |
| `--embedding_source googlenews --google_news_path GoogleNews-vectors-negative300.bin` | | Use GoogleNews vectors (default & recommended) |
| `--output_dir` | `outputs` | Output directory |
| `--lda_topics` | `20` | Number of LDA topics |

## Features Extracted

All datasets extract:
- **Text Embeddings**: 300-dim vectors (GoogleNews) or BERT embeddings
- **Sentiment**: positive, negative, neutral, compound scores (VADER)
- **LDA Topics**: topic probabilities across all documents
- **DSM-5 Symptoms**: keyword counts for depression indicators

## Supported JSON Format for WU3D

### Format 1: List of objects
```json
[
  {"text": "I feel depressed...", "other_field": "..."},
  {"text": "Another text...", ...}
]
```

### Format 2: Dictionary of objects
```json
{
  "doc_1": {"text": "I feel depressed...", ...},
  "doc_2": {"text": "Another text...", ...}
}
```

Both formats automatically receive dataset labels:
- `depressed.json` → `label=depressed`
- `normal.json` → `label=normal`
