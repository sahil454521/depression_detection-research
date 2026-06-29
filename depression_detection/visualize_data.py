"""Visualise extracted text features and embeddings from the outputs folder.

Generates publication-quality plots from:
  - outputs/features.csv          (sentiment, LDA topics, DSM-5 keyword counts)
  - outputs/embeddings.npz        (300-dim GoogleNews word2vec embeddings)
  - reddit_depression_dataset.csv (labels: 0=normal, 1=depressed)

Usage
-----
    python visualize_data.py                       # default (WU3D features)
    python visualize_data.py --source reddit       # Reddit features
    python visualize_data.py --source both         # overlay both datasets
    python visualize_data.py --max_rows 50000      # cap rows loaded (faster)

All plots are saved to  outputs/plots/
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works on headless servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# Optional: t-SNE / PCA for embedding visualisation
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

# A muted, research-friendly palette
COLORS = {
    "depressed":  "#E74C3C",   # warm red
    "normal":     "#2ECC71",   # green
    "accent":     "#3498DB",   # blue
    "muted":      "#95A5A6",   # grey
    "bg":         "#FAFAFA",
    "grid":       "#EDEDED",
}

LABEL_MAP = {0: "Normal", 1: "Depressed"}


def _apply_style():
    """Apply a clean, modern matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor":  COLORS["bg"],
        "axes.facecolor":    COLORS["bg"],
        "axes.grid":         True,
        "grid.color":        COLORS["grid"],
        "grid.linewidth":    0.6,
        "font.family":       "sans-serif",
        "font.size":         11,
        "axes.titlesize":    14,
        "axes.titleweight":  "bold",
        "axes.labelsize":    12,
        "legend.fontsize":   10,
        "figure.dpi":        150,
        "savefig.dpi":       150,
        "savefig.bbox":      "tight",
    })


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

DSM5_COLS = [
    "dsm5_depressed", "dsm5_hopeless", "dsm5_worthless", "dsm5_suicidal",
    "dsm5_anhedonia", "dsm5_fatigue", "dsm5_insomnia", "dsm5_hypersomnia",
    "dsm5_appetite", "dsm5_concentration", "dsm5_guilt", "dsm5_psychomotor",
    "dsm5_irritable", "dsm5_anxiety", "dsm5_panic", "dsm5_tearful",
    "dsm5_lonely", "dsm5_withdrawn", "dsm5_self-harm", "dsm5_no motivation",
]

SENTIMENT_COLS = ["sent_pos", "sent_neg", "sent_neu", "sent_compound"]

LDA_COLS = [f"lda_topic_{i}" for i in range(20)]


def load_features(
    features_csv: str,
    labels_csv: str | None,
    max_rows: int = 0,
) -> pd.DataFrame:
    """Load features.csv and optionally join labels from source dataset."""
    print(f"  Loading features: {features_csv}")
    nrows = max_rows if max_rows > 0 else None
    df = pd.read_csv(features_csv, nrows=nrows)
    print(f"    Rows loaded: {len(df):,}")

    # Prioritize embedded labels
    if "label" in df.columns:
        dep = (df["label"] == 1).sum()
        print(f"    Embedded labels found. Depressed: {dep:,}  |  Normal: {len(df) - dep:,}")
        return df

    # Try to join labels from the source CSV
    if labels_csv and os.path.exists(labels_csv):
        print(f"  Joining labels from: {labels_csv}")
        label_df = pd.read_csv(
            labels_csv,
            usecols=lambda c: c in {"label", ""},
            nrows=nrows,
        )
        # The source CSV may have an unnamed index column
        if "label" in label_df.columns:
            # Align by row position (features.csv row_id matches source row order)
            df["label"] = label_df["label"].values[: len(df)]
            dep = (df["label"] == 1).sum()
            print(f"    Depressed: {dep:,}  |  Normal: {len(df) - dep:,}")
    else:
        warnings.warn("No label source found — plots won't colour by class.", stacklevel=2)
        df["label"] = -1  # unknown

    return df


def load_embeddings(npz_path: str, max_rows: int = 0) -> np.ndarray | None:
    """Load embeddings.npz, returning stacked array [N, 300]."""
    if not os.path.exists(npz_path):
        print(f"  Embeddings file not found: {npz_path}")
        return None

    print(f"  Loading embeddings: {npz_path}  (this may take a minute for large files)")
    data = np.load(npz_path)
    keys = sorted(data.files, key=lambda k: int(k) if k.isdigit() else k)
    if max_rows > 0:
        keys = keys[:max_rows]
    embs = np.stack([data[k] for k in keys], axis=0)
    if embs.ndim > 2 and embs.shape[0] == 1:
        embs = embs[0]
    print(f"    Embeddings shape: {embs.shape}")
    return embs


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_sentiment_distribution(df: pd.DataFrame, ax: plt.Axes, title_prefix: str = ""):
    """Histogram of sentiment compound scores, coloured by label."""
    has_labels = (df["label"] >= 0).any()

    if has_labels:
        for label_val, label_name in LABEL_MAP.items():
            subset = df[df["label"] == label_val]["sent_compound"]
            color = COLORS["depressed"] if label_val == 1 else COLORS["normal"]
            ax.hist(subset, bins=60, alpha=0.55, color=color, label=label_name,
                    edgecolor="white", linewidth=0.3)
        ax.legend()
    else:
        ax.hist(df["sent_compound"], bins=60, alpha=0.7, color=COLORS["accent"],
                edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Compound Sentiment Score")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{title_prefix}Sentiment Compound Distribution")


def plot_sentiment_boxplot(df: pd.DataFrame, ax: plt.Axes, title_prefix: str = ""):
    """Box plot of sentiment components, split by label."""
    has_labels = (df["label"] >= 0).any()

    if has_labels:
        melted = df[SENTIMENT_COLS + ["label"]].melt(id_vars="label", var_name="component", value_name="score")
        melted["component"] = melted["component"].str.replace("sent_", "")
        labels = sorted(melted["label"].unique())
        components = sorted(melted["component"].unique())

        positions = []
        box_data = []
        box_colors = []
        tick_positions = []
        tick_labels_list = []

        for ci, comp in enumerate(components):
            for li, lbl in enumerate(labels):
                vals = melted[(melted["component"] == comp) & (melted["label"] == lbl)]["score"]
                pos = ci * (len(labels) + 1) + li
                box_data.append(vals.values)
                positions.append(pos)
                box_colors.append(COLORS["depressed"] if lbl == 1 else COLORS["normal"])
            tick_positions.append(ci * (len(labels) + 1) + 0.5)
            tick_labels_list.append(comp.capitalize())

        bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                        showfliers=False, medianprops={"color": "black", "linewidth": 1.2})
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels_list)
        # Legend
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor=COLORS["normal"], alpha=0.6, label="Normal"),
            Patch(facecolor=COLORS["depressed"], alpha=0.6, label="Depressed"),
        ])
    else:
        df[SENTIMENT_COLS].boxplot(ax=ax, patch_artist=True, showfliers=False)

    ax.set_ylabel("Score")
    ax.set_title(f"{title_prefix}Sentiment Components")


def plot_dsm5_prevalence(df: pd.DataFrame, ax: plt.Axes, title_prefix: str = ""):
    """Horizontal bar chart: fraction of samples with each DSM-5 keyword > 0."""
    has_labels = (df["label"] >= 0).any()
    short_names = [c.replace("dsm5_", "").replace("_", " ").title() for c in DSM5_COLS]

    if has_labels:
        dep_df = df[df["label"] == 1]
        nor_df = df[df["label"] == 0]

        dep_prev = [(dep_df[c] > 0).mean() for c in DSM5_COLS]
        nor_prev = [(nor_df[c] > 0).mean() for c in DSM5_COLS]

        y = np.arange(len(DSM5_COLS))
        h = 0.35
        ax.barh(y + h / 2, dep_prev, h, color=COLORS["depressed"], alpha=0.7, label="Depressed")
        ax.barh(y - h / 2, nor_prev, h, color=COLORS["normal"], alpha=0.7, label="Normal")
        ax.legend()
    else:
        prev = [(df[c] > 0).mean() for c in DSM5_COLS]
        y = np.arange(len(DSM5_COLS))
        ax.barh(y, prev, color=COLORS["accent"], alpha=0.7)

    ax.set_yticks(np.arange(len(DSM5_COLS)))
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel("Prevalence (fraction of samples)")
    ax.set_title(f"{title_prefix}DSM-5 Keyword Prevalence")
    ax.invert_yaxis()


def plot_lda_heatmap(df: pd.DataFrame, ax: plt.Axes, title_prefix: str = ""):
    """Heatmap of mean LDA topic probabilities per label."""
    has_labels = (df["label"] >= 0).any()

    if has_labels:
        grouped = df.groupby("label")[LDA_COLS].mean()
        data = grouped.values  # shape [num_labels, 20]
        ylabels = [LABEL_MAP.get(l, str(l)) for l in grouped.index]
    else:
        data = df[LDA_COLS].mean().values.reshape(1, -1)
        ylabels = ["All"]

    cmap = LinearSegmentedColormap.from_list("custom", ["#FFFFFF", COLORS["accent"], "#1A1A2E"])
    im = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(20))
    ax.set_xticklabels([f"T{i}" for i in range(20)], fontsize=8, rotation=45)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("LDA Topic")
    ax.set_title(f"{title_prefix}Mean LDA Topic Distribution")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)


def plot_dsm5_correlation(df: pd.DataFrame, ax: plt.Axes, title_prefix: str = ""):
    """Correlation heatmap of DSM-5 keyword counts."""
    corr = df[DSM5_COLS].corr()
    short_names = [c.replace("dsm5_", "").replace("_", " ") for c in DSM5_COLS]

    cmap = LinearSegmentedColormap.from_list(
        "rdbu", [COLORS["normal"], "#FFFFFF", COLORS["depressed"]]
    )
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, fontsize=7, rotation=90)
    ax.set_yticks(range(len(short_names)))
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_title(f"{title_prefix}DSM-5 Keyword Correlation")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)


def plot_label_distribution(df: pd.DataFrame, ax: plt.Axes, title_prefix: str = ""):
    """Pie chart of label distribution."""
    if (df["label"] < 0).all():
        ax.text(0.5, 0.5, "No labels available", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color=COLORS["muted"])
        ax.set_title(f"{title_prefix}Label Distribution")
        return

    counts = df["label"].value_counts().sort_index()
    labels = [LABEL_MAP.get(i, str(i)) for i in counts.index]
    colors = [COLORS["normal"] if i == 0 else COLORS["depressed"] for i in counts.index]

    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")
    ax.set_title(f"{title_prefix}Label Distribution (N={len(df):,})")


def plot_embedding_2d(
    embeddings: np.ndarray,
    labels: np.ndarray,
    ax: plt.Axes,
    method: str = "pca",
    title_prefix: str = "",
    sample_n: int = 5000,
):
    """2-D scatter plot of embeddings via PCA or t-SNE."""
    if not HAS_SKLEARN:
        ax.text(0.5, 0.5, "scikit-learn not installed\n(pip install scikit-learn)",
                transform=ax.transAxes, ha="center", va="center", fontsize=12,
                color=COLORS["muted"])
        ax.set_title(f"{title_prefix}Embedding Projection")
        return

    if getattr(embeddings, "ndim", 2) > 2:
        if embeddings.shape[0] == 1:
            embeddings = embeddings[0]
        else:
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    if getattr(labels, "ndim", 1) > 1:
        labels = labels.ravel()
    min_len = min(len(embeddings), len(labels))
    embeddings = embeddings[:min_len]
    labels = labels[:min_len]
    n = min_len
    if n > sample_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, sample_n, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    print(f"    Projecting {len(embeddings):,} embeddings with {method.upper()} ...")
    if method == "tsne":
        proj = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=500)
    else:
        proj = PCA(n_components=2, random_state=42)
    coords = proj.fit_transform(embeddings)

    has_labels = (labels >= 0).any()
    if has_labels:
        for label_val, label_name in LABEL_MAP.items():
            mask = labels == label_val
            color = COLORS["depressed"] if label_val == 1 else COLORS["normal"]
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=color, s=4, alpha=0.35, label=label_name, rasterized=True)
        ax.legend(markerscale=4, framealpha=0.8)
    else:
        ax.scatter(coords[:, 0], coords[:, 1],
                   c=COLORS["accent"], s=4, alpha=0.35, rasterized=True)

    tag = method.upper()
    if method == "pca":
        tag += f" ({proj.explained_variance_ratio_[:2].sum():.1%} var)"
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title(f"{title_prefix}Embedding Projection ({tag})")


def plot_sentiment_vs_dsm5(df: pd.DataFrame, ax: plt.Axes, title_prefix: str = ""):
    """Scatter: compound sentiment vs total DSM-5 keyword count."""
    df = df.copy()
    df["dsm5_total"] = df[DSM5_COLS].sum(axis=1)
    has_labels = (df["label"] >= 0).any()

    if has_labels:
        for label_val, label_name in LABEL_MAP.items():
            subset = df[df["label"] == label_val]
            color = COLORS["depressed"] if label_val == 1 else COLORS["normal"]
            ax.scatter(subset["sent_compound"], subset["dsm5_total"],
                       c=color, s=6, alpha=0.25, label=label_name, rasterized=True)
        ax.legend(markerscale=4, framealpha=0.8)
    else:
        ax.scatter(df["sent_compound"], df["dsm5_total"],
                   c=COLORS["accent"], s=6, alpha=0.25, rasterized=True)

    ax.set_xlabel("Compound Sentiment")
    ax.set_ylabel("Total DSM-5 Keyword Count")
    ax.set_title(f"{title_prefix}Sentiment vs DSM-5 Keywords")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualise extracted text features.")
    parser.add_argument("--source", choices=["wu3d", "reddit", "both"], default="wu3d",
                        help="Which feature set to plot (default: wu3d).")
    parser.add_argument("--max_rows", type=int, default=100000,
                        help="Max rows to load from CSV (0=all). Default 100k for speed.")
    parser.add_argument("--embed_method", choices=["pca", "tsne"], default="pca",
                        help="Projection method for embeddings (default: pca).")
    parser.add_argument("--no_embeddings", action="store_true",
                        help="Skip embedding visualisation (much faster).")
    parser.add_argument("--output_dir", default="outputs/plots",
                        help="Directory to save plots.")
    args = parser.parse_args()

    _apply_style()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Determine file paths ----
    sources = []
    if args.source in ("wu3d", "both"):
        sources.append({
            "name":       "WU3D",
            "features":   "outputs/features.csv",
            "embeddings": "outputs/embeddings.npz",
            "labels":     "reddit_depression_dataset.csv",  # same extraction order
        })
    if args.source in ("reddit", "both"):
        sources.append({
            "name":       "Reddit",
            "features":   "outputs/reddit_features.csv",
            "embeddings": "outputs/reddit_embeddings.npz",
            "labels":     "reddit_depression_dataset.csv",
        })

    for src in sources:
        name = src["name"]
        prefix = f"[{name}] "
        tag = name.lower()

        print(f"\n{'=' * 60}")
        print(f"  Processing: {name}")
        print(f"{'=' * 60}")

        if not os.path.exists(src["features"]):
            print(f"  SKIP — {src['features']} not found.")
            continue

        df = load_features(src["features"], src["labels"], max_rows=args.max_rows)

        # ---- Figure 1: Overview dashboard (2×2) ----
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"{name} Dataset — Feature Overview", fontsize=18, fontweight="bold", y=0.98)

        plot_label_distribution(df, axes[0, 0], prefix)
        plot_sentiment_distribution(df, axes[0, 1], prefix)
        plot_sentiment_boxplot(df, axes[1, 0], prefix)
        plot_sentiment_vs_dsm5(df, axes[1, 1], prefix)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        path1 = out_dir / f"{tag}_overview.png"
        fig.savefig(path1)
        plt.close(fig)
        print(f"  Saved: {path1}")

        # ---- Figure 2: DSM-5 analysis (1×2) ----
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f"{name} Dataset — DSM-5 Analysis", fontsize=18, fontweight="bold", y=0.98)

        plot_dsm5_prevalence(df, axes[0], prefix)
        plot_dsm5_correlation(df, axes[1], prefix)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        path2 = out_dir / f"{tag}_dsm5.png"
        fig.savefig(path2)
        plt.close(fig)
        print(f"  Saved: {path2}")

        # ---- Figure 3: LDA topics ----
        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        plot_lda_heatmap(df, ax, prefix)
        fig.tight_layout()
        path3 = out_dir / f"{tag}_lda.png"
        fig.savefig(path3)
        plt.close(fig)
        print(f"  Saved: {path3}")

        # ---- Figure 4: Embedding projection ----
        if not args.no_embeddings:
            embs = load_embeddings(src["embeddings"], max_rows=args.max_rows)
            if embs is not None:
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                lbl_arr = df["label"].values[: len(embs)]
                plot_embedding_2d(embs, lbl_arr, ax, method=args.embed_method,
                                  title_prefix=prefix, sample_n=5000)
                fig.tight_layout()
                path4 = out_dir / f"{tag}_embeddings_{args.embed_method}.png"
                fig.savefig(path4)
                plt.close(fig)
                print(f"  Saved: {path4}")

    print(f"\nAll plots saved to: {out_dir.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
