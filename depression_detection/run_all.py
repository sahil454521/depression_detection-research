"""Master runner - executes every layer of the depression detection system in order.

Layer execution order
---------------------
  Layer 0: Text Feature Extraction   (text_feature_extraction.py)
           -> sentiment, LDA topics, DSM-5 keywords, GoogleNews embeddings
           -> saves outputs/features.csv + outputs/embeddings.npz

  Layer 1: Dataset Construction      (multimodal_model.py)
           -> loads WU3D + Reddit + (optionally) DAIC-WOZ
           -> builds RealDepressionDataset with GoogleNews embeddings

  Layer 2: Model Initialisation      (multimodal_model.py)
           -> DepressionDetectionModel (5 encoders + fusion + prediction)

  Layer 3: Federated Learning        (federated_learning.py)
           -> FedAVG with differential privacy + secure aggregation
           -> splits dataset by source -> LocalNodes

  Layer 4: Validation                (validation_layer.py)
           -> F1 / AUC-ROC / MCC metric suite + fairness bias audit
           -> LODO cross-dataset validation (optional)

  Layer 5: Retrain Loop              (pipeline.py)
           -> if metrics below threshold -> re-run FL -> re-validate
           -> keeps best checkpoint by AUC-ROC

  Layer 6: Output Layer              (output_layer.py)
           -> PatientReport + ClinicalDashboard generation

  Layer 7: Visualisation             (visualize_data.py)
           -> sentiment / DSM-5 / LDA / embedding plots -> outputs/plots/

Usage
-----
    python run_all.py                         # default settings
    python run_all.py --skip_extraction       # skip Layer 0 if already done
    python run_all.py --skip_viz              # skip Layer 7
    python run_all.py --max_samples 500       # quick test run
    python run_all.py --help                  # full options
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(layer_num: int, title: str, char: str = "=") -> None:
    """Print a clearly visible layer banner."""
    width = 70
    print(f"\n{char * width}")
    print(f"  LAYER {layer_num}: {title}")
    print(f"{char * width}\n")


def _elapsed(start: float) -> str:
    secs = time.time() - start
    if secs < 60:
        return f"{secs:.1f}s"
    mins = int(secs // 60)
    return f"{mins}m {secs - mins * 60:.0f}s"


# ---------------------------------------------------------------------------
# Layer 0  —  Text Feature Extraction
# ---------------------------------------------------------------------------

def run_layer_0_extraction(args) -> None:
    """Extract text features from the raw datasets and save to outputs/."""
    _banner(0, "TEXT FEATURE EXTRACTION")

    from text_feature_extraction import (
        ExtractConfig,
        compute_dsm5_keyword_counts,
        compute_lda_topics,
        compute_sentiment,
        compute_text_embeddings,
        read_texts,
        write_embeddings_npz,
        write_features_csv,
    )
    import torch

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    datasets_to_extract = []

    # Reddit
    if os.path.exists(args.reddit_csv):
        datasets_to_extract.append({
            "name":    "reddit",
            "type":    "csv",
            "path":    args.reddit_csv,
            "prefix":  "reddit",
        })
    else:
        print(f"  SKIP Reddit — file not found: {args.reddit_csv}")

    # WU3D
    if os.path.exists(args.depressed_json) and os.path.exists(args.normal_json):
        datasets_to_extract.append({
            "name":    "wu3d",
            "type":    "json",
            "path":    args.depressed_json,
            "prefix":  "",  # saves as features.csv / embeddings.npz (legacy names)
        })
    else:
        print(f"  SKIP WU3D — depressed.json or normal.json not found")

    if not datasets_to_extract:
        print("  WARNING: No raw datasets found. Skipping extraction.")
        return

    cfg = ExtractConfig(
        input_path="",
        output_dir=args.output_dir,
        text_columns=("title", "body"),
        embedding_source="googlenews",
        bert_model="bert-base-uncased",
        google_news_path=args.google_news_bin,
        max_length=128,
        batch_size=16,
        lda_topics=20,
    )

    for ds in datasets_to_extract:
        t0 = time.time()
        name = ds["name"]
        prefix = ds["prefix"]
        tag = f"{prefix}_" if prefix else ""
        print(f"\n  --- Extracting: {name.upper()} ---")

        if ds["type"] == "json":
            texts, rows = read_texts(
                path=ds["path"],
                text_columns=cfg.text_columns,
                dataset_type="json",
                depressed_path=args.depressed_json,
                normal_path=args.normal_json,
                text_key=args.text_key,
            )
        else:
            texts, rows = read_texts(
                path=ds["path"],
                text_columns=cfg.text_columns,
                dataset_type="csv",
            )
        print(f"  Loaded {len(texts):,} texts")

        if len(texts) == 0:
            print(
                f"  WARNING: No texts were loaded for {name.upper()}. "
                f"Check that the files exist and contain the expected text key.\n"
                f"  For WU3D JSON files, the text key is currently set to "
                f"'tweet_content'. If your JSON uses a different key (e.g. "
                f"'text', 'content', 'post'), pass --text_key <key> on the "
                f"command line or inspect the JSON manually.\n"
                f"  Skipping {name.upper()} extraction."
            )
            continue

        print("  Computing embeddings ...")
        embeddings = compute_text_embeddings(cfg, texts, device)

        print("  Computing sentiment ...")
        sentiment = compute_sentiment(texts)

        print("  Computing LDA topics ...")
        lda = compute_lda_topics(texts, num_topics=cfg.lda_topics)

        print("  Computing DSM-5 keywords ...")
        dsm5 = compute_dsm5_keyword_counts(texts)

        emb_path = os.path.join(cfg.output_dir, f"{tag}embeddings.npz")
        feat_path = os.path.join(cfg.output_dir, f"{tag}features.csv")

        write_embeddings_npz(emb_path, embeddings)
        print(f"  Saved: {emb_path}")

        labels = []
        for r in rows:
            if ds["type"] == "json":
                val = 1 if r.get("label") == "depressed" else 0
            else:
                try:
                    val = int(r.get("label", 0))
                except (ValueError, TypeError):
                    val = 0
            labels.append(val)

        write_features_csv(feat_path, range(len(texts)), sentiment, lda, dsm5, labels=labels)
        print(f"  Saved: {feat_path}")

        print(f"  {name.upper()} extraction complete ({_elapsed(t0)})")

    print("\n  Layer 0 complete.")


# ---------------------------------------------------------------------------
# Layers 1-6  -  Pipeline (Dataset -> Model -> FL -> Validation -> Output)
# ---------------------------------------------------------------------------

def run_layers_1_to_6_pipeline(args) -> dict:
    """Run the core pipeline: dataset -> model -> FL -> validation -> output."""
    _banner(1, "DATASET CONSTRUCTION")
    _banner(2, "MODEL INITIALISATION")
    _banner(3, "FEDERATED LEARNING")
    _banner(4, "VALIDATION")
    _banner(5, "RETRAIN LOOP")
    _banner(6, "OUTPUT LAYER")

    # The above banners are printed for orientation, but the actual logging
    # comes from within DepressionDetectionPipeline.run() which prints
    # detailed progress for each phase.
    print("  Starting DepressionDetectionPipeline.run() ...\n")

    from pipeline import DepressionDetectionPipeline, PipelineConfig

    pcfg = PipelineConfig(
        # Data
        depressed_json=args.depressed_json,
        normal_json=args.normal_json,
        reddit_csv=args.reddit_csv,
        google_news_bin=args.google_news_bin,
        daic_woz_raw_dir=args.daic_woz_raw_dir,
        daic_woz_labels=args.daic_woz_labels,
        max_samples=args.max_samples,
        val_split=args.val_split,
        batch_size=args.batch_size,
        seed=args.seed,
        # Model
        hidden_dim=args.hidden_dim,
        text_embed_dim=args.text_embed_dim,
        # XAI
        run_xai=not args.no_xai,
        ig_steps=args.ig_steps,
        cf_steps=args.cf_steps,
        # Federated Learning
        fl_rounds=args.fl_rounds,
        fl_local_epochs=args.fl_local_epochs,
        fl_local_lr=args.fl_local_lr,
        fl_global_lr=args.fl_global_lr,
        fl_epsilon=args.fl_epsilon,
        fl_delta=args.fl_delta,
        fl_clip_norm=float(args.fl_clip_norm) if args.fl_clip_norm != "adaptive" else "adaptive",
        fl_sigma=args.fl_sigma,
        fl_aggregation=args.fl_aggregation,
        fl_secure_aggregation=not args.no_secure_agg,
        # Retrain
        max_retrain_attempts=args.max_retrain,
        # Validation
        target_f1=args.target_f1,
        target_auc_roc=args.target_auc,
        target_mcc=args.target_mcc,
        max_eq_odds_gap=args.max_eq_odds_gap,
        run_lodo=not args.no_lodo,
        lodo_train_epochs=args.lodo_epochs,
        lodo_lr=args.lodo_lr,
        # Output
        output_dir=args.output_dir,
        save_reports=not args.no_save,
        print_reports=not args.quiet,
    )

    pipeline = DepressionDetectionPipeline(pcfg)
    result = pipeline.run()

    print("\n  Layers 1–6 complete.")
    return result


# ---------------------------------------------------------------------------
# Layer 7  —  Visualisation
# ---------------------------------------------------------------------------

def run_layer_7_visualisation(args) -> None:
    """Generate plots from the extracted features and embeddings."""
    _banner(7, "VISUALISATION")

    from visualize_data import (
        _apply_style,
        load_features,
        load_embeddings,
        plot_label_distribution,
        plot_sentiment_distribution,
        plot_sentiment_boxplot,
        plot_sentiment_vs_dsm5,
        plot_dsm5_prevalence,
        plot_dsm5_correlation,
        plot_lda_heatmap,
        plot_embedding_2d,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_style()

    plot_dir = Path(args.output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Determine which feature sets exist
    sources = []
    feat_wu3d = os.path.join(args.output_dir, "features.csv")
    feat_reddit = os.path.join(args.output_dir, "reddit_features.csv")

    if os.path.exists(feat_wu3d):
        sources.append({
            "name": "WU3D",
            "tag":  "wu3d",
            "features":   feat_wu3d,
            "embeddings": os.path.join(args.output_dir, "embeddings.npz"),
            "labels":     args.reddit_csv,
        })
    if os.path.exists(feat_reddit):
        sources.append({
            "name": "Reddit",
            "tag":  "reddit",
            "features":   feat_reddit,
            "embeddings": os.path.join(args.output_dir, "reddit_embeddings.npz"),
            "labels":     args.reddit_csv,
        })

    if not sources:
        print("  No feature CSV files found in outputs/. Skipping visualisation.")
        return

    viz_max = args.viz_max_rows

    for src in sources:
        name = src["name"]
        tag = src["tag"]
        prefix = f"[{name}] "
        print(f"\n  --- Plotting: {name} ---")

        df = load_features(src["features"], src["labels"], max_rows=viz_max)

        # Overview figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"{name} Dataset — Feature Overview", fontsize=18,
                     fontweight="bold", y=0.98)
        plot_label_distribution(df, axes[0, 0], prefix)
        plot_sentiment_distribution(df, axes[0, 1], prefix)
        plot_sentiment_boxplot(df, axes[1, 0], prefix)
        plot_sentiment_vs_dsm5(df, axes[1, 1], prefix)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        p = plot_dir / f"{tag}_overview.png"
        fig.savefig(p)
        plt.close(fig)
        print(f"  Saved: {p}")

        # DSM-5 figure
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f"{name} Dataset — DSM-5 Analysis", fontsize=18,
                     fontweight="bold", y=0.98)
        plot_dsm5_prevalence(df, axes[0], prefix)
        plot_dsm5_correlation(df, axes[1], prefix)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        p = plot_dir / f"{tag}_dsm5.png"
        fig.savefig(p)
        plt.close(fig)
        print(f"  Saved: {p}")

        # LDA figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        plot_lda_heatmap(df, ax, prefix)
        fig.tight_layout()
        p = plot_dir / f"{tag}_lda.png"
        fig.savefig(p)
        plt.close(fig)
        print(f"  Saved: {p}")

        # Embedding projection (skip if --no_embeddings or file missing)
        if not args.no_embeddings and os.path.exists(src["embeddings"]):
            embs = load_embeddings(src["embeddings"], max_rows=viz_max)
            if embs is not None:
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                lbl_arr = df["label"].values[: len(embs)]
                plot_embedding_2d(embs, lbl_arr, ax, method=args.embed_method,
                                  title_prefix=prefix, sample_n=5000)
                fig.tight_layout()
                p = plot_dir / f"{tag}_embeddings_{args.embed_method}.png"
                fig.savefig(p)
                plt.close(fig)
                print(f"  Saved: {p}")

    print(f"\n  All plots saved to: {plot_dir.resolve()}")
    print("  Layer 7 complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the full depression detection system layer by layer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # ---- Skip flags ----
    skip = p.add_argument_group("Layer control")
    skip.add_argument("--skip_extraction", action="store_true",
                      help="Skip Layer 0 (text feature extraction). "
                           "Use if outputs/features.csv already exists.")
    skip.add_argument("--skip_pipeline", action="store_true",
                      help="Skip Layers 1-6 (model + FL + validation + output).")
    skip.add_argument("--skip_viz", action="store_true",
                      help="Skip Layer 7 (visualisation).")

    # ---- Data paths ----
    data = p.add_argument_group("Data paths")
    data.add_argument("--depressed_json",  default=os.path.join(base_dir, "depressed.json"))
    data.add_argument("--normal_json",     default=os.path.join(base_dir, "normal.json"))
    data.add_argument("--reddit_csv",      default=os.path.join(base_dir, "reddit_depression_dataset.csv"))
    data.add_argument("--google_news_bin", default=os.path.join(base_dir, "GoogleNews-vectors-negative300.bin"))
    data.add_argument("--text_key",        default="text",
                      help="JSON key holding the text body in WU3D files "
                           "(e.g. 'text', 'content', 'post', 'tweet_content'). "
                           "Default: text")
    data.add_argument("--daic_woz_raw_dir", default="",
                      help="DAIC-WOZ raw dir (empty = disabled)")
    data.add_argument("--daic_woz_labels", default=os.path.join(base_dir, "daic-woz", "labels.csv"))
    data.add_argument("--output_dir",      default=os.path.join(base_dir, "outputs"))

    # ---- Dataset / training ----
    train = p.add_argument_group("Dataset and training")
    train.add_argument("--max_samples",  type=int,   default=2000,
                       help="Cap total samples for pipeline (0=all). Default 2000.")
    train.add_argument("--val_split",    type=float, default=0.15)
    train.add_argument("--batch_size",   type=int,   default=16)
    train.add_argument("--seed",         type=int,   default=42)

    # ---- Model ----
    model = p.add_argument_group("Model")
    model.add_argument("--hidden_dim",     type=int, default=256)
    model.add_argument("--text_embed_dim", type=int, default=300)

    # ---- XAI ----
    xai = p.add_argument_group("XAI")
    xai.add_argument("--no_xai",     action="store_true",
                     help="Disable XAI explanations.")
    xai.add_argument("--ig_steps",   type=int, default=30)
    xai.add_argument("--cf_steps",   type=int, default=80)

    # ---- Federated Learning ----
    fl = p.add_argument_group("Federated Learning")
    fl.add_argument("--fl_rounds",       type=int,   default=5)
    fl.add_argument("--fl_local_epochs", type=int,   default=2)
    fl.add_argument("--fl_local_lr",     type=float, default=1e-3)
    fl.add_argument("--fl_global_lr",    type=float, default=1.0)
    fl.add_argument("--fl_epsilon",      type=float, default=None)
    fl.add_argument("--fl_delta",        type=float, default=1e-5)
    fl.add_argument("--fl_clip_norm",    type=str,   default="5.0")
    fl.add_argument("--fl_sigma",        type=float, default=1.25)
    fl.add_argument("--fl_aggregation",  default="fedavg")
    fl.add_argument("--no_secure_agg",   action="store_true")

    # ---- Retrain ----
    retrain = p.add_argument_group("Retrain loop")
    retrain.add_argument("--max_retrain", type=int, default=3)

    # ---- Validation ----
    val = p.add_argument_group("Validation")
    val.add_argument("--target_f1",       type=float, default=0.65)
    val.add_argument("--target_auc",      type=float, default=0.70)
    val.add_argument("--target_mcc",      type=float, default=0.20)
    val.add_argument("--max_eq_odds_gap", type=float, default=0.15)
    val.add_argument("--no_lodo",         action="store_true",
                     help="Skip LODO validation (faster).")
    val.add_argument("--lodo_epochs",     type=int,   default=3)
    val.add_argument("--lodo_lr",         type=float, default=1e-3)

    # ---- Visualisation ----
    viz = p.add_argument_group("Visualisation")
    viz.add_argument("--viz_max_rows",  type=int,   default=100000,
                     help="Max rows for visualisation (0=all). Default 100k.")
    viz.add_argument("--embed_method",  choices=["pca", "tsne"], default="pca")
    viz.add_argument("--no_embeddings", action="store_true",
                     help="Skip embedding plots (faster).")

    # ---- Output control ----
    out = p.add_argument_group("Output control")
    out.add_argument("--no_save", action="store_true",
                     help="Don't save model checkpoints or report JSON.")
    out.add_argument("--quiet",   action="store_true",
                     help="Suppress detailed report printing.")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    start_time = time.time()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n{'#' * 70}")
    print(f"  DEPRESSION DETECTION SYSTEM — FULL RUN")
    print(f"  Started: {ts}")
    print(f"{'#' * 70}")

    # ------------------------------------------------------------------
    # Layer 0: Text Feature Extraction
    # ------------------------------------------------------------------
    if not args.skip_extraction:
        t0 = time.time()
        run_layer_0_extraction(args)
        print(f"\n  [Time]  Layer 0 time: {_elapsed(t0)}")
    else:
        print("\n  [SKIP] Layer 0: Text Feature Extraction (--skip_extraction)")

    # ------------------------------------------------------------------
    # Layers 1-6: Pipeline (Dataset -> Model -> FL -> Validation -> Output)
    # ------------------------------------------------------------------
    result = None
    if not args.skip_pipeline:
        t1 = time.time()
        result = run_layers_1_to_6_pipeline(args)
        print(f"\n  [Time]  Layers 1-6 time: {_elapsed(t1)}")
    else:
        print("\n  [SKIP] Layers 1-6: Pipeline (--skip_pipeline)")

    # ------------------------------------------------------------------
    # Layer 7: Visualisation
    # ------------------------------------------------------------------
    if not args.skip_viz:
        t7 = time.time()
        run_layer_7_visualisation(args)
        print(f"\n  [Time]  Layer 7 time: {_elapsed(t7)}")
    else:
        print("\n  [SKIP] Layer 7: Visualisation (--skip_viz)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = _elapsed(start_time)
    print(f"\n{'#' * 70}")
    print(f"  ALL LAYERS COMPLETE")
    print(f"  Total time: {total}")
    print(f"{'#' * 70}")

    if result is not None:
        vr = result.get("validation_report")
        if vr is not None:
            m = vr.overall_metrics
            print(f"\n  Final metrics:")
            print(f"    F1:       {m.f1:.4f}")
            print(f"    AUC-ROC:  {m.auc_roc:.4f}")
            print(f"    MCC:      {m.mcc:.4f}")
            print(f"    Accuracy: {m.accuracy:.4f}")
        print(f"  Reports:    {len(result.get('patient_reports', []))} patient reports generated")
        print(f"  Outputs in: {os.path.abspath(args.output_dir)}")

    print()


if __name__ == "__main__":
    main()
