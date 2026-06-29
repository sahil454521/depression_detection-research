"""Main pipeline orchestrator for the multimodal depression detection system.

Full end-to-end pipeline:

    Dataset (WU3D + Reddit + GoogleNews)
        |
    [Model + XAI]           -- DepressionDetectionModel.forward(explain=True)
        |
    [Output Layer]          -- PatientReport + ClinicalDashboard
        |
    [Federated Learning]    -- FedAVG + DP + Secure Aggregation
        |
    [Validation Layer]  <-- retrain loop begins here
        |
        +--- F1/AUC/MCC below threshold? ------+
        |                                       |
        v  NO (satisfactory)                    v  YES
    Save checkpoint                       Back to FL training
    Generate final reports                (up to max_retrain_attempts)

Run with
--------
    python pipeline.py --help
    python pipeline.py --max_samples 1000 --fl_rounds 3 --fl_epsilon 1.0

or call PipelineRunner programmatically from another script.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader

# ---- project imports -------------------------------------------------------
from multimodal_model import (
    DatasetConfig,
    DepressionDetectionModel,
    RealDepressionDataset,
    collate_batch,
)
from prediction import PredictionConfig
from xai_module import XAIConfig, ExplanationOutput, format_explanations
from output_layer import OutputLayer, PatientReport, ClinicalDashboard
from federated_learning import FLConfig, FederatedLearningLayer
from validation_layer import (
    ValidationConfig,
    ValidationLayer,
    ValidationReport,
    format_validation_report,
)


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Unified configuration for the end-to-end pipeline."""

    # ---- Data ----
    depressed_json:   str = "depressed.json"
    normal_json:      str = "normal.json"
    reddit_csv:       str = "reddit_depression_dataset.csv"
    google_news_bin:  str = "GoogleNews-vectors-negative300.bin"
    # DAIC-WOZ — set daic_woz_raw_dir to the folder containing {pid}_P/ sub-dirs
    daic_woz_raw_dir: str = ""                     # empty = disabled
    daic_woz_labels:  str = "daic-woz/labels.csv"
    max_samples:      int = 2000
    val_split:        float = 0.15
    batch_size:       int = 16
    seed:             int = 42

    # ---- Model ----
    hidden_dim:       int = 256
    text_embed_dim:   int = 300

    # ---- XAI ----
    run_xai:          bool = True
    ig_steps:         int = 30
    cf_steps:         int = 80

    # ---- Federated Learning ----
    fl_rounds:            int = 5
    fl_local_epochs:      int = 2
    fl_local_lr:          float = 1e-3
    fl_global_lr:         float = 1.0
    fl_epsilon:           Optional[float] = None
    fl_delta:             float = 1e-5
    fl_clip_norm:         Union[float, str] = 5.0
    fl_sigma:             Optional[float] = 1.25
    fl_aggregation:       str = "fedavg"
    fl_secure_aggregation: bool = True

    # ---- Retrain loop ----
    max_retrain_attempts: int = 3
    """Maximum number of retrain attempts before accepting the current model."""

    # ---- Validation ----
    target_f1:            float = 0.65
    target_auc_roc:       float = 0.70
    target_mcc:           float = 0.20
    max_eq_odds_gap:      float = 0.15
    run_lodo:             bool = True
    lodo_train_epochs:    int = 3
    lodo_lr:              float = 1e-3

    # ---- Output ----
    output_dir:       str = "outputs"
    save_reports:     bool = True
    print_reports:    bool = True


# ---------------------------------------------------------------------------
# Retrain Loop
# ---------------------------------------------------------------------------

class RetrainLoop:
    """Drives the federated training -> validation -> retrain cycle.

    The loop:
      1. Run FL training for fl_rounds rounds (model updated in-place)
      2. Validate with ValidationLayer
      3. If metrics satisfactory: stop
      4. Else: adapt FL config (e.g. more rounds) and repeat
      5. Stop after max_retrain_attempts regardless

    The model is updated in-place. The best checkpoint (by AUC-ROC) is
    kept separately and restored at the end.
    """

    def __init__(
        self,
        fl_layer: FederatedLearningLayer,
        val_layer: ValidationLayer,
        max_attempts: int = 3,
        fl_rounds_per_attempt: int = 5,
    ):
        self.fl_layer        = fl_layer
        self.val_layer       = val_layer
        self.max_attempts    = max_attempts
        self.fl_rounds       = fl_rounds_per_attempt

    def run(
        self,
        model: torch.nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        prediction_config: PredictionConfig,
        full_dataset=None,
        model_factory: Optional[Callable] = None,
        collate_fn: Optional[Callable] = None,
    ) -> ValidationReport:
        """Execute the retrain loop.

        Returns the ValidationReport of the best (or final) checkpoint.
        """
        best_state  = copy.deepcopy(model.state_dict())
        best_report: Optional[ValidationReport] = None
        best_auc    = -1.0

        for attempt in range(self.max_attempts):
            print(
                f"\n{'=' * 60}\n"
                f"RETRAIN LOOP -- attempt {attempt + 1}/{self.max_attempts}\n"
                f"{'=' * 60}"
            )

            # --- FL training ---
            fl_rounds = self.fl_rounds * (1 + attempt // 2)  # ramp up rounds on failure
            fl_history = self.fl_layer.run(
                model=model,
                prediction_config=prediction_config,
                num_rounds=fl_rounds,
            )

            # --- Validation ---
            report = self.val_layer.validate(
                model=model,
                val_loader=val_loader,
                device=device,
                model_round=attempt,
                full_dataset=full_dataset,
                model_factory=model_factory,
                prediction_config=prediction_config,
                collate_fn=collate_fn,
            )

            # --- Track best ---
            current_auc = report.overall_metrics.auc_roc
            if current_auc > best_auc:
                best_auc    = current_auc
                best_state  = copy.deepcopy(model.state_dict())
                best_report = report
                print("  [Best checkpoint saved] AUC-ROC={:.4f}".format(best_auc))

            if not report.retrain_needed:
                print("\nConverged after {} FL+validation cycle(s).".format(attempt + 1))
                break

            # Adapt for next attempt: increase local epochs slightly
            self.fl_layer.cfg.local_epochs = min(
                self.fl_layer.cfg.local_epochs + 1, 8
            )

        # Restore best weights
        model.load_state_dict(best_state)
        print("\nRetrain loop complete. Best AUC-ROC: {:.4f}".format(best_auc))
        return best_report


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

class DepressionDetectionPipeline:
    """Full end-to-end depression detection pipeline.

    Wires together:
      Dataset -> Model -> XAI -> Output Layer -> FL -> Validation (retrain loop)

    Usage
    -----
        pipeline = DepressionDetectionPipeline(PipelineConfig())
        pipeline.run()
    """

    def __init__(self, config: PipelineConfig):
        self.cfg    = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("DepressionDetectionPipeline initialised. Device: {}".format(self.device))

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_dataset(self):
        """Build the training dataset.

        Returns a ``RealDepressionDataset`` (when DAIC-WOZ is disabled) or a
        ``torch.utils.data.ConcatDataset`` of RealDepressionDataset + DAICWOZDataset.
        The raw ``DatasetConfig`` is stored as ``self._ds_cfg`` so that
        ``_build_model`` can still access architecture parameters.
        """
        cfg = self.cfg
        ds_cfg = DatasetConfig(
            depressed_json=cfg.depressed_json,
            normal_json=cfg.normal_json,
            reddit_csv=cfg.reddit_csv,
            google_news_bin=cfg.google_news_bin,
            max_samples=cfg.max_samples,
            text_embed_dim=cfg.text_embed_dim,
            seed=cfg.seed,
        )
        # Keep ds_cfg accessible even after a ConcatDataset wrapping
        self._ds_cfg = ds_cfg

        print("\n=== Loading Datasets ===")
        base_dataset = RealDepressionDataset.build(ds_cfg)

        if cfg.daic_woz_raw_dir:
            from daic_woz_dataset import DAICWOZDataset, DAICConfig
            from torch.utils.data import ConcatDataset
            daic_cfg_train = DAICConfig(
                raw_dir=cfg.daic_woz_raw_dir,
                labels_csv=cfg.daic_woz_labels,
                split="train",
                seed=cfg.seed,
                google_news_bin=cfg.google_news_bin,
            )
            daic_cfg_dev = DAICConfig(
                raw_dir=cfg.daic_woz_raw_dir,
                labels_csv=cfg.daic_woz_labels,
                split="dev",
                seed=cfg.seed,
                google_news_bin=cfg.google_news_bin,
            )
            daic_train = DAICWOZDataset(daic_cfg_train)
            daic_dev   = DAICWOZDataset(daic_cfg_dev)
            print("  DAIC-WOZ added: {:,} train + {:,} dev participants".format(
                len(daic_train), len(daic_dev)))
            return ConcatDataset([base_dataset, daic_train])

        return base_dataset

    def _build_model(self, ds_cfg: DatasetConfig, prediction_config: PredictionConfig):
        from multimodal_model import DatasetConfig as DSCfg
        xai_cfg = XAIConfig(ig_steps=self.cfg.ig_steps, cf_steps=self.cfg.cf_steps)
        model = DepressionDetectionModel(
            vocab_size=30522,
            text_embed_dim=self.cfg.text_embed_dim,
            hidden_dim=self.cfg.hidden_dim,
            eeg_channels=ds_cfg.eeg_channels,
            wearable_dim=ds_cfg.wearable_dim,
            mfcc_dim=ds_cfg.mfcc_dim,
            audio_dim=ds_cfg.audio_dim,
            video_dim=ds_cfg.video_dim,
            clinical_dim=ds_cfg.clinical_dim,
            text_backbone_name=None,
            prediction_config=prediction_config,
            use_text_projector=True,
            use_xai=self.cfg.run_xai,
            xai_config=xai_cfg,
        ).to(self.device)
        return model

    def _model_factory(self, ds_cfg, prediction_config):
        """Return a callable that always builds a fresh identical model."""
        def _factory():
            return self._build_model(ds_cfg, prediction_config)
        return _factory

    # ------------------------------------------------------------------
    # Phase 1: Inference + Output Layer (generate patient reports)
    # ------------------------------------------------------------------

    def run_output_layer(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        dataset: RealDepressionDataset,
        max_reports: int = 50,
    ) -> List[PatientReport]:
        """Run model on loader, generate PatientReports."""
        output_layer = OutputLayer(symptom_threshold=0.45, severity_scale=8.0)
        reports: List[PatientReport] = []
        model.eval()

        global_idx = 0
        for batch_idx, batch in enumerate(loader):
            batch_device = {k: v.to(self.device) for k, v in batch.items()}

            # Forward with optional XAI
            with torch.set_grad_enabled(self.cfg.run_xai):
                out = model(
                    text_input=batch_device["text_emb"],
                    eeg=batch_device["eeg"],
                    wearable=batch_device["wearable"],
                    audio=batch_device["audio"],
                    video=batch_device["video"],
                    clinical=batch_device["clinical"],
                    mfcc=batch_device.get("mfcc"),
                    explain=self.cfg.run_xai,
                )

            xai_exp: Optional[ExplanationOutput] = out.get("explanations", None)

            B = batch_device["label"].shape[0]
            for i in range(B):
                rec = dataset.records[global_idx] if global_idx < len(dataset.records) else None
                patient_id = "PATIENT_{:05d}".format(global_idx)
                src = rec.source if rec else "unknown"
                report = output_layer.generate_report(
                    model_output=out,
                    patient_id=patient_id,
                    sample_index=i,
                    xai_explanation=xai_exp,
                    data_source=src,
                )
                reports.append(report)
                global_idx += 1
                if len(reports) >= max_reports:
                    break

            if self.cfg.print_reports and batch_idx == 0:
                # Print first report as sample
                print("\n" + output_layer.format_report(reports[0]))

            if len(reports) >= max_reports:
                break

        return reports

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """Execute the full pipeline end-to-end.

        Returns
        -------
        result : dict
            {
              'model': trained DepressionDetectionModel,
              'validation_report': ValidationReport,
              'patient_reports': List[PatientReport],
              'dashboard': ClinicalDashboard,
            }
        """
        cfg = self.cfg
        os.makedirs(cfg.output_dir, exist_ok=True)

        # ---- 1. Datasets ----
        full_dataset = self._build_dataset()
        val_size   = max(1, int(len(full_dataset) * cfg.val_split))
        train_size = len(full_dataset) - val_size
        train_data, val_data = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        print("\nTrain: {:,}  |  Val: {:,}".format(train_size, val_size))

        val_loader = DataLoader(
            val_data, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch
        )

        # ---- 1b. Class Weights ----
        train_loader_temp = DataLoader(train_data, batch_size=256, shuffle=False, collate_fn=collate_batch)
        counts = {0: 0, 1: 0}
        for b in train_loader_temp:
            for l in b["label"].long().view(-1).tolist():
                counts[l] += 1
        total = counts[0] + counts[1]
        w0 = total / (2.0 * max(counts[0], 1))
        w1 = total / (2.0 * max(counts[1], 1))
        class_weights = [w0, w1]
        print("Class weights injected: Normal={:.4f}, Depressed={:.4f}".format(w0, w1))

        # ---- 2. Model ----
        # Use self._ds_cfg (set in _build_dataset) to get architecture parameters.
        # This works whether full_dataset is a RealDepressionDataset or ConcatDataset.
        ds_cfg = getattr(full_dataset, "cfg", self._ds_cfg)
        prediction_config = PredictionConfig(
            num_symptoms=ds_cfg.num_symptoms,
            class_weights=class_weights
        )
        model = self._build_model(ds_cfg, prediction_config)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\nModel trainable parameters: {:,}".format(total_params))

        model_factory = self._model_factory(ds_cfg, prediction_config)

        # ---- 3. Federated Learning Layer ----
        fl_cfg = FLConfig(
            num_rounds=cfg.fl_rounds,
            local_epochs=cfg.fl_local_epochs,
            local_lr=cfg.fl_local_lr,
            global_lr=cfg.fl_global_lr,
            epsilon=cfg.fl_epsilon,
            delta=cfg.fl_delta,
            clip_norm=cfg.fl_clip_norm,
            sigma=cfg.fl_sigma,
            aggregation=cfg.fl_aggregation,
            use_secure_aggregation=cfg.fl_secure_aggregation,
        )
        fl_layer = FederatedLearningLayer.build_nodes_from_dataset(
            full_dataset=full_dataset,
            fl_config=fl_cfg,
            prediction_config=prediction_config,
            collate_fn=collate_batch,
            device=self.device,
            batch_size=cfg.batch_size,
            train_indices=train_data.indices,
        )

        # ---- 4. Validation Layer ----
        val_cfg = ValidationConfig(
            target_f1=cfg.target_f1,
            target_auc_roc=cfg.target_auc_roc,
            target_mcc=cfg.target_mcc,
            max_equalized_odds_gap=cfg.max_eq_odds_gap,
            run_lodo=cfg.run_lodo,
            lodo_train_epochs=cfg.lodo_train_epochs,
            lodo_lr=cfg.lodo_lr,
        )
        val_layer = ValidationLayer(config=val_cfg)

        # ---- 5. Retrain Loop ----
        retrain_loop = RetrainLoop(
            fl_layer=fl_layer,
            val_layer=val_layer,
            max_attempts=cfg.max_retrain_attempts,
            fl_rounds_per_attempt=cfg.fl_rounds,
        )
        val_report = retrain_loop.run(
            model=model,
            val_loader=val_loader,
            device=self.device,
            prediction_config=prediction_config,
            full_dataset=full_dataset if cfg.run_lodo else None,
            model_factory=model_factory if cfg.run_lodo else None,
            collate_fn=collate_batch if cfg.run_lodo else None,
        )

        # ---- 6. Validation report printout ----
        print("\n" + format_validation_report(val_report))

        # ---- 7. Output Layer: patient reports + dashboard ----
        print("\n=== Output Layer: Generating Patient Reports ===")
        # Use a small loader over the validation set for demonstration
        demo_loader = DataLoader(
            val_data, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch
        )
        patient_reports = self.run_output_layer(
            model=model,
            loader=demo_loader,
            dataset=full_dataset,
            max_reports=min(200, val_size),
        )

        output_layer = OutputLayer(symptom_threshold=0.45, severity_scale=8.0)
        dashboard = output_layer.generate_dashboard(patient_reports)

        if cfg.print_reports:
            print("\n" + output_layer.format_dashboard(dashboard))

        # ---- 8. Save artefacts ----
        if cfg.save_reports and patient_reports:
            ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = os.path.join(cfg.output_dir, "reports_{}.json".format(ts))
            output_layer.save_reports_json(patient_reports, path, dashboard=dashboard)

            # Save model
            ckpt = os.path.join(cfg.output_dir, "model_{}.pt".format(ts))
            torch.save(model.state_dict(), ckpt)
            print("Model checkpoint saved: {}".format(ckpt))

        print("\n=== Pipeline complete ===")
        return {
            "model":             model,
            "validation_report": val_report,
            "patient_reports":   patient_reports,
            "dashboard":         dashboard,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="End-to-end multimodal depression detection pipeline."
    )
    # Data
    parser.add_argument("--depressed_json",  default="depressed.json")
    parser.add_argument("--normal_json",     default="normal.json")
    parser.add_argument("--reddit_csv",      default="reddit_depression_dataset.csv")
    parser.add_argument("--google_news_bin", default="GoogleNews-vectors-negative300.bin")
    # DAIC-WOZ (optional)
    parser.add_argument("--daic_woz_raw_dir", default="",
                        help="Root dir with {pid}_P/ sub-folders (empty = disabled)")
    parser.add_argument("--daic_woz_labels",  default="daic-woz/labels.csv",
                        help="Path to labels.csv generated by daic_woz_dataset.py make_labels")
    parser.add_argument("--max_samples",     type=int,   default=1000)
    parser.add_argument("--val_split",       type=float, default=0.15)
    parser.add_argument("--batch_size",      type=int,   default=16)
    # Model
    parser.add_argument("--hidden_dim",      type=int,   default=256)
    # XAI
    parser.add_argument("--no_xai",         action="store_true")
    parser.add_argument("--ig_steps",       type=int,   default=30)
    parser.add_argument("--cf_steps",       type=int,   default=80)
    # Federated Learning
    parser.add_argument("--fl_rounds",      type=int,   default=5)
    parser.add_argument("--fl_local_epochs",type=int,   default=2)
    parser.add_argument("--fl_local_lr",    type=float, default=1e-3)
    parser.add_argument("--fl_epsilon",     type=float, default=None,
                        help="Differential privacy epsilon (lower = more privacy)")
    parser.add_argument("--fl_delta",       type=float, default=1e-5)
    parser.add_argument("--fl_clip_norm",   type=str,   default="5.0")
    parser.add_argument("--fl_sigma",       type=float, default=1.25)
    parser.add_argument("--no_secure_agg",  action="store_true",
                        help="Disable secure aggregation (for debugging)")
    # Retrain loop
    parser.add_argument("--max_retrain",    type=int,   default=3)
    # Validation
    parser.add_argument("--target_f1",      type=float, default=0.65)
    parser.add_argument("--target_auc",     type=float, default=0.70)
    parser.add_argument("--target_mcc",     type=float, default=0.20)
    parser.add_argument("--no_lodo",        action="store_true",
                        help="Skip LODO validation (much faster)")
    parser.add_argument("--lodo_epochs",    type=int,   default=3)
    # Output
    parser.add_argument("--output_dir",     default="outputs")
    parser.add_argument("--no_save",        action="store_true")
    parser.add_argument("--quiet",          action="store_true")

    args = parser.parse_args()

    cfg = PipelineConfig(
        depressed_json=args.depressed_json,
        normal_json=args.normal_json,
        reddit_csv=args.reddit_csv,
        google_news_bin=args.google_news_bin,
        daic_woz_raw_dir=args.daic_woz_raw_dir,
        daic_woz_labels=args.daic_woz_labels,
        max_samples=args.max_samples,
        val_split=args.val_split,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        run_xai=not args.no_xai,
        ig_steps=args.ig_steps,
        cf_steps=args.cf_steps,
        fl_rounds=args.fl_rounds,
        fl_local_epochs=args.fl_local_epochs,
        fl_local_lr=args.fl_local_lr,
        fl_epsilon=args.fl_epsilon,
        fl_delta=args.fl_delta,
        fl_clip_norm=float(args.fl_clip_norm) if args.fl_clip_norm != "adaptive" else "adaptive",
        fl_sigma=args.fl_sigma,
        fl_secure_aggregation=not args.no_secure_agg,
        max_retrain_attempts=args.max_retrain,
        target_f1=args.target_f1,
        target_auc_roc=args.target_auc,
        target_mcc=args.target_mcc,
        run_lodo=not args.no_lodo,
        lodo_train_epochs=args.lodo_epochs,
        output_dir=args.output_dir,
        save_reports=not args.no_save,
        print_reports=not args.quiet,
    )

    pipeline = DepressionDetectionPipeline(cfg)
    result = pipeline.run()
