import os
import glob
import json
import torch
from torch.utils.data import DataLoader, Subset
from multimodal_model import RealDepressionDataset, DatasetConfig, DepressionDetectionModel, collate_batch
from prediction import PredictionConfig
from output_layer import OutputLayer

def format_markdown_report(r) -> str:
    active_syms_str = ""
    if r.active_symptoms:
        active_syms_str = "\n".join(f"- {s} (Score: {r.symptom_scores.get(s.replace(' (Promoted)', ''), 0.0):.1%})" for s in r.active_symptoms)
    else:
        active_syms_str = "- No active symptoms detected."

    mod_list = []
    for mod, imp in r.top_features:
        mod_list.append(f"- **{mod}**: {imp:.4f}")
    mod_str = "\n".join(mod_list)

    return f"""# Patient Report: {r.patient_id}

- **Timestamp**: {r.timestamp}
- **Data Source**: {r.data_source}

## Prediction & Risk Analysis
- **Model Prediction**: **{r.prediction_label}** (Confidence: {r.confidence:.2%})
- **Risk Level**: `{r.risk_level}`
- **PHQ-9 Severity Score**: `{r.severity_score}/27` ({r.severity_category})

## DSM-5 Active Symptoms
{active_syms_str}

## Modality Attribution (SHAP/IG)
{mod_str}

## Clinical Guidance
- **Recommended Action**: {r.recommended_action}
- **Follow-up Priority**: **{r.follow_up_priority}**
- **Clinical Notes**:
  > {r.clinical_notes}

---
"""

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "outputs")
    
    # Try to determine max_samples from saved test indices first
    test_indices_path = os.path.join(output_dir, "test_indices.json")
    max_samples = 2000
    if os.path.exists(test_indices_path):
        try:
            test_info = json.load(open(test_indices_path, "r"))
            max_samples = test_info.get("total_samples", 2000)
            print(f"Detected max_samples={max_samples} from {test_indices_path}")
        except Exception as e:
            print(f"Error reading test_indices.json for max_samples: {e}")

    # Load dataset
    cfg = DatasetConfig(
        depressed_json=os.path.join(base_dir, "depressed.json"),
        normal_json=os.path.join(base_dir, "normal.json"),
        reddit_csv=os.path.join(base_dir, "reddit_depression_dataset.csv"),
        google_news_bin=os.path.join(base_dir, "GoogleNews-vectors-negative300.bin"),
        max_samples=max_samples,
        seed=42
    )
    
    print("Loading dataset...")
    full_dataset = RealDepressionDataset.build(cfg)
    
    # Try to load saved test indices from the pipeline run
    thresh = 0.5
    if os.path.exists(test_indices_path):
        try:
            test_info = json.load(open(test_indices_path, "r"))
            eval_indices = test_info["indices"]
            thresh = test_info.get("best_threshold", 0.5)
            dataset_name = "Held-Out Test Set (from pipeline)"
            print(f"Loaded {len(eval_indices)} test indices from {test_indices_path}")
            print(f"Loaded decision threshold: {thresh:.2f}")
        except Exception as e:
            print(f"Error loading {test_indices_path}: {e}")
            eval_indices = [i for i, r in enumerate(full_dataset.records) if r.source == "wu3d"]
            dataset_name = "WU3D (all samples fallback)"
            print(f"Fallback to all WU3D samples: {len(eval_indices)}")
    else:
        eval_indices = [i for i, r in enumerate(full_dataset.records) if r.source == "wu3d"]
        dataset_name = "WU3D (all samples fallback)"
        print(f"test_indices.json not found. Fallback to all WU3D samples: {len(eval_indices)}")
        
    if not eval_indices:
        print("No evaluation samples found!")
        return
        
    eval_dataset = Subset(full_dataset, eval_indices)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, collate_fn=collate_batch)
    
    # Find latest model checkpoint
    checkpoints = glob.glob(os.path.join(output_dir, "model_*.pt"))
    if not checkpoints:
        print("No checkpoints found in outputs/!")
        return
    latest_ckpt = max(checkpoints, key=os.path.getmtime)
    print(f"Loading latest checkpoint: {latest_ckpt}")
    
    # Initialize model
    prediction_config = PredictionConfig(
        num_symptoms=cfg.num_symptoms,
        class_weights=[1.0, 1.0] # default weights
    )
    
    model = DepressionDetectionModel(
        vocab_size=30522,
        text_embed_dim=cfg.text_embed_dim,
        hidden_dim=256,
        eeg_channels=cfg.eeg_channels,
        wearable_dim=cfg.wearable_dim,
        mfcc_dim=cfg.mfcc_dim,
        audio_dim=cfg.audio_dim,
        video_dim=cfg.video_dim,
        clinical_dim=cfg.clinical_dim,
        text_backbone_name=None,
        prediction_config=prediction_config,
        use_text_projector=True,
        use_xai=False
    )
    
    model.load_state_dict(torch.load(latest_ckpt, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Run evaluation
    correct = 0
    total = 0
    tp = tn = fp = fn = 0
    
    # We will also gather outputs for report generation
    reports = []
    output_layer = OutputLayer(symptom_threshold=0.45, severity_scale=1.0)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            batch_device = {k: v.to(device) for k, v in batch.items()}
            out = model(
                text_input=batch_device["text_emb"],
                eeg=batch_device["eeg"],
                wearable=batch_device["wearable"],
                audio=batch_device["audio"],
                video=batch_device["video"],
                clinical=batch_device["clinical"],
                mfcc=batch_device.get("mfcc"),
                explain=False
            )
            
            probs = out["binary_probs"].cpu()
            preds = (probs[:, 1] >= thresh).long().tolist()
            trues = batch["label"].view(-1).cpu().tolist()
            
            for p, t in zip(preds, trues):
                if p == t:
                    correct += 1
                total += 1
                if p == 1 and t == 1:
                    tp += 1
                elif p == 0 and t == 0:
                    tn += 1
                elif p == 1 and t == 0:
                    fp += 1
                elif p == 0 and t == 1:
                    fn += 1
                
            # Generate patient reports for the first 10 patients
            B = batch_device["label"].shape[0]
            for i in range(B):
                global_idx = batch_idx * 16 + i
                if len(reports) < 10 and global_idx < len(eval_indices):
                    rec_idx = eval_indices[global_idx]
                    rec = full_dataset.records[rec_idx]
                    patient_id = f"PATIENT_{rec_idx:05d}"
                    report = output_layer.generate_report(
                        model_output=out,
                        patient_id=patient_id,
                        sample_index=i,
                        xai_explanation=None,
                        data_source=rec.source
                    )
                    reports.append(report)
                    
    accuracy = correct / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n==========================================")
    print(f"Model Performance on Unseen Data")
    print(f"Evaluation Split: {dataset_name}")
    print("==========================================")
    print(f"Total Samples: {total}")
    print(f"Accuracy:      {accuracy:.4f} ({correct}/{total})")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print("------------------------------------------")
    print("Confusion Matrix:")
    print(f"              Predicted Normal  Predicted Depressed")
    print(f"True Normal:       {tn:<15}  {fp:<15}")
    print(f"True Depressed:    {fn:<15}  {tp:<15}")
    print("==========================================\n")
    
    # Generate markdown content
    md_content = f"# Unseen Data Evaluation & Patient Reports\n\n"
    md_content += f"- **Evaluation Dataset**: {dataset_name}\n"
    md_content += f"- **Total Unseen Samples**: {total}\n"
    md_content += f"- **Model Accuracy**: {accuracy:.2%} ({correct}/{total})\n"
    md_content += f"- **Precision**: {precision:.4f}\n"
    md_content += f"- **Recall**: {recall:.4f}\n"
    md_content += f"- **F1 Score**: {f1:.4f}\n\n"
    md_content += f"## Confusion Matrix\n\n"
    md_content += f"| | Predicted Normal | Predicted Depressed |\n"
    md_content += f"|---|---|---|\n"
    md_content += f"| **True Normal** | {tn} | {fp} |\n"
    md_content += f"| **True Depressed** | {fn} | {tp} |\n\n"
    md_content += f"## Generated Patient Reports (10 Samples)\n\n"
    
    for r in reports:
        md_content += format_markdown_report(r)
        
    temp_path = os.path.join(output_dir, "patient_reports_temp.md")
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved temporary markdown report to: {temp_path}")

if __name__ == "__main__":
    main()
