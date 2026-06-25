"""Auto-generate daic-woz/labels.csv by scanning extracted participant folders.

This script has TWO modes:

Mode A — From official AVEC2017 split CSVs (preferred, preserves real PHQ scores):
    python make_labels_auto.py --mode official
        --train daic-woz/train_split_Depression_AVEC2017.csv
        --dev   daic-woz/dev_split_Depression_AVEC2017.csv
        [--test daic-woz/test_split_Depression_AVEC2017.csv]
        --out   daic-woz/labels.csv

Mode B — From folder scan (fallback when split CSVs are unavailable):
    python make_labels_auto.py --mode scan
        --raw_dir daic-woz
        --out     daic-woz/labels.csv

    In scan mode ALL discovered {pid}_P folders get split=train.
    Labels default to 0 unless a per-participant .txt hint is found.
    This mode lets you use all extracted data even without the AVEC2017 CSVs.

Mode C -- Merge: official CSVs + all extracted folders (recommended):
    python make_labels_auto.py --mode merge
        --raw_dir daic-woz
        --train   daic-woz/train_split_Depression_AVEC2017.csv
        --dev     daic-woz/dev_split_Depression_AVEC2017.csv
        --out     daic-woz/labels.csv

    Merges official labels (with real PHQ scores + splits) for known PIDs,
    and adds any extra extracted folders as split=train with label=0 placeholder.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIELDNAMES = ["participant_id", "label", "phq_score", "gender", "age_bin", "split"]

_GENDER_MAP = {
    "m": "1", "male": "1", "1": "1",
    "f": "2", "female": "2", "2": "2",
}

def _norm_gender(g: str) -> str:
    return _GENDER_MAP.get(str(g).strip().lower(), "0")


def _scan_folders(raw_dir: str) -> list:
    """Return sorted list of (pid_str, folder_path) for all {pid}_P dirs."""
    found = []
    if not os.path.isdir(raw_dir):
        return found
    pat = re.compile(r"^(\d+)_P$")
    for name in sorted(os.listdir(raw_dir)):
        m = pat.match(name)
        if m:
            folder = os.path.join(raw_dir, name)
            if os.path.isdir(folder):
                found.append((m.group(1), folder))
    return found


def _read_official_csv(path: str, split_name: str) -> dict:
    """Read an AVEC2017 split CSV. Returns {pid_str: row_dict}."""
    RENAME = {
        "Participant_ID": "participant_id",
        "PHQ_Binary":     "label",
        "PHQ_Score":      "phq_score",
        "Gender":         "gender",
    }
    result = {}
    if not path or not os.path.exists(path):
        warnings.warn("Split CSV not found: {}".format(path))
        return result
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out = {"split": split_name, "age_bin": "0"}
            for old_k, new_k in RENAME.items():
                out[new_k] = row.get(old_k, row.get(new_k, "")).strip()
            out["gender"] = _norm_gender(out.get("gender", ""))
            pid = out.get("participant_id", "").strip()
            if pid:
                result[pid] = out
    return result


def _write_csv(rows: list, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Mode A: Official split CSVs only
# ---------------------------------------------------------------------------

def mode_official(args) -> None:
    from daic_woz_dataset import make_labels_csv
    test = getattr(args, "test", "") or ""
    make_labels_csv(args.train, args.dev, args.out, test)


# ---------------------------------------------------------------------------
# Mode B: Folder scan only
# ---------------------------------------------------------------------------

def mode_scan(args) -> None:
    participants = _scan_folders(args.raw_dir)
    if not participants:
        print("ERROR: No {pid}_P folders found in {}".format(args.raw_dir))
        sys.exit(1)

    rows = []
    n_train = 0
    for pid, _ in participants:
        rows.append({
            "participant_id": pid,
            "label":          "0",   # unknown — update manually or use Mode C
            "phq_score":      "0",
            "gender":         "0",
            "age_bin":        "0",
            "split":          "train",
        })
        n_train += 1

    _write_csv(rows, args.out)
    print("Scan mode: {:,} participants -> {}".format(len(rows), args.out))
    print("  WARNING: all labels default to 0. Use Mode C (--mode merge) if you")
    print("  have the official AVEC2017 split CSVs for accurate PHQ labels.")


# ---------------------------------------------------------------------------
# Mode C: Merge official + scan
# ---------------------------------------------------------------------------

def mode_merge(args) -> None:
    # Load official labels
    known: dict = {}
    for path, split_name in [(args.train, "train"), (args.dev, "dev")]:
        if path and os.path.exists(path):
            chunk = _read_official_csv(path, split_name)
            known.update(chunk)
            print("  Official {}: {:,} participants".format(split_name, len(chunk)))
    test_path = getattr(args, "test", "") or ""
    if test_path and os.path.exists(test_path):
        chunk = _read_official_csv(test_path, "test")
        known.update(chunk)
        print("  Official test: {:,} participants".format(len(chunk)))

    # Scan all extracted folders
    participants = _scan_folders(args.raw_dir)
    extra = 0
    for pid, _ in participants:
        if pid not in known:
            known[pid] = {
                "participant_id": pid,
                "label":          "0",
                "phq_score":      "0",
                "gender":         "0",
                "age_bin":        "0",
                "split":          "train",  # treat as extra training data
            }
            extra += 1

    rows = sorted(known.values(), key=lambda r: int(r["participant_id"]))
    _write_csv(rows, args.out)
    print("Merge done: {:,} official + {:,} extra scan = {:,} total -> {}".format(
        len(known) - extra, extra, len(rows), args.out))


# ---------------------------------------------------------------------------
# Summarise result
# ---------------------------------------------------------------------------

def _summarise(out_path: str) -> None:
    if not os.path.exists(out_path):
        return
    train = dev = test = dep = norm = 0
    with open(out_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sp = row.get("split", "train")
            if sp == "train": train += 1
            elif sp == "dev":  dev   += 1
            elif sp == "test": test  += 1
            if str(row.get("label", "0")) == "1": dep  += 1
            else:                                  norm += 1
    print("\nlabels.csv summary:")
    print("  train={:,}  dev={:,}  test={:,}".format(train, dev, test))
    print("  depressed={:,}  normal={:,}".format(dep, norm))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-generate daic-woz/labels.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["official", "scan", "merge"],
                        default="merge",
                        help="official=use split CSVs only | scan=folder scan only | "
                             "merge=official+scan (default)")
    parser.add_argument("--raw_dir", default="daic-woz",
                        help="Root folder containing {pid}_P/ sub-dirs")
    parser.add_argument("--train",   default="daic-woz/train_split_Depression_AVEC2017.csv")
    parser.add_argument("--dev",     default="daic-woz/dev_split_Depression_AVEC2017.csv")
    parser.add_argument("--test",    default="daic-woz/test_split_Depression_AVEC2017.csv")
    parser.add_argument("--out",     default="daic-woz/labels.csv")

    args = parser.parse_args()

    print("Mode: {}".format(args.mode))

    if args.mode == "official":
        mode_official(args)
    elif args.mode == "scan":
        mode_scan(args)
    else:
        mode_merge(args)

    _summarise(args.out)
    print("\nNext step:")
    print("  python pipeline.py --daic_woz_raw_dir {} --daic_woz_labels {}".format(
        args.raw_dir, args.out))
