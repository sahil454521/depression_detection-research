"""One-time setup script: generate daic-woz/labels.csv from official AVEC2017 split CSVs.

Usage
-----
    python make_labels.py

Expected input files (in daic-woz/ folder):
    train_split_Depression_AVEC2017.csv
    dev_split_Depression_AVEC2017.csv

Columns expected in each split file:
    Participant_ID, PHQ_Binary, PHQ_Score, Gender  (+ any extra columns, ignored)

Output
------
    daic-woz/labels.csv   -- unified labels file consumed by DAICWOZDataset

Run this script exactly once before training. Re-running is safe (overwrites).
"""

import sys
import os

# Allow running from either the depression_detection/ dir or a parent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from daic_woz_dataset import make_labels_csv

# ── Paths ──────────────────────────────────────────────────────────────────
DAIC_DIR   = os.path.join(os.path.dirname(__file__), "daic-woz")
TRAIN_CSV  = os.path.join(DAIC_DIR, "train_split_Depression_AVEC2017.csv")
DEV_CSV    = os.path.join(DAIC_DIR, "dev_split_Depression_AVEC2017.csv")
TEST_CSV   = os.path.join(DAIC_DIR, "test_split_Depression_AVEC2017.csv")
OUTPUT_CSV = os.path.join(DAIC_DIR, "labels.csv")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate daic-woz/labels.csv from AVEC2017 split CSVs"
    )
    parser.add_argument(
        "--train", default=TRAIN_CSV,
        help="Path to train_split_Depression_AVEC2017.csv"
    )
    parser.add_argument(
        "--dev",   default=DEV_CSV,
        help="Path to dev_split_Depression_AVEC2017.csv"
    )
    parser.add_argument(
        "--test",  default=TEST_CSV,
        help="Path to test_split_Depression_AVEC2017.csv (optional)"
    )
    parser.add_argument(
        "--out",   default=OUTPUT_CSV,
        help="Output path for labels.csv"
    )
    args = parser.parse_args()

    # Check which files actually exist
    test_path = args.test if os.path.exists(args.test) else ""
    if test_path:
        print("Including test split: {}".format(test_path))

    missing = [p for p in [args.train, args.dev] if not os.path.exists(p)]
    if missing:
        print("ERROR: The following required files were not found:")
        for m in missing:
            print("  {}".format(m))
        print()
        print("These are the official AVEC2017 split CSVs distributed with DAIC-WOZ.")
        print("Place them inside the daic-woz/ folder and re-run this script.")
        sys.exit(1)

    make_labels_csv(
        train_split_csv=args.train,
        dev_split_csv=args.dev,
        output_csv=args.out,
        test_split_csv=test_path,
    )
    print("Done. Now run:")
    print("  python pipeline.py --daic_woz_raw_dir daic-woz --daic_woz_labels {} ...".format(
        args.out))
