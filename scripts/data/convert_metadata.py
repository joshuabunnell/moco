"""Convert ACRIN 6664 XLSX metadata files to clean CSVs.

Reads the three Excel files provided by TCIA (no-polyp, 6-9mm, >=10mm),
normalizes column names, and writes a single combined CSV plus per-category
CSVs into the ``csv_metadata/`` directory.

The XLSX column layout for lesion files follows a repeating pattern of 5
sub-columns per lesion: location, size (mm), morphology, histology, and a
per-lesion flag.  This script extracts the largest polyp size per patient
for the combined output.

TCIA ships these as legacy ``.xls``; the ``v1/`` copies are re-saved ``.xlsx``
with byte-identical contents, so either directory reproduces the same CSVs.

Usage:
    python scripts/data/convert_metadata.py \
        --input-dir metadata/raw_metadata/v2_2026-08-24 \
        --output-dir metadata/csv_metadata
"""

import argparse
import csv
import glob
import os

import pandas as pd


def load_rows(input_dir, stem):
    """Yield the data rows of *stem* as tuples, whatever Excel format it is in."""
    matches = glob.glob(os.path.join(input_dir, stem + ".xls")) + \
        glob.glob(os.path.join(input_dir, stem + ".xlsx"))
    if not matches:
        raise FileNotFoundError(os.path.join(input_dir, stem + ".xls[x]"))
    frame = pd.read_excel(matches[0], header=0).dropna(how="all")
    return [tuple(None if pd.isna(v) else v for v in row)
            for row in frame.itertuples(index=False, name=None)]


def read_no_polyp(rows):
    """Parse the no-polyp-found file.  Single column: TCIA Patient ID."""
    patients = []
    for row in rows:
        pid = row[0]
        if pid:
            patients.append({"patient_id": str(pid).strip(), "max_polyp_mm": 0,
                             "category": "no_polyp"})
    return patients


def read_lesion_file(rows, category):
    """Parse a lesion file (6-9mm or >=10mm).

    Each row is a patient.  Lesion sub-columns repeat in groups of 5:
    [location, size_mm, morphology, histology, flag].  We extract the
    max polyp size across all lesions for each patient.
    """
    patients = []

    for row in rows:
        pid = row[0]
        if not pid:
            continue

        # Lesion data starts at column index 3 (after TCIA#, slice supine, slice prone)
        # Each lesion has 5 sub-columns; the size is at offset 1 within each group
        max_size = 0
        lesion_data = list(row[3:])
        for j in range(0, len(lesion_data), 5):
            size_val = lesion_data[j + 1] if (j + 1) < len(lesion_data) else None
            if size_val is not None:
                try:
                    max_size = max(max_size, float(size_val))
                except (ValueError, TypeError):
                    pass

        patients.append({"patient_id": str(pid).strip(),
                         "max_polyp_mm": max_size,
                         "category": category})
    return patients


def main():
    parser = argparse.ArgumentParser(description="Convert ACRIN XLSX metadata to CSV")
    parser.add_argument("--input-dir", default="metadata/raw_metadata/v2_2026-08-24",
                        help="Directory containing the TCIA polyp spreadsheets")
    parser.add_argument("--output-dir", default="metadata/csv_metadata",
                        help="Output directory for CSVs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse each Excel file
    no_polyp = read_no_polyp(
        load_rows(args.input_dir, "TCIA-CTC-no-polyp-found"))
    medium = read_lesion_file(
        load_rows(args.input_dir, "TCIA-CTC-6-to-9-mm-polyps"), "medium_6_9mm")
    large = read_lesion_file(
        load_rows(args.input_dir, "TCIA-CTC-large-10-mm-polyps"), "large_10mm_plus")

    # Write per-category CSVs
    fieldnames = ["patient_id", "max_polyp_mm", "category"]

    for name, data in [("no_polyp", no_polyp), ("medium_6_9mm", medium),
                       ("large_10mm_plus", large)]:
        path = os.path.join(args.output_dir, f"{name}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Wrote {len(data)} rows to {path}")

    # TCIA lists two patients in two files: 0011 as no-polyp and 6-9 mm (9.0 mm),
    # 0216 as 6-9 mm (10.0 mm) and large (0 mm). Keep the larger finding for each,
    # so no patient carries two labels into the splits. The per-category CSVs
    # above stay as TCIA published them.
    severity = {"no_polyp": 0, "medium_6_9mm": 1, "large_10mm_plus": 2}
    by_patient = {}
    for row in no_polyp + medium + large:
        kept = by_patient.setdefault(row["patient_id"], dict(row))
        kept["category"] = max(kept["category"], row["category"], key=severity.get)
        kept["max_polyp_mm"] = max(kept["max_polyp_mm"], row["max_polyp_mm"])
    combined = list(by_patient.values())
    combined_path = os.path.join(args.output_dir, "acrin_combined.csv")
    with open(combined_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)
    print(f"Wrote {len(combined)} combined rows to {combined_path}")

    # Summary
    counts = {c: sum(r["category"] == c for r in combined) for c in severity}
    print(f"\nSummary: {counts['no_polyp']} no-polyp, {counts['medium_6_9mm']} medium "
          f"(6-9mm), {counts['large_10mm_plus']} large (>=10mm), {len(combined)} patients")


if __name__ == "__main__":
    main()
