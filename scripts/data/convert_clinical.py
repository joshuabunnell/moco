"""Decode the ACRIN 6664 clinical-data TSV into a readable CSV.

TCIA's Version 2 release (2026-08-24) added a clinical table covering 752 of the
825 imaged subjects.  It ships as coded integers whose meaning lives in a
separate machine-readable data dictionary, so neither file is usable alone.

This script joins the two: ``List`` variables are mapped to their option labels,
column names come from the dictionary's question text, and the subject key is
renamed ``patient_id`` so the result joins to ``acrin_combined.csv`` and to
``series_catalog.csv``'s ``Subject ID``.

Usage:
    python scripts/data/convert_clinical.py \
        --input-dir metadata/raw_metadata/v2_2026-08-24 \
        --output-dir metadata/csv_metadata
"""

import argparse
import os
import re

import pandas as pd

CLINICAL = "CT-Colonography_clinical-data_v01_20260824.tsv"
DICTIONARY = "CT-Colonography_machine_readable_data_dictionary_v01_20260824-1.tsv"


def slugify(text, used):
    """Question text to a short snake_case column name, deduped against *used*."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    if len(slug) > 44:
        # Cut back to a word boundary; a mid-word truncation reads as a typo.
        slug = slug[:44].rsplit("_", 1)[0]
    slug = slug or "unnamed"
    candidate, n = slug, 2
    while candidate in used:
        candidate = f"{slug}_{n}"
        n += 1
    used.add(candidate)
    return candidate


def build_dictionary(path):
    """Read the data dictionary into {var: question} and {var: {code: label}}."""
    dd = pd.read_csv(path, sep="\t", dtype=str)
    questions = dd.drop_duplicates("Var_name").set_index("Var_name")["Question_Text"]
    options = {}
    coded = dd.dropna(subset=["coded_value"])
    for var, group in coded.groupby("Var_name"):
        options[var] = dict(zip(group["coded_value"].str.strip(),
                                group["option_list"].str.strip()))
    return questions.to_dict(), options


def main():
    parser = argparse.ArgumentParser(description="Decode ACRIN clinical data to CSV")
    parser.add_argument("--input-dir", default="metadata/raw_metadata/v2_2026-08-24",
                        help="Directory holding the clinical TSV and its dictionary")
    parser.add_argument("--output-dir", default="metadata/csv_metadata",
                        help="Output directory for the decoded CSV")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    clinical = pd.read_csv(os.path.join(args.input_dir, CLINICAL), sep="\t", dtype=str)
    questions, options = build_dictionary(os.path.join(args.input_dir, DICTIONARY))

    undocumented = [c for c in clinical.columns if c not in questions]
    if undocumented:
        print(f"Warning: {len(undocumented)} columns absent from the dictionary: "
              f"{undocumented}")

    out = pd.DataFrame(index=clinical.index)
    used = {"patient_id"}
    for col in clinical.columns:
        values = clinical[col].str.strip()
        if col in options:
            # Unmapped codes are kept verbatim rather than dropped to NaN, so a
            # dictionary that lags the data stays visible instead of silent.
            values = values.map(lambda v, o=options[col]: o.get(v, v))
        name = "patient_id" if col == "Blinded_ID" else slugify(questions.get(col, col), used)
        out[name] = values

    path = os.path.join(args.output_dir, "clinical_data.csv")
    out.to_csv(path, index=False)
    print(f"Wrote {len(out)} rows x {len(out.columns)} columns to {path}")


if __name__ == "__main__":
    main()
