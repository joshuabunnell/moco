"""Write a TCIA manifest holding only the series not yet complete on disk.

The NBIA retriever's own "download missing" mode diffs the manifest against the
``metadata.csv`` it wrote on a previous run, not against the files that are
actually present.  A scratch purge deletes the ``.dcm`` files but leaves that
CSV, so the retriever concludes everything is present and fetches almost
nothing (observed 2026-09-02: 8 series pulled against ~3300 missing).

Doing the diff here against real directories avoids that failure mode entirely,
and makes the download job resumable: rerun it and it picks up whatever is
still absent.

``series_catalog.csv`` supplies each series' relative path and expected image
count.  A series listed in the manifest but absent from the catalog is treated
as missing, so a stale catalog can only cause a redundant download, never a
silent skip.

Usage:
    python scripts/pending_series.py \\
        --manifest metadata/manifest.tcia \\
        --catalog metadata/series_catalog.csv \\
        --raw-dir /scratch/$USER/moco/raw \\
        --output /scratch/$USER/moco/tcia_pending/manifest.tcia
"""

import argparse
import csv
import os


def read_manifest(path):
    """Split a .tcia manifest into its header lines and its series UIDs."""
    header, uids = [], []
    in_series = False
    for line in open(path):
        stripped = line.strip()
        if in_series:
            if stripped:
                uids.append(stripped)
        else:
            header.append(line.rstrip("\n"))
            if stripped.startswith("ListOfSeriesToDownload"):
                in_series = True
    return header, uids


def read_catalog(path):
    """Map series UID -> (relative path, expected image count)."""
    catalog = {}
    if not os.path.isfile(path):
        return catalog
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                count = int(row["Number of Images"])
            except (KeyError, ValueError):
                continue
            catalog[row["Series UID"]] = (row["File Location"].lstrip("./"), count)
    return catalog


def is_complete(raw_dir, entry):
    """A series is complete when its directory holds at least the expected .dcm count."""
    rel, expected = entry
    path = os.path.join(raw_dir, rel)
    if not os.path.isdir(path):
        return False
    return sum(1 for f in os.listdir(path) if f.endswith(".dcm")) >= expected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    header, uids = read_manifest(args.manifest)
    catalog = read_catalog(args.catalog)
    if not catalog:
        print(f"No catalog at {args.catalog}, treating all {len(uids)} series as pending.")

    pending = [u for u in uids
               if u not in catalog or not is_complete(args.raw_dir, catalog[u])]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        fh.write("\n".join(header) + "\n")
        fh.write("\n".join(pending) + "\n")

    print(f"{len(uids) - len(pending)} of {len(uids)} series already complete; "
          f"{len(pending)} pending -> {args.output}")


if __name__ == "__main__":
    main()
