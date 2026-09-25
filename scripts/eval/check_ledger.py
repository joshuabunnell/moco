"""Check that the numbers in the docs match the ledger, and the ledger the JSONs.

Protocol rule 3 says numbers are never typed by hand: `log_experiment.py`
writes the ledger and prints the markdown row.  This checks that nothing broke
that chain afterwards, which is exactly the kind of drift an AI editing prose
can introduce without noticing:

1. Every results-table row in `docs/experiments.md` whose run id is in the
   ledger shows the ledger's values, at the precision the table prints.
2. Every ledger row still matches its eval JSON, re-flattened the same way
   `log_experiment.py` does (skipped per row when the JSON is not on disk).
3. The ImageNet bar hard-coded in `log_experiment.py` is the ledger's value.

Exit status is the number of mismatches, so it can gate a commit.

Usage:
    python scripts/eval/check_ledger.py [--eval-dir /scratch/$USER/moco/eval]
"""

import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

from log_experiment import IMAGENET_CROSS_POSITION, flatten  # noqa: E402

# Table columns as printed by log_experiment.py, with the precision it prints.
# "vs ImageNet" is derived, so it is recomputed rather than looked up.
COLUMNS = [
    ("cross_position_top1", 3), (None, None), ("any_series_top1", 3),
    ("rankme", 1), ("alignment_cos", 4), ("knn_zpos_mae", 4),
    ("knn_polyp_bal_acc", 3),
]
NUMERIC = [f for f, _ in COLUMNS if f] + [
    "cross_position_top5", "uniformity", "knn_collection_bal_acc",
    "knn_prep_bal_acc", "knn_contrast_bal_acc", "n_crops", "n_series"]


def run_key(label):
    """Normalise a table label and a ledger run_id to the same key.

    Phase 0's table predates the logger and writes ``P0:moco checkpoint_0199``
    where the ledger has ``P0:moco:checkpoint_0199.pth.tar``.
    """
    return label.strip().replace(" ", ":").replace(".pth.tar", "")


def check_tables(doc, ledger, imagenet_bar):
    errors = 0
    rows = 0
    for lineno, line in enumerate(open(doc), 1):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 1 + len(COLUMNS) or not re.match(r"^(P0r?|E\d+b?)\b", cells[0]):
            continue
        row = ledger.get(run_key(cells[0]))
        if row is None:
            continue
        rows += 1
        # A row printed against an older bar (P0 used 0.493) is judged on that.
        bar = 0.4925 if cells[0].startswith("P0:") else imagenet_bar
        for (field, places), shown in zip(COLUMNS, cells[1:]):
            if field is None:
                want = "%.2fx" % (float(row["cross_position_top1"]) / bar)
            else:
                want = "%.*f" % (places, float(row[field]))
            if shown != want:
                errors += 1
                print("%s:%d  %s  %s: doc says %s, ledger says %s"
                      % (doc, lineno, cells[0], field or "vs ImageNet", shown, want))
    print("tables: %d rows checked against the ledger" % rows)
    return errors


def check_jsons(ledger, eval_dir):
    errors = checked = 0
    for run_id, row in ledger.items():
        path = os.path.join(eval_dir, row["eval_json"])
        if not os.path.exists(path):
            continue
        # P0 rows were scored on the pre-rebuild bank; their JSONs moved aside.
        if run_id.startswith("P0:"):
            path = os.path.join(eval_dir, "p0_pt_legacy", row["eval_json"])
            if not os.path.exists(path):
                continue
        with open(path) as f:
            fresh = flatten(json.load(f), run_id, row["note"], path)
        # Several runs share one JSON name (eval_imagenet.json across P0/P0r),
        # so a mismatch is only real if the checkpoint matches too.
        if (fresh["checkpoint"] or "") != row["checkpoint"]:
            continue
        checked += 1
        for field in NUMERIC:
            a, b = fresh[field], row[field]
            if a is None and b in ("", None):
                continue
            if a is None or b in ("", None) or abs(float(a) - float(b)) > 1e-9:
                errors += 1
                print("ledger %s  %s: ledger %s, %s has %s" % (run_id, field, b, path, a))
    print("ledger: %d rows re-derived from their JSON" % checked)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ledger", default="docs/experiment_results.csv")
    parser.add_argument("--doc", default="docs/experiments.md")
    parser.add_argument("--eval-dir", default="/scratch/%s/moco/eval" % os.environ.get("USER", ""))
    args = parser.parse_args()

    with open(args.ledger, newline="") as f:
        ledger = {run_key(r["run_id"]): r for r in csv.DictReader(f)}

    errors = 0
    bar = ledger.get(run_key("P0r:imagenet"))
    if bar and abs(float(bar["cross_position_top1"]) - IMAGENET_CROSS_POSITION) > 5e-5:
        errors += 1
        print("log_experiment.IMAGENET_CROSS_POSITION = %s, ledger P0r:imagenet = %s"
              % (IMAGENET_CROSS_POSITION, bar["cross_position_top1"]))

    errors += check_tables(args.doc, ledger, IMAGENET_CROSS_POSITION)
    if os.path.isdir(args.eval_dir):
        errors += check_jsons(ledger, args.eval_dir)
    else:
        print("ledger: %s not found, JSON check skipped" % args.eval_dir)

    print("OK" if errors == 0 else "%d mismatch(es)" % errors)
    sys.exit(min(errors, 255))


if __name__ == "__main__":
    main()
