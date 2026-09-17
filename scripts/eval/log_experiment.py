"""Append frozen-encoder eval results to the experiment ledger.

`eval_repr.py` writes one JSON per scored encoder.  Transcribing those numbers
into notes by hand is where errors enter a comparison, so this flattens them
into `docs/experiment_results.csv` (upserted on ``run_id``) and prints the same
row as markdown for pasting into `docs/experiments.md`.

Alignment is also recorded as the cosine between two augmented views
(``1 - alignment/2``), because that is the form the Phase 2 kill gate is
expressed in.

Usage:
    python scripts/eval/log_experiment.py --run-id E1 --note "independent crops" \
        /scratch/$USER/moco/eval/eval_moco_e1_checkpoint_0199.json
"""

import argparse
import csv
import datetime
import json
import os

FIELDS = [
    "run_id", "date", "tag", "checkpoint", "note",
    "cross_position_top1", "cross_position_top5", "any_series_top1",
    "rankme", "alignment_cos", "uniformity",
    "knn_zpos_mae", "knn_polyp_bal_acc", "knn_collection_bal_acc",
    "n_crops", "n_series", "eval_json",
]

# Reported alongside every run: the Phase 0 bar to beat and the chance rate.
IMAGENET_CROSS_POSITION = 0.4925


def flatten(report, run_id, note, path):
    """One eval JSON to one ledger row."""
    def probe(name, key):
        block = report.get(name) or {}
        return block.get(key)

    return {
        "run_id": run_id,
        "date": datetime.date.today().isoformat(),
        "tag": report.get("tag"),
        "checkpoint": report.get("checkpoint") or "",
        "note": note,
        "cross_position_top1": probe_nested(report, "cross_position", "top1"),
        "cross_position_top5": probe_nested(report, "cross_position", "top5"),
        "any_series_top1": probe_nested(report, "any_series", "top1"),
        "rankme": report.get("rankme"),
        "alignment_cos": (None if report.get("alignment") is None
                          else 1.0 - report["alignment"] / 2.0),
        "uniformity": report.get("uniformity"),
        "knn_zpos_mae": probe("knn_zpos", "mae"),
        "knn_polyp_bal_acc": probe("knn_polyp", "balanced_accuracy"),
        "knn_collection_bal_acc": probe("knn_collection", "balanced_accuracy"),
        "n_crops": report.get("n_crops"),
        "n_series": report.get("n_series"),
        "eval_json": os.path.basename(path),
    }


def probe_nested(report, variant, key):
    block = (report.get("supine_prone") or {}).get(variant) or {}
    return block.get(key)


def upsert(ledger_path, rows):
    """Write *rows* into the ledger, replacing any row with the same run_id."""
    existing = []
    if os.path.exists(ledger_path):
        with open(ledger_path, newline="") as f:
            existing = list(csv.DictReader(f))

    incoming = {r["run_id"] for r in rows}
    merged = [r for r in existing if r["run_id"] not in incoming] + rows

    os.makedirs(os.path.dirname(ledger_path) or ".", exist_ok=True)
    with open(ledger_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(merged)
    return len(merged)


def fmt(value, places=3):
    if value in (None, ""):
        return "—"
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return str(value)


def main():
    parser = argparse.ArgumentParser(description="Log eval results to the ledger")
    parser.add_argument("eval_json", nargs="+", help="eval_repr.py JSON report(s)")
    parser.add_argument("--run-id", required=True,
                        help="Ladder id, e.g. E1.  One json: used as-is.  "
                             "Several: suffixed with each report's tag.")
    parser.add_argument("--note", default="", help="What changed in this run")
    parser.add_argument("--ledger", default="docs/experiment_results.csv")
    args = parser.parse_args()

    rows = []
    for path in args.eval_json:
        with open(path) as f:
            report = json.load(f)
        run_id = args.run_id if len(args.eval_json) == 1 \
            else f"{args.run_id}:{report.get('tag', os.path.basename(path))}"
        rows.append(flatten(report, run_id, args.note, path))

    total = upsert(args.ledger, rows)
    print(f"Ledger {args.ledger}: {len(rows)} row(s) written, {total} total\n")

    print("| Run | cross_pos top1 | vs ImageNet | any_series top1 | RankMe | "
          "align cos | zpos MAE | polyp bal-acc |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        cp = r["cross_position_top1"]
        ratio = "—" if cp is None else f"{cp / IMAGENET_CROSS_POSITION:.2f}x"
        print(f"| {r['run_id']} | {fmt(cp)} | {ratio} | {fmt(r['any_series_top1'])} | "
              f"{fmt(r['rankme'], 1)} | {fmt(r['alignment_cos'], 4)} | "
              f"{fmt(r['knn_zpos_mae'], 4)} | {fmt(r['knn_polyp_bal_acc'])} |")


if __name__ == "__main__":
    main()
