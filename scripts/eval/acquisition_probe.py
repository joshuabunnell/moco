"""Does cross-position retrieval match scanner and body instead of anatomy?

A patient's supine and prone scans come from one session: same scanner, kernel,
field of view and body.  ``eval_repr.py`` scores an encoder on picking the right
prone scan out of all 671, so part of any score could be acquisition matching.
Pre-registered in ``docs/experiments.md``, "Acquisition probe".

Two stages:

``describe`` (CPU) writes ``acquisition.csv`` beside the bank: one row per ACRIN
    series with scanner, kernel, kVp, reconstruction diameter, extent and body
    cross-section area, plus the patient's prep and contrast.  It also scores
    **A**, acquisition-only retrieval, into ``acquisition_only.json``.

``score`` (GPU) extracts one encoder's features on the bank and scores **B**,
    retrieval inside a per-query look-alike gallery: the patient's prone
    scan(s) plus the 20 other prone scans nearest to the query under A's
    distance.  Writes ``acq_<tag>.json``.

Usage:
    python scripts/eval/acquisition_probe.py describe --bank-dir $EVAL_DIR \\
        --tensor-dir $TENSOR_ACRIN --csv-dir metadata/csv_metadata
    python scripts/eval/acquisition_probe.py score --bank-dir $EVAL_DIR \\
        --encoder moco --checkpoint .../e1/checkpoint_0199.pth.tar --output ...
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from eval_repr import (  # noqa: E402
    ACRIN_COLLECTION, CONFOUNDERS, build_encoder, extract, load_clinical,
    load_index, series_embeddings)

CATEGORICAL = ["scanner", "kernel", "prep", "contrast"]
CONTINUOUS = ["kvp", "recon_diameter", "extent_y", "extent_x", "length_z",
              "body_area_25", "body_area_50", "body_area_75"]
MISMATCH_PENALTY = 10.0
LOOKALIKES = 20
BODY_HU = -500


def _header(series_path):
    import pydicom
    files = sorted(glob.glob(os.path.join(series_path, "*.dcm")))
    if not files:
        return {}
    ds = pydicom.dcmread(files[0], stop_before_pixels=True)

    def get(name):
        value = getattr(ds, name, "")
        return str(value).strip() if value is not None else ""

    return {
        "scanner": (get("Manufacturer") + " " + get("ManufacturerModelName")).strip(),
        "kernel": get("ConvolutionKernel"),
        "kvp": get("KVP"),
        "recon_diameter": get("ReconstructionDiameter"),
    }


def _body_areas(npy_path):
    volume = np.load(npy_path, mmap_mode="r")
    nz, ny, nx = volume.shape
    areas = {}
    for frac in (25, 50, 75):
        plane = np.asarray(volume[min(nz - 1, int(nz * frac / 100))])
        areas["body_area_%d" % frac] = float((plane > BODY_HU).sum())  # 1 mm voxels
    return {"extent_y": ny, "extent_x": nx, "length_z": nz, **areas}


def describe(args):
    index = load_index(args.bank_dir)
    names, first = np.unique(index["filename"], return_index=True)
    keep = index["collection"][first] == ACRIN_COLLECTION
    names, first = names[keep][:args.limit], first[keep][:args.limit]

    with open(os.path.join(args.tensor_dir, "manifest.csv")) as f:
        series_path = {r["filename"]: r["series_path"] for r in csv.DictReader(f)}
    clinical = load_clinical(os.path.join(args.csv_dir, "clinical_data.csv"))

    rows = []
    for n, (name, i) in enumerate(zip(names, first)):
        pid = index["patient_id"][i]
        row = {"filename": name, "patient_id": pid, "position": index["position"][i]}
        row.update(_header(series_path.get(name, "")))
        row.update(_body_areas(os.path.join(args.tensor_dir, name)))
        row["prep"] = clinical.get(pid, {}).get(CONFOUNDERS["knn_prep"], "")
        row["contrast"] = clinical.get(pid, {}).get(CONFOUNDERS["knn_contrast"], "")
        rows.append(row)
        if n % 200 == 0:
            print("  described %d / %d" % (n, len(names)), flush=True)

    out = os.path.join(args.bank_dir, "acquisition.csv")
    fields = ["filename", "patient_id", "position"] + CATEGORICAL + CONTINUOUS
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("Wrote", out)

    table = load_table(out)
    report = {"acquisition_only": acquisition_only(table),
              "coverage": {c: float(np.mean(table[c] != "")) for c in CATEGORICAL}}
    with open(os.path.join(args.bank_dir, "acquisition_only.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def load_table(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    table = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    cont = []
    for c in CONTINUOUS:
        v = np.array([float(x) if x not in ("", "None") else np.nan for x in table[c]])
        v = np.where(np.isnan(v), np.nanmean(v), v)
        cont.append((v - v.mean()) / (v.std() or 1.0))
    table["_cont"] = np.stack(cont, axis=1)
    return table


def acquisition_distance(table, q, g):
    """Distance from query rows *q* to gallery rows *g*: z-scored euclidean + mismatches."""
    c = table["_cont"]
    d = np.sqrt(((c[q][:, None, :] - c[g][None, :, :]) ** 2).sum(-1))
    for k in CATEGORICAL:
        d += MISMATCH_PENALTY * (table[k][q][:, None] != table[k][g][None, :])
    return d


def _queries(table):
    supine = np.where(table["position"] == "supine")[0]
    prone = np.where(table["position"] == "prone")[0]
    has_prone = np.isin(table["patient_id"][supine], table["patient_id"][prone])
    return supine[has_prone], prone


def acquisition_only(table):
    q, g = _queries(table)
    d = acquisition_distance(table, q, g)
    order = np.argsort(d, axis=1, kind="stable")
    hit = table["patient_id"][g][order] == table["patient_id"][q][:, None]
    return {"top1": float(hit[:, 0].mean()), "top5": float(hit[:, :5].any(1).mean()),
            "n_queries": int(len(q)), "n_gallery": int(len(g))}


def lookalike_retrieval(table, emb_by_name):
    """Top-1/top-5 within each query's patient-match + 20 look-alike gallery."""
    q, g = _queries(table)
    d = acquisition_distance(table, q, g)
    pat = table["patient_id"]
    emb = np.stack([emb_by_name[n] for n in table["filename"]])
    top1, top5, chance, full_top1 = [], [], [], []
    for i, qi in enumerate(q):
        own = g[pat[g] == pat[qi]]
        others = np.where(pat[g] != pat[qi])[0]
        near = g[others[np.argsort(d[i, others], kind="stable")[:LOOKALIKES]]]
        gallery = np.concatenate([own, near])
        ranked = gallery[np.argsort(-(emb[gallery] @ emb[qi]), kind="stable")]
        truth = pat[ranked] == pat[qi]
        top1.append(truth[0])
        top5.append(truth[:5].any())
        chance.append(len(own) / len(gallery))
        full = g[np.argsort(-(emb[g] @ emb[qi]), kind="stable")]
        full_top1.append(pat[full[0]] == pat[qi])
    return {"top1": float(np.mean(top1)), "top5": float(np.mean(top5)),
            "chance_top1": float(np.mean(chance)),
            "full_gallery_top1": float(np.mean(full_top1)),
            "n_queries": int(len(q)), "gallery_size": LOOKALIKES + 1}


def score(args):
    import torch
    torch.manual_seed(0)  # the random-init control is then the same network every run
    table = load_table(os.path.join(args.bank_dir, "acquisition.csv"))
    bank = np.load(os.path.join(args.bank_dir, "bank.npy"), mmap_mode="r")
    index = load_index(args.bank_dir)
    device = args.device if torch.cuda.is_available() else "cpu"

    model, norm = build_encoder(args.encoder, args.checkpoint)
    tag = args.encoder + (":" + args.checkpoint if args.checkpoint else "")
    feats = extract(model, bank, index["row"], norm, device, args.batch_size)
    names, emb = series_embeddings(feats, index)[:2]
    report = {"tag": tag, "encoder": args.encoder, "checkpoint": args.checkpoint,
              "lookalike": lookalike_retrieval(table, dict(zip(names, emb)))}
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    d = sub.add_parser("describe")
    d.add_argument("--bank-dir", required=True)
    d.add_argument("--tensor-dir", required=True)
    d.add_argument("--csv-dir", default="metadata/csv_metadata")
    d.add_argument("--limit", default=None, type=int, help="first N series only, to test")
    s = sub.add_parser("score")
    s.add_argument("--bank-dir", required=True)
    s.add_argument("--encoder", required=True, choices=["moco", "imagenet", "random"])
    s.add_argument("--checkpoint", default=None)
    s.add_argument("--output", required=True)
    s.add_argument("--batch-size", default=256, type=int)
    s.add_argument("--device", default="cuda")
    args = parser.parse_args()
    describe(args) if args.stage == "describe" else score(args)


if __name__ == "__main__":
    main()
