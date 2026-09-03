"""Phase 0 representation evaluation over the fixed crop bank.

Scores one encoder on a battery of diagnostics so that MoCo checkpoints, an
ImageNet-initialised backbone, and a randomly-initialised backbone can be
compared on byte-identical input.  Run it once per encoder and diff the JSON.

No single number is sufficient here.  ``checkpoint_0199`` scores well on RankMe
and uniformity while being useless downstream, which is exactly why the label
based probes and the retrieval task are reported alongside them.

Metrics
    rankme          Effective rank of the 2048-d feature matrix (exp of the
                    entropy of its normalised singular values).  Higher means
                    more of the representation's capacity is in use.
    alignment       Mean squared distance between two augmented views of the
                    same crop, on L2-normalised features.  Lower is better.
    uniformity      log E exp(-2 ||f(x) - f(y)||^2) over random pairs.  More
                    negative means features spread more evenly on the sphere.
    knn_polyp       Patient-level 3-class polyp probe (the real downstream
                    task).  Balanced accuracy against the majority-class floor.
    knn_collection  ACRIN vs Pediatric.  A sanity check: any encoder that
                    cannot do this is broken.
    knn_zpos        k-NN regression of depth fraction.  Tests whether features
                    encode where in the body a slab came from.
    supine_prone    Retrieve another series of the same patient out of the ACRIN
                    pool.  Needs no manual labels and is the hardest anatomy test
                    in the battery.  Reported as ``any_series`` (whole pool, where
                    an alternate reconstruction of the same acquisition is often
                    available) and ``cross_position`` (supine query, prone-only
                    gallery, where no near-duplicate exists).

Usage:
    python scripts/eval/eval_repr.py --encoder imagenet \\
        --bank-dir /scratch/$USER/moco/eval --csv-dir metadata/csv_metadata
    python scripts/eval/eval_repr.py --encoder moco \\
        --checkpoint /scratch/$USER/moco/checkpoints/base/checkpoint_0199.pth.tar \\
        --bank-dir /scratch/$USER/moco/eval --csv-dir metadata/csv_metadata
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

ACRIN_COLLECTION = "CT-COLONOGRAPHY"


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------
def build_encoder(kind, checkpoint=None, arch="resnet50"):
    """Build a 2048-d feature extractor of the requested kind.

    Returns:
        Tuple of (model in eval mode, bool saying whether to apply ImageNet
        channel normalisation to its input).
    """
    if kind == "random":
        model = models.__dict__[arch](weights=None)
        model.fc = nn.Identity()
        return model.eval(), False

    if kind == "imagenet":
        model = models.__dict__[arch](weights="IMAGENET1K_V1")
        model.fc = nn.Identity()
        # ImageNet weights expect the statistics they were trained under; the
        # MoCo runs saw raw [0, 1] HU-window values with no normalisation, so
        # this flag differs per encoder rather than being a global constant.
        return model.eval(), True

    if kind == "moco":
        if not checkpoint:
            raise ValueError("--encoder moco requires --checkpoint")
        model = models.__dict__[arch](num_classes=128)
        dim_mlp = model.fc.weight.shape[1]
        model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = {
            k.replace("module.encoder_q.", ""): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("module.encoder_q.")
        }
        model.load_state_dict(state)
        model.fc = nn.Identity()
        return model.eval(), False

    raise ValueError("unknown encoder kind: %s" % kind)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def load_index(bank_dir):
    """Read bank_index.csv into a dict of column arrays."""
    path = os.path.join(bank_dir, "bank_index.csv")
    cols = {k: [] for k in
            ["row", "filename", "patient_id", "collection", "position",
             "z_frac", "z_index", "depth"]}
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in cols:
                # Banks built before position was recorded simply have no column.
                cols[k].append(r.get(k, "unknown"))
    cols["row"] = np.array(cols["row"], dtype=np.int64)
    cols["z_frac"] = np.array(cols["z_frac"], dtype=np.float32)
    for k in ["filename", "patient_id", "collection", "position"]:
        cols[k] = np.array(cols[k])
    return cols


def augment_pair(batch, generator):
    """Two independently augmented copies of *batch*, for the alignment metric.

    Mirrors the spatial and noise transforms used in pretraining
    (``moco/ct_dataset.py``) at the tensor level, which is enough to measure how
    invariant an encoder is to them.
    """
    out = []
    for _ in range(2):
        x = batch.clone()
        # Per-sample rather than per-batch flips: a shared decision would leave
        # the two views identical ~25% of the time and inflate alignment.
        for dim in (2, 3):
            m = torch.rand(x.shape[0], generator=generator) < 0.5
            if m.any():
                x[m] = torch.flip(x[m], dims=[dim])
        x = x + 0.05 * torch.randn(x.shape, generator=generator)
        out.append(x.clamp(0, 1))
    return out


@torch.no_grad()
def extract(model, bank, rows, normalize_imagenet, device, batch_size,
            augmented=False, seed=0):
    """Forward the given bank rows through the model and return features.

    Args:
        augmented: If True, returns two feature matrices from two independently
            augmented passes over the same crops (used for alignment).
    """
    model = model.to(device)
    gen = torch.Generator().manual_seed(seed)
    outs = ([], []) if augmented else ([],)

    for start in range(0, len(rows), batch_size):
        idx = rows[start:start + batch_size]
        # Bank rows are ascending within a chunk, which np.take needs anyway,
        # and BeeGFS is much happier with sequential reads than scattered ones.
        chunk = torch.from_numpy(np.asarray(bank[idx])).float().div_(255.0)

        views = augment_pair(chunk, gen) if augmented else [chunk]
        for slot, view in enumerate(views):
            if normalize_imagenet:
                view = (view - IMAGENET_MEAN) / IMAGENET_STD
            feat = model(view.to(device, non_blocking=True))
            outs[slot].append(feat.float().cpu())

    stacked = [torch.cat(o).numpy() for o in outs]
    return stacked if augmented else stacked[0]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def l2n(x, eps=1e-8):
    """Row-wise L2 normalisation."""
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def rankme(features, eps=1e-7):
    """Effective rank: exp of the entropy of the normalised singular values."""
    s = np.linalg.svd(features, compute_uv=False)
    p = s / (s.sum() + eps)
    return float(np.exp(-(p * np.log(p + eps)).sum()))


def alignment_uniformity(fa, fb, rng, n_pairs=20000):
    """Wang & Isola alignment (on paired views) and uniformity (on singles)."""
    a, b = l2n(fa), l2n(fb)
    align = float(np.mean(np.sum((a - b) ** 2, axis=1)))

    i = rng.integers(0, len(a), n_pairs)
    j = rng.integers(0, len(a), n_pairs)
    keep = i != j
    d2 = np.sum((a[i[keep]] - a[j[keep]]) ** 2, axis=1)
    unif = float(np.log(np.mean(np.exp(-2.0 * d2)) + 1e-12))
    return align, unif


def series_embeddings(features, index):
    """Mean-pool crop features to one L2-normalised vector per series."""
    names, inverse = np.unique(index["filename"], return_inverse=True)

    # reduceat over a sorted copy rather than np.add.at, which is unbuffered and
    # roughly two orders of magnitude slower at this row count.
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(len(names)))
    pooled = np.add.reduceat(features[order], bounds, axis=0)
    counts = np.bincount(inverse, minlength=len(names)).reshape(-1, 1)
    pooled = pooled / np.maximum(counts, 1)

    first = np.zeros(len(names), dtype=np.int64)
    first[inverse[::-1]] = np.arange(len(inverse))[::-1]
    return (names, l2n(pooled), index["patient_id"][first],
            index["collection"][first], index["position"][first])


def knn_score(train_x, train_y, test_x, test_y, k=15):
    """Cosine k-NN balanced accuracy, reported against the majority-class floor."""
    if len(set(train_y)) < 2 or len(test_y) == 0:
        return None
    k = min(k, len(train_y))
    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    _, counts = np.unique(test_y, return_counts=True)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(test_y, pred)),
        "accuracy": float(np.mean(pred == test_y)),
        "majority_floor": float(counts.max() / counts.sum()),
        "n_train": int(len(train_y)),
        "n_test": int(len(test_y)),
    }


def _retrieve(emb, patients, query_idx, gallery_idx):
    """Rank *gallery_idx* against each query and score same-patient hits."""
    if len(query_idx) == 0 or len(gallery_idx) == 0:
        return None

    sims = emb[query_idx] @ emb[gallery_idx].T
    # A query that is also in the gallery must never retrieve itself.
    same = query_idx[:, None] == gallery_idx[None, :]
    sims[same] = -np.inf

    order = np.argsort(-sims, axis=1)[:, :5]
    truth = patients[query_idx][:, None] == patients[gallery_idx][order]

    # Only queries with at least one correct answer in the gallery are scorable.
    correct = (patients[query_idx][:, None] == patients[gallery_idx][None, :]) & ~same
    n_correct = correct.sum(axis=1)
    scorable = n_correct > 0
    if not scorable.any():
        return None

    # Chance is per-query: correct answers over the entries it could have picked.
    reachable = len(gallery_idx) - same.sum(axis=1)
    chance = float(np.mean(n_correct[scorable] / np.maximum(reachable[scorable], 1)))

    return {
        "top1": float(truth[scorable, 0].mean()),
        "top5": float(truth[scorable].any(axis=1).mean()),
        "chance_top1": chance,
        "n_queries": int(scorable.sum()),
        "n_gallery": int(len(gallery_idx)),
    }


def supine_prone_retrieval(emb, patients, collections, positions):
    """Retrieve another series of the same patient from the ACRIN pool.

    Reported two ways, because they measure different things.

    ``any_series`` queries every ACRIN series against every other.  Most patients
    contributed four series, and several of those are alternate reconstructions of
    the *same* acquisition, so a near-duplicate is often available as a correct
    answer.  An encoder can score well here by matching low-level texture, which
    is the same shortcut that made the pretext task collapse.

    ``cross_position`` queries supine series against a prone-only gallery.  The
    patient was re-scanned lying the other way, so gas and fluid have moved and
    collapsed bowel has opened: no near-duplicate exists and the match has to come
    from anatomy.  This is the honest number; the gap between the two says how
    much of ``any_series`` was duplicate-matching.
    """
    mask = collections == ACRIN_COLLECTION
    emb, patients, positions = emb[mask], patients[mask], positions[mask]

    uniq, counts = np.unique(patients, return_counts=True)
    multi = set(uniq[counts >= 2])
    idx = np.where(np.isin(patients, list(multi)))[0]

    supine = np.where(positions == "supine")[0]
    prone = np.where(positions == "prone")[0]

    return {
        "any_series": _retrieve(emb, patients, idx, np.arange(len(emb))),
        "cross_position": _retrieve(emb, patients, supine, prone),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_label_csv(path):
    """Read a split CSV into a dict of filename -> int label."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {r["filename"]: int(r["label"]) for r in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser(description="Phase 0 representation evaluation")
    parser.add_argument("--encoder", required=True, choices=["moco", "imagenet", "random"])
    parser.add_argument("--checkpoint", default=None, help="Required for --encoder moco")
    parser.add_argument("--bank-dir", required=True, help="Directory holding bank.npy")
    parser.add_argument("--csv-dir", default="metadata/csv_metadata",
                        help="Directory holding labels_{train,val}.csv")
    parser.add_argument("--arch", default="resnet50")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--knn-k", default=15, type=int)
    parser.add_argument("--align-crops", default=4096, type=int,
                        help="Crops sampled for the alignment/uniformity pass")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None, help="Where to write the JSON report")
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    bank = np.load(os.path.join(args.bank_dir, "bank.npy"), mmap_mode="r")
    index = load_index(args.bank_dir)
    rows = index["row"]
    print("Bank %s, %d indexed crops from %d series"
          % (bank.shape, len(rows), len(np.unique(index["filename"]))))

    model, norm = build_encoder(args.encoder, args.checkpoint, args.arch)
    tag = args.encoder + (":" + os.path.basename(args.checkpoint) if args.checkpoint else "")
    print("Encoder %s (imagenet-normalised input: %s), device %s" % (tag, norm, device))

    print("Extracting features ...")
    feats = extract(model, bank, rows, norm, device, args.batch_size)
    print("  features", feats.shape)

    report = {"encoder": args.encoder, "checkpoint": args.checkpoint,
              "tag": tag, "n_crops": int(len(feats))}

    report["rankme"] = rankme(feats)
    report["rankme_max"] = int(feats.shape[1])

    sample = np.sort(rng.choice(len(rows), min(args.align_crops, len(rows)), replace=False))
    fa, fb = extract(model, bank, rows[sample], norm, device, args.batch_size,
                     augmented=True, seed=args.seed)
    align, unif = alignment_uniformity(fa, fb, rng)
    report["alignment"] = align
    report["uniformity"] = unif

    names, emb, patients, collections, positions = series_embeddings(feats, index)
    report["n_series"] = int(len(names))

    # Collection probe: split by patient so no patient straddles the split.
    uniq_pat = np.unique(patients)
    held = set(rng.choice(uniq_pat, len(uniq_pat) // 2, replace=False))
    is_test = np.array([p in held for p in patients])
    report["knn_collection"] = knn_score(
        emb[~is_test], collections[~is_test], emb[is_test], collections[is_test], args.knn_k)

    # Depth-fraction regression, on crops rather than series (z is per crop).
    csample = np.sort(rng.choice(len(rows), min(8000, len(rows)), replace=False))
    cx, cz = l2n(feats[csample]), index["z_frac"][csample]
    split = len(csample) // 2
    reg = KNeighborsRegressor(n_neighbors=args.knn_k, metric="cosine", weights="distance")
    reg.fit(cx[:split], cz[:split])
    pred = reg.predict(cx[split:])
    report["knn_zpos"] = {
        "mae": float(np.mean(np.abs(pred - cz[split:]))),
        "mae_predict_mean": float(np.mean(np.abs(cz[split:] - cz[:split].mean()))),
    }

    # Polyp probe: patient-level splits already exist as CSVs.
    train_lbl = load_label_csv(os.path.join(args.csv_dir, "labels_train.csv"))
    val_lbl = load_label_csv(os.path.join(args.csv_dir, "labels_val.csv"))
    tr = np.array([n in train_lbl for n in names])
    va = np.array([n in val_lbl for n in names])
    if tr.any() and va.any():
        report["knn_polyp"] = knn_score(
            emb[tr], np.array([train_lbl[n] for n in names[tr]]),
            emb[va], np.array([val_lbl[n] for n in names[va]]), args.knn_k)
    else:
        report["knn_polyp"] = None
        print("WARNING: no overlap between the bank and the label CSVs")

    report["supine_prone"] = supine_prone_retrieval(emb, patients, collections, positions)

    out = args.output or os.path.join(args.bank_dir, "eval_%s.json" % args.encoder)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== %s ===" % tag)
    print("  RankMe            %8.1f / %d" % (report["rankme"], report["rankme_max"]))
    print("  Alignment         %8.4f  (lower better)" % report["alignment"])
    print("  Uniformity        %8.4f  (more negative better)" % report["uniformity"])
    if report["knn_collection"]:
        c = report["knn_collection"]
        print("  kNN collection    %8.3f bal-acc  (floor %.3f)"
              % (c["balanced_accuracy"], c["majority_floor"]))
    z = report["knn_zpos"]
    print("  kNN depth MAE     %8.4f  (predict-mean %.4f)" % (z["mae"], z["mae_predict_mean"]))
    if report["knn_polyp"]:
        p = report["knn_polyp"]
        print("  kNN polyp         %8.3f bal-acc  (floor %.3f, n=%d/%d)"
              % (p["balanced_accuracy"], p["majority_floor"], p["n_train"], p["n_test"]))
    for kind in ("any_series", "cross_position"):
        r = (report["supine_prone"] or {}).get(kind)
        if r:
            print("  retrieval %-14s %6.3f top1  %.3f top5  (chance %.5f, n=%d)"
                  % (kind, r["top1"], r["top5"], r["chance_top1"], r["n_queries"]))
    print("\nWrote %s" % out)


if __name__ == "__main__":
    main()
