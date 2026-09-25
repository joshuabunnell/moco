"""Figures that show, without jargon, what the frozen-encoder metrics measure.

Every figure is drawn from the same fixed crop bank ``eval_repr.py`` scores, so a
picture never disagrees with a ledger number.

``embed`` (GPU) saves one encoder's features on the bank to
    ``$EVAL_DIR/figures/<name>.npz``: per-scan embeddings (float32) and per-crop
    features (float16, for the depth map).
``plot`` (CPU) draws, into ``--out-dir``:

1. ``progress.png``   cross-position top-1 per run, from the ledger, with the
                      ImageNet bar and chance.  The headline.
2. ``retrieval.png``  randomly chosen supine queries (not picked for success)
                      and the prone scan each encoder ranks first.
3. ``similarity.png`` supine/prone cosine similarity, same patient vs different
                      patients.  Separation is what makes retrieval work.
4. ``depth_map.png``  UMAP of crops coloured by depth along the scan: does the
                      encoder organise crops by where they sit in the body?

Usage:
    python scripts/eval/figures.py embed --bank-dir $EVAL_DIR --encoder imagenet --name imagenet
    python scripts/eval/figures.py embed --bank-dir $EVAL_DIR --encoder moco \\
        --checkpoint .../e1/checkpoint_0199.pth.tar --name e1
    python scripts/eval/figures.py plot --bank-dir $EVAL_DIR --out-dir docs/figures \\
        --compare imagenet e1
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from eval_repr import (  # noqa: E402
    ACRIN_COLLECTION, build_encoder, extract, load_index, series_embeddings)

# Reference palette (dataviz skill, light mode): blue = our model, gray =
# reference encoders, sequential blues for depth.
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABELS = {"random": "Untrained", "imagenet": "ImageNet"}
SEED = 0


def embed(args):
    import torch
    torch.manual_seed(SEED)
    bank = np.load(os.path.join(args.bank_dir, "bank.npy"), mmap_mode="r")
    index = load_index(args.bank_dir)
    model, norm = build_encoder(args.encoder, args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feats = extract(model, bank, index["row"], norm, device, 256)
    names, emb, patients, collections, positions = series_embeddings(feats, index)
    out = os.path.join(args.bank_dir, "figures")
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, args.name + ".npz"), names=names, emb=emb,
             patients=patients, collections=collections, positions=positions,
             crops=feats.astype(np.float16), checkpoint=str(args.checkpoint))
    print("Wrote", os.path.join(out, args.name + ".npz"))


# ---------------------------------------------------------------------------
def _style(plt):
    plt.rcParams.update({
        "font.size": 11, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.dpi": 200, "savefig.bbox": "tight"})


def _label(name):
    return LABELS.get(name, "MoCo " + name.upper())


def _load(bank_dir, name):
    z = np.load(os.path.join(bank_dir, "figures", name + ".npz"), allow_pickle=False)
    return {k: z[k] for k in z.files}


def _pairs(d):
    """Supine query indices and prone gallery indices, ACRIN only, as eval_repr."""
    acr = d["collections"] == ACRIN_COLLECTION
    sup = np.where(acr & (d["positions"] == "supine"))[0]
    pro = np.where(acr & (d["positions"] == "prone"))[0]
    sup = sup[np.isin(d["patients"][sup], d["patients"][pro])]
    return sup, pro


def plot_progress(plt, ledger, out):
    rows = {r["run_id"]: r for r in csv.DictReader(open(ledger))}
    order = [("P0r:random", "Untrained"), ("P0r:imagenet", "ImageNet"),
             ("P0r:moco:checkpoint_0199.pth.tar", "MoCo, original recipe (CT colon + pediatric)"),
             ("E0:moco:checkpoint_0199.pth.tar", "E0: cleaned data"),
             ("E1:moco:checkpoint_0199.pth.tar", "E1: two overlapping crops"),
             ("E1b:moco:checkpoint_0199.pth.tar", "E1b: E1 retrained"),
             ("E2:moco:checkpoint_0199.pth.tar", "E2: + zoom jitter"),
             ("E3:moco:checkpoint_0199.pth.tar", "E3: + contrast jitter")]
    order = [(k, n) for k, n in order if k in rows]
    vals = [float(rows[k]["cross_position_top1"]) for k, _ in order]
    colors = [GRAY if k.startswith("P0r:") else BLUE for k, _ in order]
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(order) + 1.4))
    y = np.arange(len(order))[::-1]
    ax.barh(y, vals, color=colors, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + 0.012, yi, "%d%%" % round(100 * v), va="center", color=INK)
    ax.axvline(float(rows["P0r:imagenet"]["cross_position_top1"]), color=INK_2,
               lw=1, ls="--")
    ax.set_yticks(y, [n for _, n in order])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "%d%%" % (100 * v)))
    ax.set_xlabel("Supine scan matched to the same patient's prone scan (top-1, of 671)")
    ax.set_title("Pretraining on unlabelled CT: patient matching across positions",
                 loc="left", color=INK)
    ax.text(0, -0.9 / (0.5 * len(order) + 1.4), "Chance is 0.16%. Dashed line: "
            "ImageNet, the standard starting point.", transform=ax.transAxes,
            color=INK_2, fontsize=9)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(out, "progress.png"))
    plt.close(fig)


def _middle_crop(bank, index, name):
    rows = np.where(index["filename"] == name)[0]
    mid = rows[np.argmin(np.abs(index["z_frac"][rows] - 0.5))]
    return np.asarray(bank[index["row"][mid], 1])


def plot_retrieval(plt, bank_dir, names, out, n=6):
    bank = np.load(os.path.join(bank_dir, "bank.npy"), mmap_mode="r")
    index = load_index(bank_dir)
    data = [_load(bank_dir, m) for m in names]
    sup, pro = _pairs(data[0])
    rng = np.random.default_rng(SEED)
    queries = rng.choice(sup, n, replace=False)

    fig, axes = plt.subplots(n, 1 + len(names), figsize=(2.3 * (1 + len(names)), 2.5 * n))
    for r, q in enumerate(queries):
        pat = data[0]["patients"][q]
        axes[r, 0].imshow(_middle_crop(bank, index, data[0]["names"][q]), cmap="gray")
        axes[r, 0].set_title("Query: supine" if r == 0 else "", color=INK)
        for c, d in enumerate(data, start=1):
            qi = np.where(d["names"] == data[0]["names"][q])[0][0]
            _, p = _pairs(d)
            best = p[np.argmax(d["emb"][p] @ d["emb"][qi])]
            hit = d["patients"][best] == pat
            axes[r, c].imshow(_middle_crop(bank, index, d["names"][best]), cmap="gray")
            axes[r, c].set_title(("%s's pick\n" % _label(names[c - 1]) if r == 0 else "")
                                 + ("same patient" if hit else "wrong patient"),
                                 color=BLUE if hit else ORANGE, fontsize=10)
            for s in axes[r, c].spines.values():
                s.set_visible(True)
                s.set_color(BLUE if hit else ORANGE)
                s.set_linewidth(3)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    rates = []
    for d in data:
        s, p = _pairs(d)
        best = p[np.argmax(d["emb"][s] @ d["emb"][p].T, axis=1)]
        rates.append(np.mean(d["patients"][best] == d["patients"][s]))
    fig.suptitle("Find this patient's prone scan among 671. Queries drawn at random.\n"
                 "Over all 613 queries: " + ", ".join(
                     "%s %d%% right" % (_label(m), round(100 * r))
                     for m, r in zip(names, rates)),
                 x=0.02, ha="left", color=INK, y=1.01)
    fig.savefig(os.path.join(out, "retrieval.png"))
    plt.close(fig)


def plot_similarity(plt, bank_dir, names, out):
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.4))
    for ax, name in zip(np.atleast_1d(axes), names):
        d = _load(bank_dir, name)
        sup, pro = _pairs(d)
        sims = d["emb"][sup] @ d["emb"][pro].T
        same = d["patients"][sup][:, None] == d["patients"][pro][None, :]
        # Each encoder's own range: raw cosines of ReLU features all sit near 1,
        # so a shared axis hides the shape that matters.
        bins = np.linspace(np.percentile(sims, 0.5), sims.max(), 50)
        ax.hist(sims[~same], bins=bins, density=True, color=GRAY, alpha=0.8,
                label="different patients")
        ax.hist(sims[same], bins=bins, density=True, color=BLUE, alpha=0.8,
                label="same patient")
        ax.set_title(_label(name), loc="left", color=INK)
        ax.set_xlabel("Similarity of a supine and a prone scan\n(each panel on its own scale)")
        ax.set_yticks([])
        ax.grid(axis="x", color=GRID)
        ax.set_axisbelow(True)
    np.atleast_1d(axes)[0].legend(frameon=False, loc="upper left")
    fig.suptitle("How similar a patient's two scans look, against everyone else's",
                 x=0.02, ha="left", color=INK, y=1.04)
    fig.savefig(os.path.join(out, "similarity.png"))
    plt.close(fig)


def plot_depth(plt, bank_dir, names, out, n_crops=8000):
    import umap
    index = load_index(bank_dir)
    acr = np.where(index["collection"] == ACRIN_COLLECTION)[0]
    pick = np.sort(np.random.default_rng(SEED).choice(acr, n_crops, replace=False))
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 4.2))
    for ax, name in zip(np.atleast_1d(axes), names):
        x = _load(bank_dir, name)["crops"][pick].astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        xy = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                       random_state=SEED).fit_transform(x)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=index["z_frac"][pick], cmap="Blues",
                        vmin=-0.15, vmax=1.0, s=3, linewidths=0)
        ax.set_title(_label(name), loc="left", color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(sc, ax=axes, shrink=0.8, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["one end\nof scan", "middle", "other end"])
    fig.suptitle("Each dot is one crop, placed by its features; colour is its depth "
                 "in the scan.\nBoth encoders order crops along the body: a sanity "
                 "check that features track anatomy, not a difference between them.",
                 x=0.02, ha="left", color=INK, y=1.06)
    fig.savefig(os.path.join(out, "depth_map.png"))
    plt.close(fig)


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    os.makedirs(args.out_dir, exist_ok=True)
    plot_progress(plt, args.ledger, args.out_dir)
    if args.compare:
        plot_retrieval(plt, args.bank_dir, args.compare, args.out_dir)
        plot_similarity(plt, args.bank_dir, args.compare, args.out_dir)
        plot_depth(plt, args.bank_dir, args.compare, args.out_dir)
    print("Figures in", args.out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    e = sub.add_parser("embed")
    e.add_argument("--bank-dir", required=True)
    e.add_argument("--encoder", required=True, choices=["moco", "imagenet", "random"])
    e.add_argument("--checkpoint", default=None)
    e.add_argument("--name", required=True)
    p = sub.add_parser("plot")
    p.add_argument("--bank-dir", required=True)
    p.add_argument("--out-dir", default="docs/figures")
    p.add_argument("--ledger", default="docs/experiment_results.csv")
    p.add_argument("--compare", nargs="*", default=[],
                   help="embedded names to compare, e.g. imagenet e1")
    args = parser.parse_args()
    embed(args) if args.stage == "embed" else plot(args)


if __name__ == "__main__":
    main()
