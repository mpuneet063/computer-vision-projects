"""
BSDS500 edge-detection benchmark for Contours (HED): ODS, OIS, AP.

Follows the standard protocol (Arbelaez et al. / Dollar's edgesEval):
  1. Predict fused edge map on each of the 200 TEST images at original resolution
  2. Thin edges with non-maximum suppression (NMS)
  3. For 99 thresholds: binarise -> thin -> match to EVERY annotator with
     tolerance 0.0075 * image diagonal
  4. ODS = best F at one dataset-wide threshold
     OIS = best F with the threshold chosen per image
     AP  = area under the precision-recall curve

CPU-only version.

Usage (from ~/Contours/src):
  python eval_bsds.py --checkpoint ../checkpoints/hed_epoch_XX.pth
  python eval_bsds.py --checkpoint ... --thresholds 49    # ~2x faster, tiny difference
  python eval_bsds.py --skip_predict                      # re-score saved predictions

Deps: pip install scikit-image scipy tqdm
"""
import os
import argparse
import multiprocessing as mp

import numpy as np
import scipy.io as sio
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from skimage.morphology import thin
from tqdm import tqdm


# ----------------------------------------------------------------------------
# 1. Prediction
# ----------------------------------------------------------------------------
def load_model_weights(checkpoint):
    """Your checkpoints are a bundle:
         {'epoch', 'model_state', 'optimizer_state', 'scheduler_state'}
    The model only needs 'model_state'. Older checkpoints were the bare
    state_dict, so accept both."""
    import torch
    ckpt = torch.load(checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        print(f"Loaded weights from epoch {ckpt.get('epoch', '?')}")
        return ckpt["model_state"]
    return ckpt  # old format: the file IS the state_dict


def predict_all(checkpoint, image_dir, out_dir, n_threads):
    import torch
    import torchvision.transforms as T
    from model import HED

    # CPU only: let PyTorch use all available cores for one image at a time
    torch.set_num_threads(n_threads)
    device = torch.device("cpu")
    model = HED()
    model.load_state_dict(load_model_weights(checkpoint))
    model.eval()

    # Same normalisation as training (ImageNet stats); NO resize -> original size
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(image_dir) if f.endswith(".jpg"))
    with torch.inference_mode():
        for name in tqdm(names, desc="Predicting (CPU)"):
            out_path = os.path.join(out_dir, name.replace(".jpg", ".npy"))
            if os.path.exists(out_path):          # resume after an interruption
                continue
            img = Image.open(os.path.join(image_dir, name)).convert("RGB")
            x = tf(img).unsqueeze(0)
            fuse_logits = model(x)[-1]                       # fused output
            prob = torch.sigmoid(fuse_logits)[0, 0].numpy().astype(np.float32)
            np.save(out_path, prob)
    return [n.replace(".jpg", "") for n in names]


# ----------------------------------------------------------------------------
# 2. Non-maximum suppression (port of Dollar's edgesNms, r=1, s=5, m=1.01)
# ----------------------------------------------------------------------------
def edge_nms(E, r=1, s=5, m=1.01):
    # Edge orientation from a smoothed copy of the map
    Es = ndimage.gaussian_filter(E, 2)
    Ox, Oy = np.gradient(Es)[1], np.gradient(Es)[0]
    Oxx = np.gradient(Ox)[1]
    Oxy, Oyy = np.gradient(Oy)[1], np.gradient(Oy)[0]
    O = np.mod(np.arctan(Oyy * np.sign(-Oxy) / (Oxx + 1e-5)), np.pi)

    h, w = E.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cos, sin = np.cos(O), np.sin(O)
    out = E.copy()
    for d in list(range(-r, 0)) + list(range(1, r + 1)):
        # value of the neighbour along the edge normal (bilinear)
        nb = ndimage.map_coordinates(E, [ys + d * sin, xs + d * cos],
                                     order=1, mode="nearest")
        out[E * m < nb] = 0
    # suppress noisy borders
    if s > 0:
        for i in range(s):
            k = (i + 1) / (s + 1)          # linear fade-in, as in edgesNms
            out[i, :] *= k; out[-1 - i, :] *= k
            out[:, i] *= k; out[:, -1 - i] *= k
    return out


# ----------------------------------------------------------------------------
# 3. Pixel correspondence with distance tolerance
# ----------------------------------------------------------------------------
def match_pixels(pred, gt, max_dist):
    """Returns (pred_matched_mask, n_gt_matched). One-to-one matching:
    each predicted pixel can claim at most one GT pixel within max_dist."""
    p = np.argwhere(pred)
    g = np.argwhere(gt)
    matched = np.zeros_like(pred, dtype=bool)
    if len(p) == 0 or len(g) == 0:
        return matched, 0
    nbrs = cKDTree(p).query_ball_tree(cKDTree(g), max_dist)
    rows = np.repeat(np.arange(len(p)), [len(n) for n in nbrs])
    cols = np.fromiter((c for n in nbrs for c in n), dtype=np.int64, count=len(rows))
    if len(rows) == 0:
        return matched, 0
    graph = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
                       shape=(len(p), len(g)))
    m = maximum_bipartite_matching(graph, perm_type="column")  # row -> col or -1
    hit = m >= 0
    matched[p[hit, 0], p[hit, 1]] = True
    return matched, int(hit.sum())


def load_gts(mat_path):
    gts = sio.loadmat(mat_path)["groundTruth"][0]
    return [gts[i]["Boundaries"][0, 0].astype(bool) for i in range(len(gts))]


def eval_image(args):
    """For one image: per-threshold counts [cntR, sumR, cntP, sumP]."""
    name, pred_dir, gt_dir, thresholds = args
    E = np.load(os.path.join(pred_dir, name + ".npy"))
    E = edge_nms(E)
    gts = load_gts(os.path.join(gt_dir, name + ".mat"))
    max_dist = 0.0075 * np.hypot(*E.shape)

    sumR = sum(int(g.sum()) for g in gts)
    counts = np.zeros((len(thresholds), 4), dtype=np.int64)
    for k, t in enumerate(thresholds):
        B = thin(E >= t)
        any_match = np.zeros_like(B)
        cntR = 0
        for g in gts:
            mp, ng = match_pixels(B, g, max_dist)
            cntR += ng            # recall: GT pixels found, summed over annotators
            any_match |= mp       # precision: pred pixel counts if it matches ANY annotator
        counts[k] = [cntR, sumR, int(any_match.sum()), int(B.sum())]
    return counts


# ----------------------------------------------------------------------------
# 4. ODS / OIS / AP
# ----------------------------------------------------------------------------
def prf(cntR, sumR, cntP, sumP):
    R = cntR / np.maximum(sumR, 1)
    P = cntP / np.maximum(sumP, 1)
    F = 2 * P * R / np.maximum(P + R, 1e-12)
    return P, R, F


def summarise(all_counts, thresholds):
    all_counts = np.stack(all_counts)               # (n_img, n_thr, 4)

    # ODS: sum counts over images, then pick the best threshold
    tot = all_counts.sum(0)
    P, R, F = prf(*tot.T)
    k = int(F.argmax())
    ods, ods_t = F[k], thresholds[k]

    # OIS: best threshold per image, accumulate those counts
    best = []
    for c in all_counts:
        _, _, f = prf(*c.T)
        best.append(c[f.argmax()])
    _, _, ois = prf(*np.sum(best, 0))

    # AP: interpolate precision at recall 0:0.01:1 (as in edgesEvalDir)
    order = np.argsort(R)
    Rs, Ps = R[order], P[order]
    Rs, idx = np.unique(Rs, return_index=True)
    Ps = Ps[idx]
    keep = (Rs > 0) & (Rs < 1)
    Rs, Ps = Rs[keep], Ps[keep]
    if len(Rs) > 1:
        grid = np.arange(0, 1.001, 0.01)
        interp = np.interp(grid, Rs, Ps, left=np.nan, right=np.nan)
        ap = np.nansum(interp) / 100
    else:
        ap = 0.0
    return ods, ods_t, float(ois), ap, P, R


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--checkpoint", type=str)
    ap_.add_argument("--data_dir", default=os.path.join(here, "..", "data", "BSR"))
    ap_.add_argument("--pred_dir", default=os.path.join(here, "..", "eval", "pred_test"))
    ap_.add_argument("--thresholds", type=int, default=99)
    ap_.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)),
                     help="CPU cores for scoring and prediction (default: all you have)")
    ap_.add_argument("--skip_predict", action="store_true")
    args = ap_.parse_args()

    image_dir = os.path.join(args.data_dir, "images", "test")
    gt_dir = os.path.join(args.data_dir, "groundTruth", "test")

    if args.skip_predict:
        names = sorted(f[:-4] for f in os.listdir(args.pred_dir) if f.endswith(".npy"))
    else:
        assert args.checkpoint, "--checkpoint is required unless --skip_predict"
        names = predict_all(args.checkpoint, image_dir, args.pred_dir, args.workers)

    thresholds = np.linspace(1 / (args.thresholds + 1), 1 - 1 / (args.thresholds + 1),
                             args.thresholds)
    jobs = [(n, args.pred_dir, gt_dir, thresholds) for n in names]
    # "spawn" = fresh worker processes; avoids hangs from forking after PyTorch
    # has started its own CPU threads
    with mp.get_context("spawn").Pool(args.workers) as pool:
        all_counts = list(tqdm(pool.imap(eval_image, jobs), total=len(jobs),
                               desc="Scoring"))

    ods, ods_t, ois, ap, P, R = summarise(all_counts, thresholds)
    print(f"\nBSDS500 test ({len(names)} images)")
    print(f"  ODS = {ods:.3f}  (threshold {ods_t:.2f})")
    print(f"  OIS = {ois:.3f}")
    print(f"  AP  = {ap:.3f}")
    print("  HED paper (fusion output): ODS .782 | OIS .802 | AP .787")

    out_csv = os.path.join(os.path.dirname(args.pred_dir), "pr_curve.csv")
    np.savetxt(out_csv, np.c_[thresholds, P, R], delimiter=",",
               header="threshold,precision,recall", comments="")
    print(f"  PR curve saved to {out_csv}")


if __name__ == "__main__":
    main()