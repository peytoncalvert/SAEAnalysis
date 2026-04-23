"""
phase2.py  —  Geometric / Helical Analysis
==========================================
1. Extract year-token residual-stream activations across layers 4, 6, 8, 10.
2. Select the layer with the strongest PC1-year correlation.
3. At that layer: fit a circle to PC2 vs PC3, measure circularity, report
   BC/AD arc separation, and save a four-panel diagnostic plot.
"""

import numpy as np
import torch
from tqdm import tqdm

from sklearn.decomposition import PCA

from config import (
    DEVICE, OUTPUT_DIR, LAYER,
    YEAR_ITEMS, YEAR_CATEGORIES,
    load_model,
)
from plots import fit_circle_algebraic, _plot_geometric_analysis

PHASE2_LAYERS = [4, 6, 8, 10]


# ── Helpers ───────────────────────────────────────────────────────────────────

def circularity_score(points_2d: np.ndarray) -> float:
    """1 - std(radii) / mean(radii) from the fitted circle.  1 = perfect circle."""
    try:
        cx, cy, _, _ = fit_circle_algebraic(points_2d)
        radii = np.sqrt((points_2d[:, 0] - cx) ** 2 + (points_2d[:, 1] - cy) ** 2)
        return float(np.clip(1.0 - radii.std() / (radii.mean() + 1e-8), 0.0, 1.0))
    except Exception:
        return 0.0


def _extract_layer_acts(model, layer: int) -> np.ndarray:
    """Extract last-token residual-stream activations for all YEAR_ITEMS."""
    hook = f"blocks.{layer}.hook_resid_pre"
    rows = []
    for _, prompt, _ in tqdm(YEAR_ITEMS, desc=f"  L{layer}", leave=False):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook, device=DEVICE)
        rows.append(cache[hook][0, -1, :].cpu().numpy())
    return np.stack(rows)


# ── Phase 2 orchestration ─────────────────────────────────────────────────────

def run_phase2() -> None:
    print("\n" + "=" * 65)
    print("PHASE 2  -  Geometric / Helical Analysis")
    print("=" * 65)

    model       = load_model()
    year_nums   = np.array([it[2] for it in YEAR_ITEMS], dtype=float)
    year_labels = [it[0] for it in YEAR_ITEMS]

    # [1] Extract activations across layers
    print("\n[1]  Extracting year activations across layers ...")
    layer_data: dict = {}
    for layer in PHASE2_LAYERS:
        acts   = _extract_layer_acts(model, layer)
        n_pcs  = min(5, acts.shape[0] - 1, acts.shape[1])
        pca    = PCA(n_components=n_pcs)
        coords = pca.fit_transform(acts)
        var    = pca.explained_variance_ratio_
        layer_data[layer] = dict(acts=acts, pca=pca, coords=coords, var=var)
        r1 = float(np.corrcoef(year_nums, coords[:, 0])[0, 1])
        print(f"  Layer {layer:>2}:  PC1 r={r1:+.3f}  "
              f"var(PC1+PC2+PC3)={var[:3].sum():.1%}")

    # [2] Select best layer
    print("\n[2]  Selecting analysis layer by PC1-year correlation ...")
    best_layer, best_r1 = LAYER, 0.0
    for layer, ld in layer_data.items():
        r1 = float(np.corrcoef(year_nums, ld["coords"][:, 0])[0, 1])
        if abs(r1) > abs(best_r1):
            best_layer, best_r1 = layer, r1
    print(f"  Best: layer {best_layer}  (PC1 r = {best_r1:+.4f})")

    # [3] Detailed geometric analysis
    print(f"\n[3]  Detailed geometric analysis at layer {best_layer} ...")
    ld     = layer_data[best_layer]
    acts   = ld["acts"]
    coords = ld["coords"]
    var    = ld["var"]

    pc1_dir   = ld["pca"].components_[0]
    proj_pc1  = (acts @ pc1_dir[:, None]) * pc1_dir[None, :]
    acts_detr = acts - proj_pc1
    n_pcs_d   = min(3, acts_detr.shape[0] - 1, acts_detr.shape[1])
    pca_d     = PCA(n_components=n_pcs_d)
    coords_d  = pca_d.fit_transform(acts_detr)
    var_d     = pca_d.explained_variance_ratio_

    circ_raw  = circularity_score(coords[:, 1:3])
    circ_detr = circularity_score(coords_d[:, :2])
    print(f"  Circularity (raw PC2 vs PC3):         {circ_raw:.4f}")
    print(f"  Circularity (detrended PC1 vs PC2):   {circ_detr:.4f}")

    bc_mask = [i for i, c in enumerate(YEAR_CATEGORIES) if c == "ancient_bc"]
    ad_mask = [i for i, c in enumerate(YEAR_CATEGORIES) if c == "ancient_ad"]
    cx, cy, r, rmse = fit_circle_algebraic(coords[:, 1:3])
    angles  = np.arctan2(coords[:, 2] - cy, coords[:, 1] - cx)
    bc_ang  = np.degrees(angles[bc_mask])
    ad_ang  = np.degrees(angles[ad_mask])
    arc_sep = float(abs(bc_ang.mean() - ad_ang.mean()))
    arc_sep = min(arc_sep, 360 - arc_sep)
    print(f"  Circle fit (PC2 vs PC3):  cx={cx:.3f}  cy={cy:.3f}  "
          f"r={r:.3f}  RMSE={rmse:.4f}")
    print(f"  BC arc:  mean={bc_ang.mean():+.1f} deg  std={bc_ang.std():.1f} deg")
    print(f"  AD arc:  mean={ad_ang.mean():+.1f} deg  std={ad_ang.std():.1f} deg")
    print(f"  BC-AD arc separation: {arc_sep:.1f} deg  "
          f"({'opposite sides' if arc_sep > 120 else 'overlapping'})")

    for mask, label in [(bc_mask, "BC"), (ad_mask, "AD")]:
        yrs_g = year_nums[mask]
        r_g   = float(np.corrcoef(yrs_g, coords[mask, 0])[0, 1]) if len(mask) > 2 else float("nan")
        print(f"  Within-{label} PC1 vs year r:      {r_g:+.3f}")

    # [4] Save plots
    print("\n[4]  Saving plots ...")
    _plot_geometric_analysis(
        coords, var, angles, bc_mask, ad_mask,
        coords_d, var_d, year_nums, year_labels,
        best_layer,
        year_categories=YEAR_CATEGORIES,
        output_dir=OUTPUT_DIR,
    )

    print("\n" + "=" * 65)
    print("PHASE 2  COMPLETE")
    print(f"  Outputs saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 65)
