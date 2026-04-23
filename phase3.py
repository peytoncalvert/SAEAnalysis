"""
phase3.py  —  Superposition Depth Scan
=======================================
For each layer in PHASE3_LAYERS:
  1. Load the SAE for that layer.
  2. Extract year and all-item activations.
  3. Find the top discriminative cluster.
  4. Run a single-feature ablation scan across all cluster features.
  5. Report max drop and load-bearing feature count.

Produces two plots:
  phase3_superposition_depth.png   — max drop & baseline |r| vs layer
  phase3_drop_distributions.png    — violin of drop distributions per layer
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.decomposition import PCA

from config import (
    DEVICE, OUTPUT_DIR,
    YEAR_ITEMS, ALL_ITEMS, ALL_CATS,
    SAE_RELEASE, CORR_THRESHOLD,
    load_model,
)
from sae_lens import SAE
from phase1 import find_clusters

PHASE3_LAYERS = [4, 6, 8, 10]


# ── Per-layer ablation scan ───────────────────────────────────────────────────

def _ablation_scan_at_layer(model, layer: int) -> dict:
    """
    Run the full cluster ablation scan at *layer* and return summary stats.

    Returns dict with keys:
        layer, baseline_r, max_drop, top_feature,
        n_load, n_moderate, n_negligible, all_drops
    """
    hook   = f"blocks.{layer}.hook_resid_pre"
    sae_id = f"blocks.{layer}.hook_resid_pre"

    print(f"  Loading SAE for layer {layer} ...")
    sae, _, _ = SAE.from_pretrained(
        release=SAE_RELEASE, sae_id=sae_id, device=DEVICE,
    )
    sae.eval()

    # Year activations
    rows = []
    for _, prompt, _ in tqdm(YEAR_ITEMS, desc=f"  L{layer} year acts", leave=False):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook, device=DEVICE)
        rows.append(cache[hook][0, -1, :].cpu())
    acts = torch.stack(rows)

    # All-item activations (needed for clustering)
    all_rows = []
    for _, prompt, _ in tqdm(ALL_ITEMS, desc=f"  L{layer} all acts", leave=False):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook, device=DEVICE)
        all_rows.append(cache[hook][0, -1, :].cpu())
    all_acts_l = torch.stack(all_rows)

    with torch.no_grad():
        feat_acts_all  = sae.encode(all_acts_l.to(DEVICE)).cpu()
        feat_acts_year = sae.encode(acts.to(DEVICE)).cpu()

    clusters, _ = find_clusters(feat_acts_all, ALL_CATS, corr_threshold=CORR_THRESHOLD)
    if not clusters:
        clusters, _ = find_clusters(feat_acts_all, ALL_CATS, corr_threshold=0.3)
    if not clusters:
        print(f"    No clusters found at layer {layer} — skipping.")
        return dict(layer=layer, baseline_r=float("nan"), max_drop=float("nan"),
                    top_feature=-1, n_load=0, n_moderate=0, n_negligible=0,
                    all_drops=np.array([]))

    cluster_feats = clusters[0]
    W_dec        = sae.W_dec.detach().cpu()
    year_numeric = np.array([it[2] for it in YEAR_ITEMS], dtype=float)

    def _absr(fa: torch.Tensor) -> float:
        recon  = (fa @ W_dec).numpy()
        coords = PCA(n_components=1).fit_transform(recon)
        return abs(float(np.corrcoef(year_numeric, coords[:, 0])[0, 1]))

    baseline_r = _absr(feat_acts_year)

    scan = []
    for fi in tqdm(cluster_feats, desc=f"  L{layer} scan", leave=False):
        fa_abl = feat_acts_year.clone()
        fa_abl[:, fi] = 0.0
        scan.append((fi, baseline_r - _absr(fa_abl)))

    scan.sort(key=lambda x: x[1], reverse=True)
    drops    = np.array([d for _, d in scan])
    top_feat = scan[0][0]
    n_load   = int((drops > 0.10).sum())
    n_mod    = int(((drops > 0.02) & (drops <= 0.10)).sum())
    n_neg    = int((drops <= 0.02).sum())

    print(f"    Layer {layer}: baseline |r|={baseline_r:.4f}  "
          f"max_drop={drops[0]:+.4f}  top_feat={top_feat}  "
          f"load={n_load}  mod={n_mod}  neg={n_neg}")

    return dict(layer=layer, baseline_r=baseline_r, max_drop=drops[0],
                top_feature=top_feat, n_load=n_load, n_moderate=n_mod,
                n_negligible=n_neg, all_drops=drops)


# ── Phase 3 orchestration ─────────────────────────────────────────────────────

def run_phase3() -> None:
    print("\n" + "=" * 65)
    print("PHASE 3  -  Superposition Depth Scan")
    print(f"  layers={PHASE3_LAYERS}")
    print("=" * 65)

    model   = load_model()
    results = []
    for layer in PHASE3_LAYERS:
        print(f"\n--- Layer {layer} ---")
        results.append(_ablation_scan_at_layer(model, layer))

    # Summary table
    print("\n" + "=" * 65)
    print("PHASE 3  SUMMARY")
    print(f"  {'Layer':>6}  {'base |r|':>9}  {'max drop':>9}  "
          f"{'top feat':>9}  {'load':>5}  {'mod':>5}  {'neg':>5}")
    print("  " + "-" * 58)
    for r in results:
        print(f"  {r['layer']:>6}  {r['baseline_r']:>9.4f}  {r['max_drop']:>+9.4f}  "
              f"  {r['top_feature']:>8}  {r['n_load']:>5}  "
              f"{r['n_moderate']:>5}  {r['n_negligible']:>5}")

    # Plot 1: max drop & baseline |r| vs layer
    layers_valid = [r["layer"] for r in results if not np.isnan(r["max_drop"])]
    max_drops    = [r["max_drop"]    for r in results if not np.isnan(r["max_drop"])]
    base_rs      = [r["baseline_r"]  for r in results if not np.isnan(r["baseline_r"])]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(layers_valid, max_drops, "o-", color="steelblue", linewidth=2, markersize=7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Max single-feature drop in |PC1-year r|")
    ax.set_title("Concentration of temporal signal\n(higher = less superposition)")
    ax.set_xticks(layers_valid)

    ax = axes[1]
    ax.plot(layers_valid, base_rs, "s-", color="seagreen", linewidth=2, markersize=7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("|PC1-year r| (full SAE recon)")
    ax.set_title("Temporal axis strength by layer")
    ax.set_xticks(layers_valid)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    p = OUTPUT_DIR / "phase3_superposition_depth.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"\n  Saved -> {p.name}")

    # Plot 2: violin distributions per layer
    valid = [r for r in results if len(r["all_drops"]) > 0]
    if valid:
        fig, ax = plt.subplots(figsize=(8, 4))
        parts = ax.violinplot(
            [r["all_drops"] for r in valid],
            positions=range(len(valid)),
            showmedians=True, showextrema=True,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("steelblue")
            pc.set_alpha(0.6)
        ax.set_xticks(range(len(valid)))
        ax.set_xticklabels([f"Layer {r['layer']}" for r in valid])
        ax.set_ylabel("Drop in |PC1-year r|")
        ax.set_title("Distribution of single-feature ablation drops by layer\n"
                     "(narrowing = more distributed / more superposition)")
        fig.tight_layout()
        p2 = OUTPUT_DIR / "phase3_drop_distributions.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        print(f"  Saved -> {p2.name}")

    print("\n" + "=" * 65)
    print("PHASE 3  COMPLETE")
    print(f"  Outputs saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 65)
