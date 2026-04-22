#!/usr/bin/env python3
"""
Phase 1: Historical Concept Encoding in GPT-2
==============================================
Applies the clustering methodology of Engels et al. to identify candidate
multi-dimensional feature clusters when GPT-2 processes historically-grounded
tokens (years, battles, dynasties).

Steps
-----
1. Load GPT-2-small + pre-trained SAE (layer 8) via TransformerLens + SAELens.
2. Extract residual-stream activations for historical prompts at the last-token
   position (i.e. the position of the historical concept itself).
3. Collect sparse SAE feature activations; find clusters by Pearson
   co-activation correlation (following Engels et al.).
4. For each cluster reconstruct activations from the cluster's decoder
   directions and score with:
     S(f)  - separability index  (leave-one-out linear probe accuracy, above chance)
     M_eps   - eps-mixture index     (1-D GMM / full-D GMM log-likelihood ratio)
   High S(f) + low M_eps -> high irreducibility score.
5. Visualise top clusters in PCA space; plot PC components vs. numeric year
   to test linear decodability à la Gurnee & Tegmark.

Usage
-----
    python final.py

Outputs are saved to ./outputs/
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy.stats import f_oneway

from transformer_lens import HookedTransformer
from sae_lens import SAE

from plots import (
    fit_circle_algebraic,
    plot_pca_overview,
    plot_year_linearity,
    _plot_helix_scan,
    _plot_geometric_analysis,
)

# -- Configuration --------------------------------------------------------------

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR  = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME  = "gpt2"
LAYER       = 4                        # mid-to-late layer (GPT-2-small has 12)
HOOK_NAME   = f"blocks.{LAYER}.hook_resid_pre"

SAE_RELEASE = "gpt2-small-res-jb"       # Joseph Bloom's GPT-2 residual SAEs
SAE_ID      = f"blocks.{LAYER}.hook_resid_pre"

CORR_THRESHOLD = 0.5   # |Pearson r| threshold to form a co-activation edge
MIN_CLUSTER_SZ = 2     # minimum features per cluster

# -- Historical Prompts ---------------------------------------------------------
#
# Each entry: (display_label, full_prompt, numeric_year)
# The historical token appears at the END of the prompt so we extract
# the last-position residual stream activation.
# Prompts are stored in prompts.json; loaded once at module import time.

with open(Path(__file__).parent / "prompts.json") as _f:
    _prompts = json.load(_f)

BC_AD_PAIRS = [
    (e["year"], e["bc_prompt"], e["ad_prompt"])
    for e in _prompts["bc_ad_pairs"]
]

YEAR_ITEMS = (
    [(f"{yr}BC", prompt_bc, -yr) for yr, prompt_bc, _ in BC_AD_PAIRS] +
    [(f"{yr}AD", prompt_ad,  yr) for yr, _, prompt_ad in BC_AD_PAIRS] +
    [(e["label"], e["prompt"], e["year"]) for e in _prompts["extra_year_items"]]
)

YEAR_CATEGORIES = (
    ["ancient_bc"]   * len(BC_AD_PAIRS) +
    ["ancient_ad"]   * len(BC_AD_PAIRS) +
    ["medieval"]     * 7  +
    ["early_modern"] * 7  +
    ["modern"]       * 10
)
assert len(YEAR_CATEGORIES) == len(YEAR_ITEMS), (
    f"YEAR_CATEGORIES length {len(YEAR_CATEGORIES)} != YEAR_ITEMS length {len(YEAR_ITEMS)}"
)

BATTLE_EUROPE  = [(e["label"], e["prompt"], e["year"]) for e in _prompts["battle_europe"]]
BATTLE_AMERICA = [(e["label"], e["prompt"], e["year"]) for e in _prompts["battle_america"]]
BATTLE_EAST    = [(e["label"], e["prompt"], e["year"]) for e in _prompts["battle_east"]]

BATTLE_ITEMS = BATTLE_EUROPE + BATTLE_AMERICA + BATTLE_EAST

ALL_ITEMS = YEAR_ITEMS + BATTLE_ITEMS
ALL_CATS  = (
    YEAR_CATEGORIES
    + ["battle_europe"]  * len(BATTLE_EUROPE)
    + ["battle_america"] * len(BATTLE_AMERICA)
    + ["battle_east"]    * len(BATTLE_EAST)
)


# -- Model & SAE Loading --------------------------------------------------------

def load_model() -> HookedTransformer:
    print(f"Loading {MODEL_NAME} on {DEVICE} ...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    model.eval()
    print(f"  {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    return model


def load_sae():
    print(f"Loading SAE  ({SAE_RELEASE}  /  {SAE_ID}) ...")
    sae, _cfg, _sparsity = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=SAE_ID,
        device=DEVICE,
    )
    sae.eval()
    print(f"  d_in={sae.cfg.d_in},  d_sae={sae.cfg.d_sae}")
    return sae


# -- Activation Extraction ------------------------------------------------------

def extract_activations(
    model: HookedTransformer,
    items: list,
) -> tuple[torch.Tensor, list[str]]:
    """
    Tokenise each prompt and return the residual-stream activation at LAYER
    for the LAST token position (the historical concept token).

    Returns
    -------
    activations : (N, d_model) CPU tensor
    labels      : list of N display labels
    """
    activations, labels = [], []
    for label, prompt, *_ in tqdm(items, desc="  activations", leave=False):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=HOOK_NAME, device=DEVICE
            )
        activations.append(cache[HOOK_NAME][0, -1, :].cpu())
        labels.append(label)
    return torch.stack(activations), labels


# -- SAE Feature Extraction -----------------------------------------------------

def get_feature_acts(sae, activations: torch.Tensor) -> torch.Tensor:
    """Run the SAE encoder; returns (N, d_sae) sparse feature activations."""
    with torch.no_grad():
        feat_acts = sae.encode(activations.to(DEVICE))
    return feat_acts.cpu()


# -- Feature Clustering (adapted from Engels et al. for small datasets) --------
#
# Engels et al. cluster SAE features by co-activation statistics computed over
# millions of tokens.  With only ~50 tokens the raw Pearson matrix collapses
# into one giant component because any two features that fire on overlapping
# tokens appear correlated by chance.  We therefore:
#   1. Select the N_DISC most temporally discriminative features via a one-way
#      ANOVA F-test across epoch categories.
#   2. Compute each selected feature's "epoch profile" -- its mean activation
#      per epoch category (a low-dimensional fingerprint of what it responds to).
#   3. Cluster those features by the Pearson correlation of their epoch profiles.
# The resulting clusters group SAE features that respond to the same temporal
# epochs, giving semantically coherent clusters despite the small dataset size.

N_DISC = 60   # top discriminative features to select before clustering


def find_clusters(
    feat_acts: torch.Tensor,
    categories: list[str],
    corr_threshold: float = CORR_THRESHOLD,
    min_size: int = MIN_CLUSTER_SZ,
    n_disc: int = N_DISC,
) -> tuple[list[list[int]], np.ndarray]:
    """
    Find SAE feature clusters discriminative for historical epoch categories.

    Returns
    -------
    clusters   : list of clusters, each a list of global feature indices
    f_scores   : (d_sae,) ANOVA F-score for every feature (for reporting)
    """
    N, d_sae = feat_acts.shape
    cats    = sorted(set(categories))
    cat_idx = {c: [i for i, cat in enumerate(categories) if cat == c] for c in cats}

    # 1. Active features (fire on >= 2 tokens)
    freq       = (feat_acts > 0).float().sum(0)
    active_idx = freq.ge(2).nonzero(as_tuple=True)[0].tolist()
    print(f"    Active features (fire on >= 2 tokens): {len(active_idx)} / {d_sae}")

    if len(active_idx) < 2:
        return [], np.zeros(d_sae)

    # 2. ANOVA F-score per active feature
    f_scores_active = np.zeros(len(active_idx))
    for k, fi in enumerate(active_idx):
        groups = [feat_acts[cat_idx[c], fi].numpy()
                  for c in cats if len(cat_idx[c]) > 0]
        if len(groups) >= 2:
            try:
                F, _ = f_oneway(*groups)
                f_scores_active[k] = float(F) if np.isfinite(F) else 0.0
            except Exception:
                pass

    f_scores_full = np.zeros(d_sae)
    for k, fi in enumerate(active_idx):
        f_scores_full[fi] = f_scores_active[k]

    # 3. Keep top-n_disc discriminative features
    n_keep     = min(n_disc, len(active_idx))
    top_local  = np.argsort(f_scores_active)[-n_keep:][::-1]
    top_global = [active_idx[i] for i in top_local]
    print(f"    Top-{n_keep} discriminative features selected  "
          f"(max F = {f_scores_active[top_local[0]]:.2f})")

    # 4. Epoch-mean activation profile for each selected feature
    profiles = np.array([
        [feat_acts[cat_idx[c], fi].mean().item() for c in cats]
        for fi in top_global
    ])   # (n_keep, n_cats)

    stds       = profiles.std(1, keepdims=True) + 1e-8
    profiles_z = (profiles - profiles.mean(1, keepdims=True)) / stds
    corr       = (profiles_z @ profiles_z.T) / profiles.shape[1]

    # 5. Build graph and find connected components
    adj = np.abs(corr) > corr_threshold
    np.fill_diagonal(adj, False)

    visited, clusters = np.zeros(n_keep, bool), []
    for start in range(n_keep):
        if visited[start]:
            continue
        component, queue = [], [start]
        while queue:
            node = queue.pop(0)
            if visited[node]:
                continue
            visited[node] = True
            component.append(top_global[node])
            queue.extend(np.where(adj[node])[0].tolist())
        if len(component) >= min_size:
            clusters.append(component)

    return clusters, f_scores_full


# -- Irreducibility Metrics -----------------------------------------------------

def separability_index(recon: torch.Tensor, category_labels: list[str]) -> float:
    """
    S(f): leave-one-out accuracy of a linear probe predicting historical
    category from the cluster's reconstructed activations, normalised to
    [0, 1] above chance.

    High S(f) -> the cluster strongly separates different historical categories.
    """
    cats = list(set(category_labels))
    if len(cats) < 2 or len(category_labels) < 4:
        return 0.0

    X = recon.numpy()
    y = LabelEncoder().fit_transform(category_labels)

    n_pc = min(10, X.shape[0] - 1, X.shape[1])
    if n_pc < 1:
        return 0.0
    X_pc = PCA(n_components=n_pc).fit_transform(X)

    correct = 0
    for i in range(len(y)):
        train = [j for j in range(len(y)) if j != i]
        if len(set(y[train])) < 2:
            continue
        try:
            clf = LogisticRegression(max_iter=300, random_state=0)
            clf.fit(X_pc[train], y[train])
            correct += int(clf.predict(X_pc[[i]])[0] == y[i])
        except Exception:
            pass

    chance = 1.0 / len(cats)
    acc    = correct / len(y)
    return float(np.clip((acc - chance) / (1.0 - chance + 1e-8), 0.0, 1.0))


def epsilon_mixture_index(recon: torch.Tensor) -> float:
    """
    M_eps(f): fraction of variance explained by the first principal component.

    This operationalises the Engels et al. eps-mixture concept geometrically:
    a cluster is "reducible" if its activations lie near a single 1-D subspace,
    i.e. when PC1 explains most of the variance.

      M_eps = var_explained(PC1)   in [0, 1]

    High M_eps -> nearly 1-D structure -> reducible (e.g. a linear temporal axis).
    Low  M_eps -> variance spread across multiple components -> irreducible
                  (consistent with a circular or helical encoding).

    The irreducibility score S(f) * (1 - M_eps) is therefore high only when the
    cluster both separates categories well AND cannot be collapsed to a single
    direction.
    """
    X    = recon.numpy()
    N, d = X.shape
    if N < 4 or d < 2:
        return 1.0

    n_pcs = min(N - 1, d, 10)
    try:
        pca   = PCA(n_components=n_pcs)
        pca.fit(X)
        return float(pca.explained_variance_ratio_[0])
    except Exception:
        return 1.0


# -- Phase 1 Orchestration ------------------------------------------------------

def run_phase1() -> None:
    print("=" * 65)
    print("PHASE 1  -  Feature Clustering & Irreducibility Scoring")
    print(f"  model={MODEL_NAME}   layer={LAYER}   device={DEVICE}")
    print("=" * 65)

    # -- 1. Load model + SAE ---------------------------------------------------
    model = load_model()
    sae   = load_sae()

    # -- 2. Extract residual-stream activations --------------------------------
    print("\n[1]  Extracting residual-stream activations ...")
    year_acts,   year_labels   = extract_activations(model, YEAR_ITEMS)
    battle_acts, battle_labels = extract_activations(model, BATTLE_ITEMS)
    all_acts   = torch.cat([year_acts, battle_acts], dim=0)
    all_labels = year_labels + battle_labels
    print(f"  years: {year_acts.shape}  |  battles: {battle_acts.shape}  |  total: {all_acts.shape}")

    # -- 3. Baseline: PCA directly on raw residual stream ---------------------
    print("\n[2]  Baseline - PCA on raw residual stream ...")
    plot_pca_overview(
        all_acts, all_labels, ALL_CATS,
        title=f"All Historical Tokens - Raw Residual Stream  (Layer {LAYER})",
        path=OUTPUT_DIR / "phase1_all_pca.png",
    )
    plot_year_linearity(
        year_acts, YEAR_ITEMS,
        path=OUTPUT_DIR / "phase1_year_linearity.png",
        layer=LAYER,
    )
    # Report PC correlations in console too
    acts_np = year_acts.numpy()
    numeric = np.array([it[2] for it in YEAR_ITEMS], dtype=float)
    n_pc    = min(3, acts_np.shape[0] - 1, acts_np.shape[1])
    coords  = PCA(n_components=n_pc).fit_transform(acts_np)
    for pc in range(n_pc):
        r = float(np.corrcoef(numeric, coords[:, pc])[0, 1])
        print(f"    PC{pc+1} vs year:  r = {r:+.3f}")

    # -- 4. SAE feature activations --------------------------------------------
    print("\n[3]  Running SAE encoder ...")
    feat_acts  = get_feature_acts(sae, all_acts)
    mean_active = (feat_acts > 0).float().sum(1).mean().item()
    print(f"  feature acts: {feat_acts.shape}  |  mean active / token: {mean_active:.1f}")

    # -- 5. Feature clustering -------------------------------------------------
    print("\n[4]  Clustering discriminative features by epoch-profile correlation ...")
    clusters, f_scores = find_clusters(feat_acts, ALL_CATS, corr_threshold=CORR_THRESHOLD)
    if not clusters:
        print("  No clusters at threshold 0.5 - retrying at 0.3 ...")
        clusters, f_scores = find_clusters(feat_acts, ALL_CATS, corr_threshold=0.3)
    print(f"  Clusters found: {len(clusters)}")

    # Report the 5 most discriminative individual features
    top5_feat = np.argsort(f_scores)[-5:][::-1]
    print("  Top-5 discriminative features by ANOVA F-score:")
    for rank, fi in enumerate(top5_feat):
        print(f"    #{rank+1}  feature {fi:>6}   F = {f_scores[fi]:.2f}")

    # -- 6. Score clusters -----------------------------------------------------
    print("\n[5]  Scoring clusters for irreducibility ...")
    W_dec = sae.W_dec.detach().cpu()          # (d_sae, d_model)

    print(f"  {'ID':>4}  {'|f|':>5}  {'S(f)':>7}  {'M_eps':>7}  {'Irred.':>8}")
    print("  " + "-" * 37)

    scored = []
    for cid, feat_indices in enumerate(clusters):
        fi    = torch.tensor(feat_indices, dtype=torch.long)
        recon = feat_acts[:, fi] @ W_dec[fi, :]          # (N, d_model)

        S     = separability_index(recon, ALL_CATS)
        M_eps = epsilon_mixture_index(recon)
        irred = S * (1.0 - M_eps)

        scored.append(dict(
            id=cid, features=feat_indices, n=len(feat_indices),
            S=S, M_eps=M_eps, irred=irred, recon=recon,
        ))
        print(f"  {cid:>4}  {len(feat_indices):>5}  {S:>7.3f}  {M_eps:>7.3f}  {irred:>8.3f}")

    scored.sort(key=lambda x: x["irred"], reverse=True)

    # -- 7. Visualise top-3 clusters -------------------------------------------
    print("\n[6]  Visualising top-3 clusters by irreducibility ...")
    for rank, entry in enumerate(scored[:3]):
        plot_pca_overview(
            entry["recon"],
            all_labels,
            ALL_CATS,
            title=(
                f"Cluster {entry['id']}  (rank {rank+1}/{len(scored)})  "
                f"S={entry['S']:.3f}  M_eps={entry['M_eps']:.3f}  "
                f"irred={entry['irred']:.3f}"
            ),
            path=OUTPUT_DIR / f"phase1_cluster{entry['id']}_rank{rank+1}.png",
        )

    # -- 8. Feature ablation experiment ----------------------------------------
    print("\n[7]  Feature ablation experiment ...")
    KEY_FEATURES = {
        1772:  "AD token detector (orthographic)",
        16307: "BC token detector (orthographic)",
        8113:  "Historical date marker (context-gated)",
        22008: "Modern year/date context",
    }

    # Baseline: PC1-year correlation from full SAE reconstruction
    year_feat_acts = get_feature_acts(sae, year_acts)
    year_numeric   = np.array([it[2] for it in YEAR_ITEMS], dtype=float)
    W_dec_full     = sae.W_dec.detach().cpu()

    def _pc1_year_absr(fa: torch.Tensor) -> float:
        """Absolute PC1-year correlation (sign-invariant: PCA direction is arbitrary)."""
        recon  = (fa @ W_dec_full).numpy()
        coords = PCA(n_components=1).fit_transform(recon)
        return abs(float(np.corrcoef(year_numeric, coords[:, 0])[0, 1]))

    baseline_r = _pc1_year_absr(year_feat_acts)
    print(f"  Baseline |PC1-year r| (full SAE recon): {baseline_r:.4f}")
    print()
    print(f"  {'Feature':<8}  {'Description':<42}  {'|r| ablated':>11}  {'drop':>8}")
    print("  " + "-" * 73)

    ablation_results = []
    for fi, desc in KEY_FEATURES.items():
        fa_abl = year_feat_acts.clone()
        fa_abl[:, fi] = 0.0
        r_abl = _pc1_year_absr(fa_abl)
        drop  = baseline_r - r_abl        # positive = feature was helping
        ablation_results.append((fi, desc, r_abl, drop))
        print(f"  {fi:<8}  {desc:<42}  {r_abl:>11.4f}  {drop:>+8.4f}")

    # All four ablated simultaneously
    fa_all = year_feat_acts.clone()
    for fi in KEY_FEATURES:
        fa_all[:, fi] = 0.0
    r_all  = _pc1_year_absr(fa_all)
    drop_all = baseline_r - r_all
    print("  " + "-" * 73)
    print(f"  {'all 4':<8}  {'all four ablated':<42}  {r_all:>11.4f}  {drop_all:>+8.4f}")

    print()
    print("  Interpretation guide (drop = baseline |r| - ablated |r|):")
    print("    drop < 0.02  -> feature contributes negligibly (encoding is distributed)")
    print("    drop 0.02-0.1 -> modest contribution")
    print("    drop > 0.1   -> feature is load-bearing for temporal axis")

    # -- 8b. Full cluster ablation scan ----------------------------------------
    if scored:
        print("\n[8]  Full cluster ablation scan (all features individually) ...")
        cluster_feats = scored[0]["features"]   # the single top cluster
        scan_results  = []
        for fi in tqdm(cluster_feats, desc="  ablation scan", leave=False):
            fa_abl = year_feat_acts.clone()
            fa_abl[:, fi] = 0.0
            r_abl = _pc1_year_absr(fa_abl)
            scan_results.append((fi, r_abl, baseline_r - r_abl))

        scan_results.sort(key=lambda x: x[2], reverse=True)   # sort by drop desc

        print(f"  {'Rank':<5}  {'Feature':<8}  {'|r| ablated':>11}  {'drop':>8}")
        print("  " + "-" * 37)
        for rank, (fi, r_abl, drop) in enumerate(scan_results):
            marker = "  <-- load-bearing" if drop > 0.1 else ""
            print(f"  {rank+1:<5}  {fi:<8}  {r_abl:>11.4f}  {drop:>+8.4f}{marker}")

        drops      = np.array([d for _, _, d in scan_results])
        n_load     = int((drops > 0.10).sum())
        n_moderate = int(((drops > 0.02) & (drops <= 0.10)).sum())
        n_negligible = int((drops <= 0.02).sum())
        print()
        print(f"  Summary: {n_load} load-bearing (>0.10)  |  "
              f"{n_moderate} moderate (0.02-0.10)  |  "
              f"{n_negligible} negligible (<0.02)")

        # Histogram
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(drops, bins=20, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.axvline(0.10, color="crimson",  linestyle="--", linewidth=1.2,
                   label="load-bearing threshold (0.10)")
        ax.axvline(0.02, color="darkorange", linestyle="--", linewidth=1.2,
                   label="moderate threshold (0.02)")
        ax.set_xlabel("Drop in |PC1-year r| when feature ablated")
        ax.set_ylabel("Number of features")
        ax.set_title(f"Single-feature ablation across {len(cluster_feats)} cluster features\n"
                     f"baseline |r| = {baseline_r:.4f}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        hist_path = OUTPUT_DIR / "phase1_ablation_histogram.png"
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        print(f"  Saved -> {hist_path.name}")

    # -- 9. Summary ------------------------------------------------------------
    print("\n" + "=" * 65)
    print("PHASE 1  COMPLETE")
    if scored:
        b = scored[0]
        print(f"  Best cluster: #{b['id']}  ({b['n']} features)")
        print(f"    Separability   S(f) = {b['S']:.3f}")
        print(f"    eps-mixture      M_eps  = {b['M_eps']:.3f}")
        print(f"    Irreducibility      = {b['irred']:.3f}")
    print(f"  Outputs saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 65)


# ── Phase 2: Geometric / Helical Analysis ─────────────────────────────────────

PHASE2_LAYERS = [4, 6, 8, 10]   # GPT-2-small layers to examine


def circularity_score(points_2d: np.ndarray) -> float:
    """
    1 - std(radial distances) / mean(radial distances) from the fitted circle.
    Score of 1.0 = perfect circle; 0.0 = no circular structure.
    """
    try:
        cx, cy, _, _ = fit_circle_algebraic(points_2d)
        radii = np.sqrt((points_2d[:, 0] - cx) ** 2 + (points_2d[:, 1] - cy) ** 2)
        return float(np.clip(1.0 - radii.std() / (radii.mean() + 1e-8), 0.0, 1.0))
    except Exception:
        return 0.0


def _extract_layer_acts(model: HookedTransformer, layer: int) -> np.ndarray:
    """Extract last-token residual stream activations for all YEAR_ITEMS."""
    hook = f"blocks.{layer}.hook_resid_pre"
    rows = []
    for _, prompt, _ in tqdm(YEAR_ITEMS, desc=f"  L{layer}", leave=False):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook, device=DEVICE)
        rows.append(cache[hook][0, -1, :].cpu().numpy())
    return np.stack(rows)    # (N_years, 768)


def run_phase2() -> None:
    print("\n" + "=" * 65)
    print("PHASE 2  -  Geometric / Helical Analysis")
    print("=" * 65)

    model    = load_model()
    year_nums  = np.array([it[2] for it in YEAR_ITEMS], dtype=float)
    year_labels = [it[0] for it in YEAR_ITEMS]

    # ── 1. Extract activations at each analysis layer ─────────────────────────
    print("\n[1]  Extracting year activations across layers ...")
    layer_data: dict = {}
    for layer in PHASE2_LAYERS:
        acts = _extract_layer_acts(model, layer)
        n_pcs = min(5, acts.shape[0] - 1, acts.shape[1])
        pca   = PCA(n_components=n_pcs)
        coords = pca.fit_transform(acts)
        var    = pca.explained_variance_ratio_
        layer_data[layer] = dict(acts=acts, pca=pca, coords=coords, var=var)
        r1 = float(np.corrcoef(year_nums, coords[:, 0])[0, 1])
        print(f"  Layer {layer:>2}:  PC1 r={r1:+.3f}  "
              f"var(PC1+PC2+PC3)={var[:3].sum():.1%}")

    # ── 2. Select layer with strongest PC1–year correlation ──────────────────
    print("\n[2]  Selecting analysis layer by PC1-year correlation ...")
    best_layer, best_r1 = LAYER, -np.inf
    for layer, ld in layer_data.items():
        r1 = float(np.corrcoef(year_nums, ld["coords"][:, 0])[0, 1])
        if abs(r1) > abs(best_r1):
            best_layer, best_r1 = layer, r1
    print(f"  Best: layer {best_layer}  (PC1 r = {best_r1:+.4f})")

    # ── 3. Detailed geometric analysis at best layer ──────────────────────────
    print(f"\n[3]  Detailed geometric analysis at layer {best_layer} ...")
    ld     = layer_data[best_layer]
    acts   = ld["acts"]
    coords = ld["coords"]
    var    = ld["var"]

    # Detrend: remove the dominant axis (PC1) to expose residual structure
    pc1_dir   = ld["pca"].components_[0]
    proj_pc1  = (acts @ pc1_dir[:, None]) * pc1_dir[None, :]
    acts_detr = acts - proj_pc1
    n_pcs_d   = min(3, acts_detr.shape[0] - 1, acts_detr.shape[1])
    pca_d     = PCA(n_components=n_pcs_d)
    coords_d  = pca_d.fit_transform(acts_detr)
    var_d     = pca_d.explained_variance_ratio_

    # Circularity scores
    circ_raw  = circularity_score(coords[:, 1:3])
    circ_detr = circularity_score(coords_d[:, :2])
    print(f"  Circularity (raw PC2 vs PC3):         {circ_raw:.4f}")
    print(f"  Circularity (detrended PC1 vs PC2):   {circ_detr:.4f}")

    # Circle fit + arc angles for BC and AD subsets
    bc_mask = [i for i, c in enumerate(YEAR_CATEGORIES) if c == "ancient_bc"]
    ad_mask = [i for i, c in enumerate(YEAR_CATEGORIES) if c == "ancient_ad"]
    cx, cy, r, rmse = fit_circle_algebraic(coords[:, 1:3])
    angles   = np.arctan2(coords[:, 2] - cy, coords[:, 1] - cx)
    bc_ang   = np.degrees(angles[bc_mask])
    ad_ang   = np.degrees(angles[ad_mask])
    arc_sep  = float(abs(bc_ang.mean() - ad_ang.mean()))
    arc_sep  = min(arc_sep, 360 - arc_sep)
    print(f"  Circle fit (PC2 vs PC3):  cx={cx:.3f}  cy={cy:.3f}  "
          f"r={r:.3f}  RMSE={rmse:.4f}")
    print(f"  BC arc:  mean={bc_ang.mean():+.1f} deg  std={bc_ang.std():.1f} deg")
    print(f"  AD arc:  mean={ad_ang.mean():+.1f} deg  std={ad_ang.std():.1f} deg")
    print(f"  BC-AD arc separation: {arc_sep:.1f} deg  "
          f"({'opposite sides' if arc_sep > 120 else 'overlapping'})")

    # Within-group temporal ordering
    for mask, label in [(bc_mask, "BC"), (ad_mask, "AD")]:
        yrs_g = year_nums[mask]
        r_g   = float(np.corrcoef(yrs_g, coords[mask, 0])[0, 1]) if len(mask) > 2 else float("nan")
        print(f"  Within-{label} PC1 vs year r:      {r_g:+.3f}")

    # ── 4. Visualise ──────────────────────────────────────────────────────────
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


# ── Phase 3: Superposition depth scan ─────────────────────────────────────────

PHASE3_LAYERS = [4, 6, 8, 10]


def _ablation_scan_at_layer(
    model: HookedTransformer,
    layer: int,
) -> dict:
    """
    For a single layer, run the full cluster ablation scan and return summary
    statistics used to trace how superposition evolves with depth.

    Returns a dict with keys:
        layer, baseline_r, max_drop, top_feature,
        n_load, n_moderate, n_negligible, all_drops (np.ndarray)
    """
    hook     = f"blocks.{layer}.hook_resid_pre"
    sae_id   = f"blocks.{layer}.hook_resid_pre"

    # Load SAE for this layer
    print(f"  Loading SAE for layer {layer} ...")
    sae, _, _ = SAE.from_pretrained(
        release=SAE_RELEASE, sae_id=sae_id, device=DEVICE,
    )
    sae.eval()

    # Extract year activations at this layer
    rows = []
    for _, prompt, _ in tqdm(YEAR_ITEMS, desc=f"  L{layer} acts", leave=False):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook, device=DEVICE)
        rows.append(cache[hook][0, -1, :].cpu())
    acts = torch.stack(rows)                    # (N_years, 768)

    # SAE feature activations for year items + all items (for clustering)
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

    # Find clusters using epoch categories
    clusters, _ = find_clusters(feat_acts_all, ALL_CATS, corr_threshold=CORR_THRESHOLD)
    if not clusters:
        clusters, _ = find_clusters(feat_acts_all, ALL_CATS, corr_threshold=0.3)
    if not clusters:
        print(f"    No clusters found at layer {layer} — skipping.")
        return dict(layer=layer, baseline_r=float("nan"), max_drop=float("nan"),
                    top_feature=-1, n_load=0, n_moderate=0, n_negligible=0,
                    all_drops=np.array([]))

    cluster_feats = clusters[0]
    W_dec = sae.W_dec.detach().cpu()
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
        drop = baseline_r - _absr(fa_abl)
        scan.append((fi, drop))

    scan.sort(key=lambda x: x[1], reverse=True)
    drops     = np.array([d for _, d in scan])
    top_feat  = scan[0][0]
    n_load    = int((drops > 0.10).sum())
    n_mod     = int(((drops > 0.02) & (drops <= 0.10)).sum())
    n_neg     = int((drops <= 0.02).sum())

    print(f"    Layer {layer}: baseline |r|={baseline_r:.4f}  "
          f"max_drop={drops[0]:+.4f}  top_feat={top_feat}  "
          f"load={n_load}  mod={n_mod}  neg={n_neg}")

    return dict(layer=layer, baseline_r=baseline_r, max_drop=drops[0],
                top_feature=top_feat, n_load=n_load, n_moderate=n_mod,
                n_negligible=n_neg, all_drops=drops)


def run_phase3() -> None:
    import matplotlib.pyplot as plt

    print("\n" + "=" * 65)
    print("PHASE 3  -  Superposition Depth Scan")
    print(f"  layers={PHASE3_LAYERS}")
    print("=" * 65)

    model = load_model()
    results = []
    for layer in PHASE3_LAYERS:
        print(f"\n--- Layer {layer} ---")
        results.append(_ablation_scan_at_layer(model, layer))

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 3  SUMMARY")
    print(f"  {'Layer':>6}  {'base |r|':>9}  {'max drop':>9}  "
          f"{'top feat':>9}  {'load':>5}  {'mod':>5}  {'neg':>5}")
    print("  " + "-" * 58)
    for r in results:
        print(f"  {r['layer']:>6}  {r['baseline_r']:>9.4f}  {r['max_drop']:>+9.4f}  "
              f"  {r['top_feature']:>8}  {r['n_load']:>5}  "
              f"{r['n_moderate']:>5}  {r['n_negligible']:>5}")

    # ── Plot 1: max single-feature drop vs layer ──────────────────────────────
    layers_valid = [r["layer"] for r in results if not np.isnan(r["max_drop"])]
    max_drops    = [r["max_drop"] for r in results if not np.isnan(r["max_drop"])]
    base_rs      = [r["baseline_r"] for r in results if not np.isnan(r["baseline_r"])]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(layers_valid, max_drops, "o-", color="steelblue", linewidth=2, markersize=7)
    ax.axhline(0.10, color="crimson",   linestyle="--", linewidth=1.1,
               label="load-bearing threshold (0.10)")
    ax.axhline(0.02, color="darkorange", linestyle="--", linewidth=1.1,
               label="moderate threshold (0.02)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Max single-feature drop in |PC1-year r|")
    ax.set_title("Concentration of temporal signal\n(higher = less superposition)")
    ax.set_xticks(layers_valid)
    ax.legend(fontsize=8)

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

    # ── Plot 2: per-layer drop distributions (violin) ─────────────────────────
    valid = [r for r in results if len(r["all_drops"]) > 0]
    if valid:
        fig, ax = plt.subplots(figsize=(8, 4))
        data    = [r["all_drops"] for r in valid]
        lbls    = [str(r["layer"]) for r in valid]
        parts   = ax.violinplot(data, positions=range(len(valid)),
                                showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("steelblue")
            pc.set_alpha(0.6)
        ax.set_xticks(range(len(valid)))
        ax.set_xticklabels([f"Layer {l}" for l in lbls])
        ax.axhline(0.10, color="crimson",   linestyle="--", linewidth=1.1,
                   label="load-bearing (0.10)")
        ax.axhline(0.02, color="darkorange", linestyle="--", linewidth=1.1,
                   label="moderate (0.02)")
        ax.set_ylabel("Drop in |PC1-year r|")
        ax.set_title("Distribution of single-feature ablation drops by layer\n"
                     "(narrowing = more distributed / more superposition)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p2 = OUTPUT_DIR / "phase3_drop_distributions.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        print(f"  Saved -> {p2.name}")

    print("\n" + "=" * 65)
    print("PHASE 3  COMPLETE")
    print(f"  Outputs saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    run_phase1()
    run_phase2()
    run_phase3()
