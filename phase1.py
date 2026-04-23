"""
phase1.py  —  Feature Clustering & Irreducibility Scoring
==========================================================
1. Extract residual-stream activations for historical prompts at layer LAYER.
2. Run the SAE encoder to get sparse feature activations.
3. Cluster the top discriminative features by epoch-profile correlation.
4. Score each cluster with separability S(f) and eps-mixture M_eps.
5. Visualise top clusters in PCA space.
6. Ablation experiment: measure how much each of the four key features and
   every cluster feature individually contributes to the PC1-year correlation.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy.stats import f_oneway

from config import (
    DEVICE, OUTPUT_DIR, LAYER, HOOK_NAME,
    YEAR_ITEMS, YEAR_CATEGORIES, ALL_ITEMS, ALL_CATS,
    CORR_THRESHOLD, MIN_CLUSTER_SZ, N_DISC,
    load_model, load_sae,
)
from plots import plot_pca_overview, plot_year_linearity


# ── Activation extraction ─────────────────────────────────────────────────────

def extract_activations(model, items: list) -> tuple[torch.Tensor, list[str]]:
    """Return (N, d_model) residual-stream activations at the last token position."""
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


def get_feature_acts(sae, activations: torch.Tensor) -> torch.Tensor:
    """Run the SAE encoder; returns (N, d_sae) sparse feature activations."""
    with torch.no_grad():
        feat_acts = sae.encode(activations.to(DEVICE))
    return feat_acts.cpu()


# ── Feature clustering ────────────────────────────────────────────────────────

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
    clusters  : list of clusters, each a list of global feature indices
    f_scores  : (d_sae,) ANOVA F-score for every feature
    """
    N, d_sae = feat_acts.shape
    cats    = sorted(set(categories))
    cat_idx = {c: [i for i, cat in enumerate(categories) if cat == c] for c in cats}

    freq       = (feat_acts > 0).float().sum(0)
    active_idx = freq.ge(2).nonzero(as_tuple=True)[0].tolist()
    print(f"    Active features (fire on >= 2 tokens): {len(active_idx)} / {d_sae}")

    if len(active_idx) < 2:
        return [], np.zeros(d_sae)

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

    n_keep    = min(n_disc, len(active_idx))
    top_local = np.argsort(f_scores_active)[-n_keep:][::-1]
    top_global = [active_idx[i] for i in top_local]
    print(f"    Top-{n_keep} discriminative features selected  "
          f"(max F = {f_scores_active[top_local[0]]:.2f})")

    profiles = np.array([
        [feat_acts[cat_idx[c], fi].mean().item() for c in cats]
        for fi in top_global
    ])
    stds       = profiles.std(1, keepdims=True) + 1e-8
    profiles_z = (profiles - profiles.mean(1, keepdims=True)) / stds
    corr       = (profiles_z @ profiles_z.T) / profiles.shape[1]

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


# ── Irreducibility metrics ────────────────────────────────────────────────────

def separability_index(recon: torch.Tensor, category_labels: list[str]) -> float:
    cats = list(set(category_labels))
    if len(cats) < 2 or len(category_labels) < 4:
        return 0.0
    X  = recon.numpy()
    y  = LabelEncoder().fit_transform(category_labels)
    n_pc = min(10, X.shape[0] - 1, X.shape[1])
    if n_pc < 1:
        return 0.0
    X_pc  = PCA(n_components=n_pc).fit_transform(X)
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
    X    = recon.numpy()
    N, d = X.shape
    if N < 4 or d < 2:
        return 1.0
    n_pcs = min(N - 1, d, 10)
    try:
        pca = PCA(n_components=n_pcs)
        pca.fit(X)
        return float(pca.explained_variance_ratio_[0])
    except Exception:
        return 1.0


# ── Phase 1 orchestration ─────────────────────────────────────────────────────

def run_phase1() -> None:
    print("=" * 65)
    print("PHASE 1  -  Feature Clustering & Irreducibility Scoring")
    print(f"  model={__import__('config').MODEL_NAME}   layer={LAYER}   device={DEVICE}")
    print("=" * 65)

    model = load_model()
    sae   = load_sae()

    # [1] Activations
    print("\n[1]  Extracting residual-stream activations ...")
    year_acts,   year_labels   = extract_activations(model, YEAR_ITEMS)
    battle_acts, battle_labels = extract_activations(model, __import__('config').BATTLE_ITEMS)
    all_acts   = torch.cat([year_acts, battle_acts], dim=0)
    all_labels = year_labels + battle_labels
    print(f"  years: {year_acts.shape}  |  battles: {battle_acts.shape}  |  total: {all_acts.shape}")

    # [2] Baseline PCA
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
    acts_np = year_acts.numpy()
    numeric = np.array([it[2] for it in YEAR_ITEMS], dtype=float)
    n_pc    = min(3, acts_np.shape[0] - 1, acts_np.shape[1])
    coords  = PCA(n_components=n_pc).fit_transform(acts_np)
    for pc in range(n_pc):
        r = float(np.corrcoef(numeric, coords[:, pc])[0, 1])
        print(f"    PC{pc+1} vs year:  r = {r:+.3f}")

    # [3] SAE features
    print("\n[3]  Running SAE encoder ...")
    feat_acts   = get_feature_acts(sae, all_acts)
    mean_active = (feat_acts > 0).float().sum(1).mean().item()
    print(f"  feature acts: {feat_acts.shape}  |  mean active / token: {mean_active:.1f}")

    # [4] Clustering
    print("\n[4]  Clustering discriminative features by epoch-profile correlation ...")
    clusters, f_scores = find_clusters(feat_acts, ALL_CATS, corr_threshold=CORR_THRESHOLD)
    if not clusters:
        print("  No clusters at threshold 0.5 - retrying at 0.3 ...")
        clusters, f_scores = find_clusters(feat_acts, ALL_CATS, corr_threshold=0.3)
    print(f"  Clusters found: {len(clusters)}")

    top5_feat = np.argsort(f_scores)[-5:][::-1]
    print("  Top-5 discriminative features by ANOVA F-score:")
    for rank, fi in enumerate(top5_feat):
        print(f"    #{rank+1}  feature {fi:>6}   F = {f_scores[fi]:.2f}")

    # [5] Score clusters
    print("\n[5]  Scoring clusters for irreducibility ...")
    W_dec = sae.W_dec.detach().cpu()

    print(f"  {'ID':>4}  {'|f|':>5}  {'S(f)':>7}  {'M_eps':>7}  {'Irred.':>8}")
    print("  " + "-" * 37)

    scored = []
    for cid, feat_indices in enumerate(clusters):
        fi    = torch.tensor(feat_indices, dtype=torch.long)
        recon = feat_acts[:, fi] @ W_dec[fi, :]
        S     = separability_index(recon, ALL_CATS)
        M_eps = epsilon_mixture_index(recon)
        irred = S * (1.0 - M_eps)
        scored.append(dict(
            id=cid, features=feat_indices, n=len(feat_indices),
            S=S, M_eps=M_eps, irred=irred, recon=recon,
        ))
        print(f"  {cid:>4}  {len(feat_indices):>5}  {S:>7.3f}  {M_eps:>7.3f}  {irred:>8.3f}")
    scored.sort(key=lambda x: x["irred"], reverse=True)

    # [6] Visualise top-3 clusters
    print("\n[6]  Visualising top-3 clusters by irreducibility ...")
    for rank, entry in enumerate(scored[:3]):
        plot_pca_overview(
            entry["recon"], all_labels, ALL_CATS,
            title=(
                f"Cluster {entry['id']}  (rank {rank+1}/{len(scored)})  "
                f"S={entry['S']:.3f}  M_eps={entry['M_eps']:.3f}  "
                f"irred={entry['irred']:.3f}"
            ),
            path=OUTPUT_DIR / f"phase1_cluster{entry['id']}_rank{rank+1}.png",
        )

    # [7] Key-feature ablation
    print("\n[7]  Feature ablation experiment ...")
    KEY_FEATURES = {
        1772:  "AD token detector (orthographic)",
        16307: "BC token detector (orthographic)",
        8113:  "Historical date marker (context-gated)",
        22008: "Modern year/date context",
    }

    year_feat_acts = get_feature_acts(sae, year_acts)
    year_numeric   = np.array([it[2] for it in YEAR_ITEMS], dtype=float)

    def _pc1_year_absr(fa: torch.Tensor) -> float:
        recon  = (fa @ W_dec).numpy()
        coords = PCA(n_components=1).fit_transform(recon)
        return abs(float(np.corrcoef(year_numeric, coords[:, 0])[0, 1]))

    baseline_r = _pc1_year_absr(year_feat_acts)
    print(f"  Baseline |PC1-year r| (full SAE recon): {baseline_r:.4f}")
    print()
    print(f"  {'Feature':<8}  {'Description':<42}  {'|r| ablated':>11}  {'drop':>8}")
    print("  " + "-" * 73)

    for fi, desc in KEY_FEATURES.items():
        fa_abl = year_feat_acts.clone()
        fa_abl[:, fi] = 0.0
        r_abl = _pc1_year_absr(fa_abl)
        drop  = baseline_r - r_abl
        print(f"  {fi:<8}  {desc:<42}  {r_abl:>11.4f}  {drop:>+8.4f}")

    fa_all = year_feat_acts.clone()
    for fi in KEY_FEATURES:
        fa_all[:, fi] = 0.0
    r_all     = _pc1_year_absr(fa_all)
    drop_all  = baseline_r - r_all
    print("  " + "-" * 73)
    print(f"  {'all 4':<8}  {'all four ablated':<42}  {r_all:>11.4f}  {drop_all:>+8.4f}")
    print()
    print("  Interpretation guide (drop = baseline |r| - ablated |r|):")
    print("    drop < 0.02  -> negligible  (encoding is distributed)")
    print("    drop 0.02-0.1 -> modest contribution")
    print("    drop > 0.1   -> load-bearing for temporal axis")

    # [8] Full cluster ablation scan
    if scored:
        print("\n[8]  Full cluster ablation scan (all features individually) ...")
        cluster_feats = scored[0]["features"]
        scan_results  = []
        for fi in tqdm(cluster_feats, desc="  ablation scan", leave=False):
            fa_abl = year_feat_acts.clone()
            fa_abl[:, fi] = 0.0
            drop = baseline_r - _pc1_year_absr(fa_abl)
            scan_results.append((fi, _pc1_year_absr(year_feat_acts.clone().__setitem__(
                (slice(None), fi), 0.0) or year_feat_acts.clone()), drop))

        # re-run cleanly
        scan_results = []
        for fi in tqdm(cluster_feats, desc="  ablation scan", leave=False):
            fa_abl = year_feat_acts.clone()
            fa_abl[:, fi] = 0.0
            r_abl = _pc1_year_absr(fa_abl)
            scan_results.append((fi, r_abl, baseline_r - r_abl))
        scan_results.sort(key=lambda x: x[2], reverse=True)

        drops        = np.array([d for _, _, d in scan_results])
        n_load       = int((drops > 0.10).sum())
        n_moderate   = int(((drops > 0.02) & (drops <= 0.10)).sum())
        n_negligible = int((drops <= 0.02).sum())

        print(f"  {'Rank':<5}  {'Feature':<8}  {'|r| ablated':>11}  {'drop':>8}")
        print("  " + "-" * 37)
        for rank, (fi, r_abl, drop) in enumerate(scan_results):
            marker = "  <-- load-bearing" if drop > 0.1 else ""
            print(f"  {rank+1:<5}  {fi:<8}  {r_abl:>11.4f}  {drop:>+8.4f}{marker}")
        print()
        print(f"  Summary: {n_load} load-bearing (>0.10)  |  "
              f"{n_moderate} moderate (0.02-0.10)  |  "
              f"{n_negligible} negligible (<0.02)")

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(drops, bins=20, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.axvline(0.10, color="crimson",   linestyle="--", linewidth=1.2,
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

    # [9] Summary
    print("\n" + "=" * 65)
    print("PHASE 1  COMPLETE")
    if scored:
        b = scored[0]
        print(f"  Best cluster: #{b['id']}  ({b['n']} features)")
        print(f"    Separability   S(f) = {b['S']:.3f}")
        print(f"    eps-mixture    M_eps = {b['M_eps']:.3f}")
        print(f"    Irreducibility      = {b['irred']:.3f}")
    print(f"  Outputs saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 65)
