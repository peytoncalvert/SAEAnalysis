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

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy.stats import f_oneway

from transformer_lens import HookedTransformer
from sae_lens import SAE

# -- Configuration --------------------------------------------------------------

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR  = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME  = "gpt2"
LAYER       = 8                          # mid-to-late layer (GPT-2-small has 12)
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

YEAR_ITEMS = [
    # Deep antiquity (BC)
    ("3100BC", "The unification of Egypt occurred around 3100 BC",       -3100),
    ("2560BC", "The Great Pyramid of Giza was completed in 2560 BC",     -2560),
    ("1754BC", "The Code of Hammurabi was written in 1754 BC",           -1754),
    ("1274BC", "The Battle of Kadesh was fought in 1274 BC",             -1274),
    ("776BC",  "The first Olympic Games were held in 776 BC",             -776),
    ("753BC",  "Rome was traditionally founded in 753 BC",                -753),
    ("508BC",  "Athenian democracy was established in 508 BC",            -508),
    ("490BC",  "The Battle of Marathon took place in 490 BC",             -490),
    ("336BC",  "Alexander the Great became king of Macedon in 336 BC",   -336),
    ("323BC",  "Alexander the Great died in Babylon in 323 BC",          -323),
    ("264BC",  "The First Punic War began in 264 BC",                    -264),
    ("44BC",   "Julius Caesar was assassinated in 44 BC",                 -44),
    ("1BC",    "The last year before the common era was 1 BC",              -1),
    # AD / early medieval
    ("1AD",    "The first year of the common era was 1 AD",                  1),
    ("79AD",   "Mount Vesuvius erupted and buried Pompeii in 79 AD",        79),
    ("313AD",  "The Edict of Milan granted religious tolerance in 313 AD", 313),
    ("476AD",  "The Western Roman Empire fell in 476 AD",                  476),
    ("622AD",  "Muhammad's Hijra to Medina occurred in 622 AD",            622),
    ("800AD",  "Charlemagne was crowned Holy Roman Emperor in 800 AD",     800),
    # Medieval
    ("1066",   "The Norman Conquest of England occurred in 1066",         1066),
    ("1095",   "The First Crusade was called in 1095",                    1095),
    ("1215",   "The Magna Carta was signed in 1215",                      1215),
    ("1337",   "The Hundred Years War began in 1337",                     1337),
    ("1348",   "The Black Death spread across Europe in 1348",            1348),
    ("1453",   "Constantinople fell to the Ottomans in 1453",             1453),
    ("1492",   "Columbus reached the Americas in 1492",                   1492),
    # Early modern
    ("1517",   "Martin Luther published his theses in 1517",              1517),
    ("1588",   "The Spanish Armada was defeated in 1588",                 1588),
    ("1648",   "The Peace of Westphalia was signed in 1648",              1648),
    ("1687",   "Newton published Principia Mathematica in 1687",          1687),
    ("1776",   "The United States declared independence in 1776",         1776),
    ("1789",   "The French Revolution began in 1789",                     1789),
    ("1815",   "Napoleon was defeated at Waterloo in 1815",               1815),
    # Modern
    ("1848",   "The year of revolutions across Europe was 1848",          1848),
    ("1865",   "The American Civil War ended in 1865",                    1865),
    ("1903",   "The Wright Brothers first flew in 1903",                  1903),
    ("1914",   "World War I began in 1914",                               1914),
    ("1918",   "World War I ended in 1918",                               1918),
    ("1929",   "The Great Depression began in 1929",                      1929),
    ("1939",   "World War II began in 1939",                              1939),
    ("1945",   "World War II ended in 1945",                              1945),
    ("1969",   "The Moon landing occurred in 1969",                       1969),
    ("1991",   "The Soviet Union dissolved in 1991",                      1991),
]

YEAR_CATEGORIES = (
    ["deep_antiquity"] * 13 +   # 3100BC–1BC  (12 original + 1BC)
    ["early_ad"]       * 6  +   # 1AD–800AD   (1AD added, 6 total)
    ["medieval"]       * 7  +
    ["early_modern"]   * 7  +
    ["modern"]         * 10
)
assert len(YEAR_CATEGORIES) == len(YEAR_ITEMS), (
    f"YEAR_CATEGORIES length {len(YEAR_CATEGORIES)} != YEAR_ITEMS length {len(YEAR_ITEMS)}"
)

BATTLE_ITEMS = [
    ("Hastings",   "The pivotal Norman battle was the Battle of Hastings",        1066),
    ("Agincourt",  "Henry V's famous victory was the Battle of Agincourt",        1415),
    ("Waterloo",   "Napoleon's final defeat was at the Battle of Waterloo",       1815),
    ("Gettysburg", "The Civil War turning point was the Battle of Gettysburg",    1863),
    ("Somme",      "One of WWI's bloodiest battles was the Battle of the Somme",  1916),
    ("Stalingrad", "The decisive Eastern Front clash was Stalingrad",             1943),
    ("Midway",     "The Pacific turning point was the Battle of Midway",          1942),
    ("Trafalgar",  "Nelson's naval victory was the Battle of Trafalgar",          1805),
]

ALL_ITEMS = YEAR_ITEMS + BATTLE_ITEMS
ALL_CATS  = YEAR_CATEGORIES + ["battle"] * len(BATTLE_ITEMS)

# Category -> plot colour
CAT_COLORS = {
    "deep_antiquity": "#6c3483",
    "early_ad":       "#9b59b6",
    "medieval":       "#2980b9",
    "early_modern":   "#27ae60",
    "modern":         "#e74c3c",
    "battle":         "#e67e22",
}


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
    M_eps(f): ratio of best single-PC GMM log-likelihood to full-D GMM
    log-likelihood.

    Low M_eps -> the cluster is NOT well-explained by a 1-D mixture, meaning it
    is genuinely multi-dimensional (irreducible).
    """
    X   = recon.numpy()
    N, d = X.shape
    if N < 4 or d < 2:
        return 1.0

    n_comp = min(2, N // 2)

    try:
        gmm_full  = GaussianMixture(n_components=n_comp, random_state=0, max_iter=300)
        gmm_full.fit(X)
        score_full = gmm_full.score(X)
    except Exception:
        return 1.0

    n_pcs  = min(d, N - 1, 5)
    X_pca  = PCA(n_components=n_pcs).fit_transform(X)
    best_1d = -np.inf
    for pc in range(n_pcs):
        try:
            g = GaussianMixture(n_components=n_comp, random_state=0, max_iter=300)
            g.fit(X_pca[:, pc : pc + 1])
            best_1d = max(best_1d, g.score(X_pca[:, pc : pc + 1]))
        except Exception:
            pass

    if best_1d == -np.inf:
        return 1.0

    return float(np.clip(np.exp(best_1d - score_full), 0.0, 1.0))


# -- Visualisation --------------------------------------------------------------

def _cat_colors(categories: list[str]) -> list[str]:
    return [CAT_COLORS.get(c, "#7f8c8d") for c in categories]


def plot_pca_overview(
    acts: torch.Tensor,
    labels: list[str],
    categories: list[str],
    title: str,
    path: Path,
) -> None:
    """2-D and 3-D PCA scatter, coloured by historical category."""
    X    = acts.numpy()
    N    = X.shape[0]
    n_pc = min(3, N - 1, X.shape[1])

    pca    = PCA(n_components=n_pc)
    coords = pca.fit_transform(X)
    var    = pca.explained_variance_ratio_
    colors = _cat_colors(categories)

    ncols = 2 if n_pc >= 3 else 1
    fig   = plt.figure(figsize=(7 * ncols, 6))
    fig.suptitle(title, fontsize=10)

    # 2-D scatter
    ax = fig.add_subplot(1, ncols, 1)
    for i in range(N):
        ax.scatter(coords[i, 0], coords[i, 1],
                   color=colors[i], s=90, edgecolors="k", linewidths=0.5, zorder=3)
        ax.annotate(labels[i], (coords[i, 0], coords[i, 1]),
                    fontsize=7, alpha=0.85, ha="center", va="bottom")
    ax.set_xlabel(f"PC1  ({var[0]:.1%})")
    ax.set_ylabel(f"PC2  ({var[1]:.1%})" if n_pc >= 2 else "")
    ax.set_title("PC1 vs PC2")
    ax.grid(alpha=0.3)
    for cat, col in CAT_COLORS.items():
        ax.scatter([], [], color=col, label=cat, s=60)
    ax.legend(fontsize=7)

    # 3-D scatter
    if n_pc >= 3:
        ax3 = fig.add_subplot(1, 2, 2, projection="3d")
        for i in range(N):
            ax3.scatter(coords[i, 0], coords[i, 1], coords[i, 2],
                        color=colors[i], s=60, edgecolors="k", linewidths=0.3)
            ax3.text(coords[i, 0], coords[i, 1], coords[i, 2], labels[i], fontsize=6)
        ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2"); ax3.set_zlabel("PC3")
        ax3.set_title(f"3-D  ({var[:3].sum():.1%} var)")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved -> {path.name}")


def plot_year_linearity(
    acts: torch.Tensor,
    items: list,
    path: Path,
) -> None:
    """
    Plot PC1 / PC2 / PC3 vs. the numeric year value.
    A high Pearson |r| means the model linearly encodes temporal position,
    consistent with Gurnee & Tegmark.
    """
    X       = acts.numpy()
    numeric = np.array([it[2] for it in items], dtype=float)
    lbls    = [it[0] for it in items]

    n_pc   = min(3, X.shape[0] - 1, X.shape[1])
    pca    = PCA(n_components=n_pc)
    coords = pca.fit_transform(X)
    var    = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, n_pc, figsize=(5 * n_pc, 4))
    if n_pc == 1:
        axes = [axes]

    for pc, ax in enumerate(axes):
        sc = ax.scatter(numeric, coords[:, pc], c=numeric,
                        cmap="plasma", s=80, edgecolors="k", linewidths=0.5)
        for i, lbl in enumerate(lbls):
            ax.annotate(lbl, (numeric[i], coords[i, pc]), fontsize=7, alpha=0.85)
        r = float(np.corrcoef(numeric, coords[:, pc])[0, 1])
        ax.set_xlabel("Year (numeric)")
        ax.set_ylabel(f"PC{pc+1}  ({var[pc]:.1%} var)")
        ax.set_title(f"PC{pc+1} vs Year   r = {r:.3f}")
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax, label="Year")

    fig.suptitle(f"Linear Decodability of Year Value  (Layer {LAYER})", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved -> {path.name}")


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

    # -- 8. Summary ------------------------------------------------------------
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


if __name__ == "__main__":
    run_phase1()
