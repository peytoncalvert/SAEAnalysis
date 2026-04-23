"""
plots.py — all matplotlib visualisation functions for final.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

# Category -> plot colour (shared across all plots)
CAT_COLORS = {
    "ancient_bc":     "#6c3483",
    "ancient_ad":     "#a569bd",
    "medieval":       "#2980b9",
    "early_modern":   "#27ae60",
    "modern":         "#e74c3c",
    "battle_europe":  "#e67e22",
    "battle_america": "#c0392b",
    "battle_east":    "#d4ac0d",
}


def _cat_colors(categories: list[str]) -> list[str]:
    return [CAT_COLORS.get(c, "#7f8c8d") for c in categories]


def fit_circle_algebraic(points: np.ndarray) -> tuple[float, float, float, float]:
    """
    Algebraic least-squares circle fit (Coope 1993).
    Returns (cx, cy, radius, RMSE).
    """
    X, Y = points[:, 0], points[:, 1]
    A    = np.column_stack([2 * X, 2 * Y, np.ones(len(X))])
    b    = X ** 2 + Y ** 2
    res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = float(res[0]), float(res[1])
    r      = float(np.sqrt(res[2] + cx ** 2 + cy ** 2))
    radii  = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    rmse   = float(np.sqrt(np.mean((radii - r) ** 2)))
    return cx, cy, r, rmse


def plot_pca_overview(
    acts,
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
    acts,
    items: list,
    path: Path,
    layer: int,
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

    fig.suptitle(f"Linear Decodability of Year Value  (Layer {layer})", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved -> {path.name}")


def _plot_helix_scan(
    layer_data: dict,
    year_nums: np.ndarray,
    helix_periods,
    helix_r2,
    output_dir: Path,
) -> None:
    """Heatmap: helix R² across layers and candidate periods."""
    layers = list(layer_data.keys())
    matrix = np.zeros((len(layers), len(helix_periods)))
    for i, layer in enumerate(layers):
        coords3 = layer_data[layer]["coords"][:, :3]
        for j, p in enumerate(helix_periods):
            matrix[i, j] = helix_r2(coords3, year_nums, p)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Helix R2")
    ax.set_xticks(range(len(helix_periods)))
    ax.set_xticklabels([f"T={p}" for p in helix_periods])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"Layer {l}" for l in layers])
    ax.set_xlabel("Helix period (years)")
    ax.set_ylabel("GPT-2 layer")
    ax.set_title("Helix R2: does a year-helix explain the PCA structure?")
    for i in range(len(layers)):
        for j in range(len(helix_periods)):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if v > 0.6 else "black")
    plt.tight_layout()
    p = output_dir / "phase2_helix_scan.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {p.name}")


def _plot_geometric_analysis(
    coords: np.ndarray, var: np.ndarray,
    angles: np.ndarray,
    bc_mask: list, ad_mask: list,
    coords_d: np.ndarray, var_d: np.ndarray,
    year_nums: np.ndarray, year_labels: list,
    layer: int,
    year_categories: list,
    output_dir: Path,
) -> None:
    """Four-panel geometric structure plot: PCA, circular, detrended, angular."""
    cat_colors_pts = [CAT_COLORS.get(c, "#7f8c8d") for c in year_categories]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f"Phase 2 - Geometric Structure  (Layer {layer})",
        fontsize=11,
    )

    # ── Panel 1: PC1 vs PC2, coloured by numeric year ────────────────────────
    ax1 = axes[0]
    sc1 = ax1.scatter(coords[:, 0], coords[:, 1], c=year_nums,
                      cmap="plasma", s=80, edgecolors="k", linewidths=0.5)
    for i, lbl in enumerate(year_labels):
        ax1.annotate(lbl, (coords[i, 0], coords[i, 1]), fontsize=6, alpha=0.8)
    plt.colorbar(sc1, ax=ax1, label="Year")
    ax1.set_xlabel(f"PC1 ({var[0]:.1%})")
    ax1.set_ylabel(f"PC2 ({var[1]:.1%})")
    ax1.set_title("Raw: PC1 vs PC2 (by year value)")
    ax1.grid(alpha=0.3)

    # ── Panel 2: PC2 vs PC3 with fitted circle + BC/AD arcs ─────────────────
    ax2 = axes[1]
    ax2.scatter(coords[:, 1], coords[:, 2], c=cat_colors_pts,
                s=80, edgecolors="k", linewidths=0.5)
    for i, lbl in enumerate(year_labels):
        ax2.annotate(lbl, (coords[i, 1], coords[i, 2]), fontsize=6, alpha=0.8)

    # Legend handles: one patch per unique category present in the data
    import matplotlib.patches as mpatches
    seen_cats = dict.fromkeys(year_categories)   # preserves insertion order, dedupes
    cat_handles = [
        mpatches.Patch(color=CAT_COLORS.get(c, "#7f8c8d"), label=c.replace("_", " "))
        for c in seen_cats
    ]

    try:
        cx, cy, r, rmse = fit_circle_algebraic(coords[:, 1:3])
        th = np.linspace(-np.pi, np.pi, 300)
        circle_line, = ax2.plot(cx + r * np.cos(th), cy + r * np.sin(th),
                                "k--", alpha=0.45, linewidth=1.5)
        ax2.scatter([cx], [cy], color="black", marker="+", s=120, zorder=6)
        ax2.scatter(coords[bc_mask, 1], coords[bc_mask, 2],
                    color="#6c3483", s=130, edgecolors="w", linewidths=1.2, zorder=7)
        ax2.scatter(coords[ad_mask, 1], coords[ad_mask, 2],
                    color="#a569bd", s=130, marker="^", edgecolors="w",
                    linewidths=1.2, zorder=7)
        circle_handle = mpatches.Patch(
            facecolor="none", edgecolor="black",
            linestyle="--", label=f"Circle (RMSE={rmse:.3f})",
        )
        ax2.legend(handles=cat_handles + [circle_handle], fontsize=7)
    except Exception:
        ax2.legend(handles=cat_handles, fontsize=7)

    ax2.set_xlabel(f"PC2 ({var[1]:.1%})")
    ax2.set_ylabel(f"PC3 ({var[2]:.1%})")
    ax2.set_title("PC2 vs PC3 + fitted circle")
    ax2.grid(alpha=0.3)

    # ── Panel 3: Detrended (PC1 removed) ─────────────────────────────────────
    ax3 = axes[2]
    sc3 = ax3.scatter(coords_d[:, 0], coords_d[:, 1], c=year_nums,
                      cmap="plasma", s=80, edgecolors="k", linewidths=0.5)
    for i, lbl in enumerate(year_labels):
        ax3.annotate(lbl, (coords_d[i, 0], coords_d[i, 1]), fontsize=6, alpha=0.8)
    try:
        cx_d, cy_d, r_d, rmse_d = fit_circle_algebraic(coords_d[:, :2])
        th = np.linspace(-np.pi, np.pi, 300)
        ax3.plot(cx_d + r_d * np.cos(th), cy_d + r_d * np.sin(th),
                 "k--", alpha=0.45, linewidth=1.5,
                 label=f"Circle (RMSE={rmse_d:.3f})")
        ax3.legend(fontsize=7)
    except Exception:
        pass
    plt.colorbar(sc3, ax=ax3, label="Year")
    ax3.set_xlabel(f"Detr-PC1 ({var_d[0]:.1%})")
    ax3.set_ylabel(f"Detr-PC2 ({var_d[1]:.1%})")
    ax3.set_title("Detrended (dominant axis removed)")
    ax3.grid(alpha=0.3)

    # ── Panel 4: Angle around fitted circle vs year ───────────────────────────
    ax4 = axes[3]
    deg = np.degrees(angles)
    sc4 = ax4.scatter(year_nums, deg, c=year_nums,
                      cmap="plasma", s=80, edgecolors="k", linewidths=0.5)
    for i, lbl in enumerate(year_labels):
        ax4.annotate(lbl, (year_nums[i], deg[i]), fontsize=6, alpha=0.8)
    r_ang = float(np.corrcoef(year_nums, deg)[0, 1])
    ax4.set_xlabel("Year (numeric)")
    ax4.set_ylabel("Angle on fitted circle (deg)")
    ax4.set_title(f"Angular position vs year  (r = {r_ang:+.3f})")
    ax4.grid(alpha=0.3)
    plt.colorbar(sc4, ax=ax4, label="Year")

    plt.tight_layout()
    p = output_dir / f"phase2_geometric_layer{layer}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {p.name}")
