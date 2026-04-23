"""
config.py  —  shared constants, prompt data, and model-loading helpers.
Imported by phase1.py, phase2.py, and phase3.py.
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from pathlib import Path

from transformer_lens import HookedTransformer
from sae_lens import SAE

# ── Paths & device ────────────────────────────────────────────────────────────

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Model / SAE identifiers ───────────────────────────────────────────────────

MODEL_NAME = "gpt2"
LAYER      = 4                          # primary analysis layer
HOOK_NAME  = f"blocks.{LAYER}.hook_resid_pre"

SAE_RELEASE = "gpt2-small-res-jb"
SAE_ID      = f"blocks.{LAYER}.hook_resid_pre"

# ── Clustering hyper-parameters ───────────────────────────────────────────────

CORR_THRESHOLD = 0.5
MIN_CLUSTER_SZ = 2
N_DISC         = 60    # top discriminative features selected before clustering

# ── Prompt data ───────────────────────────────────────────────────────────────

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

# ── Model / SAE loaders ───────────────────────────────────────────────────────

def load_model() -> HookedTransformer:
    print(f"Loading {MODEL_NAME} on {DEVICE} ...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    model.eval()
    print(f"  {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    return model


def load_sae(layer: int | None = None):
    """Load the SAE for *layer* (defaults to LAYER)."""
    layer   = layer if layer is not None else LAYER
    sae_id  = f"blocks.{layer}.hook_resid_pre"
    print(f"Loading SAE  ({SAE_RELEASE}  /  {sae_id}) ...")
    sae, _cfg, _sparsity = SAE.from_pretrained(
        release=SAE_RELEASE, sae_id=sae_id, device=DEVICE,
    )
    sae.eval()
    print(f"  d_in={sae.cfg.d_in},  d_sae={sae.cfg.d_sae}")
    return sae
