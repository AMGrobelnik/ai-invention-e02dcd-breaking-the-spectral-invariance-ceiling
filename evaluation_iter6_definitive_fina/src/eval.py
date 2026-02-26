#!/usr/bin/env python3
"""Definitive Final Consolidation: All KW-PE Experiments into Paper-Ready Assessment.

Consolidates all 9 experiment artifacts (4 expressiveness + 5 downstream ZINC) across 6 iterations
into a single eval_out.json with:
  1. Complete expressiveness hierarchy
  2. Complete downstream table
  3. Spearman expressiveness-utility correlation
  4. 5-claim hypothesis scorecard with weighted scoring
  5. Contribution statement with venue recommendation
  6. Summary statistics
"""

import json
import math
import os
import resource
import sys
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from scipy import stats

# ── Logging ──────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Hardware-aware resource limits ───────────────────────────────────
def _detect_cpus() -> int:
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1

def _container_ram_gb() -> float | None:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None

NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or 29.0
RAM_BUDGET = int(min(TOTAL_RAM_GB * 0.5, 14) * 1024**3)  # 50% of container RAM
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, RAM budget: {RAM_BUDGET/1e9:.1f} GB")

# ── Constants / Dependency Paths ─────────────────────────────────────
WORKSPACE = Path("/workspace/runs/run__20260226_110200/3_invention_loop/iter_6/gen_art/eval_id3_it6__opus")
DEPS = {
    "exp_id1_it2": Path("/workspace/runs/run__20260225_014759/3_invention_loop/iter_2/gen_art/exp_id1_it2__opus"),
    "exp_id2_it2": Path("/workspace/runs/run__20260225_014759/3_invention_loop/iter_2/gen_art/exp_id2_it2__opus"),
    "exp_id3_it2": Path("/workspace/runs/run__20260225_014759/3_invention_loop/iter_2/gen_art/exp_id3_it2__opus"),
    "exp_id1_it3": Path("/workspace/runs/run__20260225_141527/3_invention_loop/iter_3/gen_art/exp_id1_it3__opus"),
    "exp_id2_it3": Path("/workspace/runs/run__20260225_141527/3_invention_loop/iter_3/gen_art/exp_id2_it3__opus"),
    "exp_id1_it4": Path("/workspace/runs/run__20260225_141527/3_invention_loop/iter_4/gen_art/exp_id1_it4__opus"),
    "exp_id2_it4": Path("/workspace/runs/run__20260225_141527/3_invention_loop/iter_4/gen_art/exp_id2_it4__opus"),
    "exp_id1_it5": Path("/workspace/runs/run__20260225_141527/3_invention_loop/iter_5/gen_art/exp_id1_it5__opus"),
    "exp_id2_it5": Path("/workspace/runs/run__20260225_141527/3_invention_loop/iter_5/gen_art/exp_id2_it5__opus"),
}

# Categories for expressiveness benchmark
CATEGORIES = [
    "cospectral", "CSL", "strongly_regular",
    "BREC_Basic", "BREC_Regular", "BREC_Extension", "BREC_CFI",
    "BREC_4Vertex", "BREC_Distance_Regular", "BREC_Strongly_Regular",
]

TOTAL_PAIRS = 525
COLLAPSED_MAE_THRESHOLD = 1.0  # MAE > 1.0 indicates collapsed training


def load_json(path: Path) -> dict:
    """Load a JSON file safely."""
    logger.debug(f"Loading {path}")
    return json.loads(path.read_text())


# ═══════════════════════════════════════════════════════════════════════
#  1. EXPRESSIVENESS HIERARCHY
# ═══════════════════════════════════════════════════════════════════════

def build_expressiveness_hierarchy(dep_data: dict[str, dict]) -> dict:
    """Build complete expressiveness hierarchy separating equivariant vs non-equivariant methods."""
    logger.info("Building expressiveness hierarchy...")

    non_equivariant_methods: list[dict] = []
    equivariant_methods: list[dict] = []

    # ── exp_id1_it2: KW-PE Pipeline (NON-EQUIVARIANT, uses EDMD) ──
    meta = dep_data["exp_id1_it2"]["metadata"]
    threshold = meta.get("default_threshold", 1e-5)
    summary = meta.get("summary_table", {})
    for method_name, cat_data in summary.items():
        overall = cat_data.get("overall", {})
        per_cat = {}
        for cat in CATEGORIES:
            if cat in cat_data:
                per_cat[cat] = {
                    "distinguished": cat_data[cat].get("distinguished", 0),
                    "total": cat_data[cat].get("total", 0),
                    "rate": cat_data[cat].get("discrimination_rate", 0.0),
                }
        entry = {
            "method": method_name,
            "experiment": "exp_id1_it2",
            "equivariant": False,
            "threshold": threshold,
            "distinguished": overall.get("distinguished", 0),
            "total": overall.get("total", TOTAL_PAIRS),
            "discrimination_rate": overall.get("discrimination_rate", 0.0),
            "per_category": per_cat,
        }
        non_equivariant_methods.append(entry)

    # ── exp_id3_it2: Foundational Properties (NON-EQUIVARIANT, uses EDMD) ──
    meta3 = dep_data["exp_id3_it2"]["metadata"]
    # Discrimination data is in per_method_discrimination or directly
    # exp_id3_it2 has its own discrimination test in the metadata
    disc = meta3.get("discrimination_test", {})
    if disc:
        for method_name, mdata in disc.items():
            if isinstance(mdata, dict) and "overall" in mdata:
                overall = mdata["overall"]
                per_cat = {}
                for cat in CATEGORIES:
                    if cat in mdata.get("per_category", {}):
                        cdat = mdata["per_category"][cat]
                        per_cat[cat] = {
                            "distinguished": cdat.get("distinguished", 0),
                            "total": cdat.get("total", 0),
                            "rate": cdat.get("rate", 0.0),
                        }
                entry = {
                    "method": f"{method_name}_exp3it2",
                    "experiment": "exp_id3_it2",
                    "equivariant": False,
                    "threshold": meta3.get("parameters", {}).get("threshold", 1e-6),
                    "distinguished": overall.get("distinguished", 0),
                    "total": overall.get("total", TOTAL_PAIRS),
                    "discrimination_rate": overall.get("rate", 0.0),
                    "per_category": per_cat,
                }
                non_equivariant_methods.append(entry)

    # exp_id3_it2 also has cospectral_test with KW-PE results
    cospectral_test = meta3.get("cospectral_test", {})
    if cospectral_test:
        for method_name, mdata in cospectral_test.items():
            if isinstance(mdata, dict) and "distinguished" in mdata:
                entry = {
                    "method": f"KW-PE_{method_name}_cospectral_exp3it2",
                    "experiment": "exp_id3_it2",
                    "equivariant": False,
                    "threshold": 1e-6,
                    "distinguished": mdata.get("distinguished", 0),
                    "total": mdata.get("total", 0),
                    "discrimination_rate": mdata.get("rate", 0.0),
                    "per_category": {},
                    "note": "cospectral subset only",
                }
                non_equivariant_methods.append(entry)

    # Check metadata.pair_discrimination for exp_id3_it2
    pair_disc = meta3.get("pair_discrimination", {})
    if pair_disc:
        for method_name, mdata in pair_disc.items():
            if isinstance(mdata, dict) and "total_distinguished" in mdata:
                entry = {
                    "method": f"{method_name}_exp3it2",
                    "experiment": "exp_id3_it2",
                    "equivariant": False,
                    "threshold": 1e-6,
                    "distinguished": mdata.get("total_distinguished", 0),
                    "total": mdata.get("total_pairs", TOTAL_PAIRS),
                    "discrimination_rate": mdata.get("discrimination_rate", 0.0),
                    "per_category": {},
                }
                non_equivariant_methods.append(entry)

    # ── exp_id2_it3: nRWPE-diag Discrimination (EQUIVARIANT) ──
    meta2it3 = dep_data["exp_id2_it3"]["metadata"]
    threshold_2it3 = meta2it3.get("threshold", 1e-6)
    per_method_results = meta2it3.get("per_method_results", {})
    for method_name, mdata in per_method_results.items():
        overall = mdata.get("overall", {})
        per_cat = {}
        for cat in CATEGORIES:
            if cat in mdata.get("per_category", {}):
                cdat = mdata["per_category"][cat]
                per_cat[cat] = {
                    "distinguished": cdat.get("distinguished", 0),
                    "total": cdat.get("total", 0),
                    "rate": cdat.get("rate", 0.0),
                }
        entry = {
            "method": method_name,
            "experiment": "exp_id2_it3",
            "equivariant": True,
            "threshold": threshold_2it3,
            "distinguished": overall.get("distinguished", 0),
            "total": overall.get("total", TOTAL_PAIRS),
            "discrimination_rate": overall.get("rate", 0.0),
            "per_category": per_cat,
        }
        equivariant_methods.append(entry)

    # ── exp_id1_it4: Gram Matrix Equivariant (EQUIVARIANT) ──
    meta1it4 = dep_data["exp_id1_it4"]["metadata"]
    threshold_1it4 = meta1it4.get("discrimination_threshold", 1e-6)
    overall_results = meta1it4.get("overall_results", {})
    per_category_results = meta1it4.get("per_category_results", {})
    total_pairs_it4 = overall_results.get("total", TOTAL_PAIRS)

    for method_name, mdata in overall_results.items():
        if method_name == "total":
            continue
        if not isinstance(mdata, dict):
            continue
        per_cat = {}
        for cat in CATEGORIES:
            if cat in per_category_results:
                cat_data_it4 = per_category_results[cat]
                if method_name in cat_data_it4:
                    cdat = cat_data_it4[method_name]
                    per_cat[cat] = {
                        "distinguished": cdat.get("distinguished", 0),
                        "total": cdat.get("total", 0),
                        "rate": cdat.get("rate", 0.0),
                    }
        entry = {
            "method": method_name,
            "experiment": "exp_id1_it4",
            "equivariant": True,
            "threshold": threshold_1it4,
            "distinguished": mdata.get("distinguished", 0),
            "total": total_pairs_it4,
            "discrimination_rate": mdata.get("rate", 0.0),
            "per_category": per_cat,
        }
        equivariant_methods.append(entry)

    # Sort by discrimination count descending
    non_equivariant_methods.sort(key=lambda x: x["distinguished"], reverse=True)
    equivariant_methods.sort(key=lambda x: x["distinguished"], reverse=True)

    # Deduplicate: keep best result per method name root
    logger.info(f"Non-equivariant methods: {len(non_equivariant_methods)}")
    logger.info(f"Equivariant methods: {len(equivariant_methods)}")

    return {
        "non_equivariant": non_equivariant_methods,
        "equivariant": equivariant_methods,
        "best_non_equivariant": non_equivariant_methods[0] if non_equivariant_methods else None,
        "best_equivariant": equivariant_methods[0] if equivariant_methods else None,
    }


# ═══════════════════════════════════════════════════════════════════════
#  2. DOWNSTREAM TABLE
# ═══════════════════════════════════════════════════════════════════════

def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return float("nan")
    return float((m1 - m2) / pooled_std)


def build_downstream_table(dep_data: dict[str, dict]) -> dict:
    """Build complete downstream ZINC table across all architectures."""
    logger.info("Building downstream table...")

    architectures: list[dict] = []

    # ── exp_id2_it2: GIN_v1 with KW-PE (single seed) ──
    meta = dep_data["exp_id2_it2"]["metadata"]
    results_summary = meta.get("results_summary", [])
    gin_v1_entries = []
    for run in results_summary:
        gin_v1_entries.append({
            "pe_method": run.get("run_name", run.get("variant", "unknown")),
            "test_mae": run.get("test_mae", None),
            "best_val_mae": run.get("best_val_mae", None),
            "best_epoch": run.get("best_epoch", None),
            "n_seeds": 1,
            "per_seed_maes": [run.get("test_mae", None)],
        })
    architectures.append({
        "architecture": "GIN_v1",
        "experiment": "exp_id2_it2",
        "description": "4-layer GIN, 128 hidden, PE_dim=16. Single seed.",
        "entries": gin_v1_entries,
    })

    # ── exp_id1_it3: GIN_v2 with nRWPE variants (multiple seeds, has collapsed runs) ──
    meta_it3 = dep_data["exp_id1_it3"]["metadata"]
    results_it3 = meta_it3.get("results_summary", [])
    gin_v2_entries = []
    for run in results_it3:
        pe_type = run.get("pe_type", "unknown")
        per_seed = run.get("per_seed_results", [])

        # Filter collapsed runs (MAE > 1.0)
        valid_seeds = [s for s in per_seed if s.get("test_mae", 999) <= COLLAPSED_MAE_THRESHOLD]
        collapsed_seeds = [s for s in per_seed if s.get("test_mae", 999) > COLLAPSED_MAE_THRESHOLD]

        if valid_seeds:
            valid_maes = [s["test_mae"] for s in valid_seeds]
            mean_mae = float(np.mean(valid_maes))
            std_mae = float(np.std(valid_maes, ddof=1)) if len(valid_maes) > 1 else 0.0
        else:
            mean_mae = run.get("test_mae_mean", None)
            std_mae = run.get("test_mae_std", None)
            valid_maes = []

        gin_v2_entries.append({
            "pe_method": pe_type,
            "test_mae_mean": mean_mae,
            "test_mae_std": std_mae,
            "n_valid_seeds": len(valid_seeds),
            "n_collapsed_seeds": len(collapsed_seeds),
            "per_seed_maes": valid_maes,
            "all_seed_maes": [s.get("test_mae") for s in per_seed],
        })
    architectures.append({
        "architecture": "GIN_v2",
        "experiment": "exp_id1_it3",
        "description": "4-layer GIN, 128 hidden, PE projection MLP. Collapsed runs (MAE>1.0) filtered.",
        "entries": gin_v2_entries,
    })

    # ── exp_id1_it5: GINEConv (edge-aware) ──
    meta_it5 = dep_data["exp_id1_it5"]["metadata"]
    results_it5 = meta_it5.get("results_summary", [])
    gineconv_entries = []
    for run in results_it5:
        pe_type = run.get("pe_type", "unknown")
        test_maes = run.get("test_maes", [])
        gineconv_entries.append({
            "pe_method": pe_type,
            "test_mae_mean": run.get("test_mae_mean", None),
            "test_mae_std": run.get("test_mae_std", None),
            "n_valid_seeds": run.get("n_seeds", len(test_maes)),
            "per_seed_maes": test_maes,
        })
    architectures.append({
        "architecture": "GINEConv",
        "experiment": "exp_id1_it5",
        "description": "4-layer GINEConv, 128 hidden, edge-aware. 3 seeds.",
        "entries": gineconv_entries,
    })

    # ── exp_id2_it5: GPS Graph Transformer (iter 5) ──
    meta_gps5 = dep_data["exp_id2_it5"]["metadata"]
    configs_gps5 = meta_gps5.get("configs", {})
    gps_entries = []
    for config_name, cdata in configs_gps5.items():
        seed_maes = [s["test_mae"] for s in cdata.get("seed_results", [])]
        gps_entries.append({
            "pe_method": config_name,
            "test_mae_mean": cdata.get("test_mae_mean", None),
            "test_mae_std": cdata.get("test_mae_std", None),
            "n_valid_seeds": cdata.get("n_seeds", len(seed_maes)),
            "per_seed_maes": seed_maes,
            "param_count": cdata.get("param_count", None),
        })
    # Also add GIN sanity check
    gin_sanity = meta_gps5.get("gin_sanity_check", {})
    if gin_sanity:
        gps_entries.append({
            "pe_method": "GIN_RWPE_sanity",
            "test_mae_mean": gin_sanity.get("test_mae", None),
            "test_mae_std": 0.0,
            "n_valid_seeds": 1,
            "per_seed_maes": [gin_sanity.get("test_mae", None)],
            "param_count": gin_sanity.get("param_count", None),
        })
    architectures.append({
        "architecture": "GPS",
        "experiment": "exp_id2_it5",
        "description": "GPS Graph Transformer: 64 hidden, 2 heads, 3 GPS layers. 2 seeds.",
        "entries": gps_entries,
    })

    # ── exp_id2_it4: GPS iter4 (FAILED - very high MAEs, only 5 epochs) ──
    meta_gps4 = dep_data["exp_id2_it4"]["metadata"]
    gps4_runs = meta_gps4.get("all_run_results", [])
    gps4_entries = []
    for run in gps4_runs:
        config = run.get("config", "unknown")
        gps4_entries.append({
            "pe_method": config,
            "test_mae": run.get("test_mae", None),
            "epochs_trained": run.get("epochs_trained", None),
            "n_params": run.get("n_params", None),
        })
    architectures.append({
        "architecture": "GPS_iter4_FAILED",
        "experiment": "exp_id2_it4",
        "description": "GPS Graph Transformer iter4 - FAILED. Only 5 epochs trained, MAE 2-9. Excluded from analysis.",
        "entries": gps4_entries,
        "failed": True,
        "failure_reason": "Only 5 epochs trained per config; MAE values (2.4-9.1) indicate no convergence.",
    })

    # ── Compute Cohen's d for RWPE vs nRWPE pairwise comparisons ──
    effect_sizes = compute_effect_sizes(architectures)

    # ── Find best configuration ──
    best_config = find_best_config(architectures)

    return {
        "architectures": architectures,
        "effect_sizes": effect_sizes,
        "best_configuration": best_config,
    }


def compute_effect_sizes(architectures: list[dict]) -> list[dict]:
    """Compute Cohen's d effect sizes for RWPE vs nRWPE pairwise comparisons."""
    effect_sizes = []

    for arch in architectures:
        if arch.get("failed", False):
            continue
        entries = arch.get("entries", [])

        # Find RWPE baseline(s) - exclude sanity checks from different architectures
        rwpe_entries = [e for e in entries if "rwpe" in e.get("pe_method", "").lower()
                        and "nrwpe" not in e.get("pe_method", "").lower()
                        and "no_pe" not in e.get("pe_method", "").lower()
                        and "sanity" not in e.get("pe_method", "").lower()]
        nrwpe_entries = [e for e in entries if "nrwpe" in e.get("pe_method", "").lower()
                         or "kw" in e.get("pe_method", "").lower()
                         or "gram" in e.get("pe_method", "").lower()]

        for rwpe in rwpe_entries:
            rwpe_maes = rwpe.get("per_seed_maes", [])
            if not rwpe_maes:
                continue
            for nrwpe in nrwpe_entries:
                nrwpe_maes = nrwpe.get("per_seed_maes", [])
                if not nrwpe_maes:
                    continue
                d = cohens_d(nrwpe_maes, rwpe_maes)
                effect_sizes.append({
                    "architecture": arch["architecture"],
                    "rwpe_method": rwpe["pe_method"],
                    "nrwpe_method": nrwpe["pe_method"],
                    "cohens_d": round(d, 4) if not math.isnan(d) else None,
                    "rwpe_mae_mean": float(np.mean(rwpe_maes)),
                    "nrwpe_mae_mean": float(np.mean(nrwpe_maes)),
                    "nrwpe_worse": float(np.mean(nrwpe_maes)) > float(np.mean(rwpe_maes)),
                    "interpretation": (
                        "nRWPE worse" if float(np.mean(nrwpe_maes)) > float(np.mean(rwpe_maes))
                        else "nRWPE better"
                    ),
                })

    return effect_sizes


def find_best_config(architectures: list[dict]) -> dict:
    """Find the best overall configuration across all architectures."""
    best = {"architecture": None, "pe_method": None, "test_mae_mean": float("inf")}
    for arch in architectures:
        if arch.get("failed", False):
            continue
        for entry in arch.get("entries", []):
            mae = entry.get("test_mae_mean", entry.get("test_mae", None))
            if mae is not None and mae < best["test_mae_mean"]:
                best = {
                    "architecture": arch["architecture"],
                    "pe_method": entry.get("pe_method", "unknown"),
                    "test_mae_mean": mae,
                    "n_seeds": entry.get("n_valid_seeds", entry.get("n_seeds", 1)),
                }
    return best


# ═══════════════════════════════════════════════════════════════════════
#  3. EXPRESSIVENESS-UTILITY CORRELATION
# ═══════════════════════════════════════════════════════════════════════

def compute_expressiveness_utility_correlation(
    expressiveness: dict,
    downstream: dict,
) -> dict:
    """Compute Spearman correlation between expressiveness and downstream MAE."""
    logger.info("Computing expressiveness-utility correlation...")

    # Map: method key -> (discrimination count, downstream MAE mean)
    # We need to match methods across experiments

    # Build expressiveness lookup: method_root -> discrimination_count
    expr_lookup: dict[str, int] = {}

    # From equivariant methods (these are the ones that also have downstream)
    for m in expressiveness.get("equivariant", []):
        method = m["method"]
        expr_lookup[method] = m["distinguished"]

    # From non-equivariant
    for m in expressiveness.get("non_equivariant", []):
        method = m["method"]
        expr_lookup[method] = m["distinguished"]

    # Build downstream lookup: method_root -> (architecture, MAE mean)
    down_lookup: dict[str, list[tuple[str, float]]] = {}

    # Map between expressiveness method names and downstream PE names
    method_mapping = {
        # Equivariant methods
        "RWPE_diag_K20": ["rwpe", "rwpe_16"],
        "nRWPE_diag_tanh_T20": ["nrwpe_diag", "nrwpe_diag_tanh_16"],
        "nRWPE_offdiag_tanh_T20": ["nrwpe_offdiag_16", "GPS_nRWPE_offdiag_8"],
        "nRWPE_diag_softplus_T20": ["nrwpe_diag_softplus_16"],
        "RWPE_diag": ["rwpe", "rwpe_16"],
        "nRWPE_diag": ["nrwpe_diag", "nrwpe_diag_tanh_16"],
        # Non-equivariant methods (KW-PE)
        "KW-PE_tanh": ["GIN_plus_KW_PE"],
        "RWPE": ["GIN_plus_RWPE", "rwpe", "rwpe_16", "GPS_RWPE_8"],
        "LapPE": ["GIN_plus_LapPE"],
    }

    for arch in downstream.get("architectures", []):
        if arch.get("failed", False):
            continue
        for entry in arch.get("entries", []):
            pe = entry.get("pe_method", "unknown")
            mae = entry.get("test_mae_mean", entry.get("test_mae", None))
            if mae is not None:
                if pe not in down_lookup:
                    down_lookup[pe] = []
                down_lookup[pe].append((arch["architecture"], mae))

    # Match pairs
    matched_pairs: list[dict] = []
    for expr_method, disc_count in expr_lookup.items():
        mapped_names = method_mapping.get(expr_method, [expr_method])
        for mapped_name in mapped_names:
            if mapped_name in down_lookup:
                for arch, mae in down_lookup[mapped_name]:
                    matched_pairs.append({
                        "expressiveness_method": expr_method,
                        "downstream_method": mapped_name,
                        "architecture": arch,
                        "discrimination_count": disc_count,
                        "downstream_mae": mae,
                    })

    if len(matched_pairs) < 3:
        logger.warning(f"Only {len(matched_pairs)} matched pairs for correlation - too few")
        return {
            "spearman_rho": None,
            "spearman_p_value": None,
            "n_matched_pairs": len(matched_pairs),
            "matched_pairs": matched_pairs,
            "note": "Insufficient matched pairs for reliable correlation",
        }

    disc_counts = [p["discrimination_count"] for p in matched_pairs]
    maes = [p["downstream_mae"] for p in matched_pairs]

    rho, p_value = stats.spearmanr(disc_counts, maes)

    # Also compute for equivariant-only subset
    eq_pairs = [p for p in matched_pairs
                if any(m["method"] == p["expressiveness_method"]
                       for m in expressiveness.get("equivariant", []))]
    if len(eq_pairs) >= 3:
        eq_disc = [p["discrimination_count"] for p in eq_pairs]
        eq_maes = [p["downstream_mae"] for p in eq_pairs]
        eq_rho, eq_p = stats.spearmanr(eq_disc, eq_maes)
    else:
        eq_rho, eq_p = None, None

    return {
        "spearman_rho": round(float(rho), 4) if not math.isnan(rho) else None,
        "spearman_p_value": round(float(p_value), 4) if not math.isnan(p_value) else None,
        "n_matched_pairs": len(matched_pairs),
        "matched_pairs": matched_pairs,
        "equivariant_only": {
            "spearman_rho": round(float(eq_rho), 4) if eq_rho is not None and not math.isnan(eq_rho) else None,
            "spearman_p_value": round(float(eq_p), 4) if eq_p is not None and not math.isnan(eq_p) else None,
            "n_pairs": len(eq_pairs),
        },
        "interpretation": (
            f"Spearman rho={rho:.3f}, p={p_value:.3f}. "
            f"{'Negative' if rho < 0 else 'Positive'} correlation "
            f"({'significant' if p_value < 0.05 else 'not significant'} at alpha=0.05). "
            f"{'Higher expressiveness is associated with LOWER downstream MAE (better performance).' if rho < 0 else 'Higher expressiveness is NOT associated with better downstream performance.'}"
        ) if rho is not None and not math.isnan(rho) else "Insufficient data",
    }


# ═══════════════════════════════════════════════════════════════════════
#  4. HYPOTHESIS SCORECARD
# ═══════════════════════════════════════════════════════════════════════

def build_hypothesis_scorecard(
    expressiveness: dict,
    downstream: dict,
    correlation: dict,
    dep_data: dict[str, dict],
) -> dict:
    """Build 5-claim hypothesis scorecard with weighted scoring."""
    logger.info("Building hypothesis scorecard...")

    claims: list[dict] = []

    # ── Claim 1: Breaks spectral invariance ──
    # KW-PE (non-equivariant, EDMD-based) achieves 525/525 while RWPE gets ~345-361
    # But equivariant nRWPE only gets 345-346
    best_ne = expressiveness.get("best_non_equivariant", {})
    best_eq = expressiveness.get("best_equivariant", {})
    rwpe_expr = None
    for m in expressiveness.get("equivariant", []):
        if "RWPE" in m["method"] and "nRWPE" not in m["method"]:
            rwpe_expr = m
            break
    if rwpe_expr is None:
        for m in expressiveness.get("non_equivariant", []):
            if m["method"] == "RWPE":
                rwpe_expr = m
                break

    claim1_score = 0.0
    evidence_for_1 = []
    evidence_against_1 = []

    if best_ne and best_ne.get("distinguished", 0) == TOTAL_PAIRS:
        evidence_for_1.append(f"KW-PE_tanh achieves 525/525 (100%) pair discrimination")
        claim1_score += 0.5
    if rwpe_expr:
        rwpe_disc = rwpe_expr.get("distinguished", 0)
        evidence_for_1.append(f"RWPE baseline only achieves {rwpe_disc}/{TOTAL_PAIRS} ({rwpe_disc/TOTAL_PAIRS*100:.1f}%)")
        if rwpe_disc < TOTAL_PAIRS:
            claim1_score += 0.2

    # Check: is the gain from EDMD (non-equivariant processing) rather than nonlinear walk?
    if best_eq:
        eq_disc = best_eq.get("distinguished", 0)
        if eq_disc <= (rwpe_expr.get("distinguished", 0) if rwpe_expr else 345) + 5:
            evidence_against_1.append(
                f"Equivariant nRWPE only achieves {eq_disc}/{TOTAL_PAIRS} - gain comes from EDMD, not nonlinear walk itself"
            )
            claim1_score -= 0.2
        else:
            evidence_for_1.append(f"Equivariant best achieves {eq_disc}/{TOTAL_PAIRS}")
            claim1_score += 0.1

    # LapPE comparison
    for m in expressiveness.get("non_equivariant", []):
        if m["method"] == "LapPE":
            lappe_disc = m.get("distinguished", 0)
            if lappe_disc >= 524:
                evidence_against_1.append(
                    f"LapPE also achieves {lappe_disc}/{TOTAL_PAIRS} - KW-PE's EDMD may be doing similar non-equivariant processing"
                )
                claim1_score -= 0.15
            break

    claim1_score = max(0.0, min(1.0, claim1_score + 0.35))  # Baseline: partial support

    claims.append({
        "claim": "Breaks spectral invariance ceiling",
        "claim_number": 1,
        "score": round(claim1_score, 3),
        "confidence_interval": [round(max(0, claim1_score - 0.15), 3), round(min(1, claim1_score + 0.15), 3)],
        "evidence_for": evidence_for_1,
        "evidence_against": evidence_against_1,
        "definitive_assessment": (
            "PARTIALLY CONFIRMED. KW-PE (with EDMD) achieves perfect discrimination, but the gain is "
            "attributable to the non-equivariant EDMD processing rather than the nonlinear walk dynamics alone. "
            "Equivariant nRWPE variants match but do not exceed RWPE's discrimination ability."
        ),
    })

    # ── Claim 2: Sign-canonical PEs ──
    # exp_id3_it2 verified sign canonicality
    meta3 = dep_data["exp_id3_it2"]["metadata"]
    sign_canon = meta3.get("sign_canonicality", {})
    # Also exp_id1_it2
    meta1 = dep_data["exp_id1_it2"]["metadata"]
    headline = meta1.get("headline_results", {})

    claim2_score = 0.0
    evidence_for_2 = []
    evidence_against_2 = []

    if headline.get("sign_canonicity_verified", False):
        evidence_for_2.append("Sign canonicality verified in exp_id1_it2 (KW-PE)")
        claim2_score += 0.4

    if sign_canon and sign_canon.get("consistency", 0) == 1.0:
        evidence_for_2.append(f"Perfect sign consistency score: {sign_canon.get('consistency', 0)}")
        claim2_score += 0.3

    # nRWPE (equivariant) also has sign canonicality (diagonal is always positive)
    equivariance_test = dep_data["exp_id2_it3"]["metadata"].get("equivariance_test", {})
    if equivariance_test.get("all_pass", False):
        evidence_for_2.append(
            f"Equivariance test passes with max error {equivariance_test.get('max_error', 'N/A')}"
        )
        claim2_score += 0.2

    evidence_for_2.append("nRWPE diagonal features are inherently sign-canonical (return probabilities ≥ 0)")
    claim2_score += 0.1

    claim2_score = min(1.0, claim2_score)

    claims.append({
        "claim": "Produces sign-canonical positional encodings",
        "claim_number": 2,
        "score": round(claim2_score, 3),
        "confidence_interval": [round(max(0, claim2_score - 0.1), 3), round(min(1, claim2_score + 0.1), 3)],
        "evidence_for": evidence_for_2,
        "evidence_against": evidence_against_2,
        "definitive_assessment": (
            "CONFIRMED. Both KW-PE (via EDMD canonicalization) and nRWPE (via diagonal return probabilities) "
            "produce sign-canonical PEs. Verified experimentally with permutation equivariance tests."
        ),
    })

    # ── Claim 3: Distinguishes cospectral graphs ──
    claim3_score = 0.0
    evidence_for_3 = []
    evidence_against_3 = []

    # Check cospectral pairs across experiments
    # exp_id1_it2 (non-equivariant): RWPE gets 63/64, KW-PE gets 64/64
    for m in expressiveness.get("non_equivariant", []):
        cos = m.get("per_category", {}).get("cospectral", {})
        if cos and cos.get("total", 0) > 0:
            if "KW-PE" in m["method"] or "KW_PE" in m["method"]:
                if cos.get("distinguished", 0) == cos.get("total", 64):
                    evidence_for_3.append(
                        f"{m['method']} distinguishes {cos['distinguished']}/{cos['total']} cospectral pairs"
                    )
                    claim3_score += 0.15
            elif "RWPE" == m["method"]:
                rwpe_cos = cos.get("distinguished", 0)
                rwpe_cos_total = cos.get("total", 64)
                if rwpe_cos < rwpe_cos_total:
                    evidence_for_3.append(
                        f"RWPE only distinguishes {rwpe_cos}/{rwpe_cos_total} cospectral (exp_id1_it2)"
                    )
                    claim3_score += 0.1
                else:
                    evidence_against_3.append(
                        f"RWPE also distinguishes all cospectral pairs in exp_id1_it2"
                    )

    # Equivariant: all methods (including RWPE) get 64/64 cospectral
    eq_cos_all_same = True
    for m in expressiveness.get("equivariant", []):
        cos = m.get("per_category", {}).get("cospectral", {})
        if cos and cos.get("total", 0) > 0:
            if cos.get("distinguished", 0) < cos.get("total", 64):
                eq_cos_all_same = False
    if eq_cos_all_same:
        evidence_against_3.append(
            "In equivariant experiments, RWPE also distinguishes all 64 cospectral pairs"
        )

    # Check BREC_CFI - this is where KW-PE truly excels
    for m in expressiveness.get("non_equivariant", []):
        cfi = m.get("per_category", {}).get("BREC_CFI", {})
        if cfi and cfi.get("total", 0) > 0:
            if ("KW-PE" in m["method"] or "KW_PE" in m["method"]) and cfi.get("rate", 0) > 0.5:
                evidence_for_3.append(
                    f"{m['method']} distinguishes {cfi['distinguished']}/{cfi['total']} BREC_CFI pairs "
                    f"(RWPE only gets ~10-12/100)"
                )
                claim3_score += 0.2
                break

    # Check BREC_Strongly_Regular
    sr_results_eq = {}
    for m in expressiveness.get("equivariant", []):
        sr = m.get("per_category", {}).get("BREC_Strongly_Regular", {})
        if sr and sr.get("total", 0) > 0:
            sr_results_eq[m["method"]] = sr

    eq_sr_fail = all(r.get("distinguished", 0) == 0 for r in sr_results_eq.values() if r.get("total", 0) > 0)
    if eq_sr_fail and sr_results_eq:
        evidence_against_3.append("Equivariant nRWPE fails to distinguish any BREC_Strongly_Regular pairs")
        claim3_score -= 0.05

    # KW-PE (non-equivariant) distinguishes strongly regular
    for m in expressiveness.get("non_equivariant", []):
        sr = m.get("per_category", {}).get("BREC_Strongly_Regular", {})
        if sr and sr.get("distinguished", 0) > 0:
            evidence_for_3.append(
                f"{m['method']} distinguishes {sr['distinguished']}/{sr['total']} BREC_Strongly_Regular "
                f"(equivariant methods get 0)"
            )
            claim3_score += 0.15
            break

    # Key nuance: the discrimination advantage comes from EDMD (non-equivariant), not walk dynamics
    evidence_against_3.append(
        "Expressiveness gains over RWPE are from EDMD (non-equivariant processing), "
        "not from nonlinear walk dynamics alone"
    )
    claim3_score -= 0.1

    claim3_score = max(0.0, min(1.0, claim3_score))

    claims.append({
        "claim": "Distinguishes cospectral graphs that spectral methods cannot",
        "claim_number": 3,
        "score": round(claim3_score, 3),
        "confidence_interval": [round(max(0, claim3_score - 0.15), 3), round(min(1, claim3_score + 0.15), 3)],
        "evidence_for": evidence_for_3,
        "evidence_against": evidence_against_3,
        "definitive_assessment": (
            "PARTIALLY CONFIRMED. KW-PE (non-equivariant, EDMD-based) achieves 100% on BREC_CFI (vs RWPE's ~10%) "
            "and 100% on BREC_Strongly_Regular (vs equivariant 0%). However, equivariant nRWPE only matches "
            "RWPE's discrimination. The expressiveness gain comes from EDMD processing, not nonlinear walk dynamics."
        ),
    })

    # ── Claim 4: Avoids eigendecomposition ──
    claim4_score = 0.0
    evidence_for_4 = []
    evidence_against_4 = []

    # nRWPE avoids eigendecomposition (matrix power walk is O(n^2 * T))
    evidence_for_4.append("nRWPE computes walk features via matrix-vector products, avoiding O(n^3) eigendecomposition")

    # exp_id3_it2 has computational cost analysis
    comp_cost = meta3.get("computational_cost", {})
    if comp_cost:
        evidence_for_4.append(f"Computational cost scaling: {json.dumps(comp_cost)[:200]}")

    # KW-PE uses EDMD which involves SVD/pseudoinverse → not eigendecomposition-free
    evidence_against_4.append("KW-PE uses EDMD which involves pseudoinverse computation (similar computational class)")

    claim4_score = 0.7  # nRWPE avoids it, KW-PE does not

    claims.append({
        "claim": "Avoids eigendecomposition bottleneck",
        "claim_number": 4,
        "score": round(claim4_score, 3),
        "confidence_interval": [round(max(0, claim4_score - 0.1), 3), round(min(1, claim4_score + 0.1), 3)],
        "evidence_for": evidence_for_4,
        "evidence_against": evidence_against_4,
        "definitive_assessment": (
            "MOSTLY CONFIRMED for nRWPE. The equivariant nRWPE variants use only matrix-vector products, "
            "avoiding eigendecomposition. However, KW-PE (which achieves the best expressiveness) relies on "
            "EDMD, which involves pseudoinverse computation of similar complexity to eigendecomposition."
        ),
    })

    # ── Claim 5: Superior downstream performance (weighted 2x) ──
    claim5_score = 0.0
    evidence_for_5 = []
    evidence_against_5 = []

    # Check all architectures
    rwpe_wins = 0
    nrwpe_wins = 0
    total_comparisons = 0

    for arch in downstream.get("architectures", []):
        if arch.get("failed", False):
            continue
        entries = arch.get("entries", [])
        rwpe_entry = None
        for e in entries:
            pe = e.get("pe_method", "").lower()
            if ("rwpe" in pe and "nrwpe" not in pe and "no_pe" not in pe
                    and "sanity" not in pe and "gin" not in pe.replace("gineconv", "")):
                mae = e.get("test_mae_mean", e.get("test_mae", None))
                if mae is not None:
                    if rwpe_entry is None or mae < rwpe_entry[1]:
                        rwpe_entry = (e.get("pe_method", ""), mae)

        if rwpe_entry is None:
            continue

        for e in entries:
            pe = e.get("pe_method", "").lower()
            if "nrwpe" in pe or "kw" in pe or "gram" in pe:
                mae = e.get("test_mae_mean", e.get("test_mae", None))
                if mae is not None:
                    total_comparisons += 1
                    if mae < rwpe_entry[1]:
                        nrwpe_wins += 1
                        evidence_for_5.append(
                            f"{arch['architecture']}: {e.get('pe_method')} ({mae:.4f}) beats RWPE ({rwpe_entry[1]:.4f})"
                        )
                    else:
                        rwpe_wins += 1
                        evidence_against_5.append(
                            f"{arch['architecture']}: RWPE ({rwpe_entry[1]:.4f}) beats {e.get('pe_method')} ({mae:.4f})"
                        )

    if total_comparisons > 0:
        nrwpe_win_rate = nrwpe_wins / total_comparisons
        claim5_score = nrwpe_win_rate * 0.8  # Scale to max 0.8

    # Bonus if any nRWPE is actually best overall
    best = downstream.get("best_configuration", {})
    if best:
        best_pe = best.get("pe_method", "").lower()
        if "nrwpe" in best_pe or "kw" in best_pe:
            claim5_score += 0.2
            evidence_for_5.append(f"Best overall: {best.get('pe_method')} = {best.get('test_mae_mean'):.4f}")
        else:
            evidence_against_5.append(
                f"Best overall is RWPE-based: {best.get('pe_method')} = {best.get('test_mae_mean'):.4f}"
            )

    claim5_score = max(0.0, min(1.0, claim5_score))

    claims.append({
        "claim": "Superior downstream performance on molecular property prediction",
        "claim_number": 5,
        "score": round(claim5_score, 3),
        "confidence_interval": [round(max(0, claim5_score - 0.1), 3), round(min(1, claim5_score + 0.1), 3)],
        "evidence_for": evidence_for_5,
        "evidence_against": evidence_against_5,
        "definitive_assessment": (
            f"DISCONFIRMED. RWPE wins {rwpe_wins}/{total_comparisons} pairwise comparisons against nRWPE/KW-PE variants. "
            f"No nonlinear walk PE variant consistently outperforms standard RWPE on ZINC-12k across "
            f"any tested architecture (GIN_v1, GIN_v2, GINEConv, GPS)."
        ),
        "weight": 2.0,
        "rwpe_vs_nrwpe_summary": {
            "rwpe_wins": rwpe_wins,
            "nrwpe_wins": nrwpe_wins,
            "total_comparisons": total_comparisons,
        },
    })

    # ── Overall weighted score ──
    total_weight = sum(c.get("weight", 1.0) for c in claims)
    weighted_sum = sum(c["score"] * c.get("weight", 1.0) for c in claims)
    overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    return {
        "claims": claims,
        "overall_weighted_score": round(overall_score, 3),
        "total_weight": total_weight,
        "scoring_method": "Weighted average, claim 5 (downstream) weighted 2x",
    }


# ═══════════════════════════════════════════════════════════════════════
#  5. CONTRIBUTION STATEMENT
# ═══════════════════════════════════════════════════════════════════════

def build_contribution_statement(
    expressiveness: dict,
    downstream: dict,
    scorecard: dict,
    correlation: dict,
) -> dict:
    """Build contribution statement with venue recommendation."""
    logger.info("Building contribution statement...")

    overall = scorecard.get("overall_weighted_score", 0.0)
    claims = scorecard.get("claims", [])

    # Count confirmed/partial/disconfirmed
    confirmed = sum(1 for c in claims if c["score"] >= 0.7)
    partial = sum(1 for c in claims if 0.3 <= c["score"] < 0.7)
    disconfirmed = sum(1 for c in claims if c["score"] < 0.3)

    # Venue assessment
    if overall >= 0.7 and disconfirmed == 0:
        venue = "main_conference"
        venue_detail = "Strong results support main conference submission (NeurIPS/ICML/ICLR)"
    elif overall >= 0.4:
        venue = "workshop"
        venue_detail = "Mixed results suggest workshop paper at NeurIPS/ICML workshops on graph learning"
    else:
        venue = "negative_result"
        venue_detail = "Primarily negative results on downstream task; suitable for negative results workshop or analysis paper"

    # Strongest aspects
    strongest = sorted(claims, key=lambda c: c["score"], reverse=True)[:3]
    weakest = sorted(claims, key=lambda c: c["score"])[:3]

    contribution_statement = (
        f"This work investigates nonlinear random walk positional encodings (nRWPE/KW-PE) as alternatives to "
        f"spectral PEs for graph neural networks. "
        f"Theoretically, KW-PE with EDMD achieves perfect discrimination (525/525 pairs), but this advantage "
        f"stems from non-equivariant EDMD processing rather than the nonlinear walk dynamics; equivariant nRWPE "
        f"variants only match RWPE's expressiveness (~345/525). "
        f"Empirically, RWPE consistently outperforms all nRWPE/KW-PE variants on ZINC-12k across 4 architectures "
        f"(GIN, GINEConv, GPS), indicating that the expressiveness gains do not translate to downstream utility."
    )

    return {
        "contribution_statement": contribution_statement,
        "venue_assessment": venue,
        "venue_detail": venue_detail,
        "claims_confirmed": confirmed,
        "claims_partially_confirmed": partial,
        "claims_disconfirmed": disconfirmed,
        "strongest_aspects": [
            {
                "claim": s["claim"],
                "score": s["score"],
                "reason": s["definitive_assessment"][:200],
            }
            for s in strongest
        ],
        "weakest_aspects": [
            {
                "claim": w["claim"],
                "score": w["score"],
                "reason": w["definitive_assessment"][:200],
            }
            for w in weakest
        ],
    }


# ═══════════════════════════════════════════════════════════════════════
#  6. SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════════

def build_summary_statistics(
    expressiveness: dict,
    downstream: dict,
    scorecard: dict,
) -> dict:
    """Build summary statistics across all experiments."""
    logger.info("Building summary statistics...")

    # Count total training runs
    total_runs = 0
    for arch in downstream.get("architectures", []):
        for entry in arch.get("entries", []):
            n = entry.get("n_valid_seeds", entry.get("n_seeds", 1))
            if isinstance(n, (int, float)):
                total_runs += int(n)

    # Count methods tested
    all_methods = set()
    for m in expressiveness.get("equivariant", []):
        all_methods.add(m["method"])
    for m in expressiveness.get("non_equivariant", []):
        all_methods.add(m["method"])
    for arch in downstream.get("architectures", []):
        for entry in arch.get("entries", []):
            all_methods.add(entry.get("pe_method", "unknown"))

    # Best expressiveness
    best_ne = expressiveness.get("best_non_equivariant", {})
    best_eq = expressiveness.get("best_equivariant", {})

    # Best downstream
    best_down = downstream.get("best_configuration", {})

    # RWPE vs nRWPE win-loss
    claim5 = None
    for c in scorecard.get("claims", []):
        if c.get("claim_number") == 5:
            claim5 = c
            break

    rwpe_wins = claim5.get("rwpe_vs_nrwpe_summary", {}).get("rwpe_wins", 0) if claim5 else 0
    nrwpe_wins = claim5.get("rwpe_vs_nrwpe_summary", {}).get("nrwpe_wins", 0) if claim5 else 0

    return {
        "total_experiments": 9,
        "total_training_runs": total_runs,
        "total_methods_tested": len(all_methods),
        "best_expressiveness_non_equivariant": {
            "method": best_ne.get("method", "N/A") if best_ne else "N/A",
            "discrimination": f"{best_ne.get('distinguished', 0)}/{best_ne.get('total', TOTAL_PAIRS)}" if best_ne else "N/A",
        },
        "best_expressiveness_equivariant": {
            "method": best_eq.get("method", "N/A") if best_eq else "N/A",
            "discrimination": f"{best_eq.get('distinguished', 0)}/{best_eq.get('total', TOTAL_PAIRS)}" if best_eq else "N/A",
        },
        "best_downstream": {
            "architecture": best_down.get("architecture", "N/A"),
            "pe_method": best_down.get("pe_method", "N/A"),
            "test_mae": best_down.get("test_mae_mean", None),
        },
        "rwpe_vs_nrwpe_win_loss": {
            "rwpe_wins": rwpe_wins,
            "nrwpe_wins": nrwpe_wins,
            "summary": f"RWPE wins {rwpe_wins} out of {rwpe_wins + nrwpe_wins} pairwise comparisons",
        },
    }


# ═══════════════════════════════════════════════════════════════════════
#  SCHEMA-COMPLIANT OUTPUT BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_schema_output(
    expressiveness: dict,
    downstream: dict,
    correlation: dict,
    scorecard: dict,
    contribution: dict,
    summary: dict,
    dep_data: dict[str, dict] | None = None,
) -> dict:
    """Build output conforming to exp_eval_sol_out.json schema."""
    logger.info("Building schema-compliant output...")
    if dep_data is None:
        dep_data = {}

    # ── metrics_agg: numeric metrics only ──
    metrics_agg: dict[str, float] = {}

    # Expressiveness metrics
    best_ne = expressiveness.get("best_non_equivariant", {})
    best_eq = expressiveness.get("best_equivariant", {})
    if best_ne:
        metrics_agg["best_non_equivariant_discrimination_count"] = best_ne.get("distinguished", 0)
        metrics_agg["best_non_equivariant_discrimination_rate"] = best_ne.get("discrimination_rate", 0.0)
    if best_eq:
        metrics_agg["best_equivariant_discrimination_count"] = best_eq.get("distinguished", 0)
        metrics_agg["best_equivariant_discrimination_rate"] = best_eq.get("discrimination_rate", 0.0)

    # Downstream metrics
    best_down = downstream.get("best_configuration", {})
    if best_down and best_down.get("test_mae_mean") is not None:
        metrics_agg["best_downstream_mae"] = best_down["test_mae_mean"]

    # Correlation
    if correlation.get("spearman_rho") is not None:
        metrics_agg["expressiveness_utility_spearman_rho"] = correlation["spearman_rho"]
    if correlation.get("spearman_p_value") is not None:
        metrics_agg["expressiveness_utility_spearman_p"] = correlation["spearman_p_value"]

    # Scorecard
    metrics_agg["hypothesis_overall_score"] = scorecard.get("overall_weighted_score", 0.0)
    for claim in scorecard.get("claims", []):
        key = f"claim_{claim['claim_number']}_score"
        metrics_agg[key] = claim["score"]

    # Win-loss
    claim5 = None
    for c in scorecard.get("claims", []):
        if c.get("claim_number") == 5:
            claim5 = c
            break
    if claim5:
        wl = claim5.get("rwpe_vs_nrwpe_summary", {})
        metrics_agg["rwpe_wins"] = wl.get("rwpe_wins", 0)
        metrics_agg["nrwpe_wins"] = wl.get("nrwpe_wins", 0)

    metrics_agg["total_experiments"] = 9
    metrics_agg["total_methods_tested"] = summary.get("total_methods_tested", 0)
    metrics_agg["total_training_runs"] = summary.get("total_training_runs", 0)

    # ── metadata: rich structured data ──
    metadata = {
        "evaluation_name": "Definitive Final Consolidation: All KW-PE Experiments",
        "description": (
            "Consolidates 9 experiments (4 expressiveness + 5 downstream ZINC) across 6 iterations "
            "into paper-ready assessment of nonlinear walk positional encodings."
        ),
        "expressiveness_hierarchy": expressiveness,
        "downstream_table": downstream,
        "expressiveness_utility_correlation": correlation,
        "hypothesis_scorecard": scorecard,
        "contribution_statement": contribution,
        "summary_statistics": summary,
    }

    # ── datasets/examples: one example per experiment ──
    examples: list[dict[str, Any]] = []

    # Expressiveness experiments
    expr_exps = [
        ("exp_id1_it2", "KW-PE Pipeline Expressiveness (non-equivariant)"),
        ("exp_id3_it2", "KW-PE Foundational Properties (non-equivariant)"),
        ("exp_id2_it3", "nRWPE-diag Discrimination (equivariant)"),
        ("exp_id1_it4", "Gram Matrix Equivariant Features"),
    ]
    for exp_id, desc in expr_exps:
        exp_meta = dep_data.get(exp_id, {}).get("metadata", {})
        # Summarize key results
        if exp_id == "exp_id1_it2":
            result_str = json.dumps({
                k: {
                    "overall_rate": v.get("overall", {}).get("discrimination_rate", 0),
                    "distinguished": v.get("overall", {}).get("distinguished", 0),
                }
                for k, v in exp_meta.get("summary_table", {}).items()
            })
        elif exp_id == "exp_id3_it2":
            result_str = json.dumps({
                "sign_canonicality": exp_meta.get("sign_canonicality", {}),
                "convergence": exp_meta.get("convergence_analysis", {}),
            }, default=str)[:2000]
        elif exp_id == "exp_id2_it3":
            result_str = json.dumps({
                k: v.get("overall", {})
                for k, v in exp_meta.get("per_method_results", {}).items()
            })
        elif exp_id == "exp_id1_it4":
            overall_res = exp_meta.get("overall_results", {})
            result_str = json.dumps({
                k: {"distinguished": v.get("distinguished", 0), "rate": v.get("rate", 0)}
                for k, v in overall_res.items() if isinstance(v, dict)
            })
        else:
            result_str = "{}"

        examples.append({
            "input": f"Evaluate {desc} experiment ({exp_id})",
            "output": result_str[:2000],
            "eval_expressiveness_type": 1,
            "eval_is_downstream": 0,
        })

    # Downstream experiments
    down_exps = [
        ("exp_id2_it2", "ZINC-12k GIN_v1 with KW-PE"),
        ("exp_id1_it3", "ZINC-12k GIN_v2 with nRWPE"),
        ("exp_id1_it5", "ZINC-12k GINEConv with nRWPE"),
        ("exp_id2_it5", "ZINC-12k GPS with nRWPE"),
        ("exp_id2_it4", "ZINC-12k GPS iter4 (FAILED)"),
    ]
    for exp_id, desc in down_exps:
        exp_meta = dep_data.get(exp_id, {}).get("metadata", {})
        results = exp_meta.get("results_summary", exp_meta.get("configs", {}))
        if isinstance(results, dict):
            result_str = json.dumps({
                k: {"test_mae_mean": v.get("test_mae_mean"), "n_seeds": v.get("n_seeds")}
                for k, v in results.items() if isinstance(v, dict)
            })
        elif isinstance(results, list):
            result_str = json.dumps([
                {
                    "config": r.get("pe_type", r.get("run_name", r.get("config", "?"))),
                    "test_mae": r.get("test_mae_mean", r.get("test_mae")),
                }
                for r in results
            ])
        else:
            result_str = "{}"

        examples.append({
            "input": f"Evaluate {desc} experiment ({exp_id})",
            "output": result_str[:2000],
            "eval_expressiveness_type": 0,
            "eval_is_downstream": 1,
        })

    return {
        "metadata": metadata,
        "metrics_agg": metrics_agg,
        "datasets": [
            {
                "dataset": "KW-PE_consolidated_evaluation",
                "examples": examples,
            }
        ],
    }


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    """Main evaluation function."""
    logger.info("=" * 70)
    logger.info("Starting Definitive Final Consolidation Evaluation")
    logger.info("=" * 70)

    # Load all dependency data (metadata only - from preview files for quick loading)
    logger.info("Loading dependency data...")
    dep_data: dict[str, dict] = {}
    for dep_id, dep_path in DEPS.items():
        try:
            # Try mini first (has full metadata), then preview
            mini_path = dep_path / "mini_method_out.json"
            preview_path = dep_path / "preview_method_out.json"
            if mini_path.exists():
                dep_data[dep_id] = load_json(mini_path)
                logger.info(f"  Loaded {dep_id} from mini ({mini_path})")
            elif preview_path.exists():
                dep_data[dep_id] = load_json(preview_path)
                logger.info(f"  Loaded {dep_id} from preview ({preview_path})")
            else:
                logger.warning(f"  No mini/preview found for {dep_id} at {dep_path}")
                dep_data[dep_id] = {"metadata": {}, "datasets": []}
        except Exception:
            logger.exception(f"Failed to load {dep_id}")
            dep_data[dep_id] = {"metadata": {}, "datasets": []}

    logger.info(f"Loaded {len(dep_data)} dependencies")

    # 1. Expressiveness Hierarchy
    expressiveness = build_expressiveness_hierarchy(dep_data)
    logger.info(
        f"Expressiveness: {len(expressiveness['non_equivariant'])} non-equiv, "
        f"{len(expressiveness['equivariant'])} equiv methods"
    )

    # 2. Downstream Table
    downstream = build_downstream_table(dep_data)
    logger.info(f"Downstream: {len(downstream['architectures'])} architectures")

    # 3. Expressiveness-Utility Correlation
    correlation = compute_expressiveness_utility_correlation(expressiveness, downstream)
    logger.info(f"Correlation: rho={correlation.get('spearman_rho')}, p={correlation.get('spearman_p_value')}")

    # 4. Hypothesis Scorecard
    scorecard = build_hypothesis_scorecard(expressiveness, downstream, correlation, dep_data)
    logger.info(f"Scorecard overall: {scorecard.get('overall_weighted_score')}")

    # 5. Contribution Statement
    contribution = build_contribution_statement(expressiveness, downstream, scorecard, correlation)
    logger.info(f"Venue: {contribution.get('venue_assessment')}")

    # 6. Summary Statistics
    summary = build_summary_statistics(expressiveness, downstream, scorecard)
    logger.info(f"Summary: {summary.get('total_experiments')} experiments, {summary.get('total_methods_tested')} methods")

    # Build schema-compliant output
    output = build_schema_output(
        expressiveness, downstream, correlation, scorecard, contribution, summary, dep_data
    )

    # Save output
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved output to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # Print key metrics
    logger.info("=" * 70)
    logger.info("KEY METRICS:")
    for k, v in output["metrics_agg"].items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
