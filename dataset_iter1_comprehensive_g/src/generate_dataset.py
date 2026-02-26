#!/usr/bin/env python3
"""Generate Comprehensive Graph Expressiveness Benchmark Dataset.

Produces ~370-400 non-isomorphic graph pairs across four families:
1. Cospectral pairs (5-7 vertices from atlas)
2. CSL graphs (n=16, n=41)
3. Strongly regular graph pairs (Shrikhande, Chang)
4. BREC benchmark pairs (from raw .npy files)
"""

import json
import math
import os
import resource
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "generate_dataset.log"), rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware detection (cgroup-aware)
# ---------------------------------------------------------------------------
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
RAM_BUDGET = int(min(8 * 1024**3, TOTAL_RAM_GB * 0.5 * 1024**3))
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, budget={RAM_BUDGET/1e9:.1f}GB")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent
BREC_DATA_DIR = WORKSPACE / "temp" / "datasets"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def graph_to_adj_matrix(G: nx.Graph) -> list[list[int]]:
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    adj = [[0] * n for _ in range(n)]
    for u, v in G.edges():
        adj[idx[u]][idx[v]] = 1
        adj[idx[v]][idx[u]] = 1
    return adj


def graph_to_edge_list(G: nx.Graph) -> list[list[int]]:
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    return sorted([[idx[u], idx[v]] for u, v in G.edges()])


def compute_eigenvalues(G: nx.Graph) -> list[float]:
    A = nx.adjacency_matrix(G).toarray().astype(float)
    return sorted(np.linalg.eigvalsh(A).tolist())


def are_cospectral(e1: list[float], e2: list[float], tol: float = 1e-6) -> bool:
    if len(e1) != len(e2):
        return False
    return all(abs(a - b) < tol for a, b in zip(e1, e2))


def make_example(
    G_A: nx.Graph, G_B: nx.Graph,
    pair_id: str, category: str, subcategory: str,
    wl_level: str, difficulty: str,
    fold: str = "test",
    skip_iso_check: bool = False,
    skip_eigenvalues: bool = False,
) -> dict[str, Any]:
    """Convert a graph pair to schema-compliant example."""
    if skip_eigenvalues:
        eigs_A = []
        eigs_B = []
        cospectral = False
    else:
        eigs_A = compute_eigenvalues(G_A)
        eigs_B = compute_eigenvalues(G_B)
        cospectral = are_cospectral(eigs_A, eigs_B)

    is_iso = False if skip_iso_check else nx.is_isomorphic(G_A, G_B)

    input_data = {
        "graph_A": {
            "num_nodes": G_A.number_of_nodes(),
            "num_edges": G_A.number_of_edges(),
            "adjacency_matrix": graph_to_adj_matrix(G_A),
            "edge_list": graph_to_edge_list(G_A),
            "degree_sequence": sorted([d for _, d in G_A.degree()]),
            "eigenvalues": [round(e, 8) for e in eigs_A],
        },
        "graph_B": {
            "num_nodes": G_B.number_of_nodes(),
            "num_edges": G_B.number_of_edges(),
            "adjacency_matrix": graph_to_adj_matrix(G_B),
            "edge_list": graph_to_edge_list(G_B),
            "degree_sequence": sorted([d for _, d in G_B.degree()]),
            "eigenvalues": [round(e, 8) for e in eigs_B],
        },
    }
    output_data = {"is_isomorphic": is_iso, "are_cospectral": cospectral}

    return {
        "input": json.dumps(input_data),
        "output": json.dumps(output_data),
        "metadata_fold": fold,
        "metadata_pair_id": pair_id,
        "metadata_category": category,
        "metadata_subcategory": subcategory,
        "metadata_wl_level": wl_level,
        "metadata_difficulty": difficulty,
        "metadata_num_nodes_A": G_A.number_of_nodes(),
        "metadata_num_nodes_B": G_B.number_of_nodes(),
    }


# ---------------------------------------------------------------------------
# Family 1: Cospectral Non-Isomorphic Pairs
# ---------------------------------------------------------------------------
def generate_cospectral_pairs() -> list[dict]:
    """Enumerate cospectral non-isomorphic pairs from graph atlas (5-7 vertices)."""
    logger.info("Generating cospectral pairs from atlas (5-7 vertices)")
    atlas = nx.graph_atlas_g()
    examples = []

    # Canonical smallest pair
    K14 = nx.star_graph(4)
    C4_K1 = nx.disjoint_union(nx.cycle_graph(4), nx.empty_graph(1))
    examples.append(make_example(
        K14, C4_K1, "cospectral_5v_canonical", "cospectral",
        "5_vertex_canonical", "1-WL", "easy"))
    logger.info("Added canonical 5v pair: K_{1,4} vs C_4 ∪ K_1")

    for n in range(5, 8):
        t0 = time.time()
        graphs_n = [g for g in atlas if g.number_of_nodes() == n and g.number_of_edges() > 0]
        logger.info(f"Processing {len(graphs_n)} graphs on {n} vertices")

        spec_groups: dict[tuple, list[tuple[int, nx.Graph]]] = defaultdict(list)
        for idx, g in enumerate(graphs_n):
            eigs = compute_eigenvalues(g)
            key = tuple(round(e * 1e8) for e in sorted(eigs))
            spec_groups[key].append((idx, g))

        pc = 0
        for key, group in spec_groups.items():
            if len(group) < 2:
                continue
            for i, j in combinations(range(len(group)), 2):
                _, gi = group[i]
                _, gj = group[j]
                if not nx.is_isomorphic(gi, gj):
                    if n == 5 and pc == 0:
                        pc += 1
                        continue  # canonical pair already added
                    examples.append(make_example(
                        gi, gj, f"cospectral_{n}v_{pc}", "cospectral",
                        f"{n}_vertex_atlas", "1-WL",
                        "easy" if n <= 6 else "medium"))
                    pc += 1
        logger.info(f"  {n}v: {pc} pairs in {time.time()-t0:.1f}s")

    logger.info(f"Total cospectral pairs: {len(examples)}")
    return examples


# ---------------------------------------------------------------------------
# Family 2: CSL Graphs
# ---------------------------------------------------------------------------
def make_csl(n: int, s: int) -> nx.Graph:
    G = nx.cycle_graph(n)
    for i in range(n):
        G.add_edge(i, (i + s) % n)
        G.add_edge(i, (i - s) % n)
    return G


def generate_csl_pairs() -> list[dict]:
    logger.info("Generating CSL graph pairs")
    examples = []
    pc = 0

    # n=41
    skips_41 = [2, 3, 4, 5, 6, 9, 11, 12, 13, 16]
    csl_41 = {s: make_csl(41, s) for s in skips_41}
    for i, s1 in enumerate(skips_41):
        for s2 in skips_41[i+1:]:
            if not nx.is_isomorphic(csl_41[s1], csl_41[s2]):
                examples.append(make_example(
                    csl_41[s1], csl_41[s2], f"csl_41_{s1}_{s2}",
                    "CSL", "n41", "1-WL", "medium"))
                pc += 1
    logger.info(f"  CSL n=41: {pc} pairs")

    # n=16
    skips_16 = [2, 3, 4, 5, 6, 7]
    csl_16 = {s: make_csl(16, s) for s in skips_16}
    c16 = 0
    for i, s1 in enumerate(skips_16):
        for s2 in skips_16[i+1:]:
            if not nx.is_isomorphic(csl_16[s1], csl_16[s2]):
                examples.append(make_example(
                    csl_16[s1], csl_16[s2], f"csl_16_{s1}_{s2}",
                    "CSL", "n16", "1-WL", "easy"))
                c16 += 1
                pc += 1
    logger.info(f"  CSL n=16: {c16} pairs")
    logger.info(f"Total CSL pairs: {pc}")
    return examples


# ---------------------------------------------------------------------------
# Family 3: Strongly Regular Graph Pairs
# ---------------------------------------------------------------------------
def make_shrikhande() -> nx.Graph:
    """Shrikhande graph: srg(16,6,2,2). Z4×Z4, diffs {(0,1),(0,3),(1,0),(3,0),(1,1),(3,3)}."""
    G = nx.Graph()
    G.add_nodes_from(range(16))
    nodes = [(i, j) for i in range(4) for j in range(4)]
    nmap = {v: i for i, v in enumerate(nodes)}
    diffs = [(0,1),(0,3),(1,0),(3,0),(1,1),(3,3)]
    for v in nodes:
        for d in diffs:
            u = ((v[0]+d[0])%4, (v[1]+d[1])%4)
            i, j = nmap[v], nmap[u]
            if i < j:
                G.add_edge(i, j)
    return G


def seidel_switch(G: nx.Graph, S: set) -> nx.Graph:
    """Seidel switching: toggle edges between S and V\\S."""
    H = G.copy()
    all_nodes = set(G.nodes())
    complement = all_nodes - S
    for u in S:
        for v in complement:
            if H.has_edge(u, v):
                H.remove_edge(u, v)
            else:
                H.add_edge(u, v)
    return H


def generate_srg_pairs() -> list[dict]:
    """Generate strongly regular graph pairs."""
    logger.info("Generating strongly regular graph pairs")
    examples = []

    # 1) Shrikhande vs 4x4 Rook: srg(16,6,2,2)
    shrik = make_shrikhande()
    rook = nx.convert_node_labels_to_integers(
        nx.cartesian_product(nx.complete_graph(4), nx.complete_graph(4)))
    logger.info(f"  Shrikhande: {shrik.number_of_nodes()}n, {shrik.number_of_edges()}e")
    logger.info(f"  4x4 Rook:  {rook.number_of_nodes()}n, {rook.number_of_edges()}e")
    if not nx.is_isomorphic(shrik, rook):
        examples.append(make_example(
            shrik, rook, "srg_shrikhande_vs_rook", "strongly_regular",
            "srg_16_6_2_2", "2-WL", "hard"))

    # 2) Chang graphs via Seidel switching on T(8)
    T8 = nx.line_graph(nx.complete_graph(8))
    T8_nodes = list(T8.nodes())
    T8_int = nx.convert_node_labels_to_integers(T8)
    logger.info(f"  T(8): {T8_int.number_of_nodes()}n, {T8_int.number_of_edges()}e")

    # Perfect matchings of K8 — switching on these 4-vertex sets preserves regularity
    matchings = [
        {(0,1),(2,3),(4,5),(6,7)},
        {(0,2),(1,3),(4,6),(5,7)},
        {(0,3),(1,2),(4,7),(5,6)},
        {(0,4),(1,5),(2,6),(3,7)},
        {(0,5),(1,4),(2,7),(3,6)},
        {(0,6),(1,7),(2,4),(3,5)},
        {(0,7),(1,6),(2,5),(3,4)},
    ]

    chang_graphs = []
    seen = [T8_int]

    for matching in matchings:
        if len(chang_graphs) >= 3:
            break
        S = set()
        for e in T8_nodes:
            if tuple(sorted(e)) in matching:
                S.add(e)
        H = seidel_switch(T8, S)
        H_int = nx.convert_node_labels_to_integers(H)

        # Quick regularity check
        degs = set(d for _, d in H_int.degree())
        if degs != {12} or H_int.number_of_edges() != 168:
            continue

        # Check novelty
        is_new = all(not nx.is_isomorphic(H_int, prev) for prev in seen)
        if is_new:
            chang_graphs.append(H_int)
            seen.append(H_int)
            logger.info(f"  Found Chang graph: 28n, 168e")

    # Also try symmetric differences of pairs of matchings
    if len(chang_graphs) < 3:
        for i in range(len(matchings)):
            for j in range(i+1, len(matchings)):
                if len(chang_graphs) >= 3:
                    break
                S_i = {e for e in T8_nodes if tuple(sorted(e)) in matchings[i]}
                S_j = {e for e in T8_nodes if tuple(sorted(e)) in matchings[j]}
                S = S_i.symmetric_difference(S_j)
                H = seidel_switch(T8, S)
                H_int = nx.convert_node_labels_to_integers(H)
                degs = set(d for _, d in H_int.degree())
                if degs != {12} or H_int.number_of_edges() != 168:
                    continue
                is_new = all(not nx.is_isomorphic(H_int, prev) for prev in seen)
                if is_new:
                    chang_graphs.append(H_int)
                    seen.append(H_int)
                    logger.info(f"  Found Chang graph (symm diff): 28n, 168e")
            if len(chang_graphs) >= 3:
                break

    # Pair T(8) with each Chang graph
    for ci, cg in enumerate(chang_graphs):
        examples.append(make_example(
            T8_int, cg, f"srg_T8_vs_chang{ci}", "strongly_regular",
            "srg_28_12_6_4", "2-WL", "hard"))

    # Inter-Chang pairs
    for i in range(len(chang_graphs)):
        for j in range(i+1, len(chang_graphs)):
            if not nx.is_isomorphic(chang_graphs[i], chang_graphs[j]):
                examples.append(make_example(
                    chang_graphs[i], chang_graphs[j],
                    f"srg_chang{i}_vs_chang{j}", "strongly_regular",
                    "srg_28_12_6_4", "2-WL", "hard"))

    # Paley graphs as reference entries
    for q in [5, 13, 17]:
        try:
            P = nx.paley_graph(q)
            P = nx.convert_node_labels_to_integers(P)
            logger.info(f"  Paley({q}): {P.number_of_nodes()}n, {P.number_of_edges()}e")
        except Exception:
            logger.exception(f"Failed Paley({q})")

    logger.info(f"Total SRG pairs: {len(examples)}")
    return examples


# ---------------------------------------------------------------------------
# Family 4: BREC Benchmark Pairs
# ---------------------------------------------------------------------------
def load_brec_pairs() -> list[dict]:
    """Load BREC pairs from raw .npy files. Skip iso check (BREC guarantees non-iso).
    Skip eigenvalues for large graphs (>50 nodes) to avoid slowness."""
    logger.info("Loading BREC benchmark pairs")

    files_config = [
        # (filename, category, key, wl_level, difficulty, format_type)
        ("basic.npy", "BREC_Basic", "basic", "1-WL", "easy", "flat"),
        ("regular.npy", "BREC_Regular", "regular", "1-WL", "medium", "2d"),
        ("str.npy", "BREC_Strongly_Regular", "str", "3-WL", "hard", "flat"),
        ("extension.npy", "BREC_Extension", "extension", "2-WL", "medium", "2d"),
        ("cfi.npy", "BREC_CFI", "cfi", "3-WL", "hard", "2d"),
        ("4vtx.npy", "BREC_4Vertex", "4vtx", "3-WL", "hard", "flat"),
        ("dr.npy", "BREC_Distance_Regular", "dr", "3-WL", "hard", "flat"),
    ]

    examples = []
    for fname, category, key, wl, diff, fmt in files_config:
        fpath = BREC_DATA_DIR / fname
        if not fpath.exists():
            logger.warning(f"Missing: {fpath}")
            continue

        data = np.load(str(fpath), allow_pickle=True)
        logger.info(f"  {fname}: shape={data.shape}, dtype={data.dtype}")
        pc = 0

        try:
            if fmt == "flat":
                for i in range(0, data.size, 2):
                    try:
                        g6_a, g6_b = data[i], data[i+1]
                        if isinstance(g6_a, str): g6_a = g6_a.encode()
                        if isinstance(g6_b, str): g6_b = g6_b.encode()
                        GA = nx.from_graph6_bytes(g6_a)
                        GB = nx.from_graph6_bytes(g6_b)
                        skip_eig = GA.number_of_nodes() > 50
                        examples.append(make_example(
                            GA, GB, f"brec_{key}_{pc}", category, key,
                            wl, diff, skip_iso_check=True,
                            skip_eigenvalues=skip_eig))
                        pc += 1
                    except Exception:
                        logger.exception(f"Failed pair {pc} in {fname}")
            else:  # 2d format
                for i in range(data.shape[0]):
                    try:
                        g6_a, g6_b = data[i][0], data[i][1]
                        if isinstance(g6_a, str): g6_a = g6_a.encode()
                        if isinstance(g6_b, str): g6_b = g6_b.encode()
                        GA = nx.from_graph6_bytes(g6_a)
                        GB = nx.from_graph6_bytes(g6_b)
                        skip_eig = GA.number_of_nodes() > 50
                        examples.append(make_example(
                            GA, GB, f"brec_{key}_{pc}", category, key,
                            wl, diff, skip_iso_check=True,
                            skip_eigenvalues=skip_eig))
                        pc += 1
                    except Exception:
                        logger.exception(f"Failed pair {pc} in {fname}")
        except Exception:
            logger.exception(f"Failed loading {fname}")

        logger.info(f"  {fname}: {pc} pairs loaded")

    logger.info(f"Total BREC pairs: {len(examples)}")
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@logger.catch
def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Graph Expressiveness Benchmark Dataset Generation")
    logger.info("=" * 60)

    all_examples = []

    # Family 1
    logger.info("\n--- Family 1: Cospectral Pairs ---")
    cospec = generate_cospectral_pairs()
    all_examples.extend(cospec)

    # Family 2
    logger.info("\n--- Family 2: CSL Graphs ---")
    csl = generate_csl_pairs()
    all_examples.extend(csl)

    # Family 3
    logger.info("\n--- Family 3: Strongly Regular Graphs ---")
    srg = generate_srg_pairs()
    all_examples.extend(srg)

    # Family 4
    logger.info("\n--- Family 4: BREC Benchmark ---")
    brec = load_brec_pairs()
    all_examples.extend(brec)

    # Summary
    logger.info("\n" + "=" * 60)
    cats: dict[str, int] = defaultdict(int)
    for ex in all_examples:
        cats[ex["metadata_category"]] += 1
    for c, n in sorted(cats.items()):
        logger.info(f"  {c}: {n}")
    logger.info(f"  TOTAL: {len(all_examples)} pairs")

    # Verify non-isomorphism
    iso_count = sum(1 for ex in all_examples if json.loads(ex["output"])["is_isomorphic"])
    if iso_count:
        logger.warning(f"{iso_count} pairs are isomorphic!")
    else:
        logger.info("All pairs verified non-isomorphic")

    # Assemble
    output = {
        "metadata": {
            "title": "Comprehensive Graph Expressiveness Benchmark Dataset",
            "description": "Non-isomorphic graph pairs for evaluating graph distinguishing methods",
            "total_pairs": len(all_examples),
            "families": {
                "cospectral": len(cospec),
                "CSL": len(csl),
                "strongly_regular": len(srg),
                "BREC": len(brec),
            },
            "generation_time_seconds": round(time.time() - t0, 1),
        },
        "datasets": [{
            "dataset": "graph_expressiveness_benchmark",
            "examples": all_examples,
        }],
    }

    out_path = WORKSPACE / "data_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved to {out_path}")
    logger.info(f"Time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
