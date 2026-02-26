#!/usr/bin/env python3
"""Prepare ZINC-12k molecular graph regression benchmark dataset.

Loads the full ZINC dataset from HuggingFace downloads in temp/datasets/,
applies the canonical 12k subset indices from benchmarking-gnns repo,
and outputs in exp_sel_data_out.json schema format.

Each graph becomes one example with:
- input: JSON-stringified graph structure (edge_index, node_feat, edge_attr, num_nodes)
- output: regression target y as string
- metadata_fold: "train"/"val"/"test" split label
- metadata_task_type: "regression"
- metadata_row_index: original index in the full dataset split
- metadata_num_nodes: number of nodes in the graph
- metadata_num_edges: number of edges in the graph
"""

import gc
import json
import math
import os
import resource
import statistics
import sys
from collections import Counter
from pathlib import Path

from loguru import logger

# --- Logging setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "data.log"), rotation="30 MB", level="DEBUG")

# --- Hardware detection ---
def _detect_cpus() -> int:
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return math.ceil(q / p)
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

# --- Memory limits ---
RAM_BUDGET = int(4 * 1024**3)  # 4GB
import psutil
_avail = psutil.virtual_memory().available
assert RAM_BUDGET < _avail, f"Budget {RAM_BUDGET/1e9:.1f}GB > available {_avail/1e9:.1f}GB"
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# --- Paths ---
TEMP_DIR = SCRIPT_DIR / "temp" / "datasets"
OUTPUT_DIR = SCRIPT_DIR

SPLIT_MAP = {
    "train": [
        "full_graphs-datasets_ZINC_train_part_001.json",
        "full_graphs-datasets_ZINC_train_part_002.json",
        "full_graphs-datasets_ZINC_train_part_003.json",
    ],
    "validation": ["full_graphs-datasets_ZINC_validation.json"],
    "test": ["full_graphs-datasets_ZINC_test.json"],
}
INDEX_MAP = {
    "train": "train.index",
    "val": "val.index",
    "test": "test.index",
}
# HuggingFace split name → index file split name mapping
HF_TO_SPLIT = {
    "train": "train",
    "validation": "val",
    "test": "test",
}


def load_index_file(path: Path) -> list[int]:
    """Load comma-separated index file."""
    text = path.read_text().strip()
    indices = [int(x) for x in text.split(",")]
    logger.info(f"Loaded {len(indices)} indices from {path.name}")
    return indices


def load_json_split(filenames: list[str]) -> list[dict]:
    """Load JSON array file(s) from temp/datasets. Handles split files."""
    data = []
    for fname in filenames:
        fpath = TEMP_DIR / fname
        logger.info(f"Loading {fpath.name} ...")
        part = json.loads(fpath.read_text())
        logger.info(f"Loaded {len(part)} graphs from {fpath.name}")
        data.extend(part)
    logger.info(f"Total loaded: {len(data)} graphs from {len(filenames)} file(s)")
    return data


def convert_graph_to_example(
    graph: dict,
    split_label: str,
    original_index: int,
) -> dict:
    """Convert a HuggingFace graph record to exp_sel_data_out example format.

    HF format:
      node_feat: [[0], [1], ...]  (1-element lists)
      edge_index: [[src_nodes], [dst_nodes]]
      edge_attr: [[1], [2], ...]  (1-element lists)
      y: [float]  (1-element list)
      num_nodes: int

    Output:
      input: JSON string of {edge_index, node_feat, edge_attr, num_nodes}
      output: string of the regression target y
      metadata_*: per-example metadata
    """
    # Flatten node features from [[0], [1], ...] to [0, 1, ...]
    node_feat = [nf[0] for nf in graph["node_feat"]]

    # Flatten edge attributes from [[1], [2], ...] to [1, 2, ...]
    edge_attr = [ea[0] for ea in graph["edge_attr"]]

    # Extract scalar y
    y_val = graph["y"][0] if isinstance(graph["y"], list) else graph["y"]

    # Number of edges (each direction counted)
    num_edges = len(edge_attr)

    # Build input as JSON string
    input_obj = {
        "edge_index": graph["edge_index"],
        "node_feat": node_feat,
        "edge_attr": edge_attr,
        "num_nodes": graph["num_nodes"],
    }

    return {
        "input": json.dumps(input_obj, separators=(",", ":")),
        "output": str(y_val),
        "metadata_fold": split_label,
        "metadata_task_type": "regression",
        "metadata_row_index": original_index,
        "metadata_num_nodes": graph["num_nodes"],
        "metadata_num_edges": num_edges,
    }


def validate_examples(examples: list[dict]) -> bool:
    """Validate ZINC-12k examples against known statistics."""
    logger.info("=== Validating dataset ===")
    errors = []

    # Split counts
    split_counts = Counter(ex["metadata_fold"] for ex in examples)
    logger.info(f"Split counts: {dict(split_counts)}")
    if split_counts.get("train", 0) != 10000:
        errors.append(f"Train count {split_counts.get('train', 0)} != 10000")
    if split_counts.get("val", 0) != 1000:
        errors.append(f"Val count {split_counts.get('val', 0)} != 1000")
    if split_counts.get("test", 0) != 1000:
        errors.append(f"Test count {split_counts.get('test', 0)} != 1000")

    all_node_feats = []
    all_edge_attrs = []
    all_num_nodes = []
    all_y = []

    for i, ex in enumerate(examples):
        inp = json.loads(ex["input"])

        # Collect stats
        all_node_feats.extend(inp["node_feat"])
        all_edge_attrs.extend(inp["edge_attr"])
        all_num_nodes.append(inp["num_nodes"])
        all_y.append(float(ex["output"]))

        # Validate node features in [0, 27]
        for nf in inp["node_feat"]:
            if not (0 <= nf <= 27):
                errors.append(f"Example {i}: node_feat {nf} not in [0, 27]")

        # Validate edge attributes in [1, 3]
        for ea in inp["edge_attr"]:
            if not (1 <= ea <= 3):
                errors.append(f"Example {i}: edge_attr {ea} not in [1, 3]")

        # Validate edge_index consistency
        src = inp["edge_index"][0]
        dst = inp["edge_index"][1]
        if len(src) != len(dst):
            errors.append(f"Example {i}: edge_index lengths mismatch {len(src)} != {len(dst)}")
        if len(src) != len(inp["edge_attr"]):
            errors.append(f"Example {i}: edge count {len(src)} != edge_attr count {len(inp['edge_attr'])}")

    # Statistics
    avg_nodes = statistics.mean(all_num_nodes)
    min_nodes = min(all_num_nodes)
    max_nodes = max(all_num_nodes)
    avg_y = statistics.mean(all_y)
    std_y = statistics.stdev(all_y)

    logger.info(f"Node features unique: {sorted(set(all_node_feats))}")
    logger.info(f"Edge attr unique: {sorted(set(all_edge_attrs))}")
    logger.info(f"Num nodes: min={min_nodes}, max={max_nodes}, avg={avg_nodes:.1f}")
    logger.info(f"Target y: min={min(all_y):.3f}, max={max(all_y):.3f}, mean={avg_y:.3f}, std={std_y:.3f}")
    logger.info(f"Total examples: {len(examples)}")

    # Undirected check on first 100
    undirected_ok = 0
    undirected_fail = 0
    for ex in examples[:100]:
        inp = json.loads(ex["input"])
        edges = set(zip(inp["edge_index"][0], inp["edge_index"][1]))
        for u, v in list(edges):
            if (v, u) in edges:
                undirected_ok += 1
            else:
                undirected_fail += 1
    logger.info(f"Undirected check (100 graphs): {undirected_ok} bidir, {undirected_fail} unidir")

    if errors:
        for e in errors[:20]:
            logger.error(e)
        logger.error(f"Total errors: {len(errors)}")
        return False

    logger.info("✓ Validation PASSED")
    return True


@logger.catch
def main():
    logger.info(f"=== ZINC-12k Dataset Preparation ===")
    logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM")
    logger.info(f"RAM budget: {RAM_BUDGET/1e9:.1f}GB")

    examples = []

    # Process each split: load full data, apply subset indices, convert
    for hf_split, split_label in [("train", "train"), ("validation", "val"), ("test", "test")]:
        logger.info(f"--- Processing split: {hf_split} → {split_label} ---")

        # Load index file
        idx_file = INDEX_MAP[split_label]
        indices = load_index_file(TEMP_DIR / idx_file)

        # Load full split (may be split across multiple files)
        data_files = SPLIT_MAP[hf_split]
        full_data = load_json_split(data_files)

        # Verify indices in bounds
        max_idx = max(indices)
        assert max_idx < len(full_data), (
            f"{split_label} max index {max_idx} >= data size {len(full_data)}"
        )

        # Apply subset indices and convert
        for orig_idx in indices:
            ex = convert_graph_to_example(full_data[orig_idx], split_label, orig_idx)
            examples.append(ex)
        logger.info(f"Converted {len(indices)} {split_label} examples")

        # Free memory
        del full_data
        gc.collect()

    logger.info(f"Total examples: {len(examples)}")

    # Validate
    is_valid = validate_examples(examples)
    if not is_valid:
        logger.error("Validation FAILED — check errors above")

    # Build exp_sel_data_out schema
    output = {
        "datasets": [
            {
                "dataset": "ZINC-12k",
                "examples": examples,
            }
        ]
    }

    # Write full_data_out.json
    out_path = OUTPUT_DIR / "full_data_out.json"
    out_path.write_text(json.dumps(output, separators=(",", ":")))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"Wrote {out_path.name}: {len(examples)} examples, {size_mb:.1f} MB")

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
