#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["loguru", "numpy", "networkx"]
# ///
"""Standardize graph expressiveness benchmark data into exp_sel_data_out schema.

Loads the pre-generated data_out.json (525 graph pairs across 4 families),
combines into a single dataset, assigns fold numbers, and outputs
full_data_out.json compliant with the exp_sel_data_out.json schema.

Single unified dataset: graph_expressiveness_benchmark
- 125 programmatic pairs (cospectral, CSL, strongly regular)
- 400 BREC benchmark pairs (Basic, Regular, Extension, CFI, 4-Vertex, Distance Regular)
"""

import json
import sys
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "data.log"), rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent
INPUT_PATH = WORKSPACE / "data_out.json"
OUTPUT_PATH = WORKSPACE / "full_data_out.json"


@logger.catch
def main():
    logger.info(f"Loading data from {INPUT_PATH}")
    raw = json.loads(INPUT_PATH.read_text())
    all_examples = raw["datasets"][0]["examples"]
    logger.info(f"Total examples loaded: {len(all_examples)}")

    # Assign fold numbers and row indices
    enriched = []
    for i, ex in enumerate(all_examples):
        ex_copy = dict(ex)
        ex_copy["metadata_fold"] = i % 5
        ex_copy["metadata_row_index"] = i
        ex_copy["metadata_task_type"] = "graph_pair_classification"
        enriched.append(ex_copy)

    # Validate structure
    for ex in enriched:
        assert "input" in ex and isinstance(ex["input"], str)
        assert "output" in ex and isinstance(ex["output"], str)
        for key in ex:
            if key not in ("input", "output") and not key.startswith("metadata_"):
                raise ValueError(f"Invalid key '{key}' in example {ex.get('metadata_pair_id')}")

    # Build single-dataset output
    output = {
        "metadata": {
            "title": "Comprehensive Graph Expressiveness Benchmark Dataset",
            "description": (
                "525 non-isomorphic graph pairs across 4 families for evaluating "
                "graph distinguishing methods (Koopman Walk PE hypothesis). "
                "Families: cospectral (64), CSL (59), strongly regular (2), BREC (400). "
                "Each pair includes adjacency matrices, edge lists, eigenvalues, "
                "cospectrality labels, and WL-level annotations."
            ),
            "total_pairs": len(enriched),
            "sources": {
                "programmatic": "Generated via NetworkX (cospectral atlas enumeration, CSL circulant graphs, Shrikhande/Chang SRG pairs)",
                "brec": "Official BREC benchmark (GraphPKU/BREC, arXiv:2304.07702)",
            },
        },
        "datasets": [
            {
                "dataset": "graph_expressiveness_benchmark",
                "examples": enriched,
            }
        ],
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1e6:.1f} MB)")
    logger.info(f"  graph_expressiveness_benchmark: {len(enriched)} examples")
    logger.info("Done.")


if __name__ == "__main__":
    main()
