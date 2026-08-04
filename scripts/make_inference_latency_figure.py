"""Regenerate the paper's RTX 5080 inference-latency figure."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SYSTEMS = (
    ("cortex_compact", "Cortex (5×8)"),
    ("cortex_full_dino", "Cortex (25×40)"),
    ("p2p_150m", "P2P-150M"),
    ("nitrogen", "NitroGen (18 actions)"),
)
VISION = "#4C78A8"
POLICY = "#F58518"


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    with (repo / "paper" / "results_data.json").open() as file:
        data = json.load(file)["inference_latency_rtx5080"]["systems"]

    labels = [label for _, label in SYSTEMS]
    vision = np.asarray([data[key]["vision_p50_ms"] for key, _ in SYSTEMS])
    policy = np.asarray([data[key]["policy_p50_ms"] for key, _ in SYSTEMS])
    total = np.asarray([data[key]["total_p50_ms"] for key, _ in SYSTEMS])
    p95 = np.asarray([data[key]["total_p95_ms"] for key, _ in SYSTEMS])
    y = np.arange(len(SYSTEMS))

    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    ax.barh(y, vision, color=VISION, height=0.62, label="Vision encoder")
    ax.barh(
        y,
        policy,
        left=vision,
        color=POLICY,
        height=0.62,
        label="Policy and action generation",
    )
    ax.scatter(
        p95,
        y,
        marker="|",
        s=170,
        linewidth=1.7,
        color="#333333",
        label="Total p95",
        zorder=3,
    )

    for index, value in enumerate(total):
        ax.text(value + 0.7, index, f"{value:.2f} ms", va="center", fontsize=9)

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Batch-one inference latency on RTX 5080 (ms)")
    ax.set_xlim(0, 58)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Patch(facecolor=VISION, label="Vision encoder"),
            Patch(facecolor=POLICY, label="Policy and action generation"),
            Line2D(
                [],
                [],
                color="#333333",
                marker="|",
                markersize=12,
                linestyle="None",
                label="Total p95",
            ),
        ],
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
    )
    fig.tight_layout()

    output = repo / "paper" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output / "inference_latency_rtx5080.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(output / "inference_latency_rtx5080.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
