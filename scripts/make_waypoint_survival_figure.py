"""Regenerate the paper's E1M1 waypoint-survival figure."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SERIES = {
    "Cortex A (n=20)": "cortex_production",
    "Cortex B (n=20)": "cortex_replication",
    "P2P-150M (n=5)": "p2p_150m",
    "NitroGen (n=5)": "nitrogen",
}

COLORS = {
    "Cortex A (n=20)": "#d97706",
    "Cortex B (n=20)": "#b45309",
    "P2P-150M (n=5)": "#2563eb",
    "NitroGen (n=5)": "#0891b2",
}

STYLES = {
    "Cortex A (n=20)": "-",
    "Cortex B (n=20)": "--",
    "P2P-150M (n=5)": "-.",
    "NitroGen (n=5)": ":",
}


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    with (repo / "paper" / "results_data.json").open() as f:
        data = json.load(f)["e1m1_fresh_spawn_120s"]

    waypoints = np.arange(15)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    for label, series in SERIES.items():
        values = np.asarray(data[series]["routes"])
        survival = np.asarray([(values >= k).mean() for k in waypoints])
        ax.step(
            waypoints,
            survival,
            where="post",
            label=label,
            color=COLORS[label],
            linestyle=STYLES[label],
            linewidth=2.2,
        )
        ax.scatter(waypoints, survival, color=COLORS[label], s=14, zorder=3)

    ax.set_xlabel("Maximum E1M1 waypoint index reached")
    ax.set_ylabel("Fraction of episodes reaching index or higher")
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xticks(waypoints)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()

    out = repo / "paper" / "figures"
    fig.savefig(out / "waypoint_survival.pdf", bbox_inches="tight")
    fig.savefig(out / "waypoint_survival.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
