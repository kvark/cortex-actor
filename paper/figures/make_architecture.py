"""Architecture diagram for the Cortex paper (figures/architecture.{pdf,png})."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FROZEN = "#dbe9f4"
FROZEN_EDGE = "#5b8db8"
TRAIN = "#fdebd3"
TRAIN_EDGE = "#c77d2e"
NEUTRAL = "#f2f2f2"
NEUTRAL_EDGE = "#888888"

fig, ax = plt.subplots(figsize=(10.5, 3.6))
ax.set_xlim(0, 10.5)
ax.set_ylim(0, 3.6)
ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=8.5, bold_first=True):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.08",
            facecolor=fc, edgecolor=ec, linewidth=1.2,
        )
    )
    lines = text.split("\n")
    if bold_first and len(lines) > 1:
        ax.text(x + w / 2, y + h / 2 + 0.02, lines[0], ha="center",
                va="bottom", fontsize=fs, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.03, "\n".join(lines[1:]),
                ha="center", va="top", fontsize=fs - 0.8)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs)


def arrow(x0, y0, x1, y1):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>", mutation_scale=11,
            linewidth=1.1, color="#444444", shrinkA=2, shrinkB=2,
        )
    )


# input frames (stacked)
for i, off in enumerate((0.18, 0.12, 0.06, 0.0)):
    ax.add_patch(
        FancyBboxPatch(
            (0.25 + off, 1.35 + off), 1.05, 0.85,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor="white", edgecolor="#666666", linewidth=0.9,
        )
    )
ax.text(0.85, 1.05, "last 4 frames\n640×400 @ 100 ms\n(300 ms span)",
        ha="center", va="top", fontsize=8)

arrow(1.6, 1.85, 2.15, 1.85)

# frozen encoder
box(2.2, 1.25, 1.75, 1.2,
    "DINOv3 ViT-S+/16\nfrozen ❄\n28.7M params, bf16",
    FROZEN, FROZEN_EDGE)

arrow(4.02, 1.85, 4.55, 1.85)

# tokens
box(4.6, 1.25, 1.7, 1.2,
    "41 tokens × 384\nper frame:\nCLS + 5×8 patch sample",
    NEUTRAL, NEUTRAL_EDGE)
ax.text(5.45, 0.98, "+ learned spatial &\ntemporal embeddings",
        ha="center", va="top", fontsize=7.3, style="italic", color="#555555")

arrow(6.37, 1.85, 6.9, 1.85)

# trunk
box(6.95, 1.25, 1.75, 1.2,
    "Transformer trunk\n6 layers, d=384, 6 heads\n164 tokens, bidirectional",
    TRAIN, TRAIN_EDGE)

# readout point
arrow(8.77, 2.15, 9.3, 2.6)
arrow(8.77, 1.55, 9.3, 1.1)
ax.text(8.82, 1.86, "last-frame\nCLS readout", ha="left", va="center",
        fontsize=7.0, color="#555555")

# heads
box(9.05, 2.55, 1.3, 0.85,
    "held-state head\n36 Bernoulli logits\nsample @ T=1 → diff\n→ key events",
    TRAIN, TRAIN_EDGE, fs=7.6)
box(9.05, 0.35, 1.3, 0.85,
    "mouse head\ntanh dx, dy\n±500 / ±250 counts",
    TRAIN, TRAIN_EDGE, fs=7.6)

# legend
ax.add_patch(FancyBboxPatch((0.25, 2.95), 0.28, 0.22,
                            boxstyle="round,pad=0.02,rounding_size=0.03",
                            facecolor=FROZEN, edgecolor=FROZEN_EDGE, lw=1))
ax.text(0.62, 3.06, "frozen (28.7M)", va="center", fontsize=8)
ax.add_patch(FancyBboxPatch((1.85, 2.95), 0.28, 0.22,
                            boxstyle="round,pad=0.02,rounding_size=0.03",
                            facecolor=TRAIN, edgecolor=TRAIN_EDGE, lw=1))
ax.text(2.22, 3.06, "trainable (10.98M)", va="center", fontsize=8)

fig.tight_layout(pad=0.3)
for ext in ("pdf", "png"):
    fig.savefig(f"architecture.{ext}", dpi=220, bbox_inches="tight")
print("wrote architecture.pdf/.png")
