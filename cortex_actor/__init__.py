"""Cortex: a compact behavioral-cloning game policy over frozen DINOv3 features.

10.98M trainable parameters (39.7M executed including the frozen encoder):
four frames of DINOv3 CLS + 5x8 patch tokens through a 6-layer transformer,
absolute held-state logits, and continuous mouse regression. See the paper and
the repository README for results and the evaluation protocol.
"""

from .hub import from_hub
from .model import (
    COMPACT_PATCH_GRID,
    DINO_PATCH_GRID,
    MULTISCALE_PATCH_GRID,
    Cortex,
    cortex_from_args,
)
from .schema import (
    KEYS,
    KEY_INDEX,
    MOUSE_BUTTONS,
    MOUSE_DX_SCALE,
    MOUSE_DY_SCALE,
    N_HELD_STATE,
    N_KEYS,
    N_MOUSE_BUTTONS,
)

__all__ = [
    "COMPACT_PATCH_GRID",
    "DINO_PATCH_GRID",
    "MULTISCALE_PATCH_GRID",
    "Cortex",
    "cortex_from_args",
    "from_hub",
    "KEYS",
    "KEY_INDEX",
    "MOUSE_BUTTONS",
    "MOUSE_DX_SCALE",
    "MOUSE_DY_SCALE",
    "N_HELD_STATE",
    "N_KEYS",
    "N_MOUSE_BUTTONS",
]
