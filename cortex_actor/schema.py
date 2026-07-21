"""Action-channel contract shared by training, deployment, and checkpoints.

The key roster is ordered: the index in ``KEYS`` is the channel index in the
model's held-state logits tensor. Changing the order or length invalidates
every existing checkpoint, so treat this file as part of the model
architecture, not as configuration.
"""

# Canonical key roster — every key the model can output.
# Grouped by function so related keys are nearby in the embedding.
KEYS: list[str] = [
    # WASD movement
    "w",
    "a",
    "s",
    "d",
    # Arrow keys
    "Up",
    "Down",
    "Left",
    "Right",
    # Common game keys
    "space",
    "Shift_L",
    "Control_L",
    "Alt_L",
    "Tab",
    "Escape",
    "Return",
    "e",
    "f",
    "g",
    "q",
    "r",
    "t",
    # Number row (weapon select, etc.)
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "0",
    # Misc
    "Delete",
    "F12",
]

KEY_INDEX: dict[str, int] = {k: i for i, k in enumerate(KEYS)}
N_KEYS = len(KEYS)

MOUSE_BUTTONS: list[int] = [1, 2, 3]  # left, middle, right
N_MOUSE_BUTTONS = len(MOUSE_BUTTONS)

# Held-state vector layout: N_KEYS key channels then N_MOUSE_BUTTONS button
# channels.
N_HELD_STATE = N_KEYS + N_MOUSE_BUTTONS

# Continuous mouse outputs are tanh-squashed to [-1, 1] and scaled to raw
# mouse counts ("mickeys") at deploy time.
MOUSE_DX_SCALE = 500.0
MOUSE_DY_SCALE = 250.0
