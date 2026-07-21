import torch

from cortex_actor import Cortex, cortex_from_args, N_HELD_STATE
from cortex_actor.schema import KEYS, KEY_INDEX, N_KEYS


def test_schema_contract():
    assert N_KEYS == 33
    assert N_HELD_STATE == 36
    assert len(set(KEYS)) == len(KEYS)
    assert all(KEY_INDEX[k] == i for i, k in enumerate(KEYS))


def test_forward_smoke():
    # Production-shaped model (compact 5x8 patch grid, no ego branch),
    # reduced to one layer so the test runs in seconds on CPU.
    args = {
        "seq_len": 4,
        "vision_tokens": "patches",
        "patch_grid": [5, 8],
        "no_ego": True,
        "n_layers": 1,
    }
    model = cortex_from_args(args)
    assert isinstance(model, Cortex)
    model.eval()
    cls = torch.randn(2, 4, 384)
    patches = torch.randn(2, 4, 5, 8, 384)
    with torch.no_grad():
        out = model(cls, patches=patches)
    assert out["held_logits"].shape == (2, N_HELD_STATE)
    assert out["mouse_dx"].shape == (2,)
    assert out["mouse_dy"].shape == (2,)
