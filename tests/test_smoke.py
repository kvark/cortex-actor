import numpy as np
import torch

from cortex_actor import (
    Cortex,
    N_HELD_STATE,
    PixelCortex,
    aggregate_spatial_patches_np,
    aggregate_spatial_patches_torch,
    cortex_from_args,
)
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


def test_pixel_forward_smoke():
    args = {
        "seq_len": 2,
        "vision_tokens": "pixels",
        "patch_grid": [2, 3],
        "no_ego": True,
        "d_model": 24,
        "n_layers": 1,
        "n_heads": 4,
        "pixel_encoder_width": 0.25,
    }
    model = cortex_from_args(args)
    assert isinstance(model, PixelCortex)
    out = model(torch.zeros(2, 2, 16, 24, 3, dtype=torch.uint8))
    assert out["held_logits"].shape == (2, N_HELD_STATE)
    out["held_logits"].sum().backward()
    assert model.pixel_encoder.stages[0][0].weight.grad is not None


def test_mean_patch_aggregation_matches_numpy_and_torch():
    source = np.arange(2 * 10 * 16 * 3, dtype=np.float32).reshape(2, 10, 16, 3)
    numpy_result = aggregate_spatial_patches_np(source, (5, 8), aggregation="mean")
    torch_result = aggregate_spatial_patches_torch(
        torch.from_numpy(source),
        (5, 8),
        aggregation="mean",
    )

    np.testing.assert_array_equal(numpy_result, torch_result.numpy())
