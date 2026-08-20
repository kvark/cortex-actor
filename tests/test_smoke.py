import numpy as np
import torch

from cortex_actor import Cortex, cortex_from_args, N_HELD_STATE
from cortex_actor.model import (
    center_mean_spatial_patches_np,
    center_mean_spatial_patches_torch,
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


def test_complete_action_codebook_forward():
    model = Cortex(
        d_model=24,
        n_layers=1,
        n_heads=4,
        seq_len=2,
        use_patches=False,
        use_ego=False,
        mouse_mode="regress",
        action_codes=17,
    ).eval()
    cls = torch.randn(3, 2, 384)

    with torch.no_grad():
        out = model(cls)

    assert set(out) == {"action_code_logits"}
    assert out["action_code_logits"].shape == (3, 17)
    assert model.action_code_held.shape == (17, N_HELD_STATE)
    assert model.action_code_tap.shape == (17, N_HELD_STATE)
    assert model.action_code_mouse.shape == (17, 2)


def test_center_mean_is_identical_offline_and_live():
    patches = np.arange(2 * 25 * 40 * 3, dtype=np.float32).reshape(2, 25, 40, 3)
    offline = center_mean_spatial_patches_np(patches)
    live = center_mean_spatial_patches_torch(torch.from_numpy(patches)).numpy()
    assert offline.shape == (2, 10, 8, 3)
    np.testing.assert_allclose(offline, live)


def test_multiscale_model_accepts_native_or_preaggregated_grid():
    torch.manual_seed(4)
    model = Cortex(
        d_model=24,
        n_layers=1,
        n_heads=4,
        seq_len=2,
        use_patches=True,
        patch_grid=(10, 8),
        multiscale_patches=True,
        use_ego=False,
        mouse_mode="regress",
    ).eval()
    cls = torch.randn(1, 2, 384)
    native = torch.randn(1, 2, 25, 40, 384)
    packed = center_mean_spatial_patches_torch(native)
    with torch.no_grad():
        from_native = model(cls, patches=native)
        from_packed = model(cls, patches=packed)
    for name in from_native:
        torch.testing.assert_close(from_native[name], from_packed[name])
