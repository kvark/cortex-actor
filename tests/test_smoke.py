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


def test_action_context_preserves_common_initialization():
    kwargs = dict(
        d_model=24,
        n_layers=1,
        n_heads=4,
        seq_len=2,
        use_patches=False,
        use_ego=False,
        mouse_mode="regress",
        action_codes=17,
    )
    torch.manual_seed(5)
    baseline = Cortex(**kwargs)
    torch.manual_seed(5)
    contextual = Cortex(**kwargs, use_action_context=True)

    contextual_state = contextual.state_dict()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, contextual_state[name])


def test_held_duration_head_preserves_and_detaches_common_policy():
    kwargs = dict(
        d_model=24,
        n_layers=1,
        n_heads=4,
        seq_len=2,
        use_patches=False,
        use_ego=False,
        mouse_mode="regress",
        action_codes=17,
    )
    torch.manual_seed(7)
    baseline = Cortex(**kwargs)
    torch.manual_seed(7)
    duration = Cortex(**kwargs, held_duration_groups=5, held_duration_bins=4)

    duration_state = duration.state_dict()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, duration_state[name])

    out = duration(torch.zeros(2, 2, 384))
    assert out["held_duration_logits"].shape == (2, 5, 4)
    out["held_duration_logits"].sum().backward()
    assert duration.held_duration_head.weight.grad is not None
    for name, parameter in duration.named_parameters():
        if not name.startswith("held_duration_head."):
            assert parameter.grad is None, name


def test_held_duration_training_leaves_control_policy_exact():
    kwargs = dict(
        d_model=24,
        n_layers=1,
        n_heads=4,
        seq_len=2,
        use_patches=False,
        use_ego=False,
        mouse_mode="regress",
        action_codes=17,
    )
    torch.manual_seed(11)
    baseline = Cortex(**kwargs)
    torch.manual_seed(11)
    duration = Cortex(**kwargs, held_duration_groups=5, held_duration_bins=4)
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=1e-3)
    duration_optimizer = torch.optim.AdamW(duration.parameters(), lr=1e-3)
    cls = torch.randn(3, 2, 384)
    action_target = torch.tensor([1, 4, 9])
    duration_target = torch.tensor([0, 2, 1])
    group = torch.tensor([0, 3, 2])

    baseline_out = baseline(cls)
    baseline_loss = torch.nn.functional.cross_entropy(
        baseline_out["action_code_logits"], action_target
    )
    baseline_loss.backward()
    torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.0)
    baseline_optimizer.step()

    duration_out = duration(cls)
    duration_loss = torch.nn.functional.cross_entropy(
        duration_out["action_code_logits"], action_target
    ) + torch.nn.functional.cross_entropy(
        duration_out["held_duration_logits"][torch.arange(3), group], duration_target
    )
    duration_loss.backward()
    policy_parameters = [
        parameter
        for name, parameter in duration.named_parameters()
        if not name.startswith("held_duration_head.")
    ]
    torch.nn.utils.clip_grad_norm_(policy_parameters, 1.0)
    torch.nn.utils.clip_grad_norm_(duration.held_duration_head.parameters(), 1.0)
    duration_optimizer.step()

    duration_state = duration.state_dict()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, duration_state[name], rtol=0, atol=0)


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
