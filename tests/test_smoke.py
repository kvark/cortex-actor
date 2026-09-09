import pytest
import torch
from cortex_actor import N_HELD_STATE, Cortex, PixelCortex, cortex_from_args
from cortex_actor.schema import KEY_INDEX, KEYS, N_KEYS


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


def test_causal_action_state_is_an_exact_zero_initialized_extension():
    args = {
        "seq_len": 2,
        "vision_tokens": "cls",
        "no_ego": True,
        "d_model": 24,
        "n_layers": 1,
        "n_heads": 4,
    }
    torch.manual_seed(11)
    baseline = cortex_from_args(args).eval()
    torch.manual_seed(11)
    conditioned = cortex_from_args({**args, "use_action_state": True}).eval()

    baseline_state = baseline.state_dict()
    conditioned_state = conditioned.state_dict()
    for name, tensor in baseline_state.items():
        torch.testing.assert_close(conditioned_state[name], tensor, rtol=0, atol=0)
    assert torch.count_nonzero(conditioned.action_state_proj.weight) == 0

    cls = torch.randn(3, 2, 384)
    causal_held = torch.randint(0, 2, (3, N_HELD_STATE)).float()
    with torch.no_grad():
        baseline_output = baseline(cls)
        conditioned_output = conditioned(cls, causal_held=causal_held)
    for name, tensor in baseline_output.items():
        torch.testing.assert_close(conditioned_output[name], tensor, rtol=0, atol=0)

    with pytest.raises(ValueError, match="causal held state"):
        conditioned(cls)
    with pytest.raises(ValueError, match="causal held state"):
        conditioned(cls, causal_held=causal_held[:, :-1])
