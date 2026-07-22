"""Cortex behavioral-cloning policy.

The production baseline is deliberately small: a frozen DINOv3 token window,
a transformer, direct held-state logits, and mouse prediction.  Spatial tokens,
previous actions, and egocentric pose history are checkpoint-selected
ablations implemented by the same model so training and deployment cannot
silently choose different architectures.

``build_ego_history`` is the shared train/deploy feature builder for the
optional pose-history branch.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .schema import N_KEYS, N_HELD_STATE


# DINOv3 emits a 25x40 grid for the canonical 400x640 observation. A fixed 5x8
# center sample preserves coarse spatial layout with 25x fewer tokens and is
# shared by the dataset and live actor. This is a representation contract, not
# a game rule.
DINO_PATCH_GRID = (25, 40)
COMPACT_PATCH_GRID = (5, 8)


def register_tokens_to_grid(registers):
    """Lay out DINO's ordered register tokens as a one-row token grid.

    The helper intentionally works for both NumPy arrays used by the dataset
    and Torch tensors used by the live actor.  Keeping this reshape shared is
    important: ``Cortex`` treats the two grid dimensions as part of the
    checkpoint's representation contract.
    """
    if registers.ndim < 2:
        raise ValueError("register tokens must have (..., registers, channels) shape")
    return registers[..., None, :, :]


def spatial_sample_indices(source: int, target: int) -> np.ndarray:
    """Return the center index of each cell in a uniform spatial partition."""
    if source < 1 or target < 1 or target > source:
        raise ValueError(f"invalid spatial sample {source} -> {target}")
    return np.floor((np.arange(target) + 0.5) * source / target).astype(np.int64)


def sample_spatial_patches_np(
    patches: np.ndarray,
    target_grid: tuple[int, int] = COMPACT_PATCH_GRID,
) -> np.ndarray:
    """Select a compact grid from an array shaped ``(..., H, W, C)``.

    ``np.take`` touches only the selected mmap rows instead of materializing the
    full patch grid, which is the main full-corpus I/O saving.
    """
    source_grid = patches.shape[-3:-1]
    if tuple(source_grid) == tuple(target_grid):
        return patches
    rows = spatial_sample_indices(source_grid[0], target_grid[0])
    columns = spatial_sample_indices(source_grid[1], target_grid[1])
    return np.take(np.take(patches, rows, axis=-3), columns, axis=-2)


def sample_spatial_patches_torch(
    patches: torch.Tensor,
    target_grid: tuple[int, int] = COMPACT_PATCH_GRID,
) -> torch.Tensor:
    """Torch equivalent of :func:`sample_spatial_patches_np`."""
    source_grid = patches.shape[-3:-1]
    if tuple(source_grid) == tuple(target_grid):
        return patches
    rows = torch.as_tensor(
        spatial_sample_indices(source_grid[0], target_grid[0]),
        device=patches.device,
    )
    columns = torch.as_tensor(
        spatial_sample_indices(source_grid[1], target_grid[1]),
        device=patches.device,
    )
    return patches.index_select(-3, rows).index_select(-2, columns)

# Bin edges for mouse classification (mouse_mode='bins').
# Edges define n+1 boundaries for n bins on normalized [-1, 1].
# Asymmetric edges around 0 because P2P mouse distribution is heavy near 0.
MOUSE_BIN_EDGES_5 = [-1.001, -0.30, -0.05, 0.05, 0.30, 1.001]  # 5 bins
MOUSE_BIN_CENTERS_5 = [-0.55, -0.175, 0.0, 0.175, 0.55]

MOUSE_BIN_EDGES_7 = [-1.001, -0.50, -0.15, -0.04, 0.04, 0.15, 0.50, 1.001]  # 7 bins
MOUSE_BIN_CENTERS_7 = [-0.75, -0.30, -0.085, 0.0, 0.085, 0.30, 0.75]

# 9 bins, power-style: tighter resolution near zero (where ~95% of data lives),
# wider tails for fast turns. Reduces adjacent-bin jumps in the common motion
# regime to ~25-45 mickeys (vs ~87 with linear 5-bin), smoothing the camera.
MOUSE_BIN_EDGES_9P = [-1.001, -0.50, -0.20, -0.075, -0.02, 0.02, 0.075, 0.20, 0.50, 1.001]
MOUSE_BIN_CENTERS_9P = [-0.75, -0.35, -0.14, -0.05, 0.0, 0.05, 0.14, 0.35, 0.75]


def get_mouse_bins(n_bins: int):
    if n_bins == 5:
        return MOUSE_BIN_EDGES_5, MOUSE_BIN_CENTERS_5
    if n_bins == 7:
        return MOUSE_BIN_EDGES_7, MOUSE_BIN_CENTERS_7
    if n_bins == 9:
        return MOUSE_BIN_EDGES_9P, MOUSE_BIN_CENTERS_9P
    raise ValueError(f"unsupported n_bins: {n_bins}")


def cortex_feature_flags(args) -> tuple[bool, bool]:
    """Resolve vision/ego feature flags across current and legacy checkpoints.

    Current checkpoints store negative ``no_*`` flags. Cortex-mini checkpoints
    predate that schema and store only ``use_patches``; they never had an ego
    branch. Keeping the compatibility rule here prevents the trainer/model and
    deployment actor from reconstructing different architectures.
    """
    a = vars(args) if hasattr(args, "__dict__") else args
    if "vision_tokens" in a:
        use_patches = a["vision_tokens"] != "cls"
    else:
        use_patches = (
            not bool(a["no_patches"])
            if "no_patches" in a
            else bool(a.get("use_patches", False))
        )
    use_ego = (
        not bool(a["no_ego"])
        if "no_ego" in a
        else bool(a.get("use_ego", False))
    )
    return use_patches, use_ego


def upgrade_legacy_cortex_state_dict(state_dict):
    """Pad pre-fire Cortex-mini key heads to the universal held schema.

    Cortex-mini v1-v12 predicted keyboard state only. Later checkpoints add
    three mouse-button rows. Zero-logit padding preserves the legacy keyboard
    policy and leaves every added button released under deterministic decode.
    """
    weight = state_dict.get("key_head.weight")
    bias = state_dict.get("key_head.bias")
    if weight is None or bias is None or weight.shape[0] != N_KEYS:
        return state_dict
    upgraded = dict(state_dict)
    extra = N_HELD_STATE - N_KEYS
    upgraded["key_head.weight"] = torch.cat(
        [weight, weight.new_zeros(extra, weight.shape[1])], dim=0
    )
    upgraded["key_head.bias"] = torch.cat(
        [bias, bias.new_zeros(extra)], dim=0
    )
    return upgraded


# mouse_mode='chunk' defaults: 1s of future mouse at 10Hz, 64 k-means templates.
CHUNK_HORIZON = 10
CHUNK_CODES = 64

# Egocentric history feature layout, per sampled past step:
#   [rel_x, rel_y, rel_z, fwd_x, fwd_y, fwd_z]
#   rel_*  = past camera center in the CURRENT camera frame, / EGO_POS_SCALE
#   fwd_*  = past camera forward axis, rotated into the CURRENT camera frame
EGO_FEAT_DIM = 6
EGO_FWD_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # CUT3R camera looks +Z

# Yaw-aux normalization: corpus per-frame CUT3R yaw-delta std is ~0.197 rad.
YAW_AUX_SCALE = 0.2
# Future-yaw normalization: corpus 10-frame (1s) cumulative yaw std is ~0.98 rad.
FUTURE_YAW_SCALE = 1.0
# Big-turn class boundary for the 3-class future-yaw head (~20 deg over 1s),
# matching the initiation-probe definition.
FUTURE_YAW_THRESH = 0.35


def yaw_deltas(pose_R: np.ndarray) -> np.ndarray:
    """Per-transition camera yaw change (rad), length N-1. Rotation of the +Z
    view axis in the camera xz plane between consecutive frames."""
    R = pose_R.astype(np.float64)
    rel = np.einsum("tij,tik->tjk", R[:-1], R[1:])
    return np.arctan2(rel[:, 0, 2], rel[:, 2, 2]).astype(np.float32)


# Motion-input normalization: big turns shift the patch grid by ~1-3 columns.
MOTION_SHIFT_SCALE = 2.0


@torch.no_grad()
def patch_shift_estimate(patches: torch.Tensor, max_shift: int = 5) -> torch.Tensor:
    """Horizontal patch-grid shift between consecutive frames, (B,T,H,W,C) ->
    (B,T-1) in patch-column units (quadratic peak interpolation, ~[-5, 5]).

    This is the naive cross-correlator that recovers camera turn direction at
    0.84 sign-agreement from the stored DINOv3 patches — the trunk demonstrably
    fails to learn this computation itself (yaw-aux probe: chance even at
    weight 10), so it is fed as an explicit input instead.
    """
    x = patches / (patches.norm(dim=-1, keepdim=True) + 1e-6)
    A, B = x[:, :-1], x[:, 1:]                         # (B,T-1,H,W,C)
    W = A.shape[3]
    scores = []
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            sim = (A[:, :, :, s:] * B[:, :, :, :W - s]).sum(-1).mean((-1, -2))
        else:
            sim = (A[:, :, :, :W + s] * B[:, :, :, -s:]).sum(-1).mean((-1, -2))
        scores.append(sim)
    S = torch.stack(scores, dim=-1)                    # (B,T-1,2*max_shift+1)
    smax = S.argmax(-1)
    lo = (smax - 1).clamp(min=0)
    hi = (smax + 1).clamp(max=2 * max_shift)
    y0 = S.gather(-1, lo.unsqueeze(-1)).squeeze(-1)
    y1 = S.gather(-1, smax.unsqueeze(-1)).squeeze(-1)
    y2 = S.gather(-1, hi.unsqueeze(-1)).squeeze(-1)
    frac = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-9)
    frac = torch.where((smax > 0) & (smax < 2 * max_shift), frac, torch.zeros_like(frac))
    return (smax - max_shift).float() + frac


def build_ego_history(
    pose_xyz: np.ndarray,   # (N, 3) CUT3R camera centers in world
    pose_R: np.ndarray,     # (N, 3, 3) camera->world rotations
    t: int,                 # current frame index
    k: int = 16,            # number of strided history samples
    horizon: int = 50,      # frames back the oldest sample reaches (5 s @ 10 fps)
    pos_scale: float = 5.0,  # CUT3R is approximately metric; normalize to O(1)
    log_spaced: bool = False,  # see below
) -> np.ndarray:
    """Last `k` camera poses, in the current camera frame. Newest sample last.
    Clamps to spawn when history is short, so the same function works from frame
    0 onward at deploy.

    Two sampling modes:
      * linear (default): `k` evenly spaced offsets over the last `horizon`
        frames — a short, uniform window (the v1 behaviour).
      * log_spaced: the oldest sample reaches back the WHOLE available path
        (offset = min(t, horizon)); offsets are geometrically spaced so recent
        frames are dense and the distant past is coarse. `horizon` here is a CAP
        (set to the training-shard length so deploy never produces offsets the
        model never saw — the train==deploy invariant).

    Returns (k, EGO_FEAT_DIM) float32.
    """
    if log_spaced:
        max_off = min(int(horizon), int(t)) if horizon and horizon > 0 else int(t)
        max_off = max(max_off, 1)
        # geomspace(1, max_off+1) - 1 -> ages 0..max_off, dense near 0 (recent);
        # reverse to old -> new so the newest sample is last (matches readout).
        ages = np.geomspace(1.0, max_off + 1.0, k) - 1.0
        offsets = ages[::-1].round().astype(np.int64)
    else:
        offsets = np.linspace(horizon, 0, k).round().astype(np.int64)  # old -> new
    Rt = pose_R[t].astype(np.float32)
    pt = pose_xyz[t].astype(np.float32)
    RtT = Rt.T
    feats = np.zeros((k, EGO_FEAT_DIM), dtype=np.float32)
    for j, off in enumerate(offsets):
        s = t - int(off)
        if s < 0:
            s = 0  # clamp to oldest available (spawn) — no fabricated history
        rel = RtT @ (pose_xyz[s].astype(np.float32) - pt)
        fdir = RtT @ (pose_R[s].astype(np.float32) @ EGO_FWD_AXIS)
        feats[j, :3] = rel / pos_scale
        feats[j, 3:] = fdir
    return feats


class Cortex(nn.Module):
    """CLS(+patches) vision tokens + K egocentric history tokens -> action.

    Output (last frame only), matching CortexMini:
      held_logits (B, N_HELD_STATE), mouse heads (regress or bins).
    """

    def __init__(
        self,
        d_model: int = 384,
        n_layers: int = 6,
        n_heads: int = 6,
        seq_len: int = 8,
        ego_k: int = 16,
        mouse_scale: float = 1.0,
        mouse_mode: str = "bins",
        mouse_n_bins: int = 9,
        chunk_codes: int = CHUNK_CODES,
        chunk_horizon: int = CHUNK_HORIZON,
        use_patches: bool = True,
        use_ego: bool = True,
        patch_grid: tuple[int, int] = (25, 40),
        yaw_aux: bool = False,
        future_yaw: bool = False,
        motion_input: bool = False,
        use_action_context: bool = False,
        action_persistence_skip: bool = False,
        predict_taps: bool = False,
        use_goal: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.ego_k = ego_k
        self.mouse_scale = float(mouse_scale)
        self.n_keys = N_KEYS
        self.n_held = N_HELD_STATE
        self.mouse_mode = mouse_mode
        self.mouse_n_bins = int(mouse_n_bins)
        self.use_patches = use_patches
        self.patch_grid = tuple(patch_grid)
        self.n_patches = patch_grid[0] * patch_grid[1]
        self.tokens_per_frame = (1 + self.n_patches) if use_patches else 1
        assert mouse_mode in ("regress", "bins", "chunk"), mouse_mode
        self.chunk_codes = int(chunk_codes)
        self.chunk_horizon = int(chunk_horizon)

        self.cls_proj = nn.Linear(384, d_model)
        if use_patches:
            self.patch_proj = nn.Linear(384, d_model)
            self.spatial_pos = nn.Parameter(
                torch.randn(self.tokens_per_frame, d_model) * 0.02
            )
        self.pos_emb = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)

        # Egocentric history branch: project each step's 6-d feature to a token,
        # add a per-step positional embedding, prepend the K tokens to the
        # vision sequence so attention can fuse "where I've been" with "what I see".
        # use_ego=False drops this branch entirely -> a pure CortexMini (the clean
        # BC baseline, no SLAM input); the ego version is then a one-variable A/B.
        self.use_ego = use_ego
        if use_ego:
            self.ego_proj = nn.Linear(EGO_FEAT_DIM, d_model)
            self.ego_pos = nn.Parameter(torch.randn(ego_k, d_model) * 0.02)

        # Optional goal token: the DINOv3 CLS of a desired future view,
        # hindsight-sampled during training. The goal is real data, not a
        # learned latent — there is no codebook to collapse. When no goal is
        # provided (goal dropout in training, unconditional deployment) a
        # learned placeholder embedding takes the token's place, so the same
        # checkpoint runs conditionally and unconditionally.
        self.use_goal = use_goal
        if use_goal:
            self.goal_proj = nn.Linear(384, d_model)
            self.goal_pos = nn.Parameter(torch.randn(1, d_model) * 0.02)
            self.no_goal_emb = nn.Parameter(torch.zeros(d_model))

        # Optional previous-action token. The minimal BC baseline omits it;
        # older contextual checkpoints reconstruct the branch from metadata.
        self.use_action_context = use_action_context
        if use_action_context:
            self.action_proj = nn.Linear(N_HELD_STATE, d_model)
            self.action_pos = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.action_persistence_skip = bool(action_persistence_skip)
        if self.action_persistence_skip:
            if not self.use_action_context:
                raise ValueError("action persistence skip requires action context")
            # Start from a generic persistence prior, then learn one strength
            # per universal held channel. The visual head only has to override
            # this skip on real press/release transitions.
            self.action_persistence = nn.Parameter(
                torch.full((N_HELD_STATE,), 2.0)
            )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.ln_f = nn.LayerNorm(d_model)

        # Motion input: per-transition patch-shift estimate (computed in
        # forward from the patches themselves -> train/deploy bit-identical),
        # concatenated DIRECTLY into the head inputs. Prepended motion tokens
        # were tried first and failed: attention never learned to route them
        # to the readouts (yaw head stayed at chance despite the answer being
        # an input); the concat path is linear so the gradient is immediate.
        self.motion_input = motion_input
        if motion_input:
            assert use_patches, "motion_input requires patches"
        mouse_extra = (seq_len - 1) if motion_input else 0

        # Yaw-aux: per-frame head predicting the camera yaw delta of the
        # transition leaving each frame (frame i -> i+1, / YAW_AUX_SCALE).
        # Deltas 0..T-2 are observable ego-motion (forces the trunk to extract
        # turn direction from the frame stream); delta T-1 is the FUTURE,
        # action-aligned turn. Train-only signal; deploy ignores it.
        self.yaw_aux = yaw_aux
        if yaw_aux:
            self.yaw_head = nn.Linear(d_model + (1 if motion_input else 0), 1)

        # Future-yaw: per-frame 3-class head over the CUMULATIVE camera yaw in
        # the next H frames: right (< -thresh) / none / left (> +thresh).
        # Classification, not regression: the predictable component of future
        # yaw is small vs its std (~1 rad), so L1 regresses to ~0 and buries
        # the direction signal (validated: 10k regression steps stayed at
        # chance sign while a frozen-CLS classifier probe hit 0.57/0.61).
        self.future_yaw = future_yaw
        if future_yaw:
            self.fyaw_head = nn.Linear(d_model + (1 if motion_input else 0), 3)

        self.key_head = nn.Linear(d_model, N_HELD_STATE)
        self.predict_taps = bool(predict_taps)
        if self.predict_taps:
            # Resulting held state cannot represent a press+release contained
            # in one decision interval. Preserve that transient device event
            # with one small parallel head while keeping held-state BC simple.
            self.tap_head = nn.Linear(d_model, N_HELD_STATE)
        if mouse_mode == "regress":
            self.mouse_dx_head = nn.Linear(d_model + mouse_extra, 1)
            self.mouse_dy_head = nn.Linear(d_model + mouse_extra, 1)
            for h in (self.mouse_dx_head, self.mouse_dy_head):
                nn.init.zeros_(h.bias)
                nn.init.normal_(h.weight, std=0.01)
        elif mouse_mode == "bins":
            self.mouse_dx_head = nn.Linear(d_model + mouse_extra, self.mouse_n_bins)
            self.mouse_dy_head = nn.Linear(d_model + mouse_extra, self.mouse_n_bins)
        else:
            # Chunk mode: classify the next short mouse trajectory into one of K
            # fixed k-means templates (codebook built by the trainer from the
            # corpus, stored as a buffer so it travels with the ckpt). Per-step
            # sampling re-flips the direction coin 10x/s and cancels into yaw
            # noise; a sampled template is a committed micro-trajectory. The
            # actor holds a drawn code across steps (receding-horizon sticky
            # sampling) — commitment lives in the sampler, not this head.
            self.mouse_chunk_head = nn.Linear(d_model + mouse_extra, self.chunk_codes)
            self.register_buffer(
                "chunk_codebook",
                torch.zeros(self.chunk_codes, self.chunk_horizon, 2))

    def forward(
        self,
        cls: torch.Tensor,                 # (B, T, 384)
        ego: torch.Tensor | None = None,   # (B, K, EGO_FEAT_DIM); ignored if use_ego=False
        patches: torch.Tensor | None = None,  # (B, T, H, W, 384)
        return_hidden: bool = False,
        action_context: torch.Tensor | None = None,  # (B, N_HELD_STATE), previous state
        goal: torch.Tensor | None = None,            # (B, 384) goal-frame CLS
        goal_mask: torch.Tensor | None = None,       # (B,) bool; False -> no-goal placeholder
    ) -> dict[str, torch.Tensor]:
        B, T, _ = cls.shape
        if T > self.seq_len:
            cls = cls[:, -self.seq_len:]
            if patches is not None:
                patches = patches[:, -self.seq_len:]
            T = self.seq_len

        if self.use_patches:
            assert patches is not None, "use_patches=True requires patches"
            if patches.shape[-3:-1] != self.patch_grid:
                patches = sample_spatial_patches_torch(patches, self.patch_grid)
            B_, T_, H, W, C = patches.shape
            assert (H, W) == self.patch_grid, f"patches {(H,W)} != {self.patch_grid}"
            patches_flat = patches.reshape(B, T, self.n_patches, 384)
            cls_h = self.cls_proj(cls).unsqueeze(2)            # (B,T,1,d)
            patches_h = self.patch_proj(patches_flat)          # (B,T,n_patches,d)
            vis = torch.cat([cls_h, patches_h], dim=2)         # (B,T,tpf,d)
            vis = vis + self.spatial_pos.view(1, 1, self.tokens_per_frame, vis.size(-1))
            vis = vis + self.pos_emb[:T].view(1, T, 1, vis.size(-1))
            vis = vis.reshape(B, T * self.tokens_per_frame, -1)
        else:
            vis = self.cls_proj(cls) + self.pos_emb[:T]        # (B,T,d)

        mshift = None
        if self.motion_input:
            mshift = patch_shift_estimate(patches) / MOTION_SHIFT_SCALE  # (B,T-1)

        prefixes = []
        if self.use_goal:
            if goal is None:
                goal_tok = self.no_goal_emb.view(1, 1, -1).expand(B, 1, -1)
            else:
                projected = self.goal_proj(goal)
                if goal_mask is not None:
                    projected = torch.where(
                        goal_mask[:, None], projected, self.no_goal_emb
                    )
                goal_tok = projected.unsqueeze(1)
            prefixes.append(goal_tok + self.goal_pos.unsqueeze(0))
        if self.use_action_context:
            if action_context is None:
                raise ValueError(
                    "this checkpoint requires the previously executed held state"
                )
            action_tok = self.action_proj(action_context).unsqueeze(1)
            prefixes.append(action_tok + self.action_pos.unsqueeze(0))
        if self.use_ego:
            ego_tok = self.ego_proj(ego) + self.ego_pos.unsqueeze(0)  # (B,K,d)
            prefixes.append(ego_tok)
        if prefixes:
            prefix = torch.cat(prefixes, dim=1)
            h = torch.cat([prefix, vis], dim=1)
            n_prefix = prefix.shape[1]
        else:
            h = vis
            n_prefix = 0
        last_cls_idx = n_prefix + (T - 1) * self.tokens_per_frame

        # Bidirectional (no causal mask): only the last-frame CLS token is read
        # out, so flash-attention dispatch is safe and fast (see CortexMini note).
        h = self.transformer(h)
        h_last = self.ln_f(h[:, last_cls_idx])

        held_logits = self.key_head(h_last)
        if self.action_persistence_skip:
            held_logits = held_logits + self.action_persistence * (
                2.0 * action_context - 1.0
            )
        out = {"held_logits": held_logits}
        if self.predict_taps:
            out["tap_logits"] = self.tap_head(h_last)
        if self.yaw_aux or self.future_yaw:
            idxs = n_prefix + torch.arange(T, device=h.device) * self.tokens_per_frame
            h_frames = self.ln_f(h[:, idxs])                    # (B, T, d)
            if self.motion_input:
                # Per-frame shift, aligned so frame i sees the estimate for its
                # own outgoing transition; the last frame (FUTURE target) sees
                # the most recent observable one — the continuation prior.
                sh = torch.cat([mshift, mshift[:, -1:]], dim=1).unsqueeze(-1)
                h_frames = torch.cat([h_frames, sh], dim=-1)
            if self.yaw_aux:
                out["yaw_pred"] = self.yaw_head(h_frames).squeeze(-1)   # (B, T)
            if self.future_yaw:
                out["fyaw_logits"] = self.fyaw_head(h_frames)  # (B, T, 3)
        if return_hidden:
            out["hidden"] = h_last                  # (B, d) — for the RL value head
        h_mouse = torch.cat([h_last, mshift], dim=-1) if self.motion_input else h_last
        if self.mouse_mode == "regress":
            out["mouse_dx"] = torch.tanh(self.mouse_dx_head(h_mouse).squeeze(-1)) * self.mouse_scale
            out["mouse_dy"] = torch.tanh(self.mouse_dy_head(h_mouse).squeeze(-1)) * self.mouse_scale
        elif self.mouse_mode == "bins":
            out["mouse_dx_bins"] = self.mouse_dx_head(h_mouse)
            out["mouse_dy_bins"] = self.mouse_dy_head(h_mouse)
        else:
            out["mouse_chunk_logits"] = self.mouse_chunk_head(h_mouse)  # (B, K)
        return out


def cortex_from_args(args) -> Cortex:
    """Reconstruct the exact Cortex architecture recorded in a checkpoint."""
    a = vars(args) if hasattr(args, "__dict__") else args
    use_patches, use_ego = cortex_feature_flags(a)
    patch_grid = tuple(a.get("patch_grid") or DINO_PATCH_GRID)
    return Cortex(
        d_model=int(a.get("d_model", 384)),
        n_layers=int(a.get("n_layers", 6)),
        n_heads=int(a.get("n_heads", 6)),
        seq_len=int(a["seq_len"]),
        ego_k=int(a.get("ego_k", 16)),
        mouse_scale=float(a.get("mouse_scale", 1.0)),
        mouse_mode=a.get("mouse_mode", "regress"),
        mouse_n_bins=int(a.get("mouse_n_bins", 9)),
        chunk_codes=int(a.get("chunk_codes", 64)),
        chunk_horizon=int(
            a.get("mouse_chunk_horizon", a.get("future_yaw_horizon", 10))
        ),
        use_patches=use_patches,
        patch_grid=patch_grid,
        use_ego=use_ego,
        yaw_aux=float(a.get("yaw_aux_weight", 0.0)) > 0,
        future_yaw=float(a.get("future_yaw_weight", 0.0)) > 0,
        motion_input=bool(a.get("motion_input", False)),
        use_action_context=bool(a.get("action_context", False)),
        action_persistence_skip=bool(a.get("action_persistence_skip", False)),
        predict_taps=bool(a.get("tap_events", False)),
        use_goal=bool(a.get("goal_conditioning", False)),
    )
