# Cortex: Compact Behavior Cloning for Quake with Frozen Visual Features

Dzmitry Malyshau — [ORCID 0009-0005-6410-4276](https://orcid.org/0009-0005-6410-4276) — August 2026

[Code (`b4de4f6`)](https://github.com/kvark/cortex-actor/tree/b4de4f66420df2c408ec42b5c01c91a088d8b63d) · [Weights](https://huggingface.co/mad-bot/cortex) · [Video](https://youtu.be/Ou9NAmFoCOM)

## Abstract

We study how far a deliberately simple behavioral-cloning policy can progress in a visually rich first-person game before adding reinforcement learning or explicit memory. **Cortex** has 10.98M trainable parameters in a six-layer transformer over a frozen 28.7M-parameter DINOv3 ViT-S+/16 encoder. It is trained on the Quake subset of the public Pixels2Play corpus: 6,849 recordings (~474.7 hours), represented as 17.09M cached 10 Hz decision frames with ground-truth keyboard and mouse actions. The complete training run is one sampled epoch—517,048 four-frame windows, or 3.27% of all valid windows in the training split—requiring 32,320 optimizer updates and 3.3 minutes of policy-head optimization on one RTX 5080; evaluation uses its 30,000-update checkpoint. This timing excludes one-time feature extraction.

In two independent batches of 20 stochastic, 120-second E1M1 episodes, Cortex completes the level 0/20 times in each batch, reaches pose-based proxies for the opening door, button room, and gate descent in 20/20, has route-waypoint medians 5 and 6 (maximum 9), and records at least one kill in 19/20 episodes in each batch. Under the same time-controlled environment, released P2P-150M and NitroGen checkpoints each complete 0/5 matched-duration episodes and have route median 1.

These reference comparisons are limited by small samples, different native action and observation interfaces, and a custom gamepad-to-keyboard adapter for NitroGen; they measure the evaluated releases under this harness, not generalist ability in the aggregate. Additional-map and mid-map-start experiments are mixed. Controlled ablations show that denser visual tokens improve combat and survival but not route reliability, while longer optimization and naive action history improve offline metrics without consistently improving play. The remaining failures are consistent with covariate shift and motivate targeted corrective data. We release the compact policy implementation, checkpoint, and a representative rollout.

## 1. Introduction

Large gaming policies demonstrate impressive breadth. NitroGen trains a roughly 500M-parameter gamepad policy on 40,000 hours across more than 1,000 games; Pixels2Play (P2P) releases keyboard-and-mouse policies trained on 8,300+ hours. Breadth and depth are different questions. Before adding reinforcement learning, navigation memory, or goal conditioning, we ask what a compact specialist can learn from ordinary supervised imitation on one target game.

Quake E1M1 combines locomotion, door interaction, a wall button, combat, vertical transitions, and recovery from collisions. Completion is a long-horizon binary outcome, while intermediate progress can be measured from an engine pose stream. The level exposes both useful imitation and the familiar behavioral-cloning failure mode: small errors lead to observations absent or rare in the demonstration distribution.

We factor generality into a frozen visual representation and a small learned controller. Cortex receives no privileged state, map, pose, goal, or Quake-specific behavior rule. Sharing source-data provenance makes P2P an informative reference, but it does **not** isolate architecture: P2P has a different multi-game distribution, objective, visual encoder, action decoder, temporal context, and correction data. NitroGen is a looser reference because its native action space is a gamepad and our Quake mapping is an adapter choice.

Our contributions are:

- A compact baseline with 10.98M trainable policy parameters over a frozen visual encoder, direct held-state and mouse prediction, and no learned action history.
- An exact account of source frames, cached decision frames, sample coverage, optimization time, and parameter accounting.
- Engine-observed evaluation with authoritative completion and death signals, pose-derived intermediate progress, two N=20 Cortex batches, and matched-duration reference batches.
- Controlled, retained ablations of spatial density, temporal sampling, optimization exposure, action history, and recovery-weighted data selection.
- Public policy code, exact evaluated weights, and a representative gameplay video.

## 2. Related work

**Generalist gaming policies.** NitroGen couples a SigLIP2-L vision tower with a flow-matching action decoder. Its 40,000-hour dataset comes from internet videos displaying controller overlays: the overlay is localized and parsed with a trained segmentation/classification model, followed by action-density filtering. It is inaccurate to describe these labels as outputs of a gameplay inverse-dynamics model. The NitroGen paper trains with one context frame and 16-action chunks; the released checkpoint evaluated here serializes an 18-action horizon.

P2P is a decoder-only transformer over an EfficientNet-B0 visual token with a 200-frame history and an autoregressive keyboard/mouse decoder. Its public corpus contains 8,300+ hours of recorded human inputs. P2P explicitly conditions its backbone on past ground-truth action tokens during training and mixes in human correction trajectories constituting less than 1% of annotated data.

SIMA 2 combines Gemini with a task-conditioned embodied agent that follows language and image goals, transfers to held-out 3D worlds, and improves from self-generated experience. It is relevant to the long-term goal of broadly capable game agents, but differs from our unconditional low-level imitation setting and was announced as a [limited research preview](https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/); no compatible released checkpoint was available for this harness.

**FPS agents.** ViZDoom established Doom as a platform for visual RL; population-based RL reached human-level capture-the-flag play in Quake III Arena; and large-scale behavioral cloning produced competent Counter-Strike deathmatch behavior. Relative to the short, combat-centric ViZDoom scenarios commonly used as benchmarks, E1M1 requires a convoluted multi-level route, doors and a wall button, lifts and other vertical transitions, vertical aiming, and recovery amid low-fidelity, visually repetitive textures. This characterizes the task used here, not Doom as a whole, which can also support navigation-rich scenarios.

**Frozen features and imitation.** Frozen pretrained vision models can be competitive control representations. DINOv3 supplies dense self-supervised features; the complete Cortex deployment still includes this frozen encoder. DAgger formalizes compounding error and interactive data aggregation; P2P's correction trajectories motivate a related corrective-data direction. Robomimic reports that validation loss can poorly predict closed-loop manipulation success. ACT and Diffusion Policy motivate coherent action-sequence prediction, but the baseline studied here deliberately keeps a simpler per-decision head.

## 3. Architecture

Per 100 ms policy decision:

- **Vision:** frozen DINOv3 ViT-S+/16 encodes 640×400 pixels into CLS plus a 25×40 patch grid. The policy consumes CLS and an ordered 5×8 uniform patch sample: 41 tokens of dimension 384 per frame.
- **Trunk:** four frames spanning 300 ms yield 164 tokens with learned spatial and temporal embeddings. A six-layer, 384-dimensional, six-head bidirectional transformer (FF dimension 1536) reads the last-frame CLS position.
- **Heads:** one linear head predicts 36 independent held-state logits (33 keys and three mouse buttons). Two linear heads predict tanh-squashed relative mouse dx/dy.
- **Execution:** held states are sampled independently at temperature 1, masked by the game adapter’s legal schema, and differenced against the previously executed state to produce device events. There is no previous-action, pose, map, task-text, or privileged-observer input.

*Generate the architecture figure with
[`make_architecture_figure.py`](../scripts/make_architecture_figure.py), or run
`make paper` for the complete rendered manuscript.*

The released state dictionary contains exactly 10,975,142 policy parameters. DINOv3 ViT-S+/16 adds approximately 28.7M frozen parameters, so the pixel-to-action system executes about 39.7M parameters per decision. P2P-150M and NitroGen’s roughly 493M parameters include their visual stacks; these size labels are not matched compute measurements. We report both 10.98M trainable and 39.7M total and make no claim that parameter count alone causes the measured behavior.

| Component | Value |
| --- | --- |
| Vision encoder | DINOv3 ViT-S+/16, frozen, ~28.7M, 640×400 |
| Tokens per frame | 1 CLS + 40 sampled patches, 384-d |
| Context | 4 frames / 300 ms |
| Policy trunk | 6 layers, d=384, 6 heads, FF 1536 |
| Action heads | 36 held-state logits + continuous mouse dx/dy |
| Decision rate | 10 Hz simulated time |
| Trainable / total parameters | 10.98M / ~39.7M |
| Sampled epoch | 32,320 updates, 517,048 examples, maximum batch 16 |
| Evaluated checkpoint | update 30,000 |
| Measured throughput | 2,601 examples/s; 198.8 s on RTX 5080 |

Feature extraction is excluded from the 198.8-second timing.

## 4. Data and training

We use the Quake subset of the public Pixels2Play corpus (`elefantai/p2p-full-data`) with recorded human keyboard and mouse input. The packed index contains **6,849 recordings** and approximately **474.7 hours**. At the source rate this corresponds to approximately 34.18M frames at 20 fps. We retain every second source frame for the 10 Hz policy, producing exactly **17,087,846 cached decision frames**.

Observation *i* is paired with the action interval beginning at *i+1*. The source rate, stride 2, and offset 1 are recorded in the packed index and checkpoint contract. The training split contains 129,262 chunks and 15,828,466 valid four-frame windows. We select four deterministic, distinct windows per chunk, or 517,048 examples (3.27% of valid windows). Thus “one epoch” means one pass over this selected sample, not all possible windows.

We hold out 342 whole recordings, yielding 844,831 valid validation windows before evaluation subsampling. Frames from one recording never cross splits. Production uses no augmentation, class reweighting, route labels, door oversampling, or privileged game state.

At 2,601 examples/s, the complete sampled epoch takes 198.8 seconds; the evaluated snapshot is saved at update 30,000. This excludes DINOv3 extraction, packing, validation, and rollouts. At a measured selective-encoder rate of approximately 207 frames/s, encoding all retained frames is projected to take approximately 23 GPU-hours; this is not an end-to-end timing of a complete extraction run.

## 5. Evaluation protocol

All systems play the same vkQuake build through the same capture path and virtual input device. A read-only engine channel exports pose, health, kills, and client intermission state without changing game rules or physics. The environment advances virtual time in 1/60-second substeps and uses a 61.2 Hz render ceiling for scheduling headroom. Cortex holds one decision for six substeps (100 ms), P2P for three (50 ms), and NitroGen’s queued actions are applied one per substep.

**Time control.** The game waits while a policy computes, so cadence is simulated rather than wall-clock time. This removes stale-frame and missed-decision effects but does not measure real-time deployment throughput. P2P targets real-time 20 Hz inference. NitroGen’s own simulator also freezes game time during inference, so our timing model aligns with NitroGen rather than departing from it.

**Reference implementations.** Both references use released checkpoints and upstream model-side preprocessing/decoding, wrapped by compatibility glue and our game adapter.

- **P2P-150M:** released step-500k checkpoint, upstream KV-cache state, 192×192 Hamming preprocessing, autoregressive decoding at temperature 1.0, truncated-normal mouse-bin dequantization, 20 Hz, and Quake sensitivity 3.5. Its rolling context is reset at episode boundaries but not periodically within an episode.
- **NitroGen:** released checkpoint and processor, 256×256 input, bf16, and the checkpoint’s serialized 18-action horizon at 60 Hz. We supply a custom mapping from left stick to thresholded WASD, right stick to mouse counts, and SOUTH to fire. No official NitroGen-to-Quake binding was released, and we bind no jump button. Results are adapter-dependent.

Cortex and NitroGen use Quake sensitivity 6.0; P2P retains 3.5. The systems share the environment, observer, capture, and input injection but retain model-native preprocessing, cadence, action representation, and sensitivity.

**Outcomes and progress.**

- **Completion** requires the engine’s level-to-intermission transition. Death is health ≤ 0 from the same observer.
- **Route waypoint** is the largest index among 15 reference points (0–14) whose 128-unit radius is touched at any time. Earlier points are not prerequisites. “Opening sequence” is shorthand for reaching the door, button-room, and gate-descent regions; it is not a direct button-event log.
- **Batches:** each Cortex batch has 20 episodes (four lanes and five policy seeds), 120 simulated seconds, stochastic decoding, and no episode selection. Each reference has five matched-duration episodes.
- **Artifacts:** videos were recorded and inspected. The retention policy later removed some raw videos after preserving contact sheets, telemetry, summaries, and manifests.
- **Statistics:** binomial rates carry Wilson 95% intervals. Five-episode reference batches are descriptive and cannot establish a broad ranking.

## 6. Inference performance on one RTX 5080

We measure the four evaluated pixel-to-action stacks on the same RTX 5080, in isolation, at batch size one. A retained E1M1 frame is converted once to each model’s native input; capture, video decoding, model-external CPU image preprocessing, game execution, and input injection are outside the timed region. The input tensor is already GPU-resident. We use each evaluator’s eager, model-native precision path, warm up Cortex for 20 calls, NitroGen for 10, and P2P for 210 so its 200-frame rolling KV cache is full, then record 100 calls with CUDA events. The vision interval brackets each implementation’s native visual module. “Policy/action” is the same-call residual and includes downstream tensor operations, policy inference, and native action generation. These are measurements of the evaluated implementations, not hardware-independent architectural lower bounds.

*Generate the latency figure with
[`make_inference_latency_figure.py`](../scripts/make_inference_latency_figure.py),
or run `make paper` for the complete rendered manuscript.*

| System | Vision p50 | Policy/action p50 | Total p50 | Total p95 | Native output |
| --- | ---: | ---: | ---: | ---: | --- |
| Cortex 5×8 | 2.751 ms | 1.363 ms | **4.114 ms** | 4.141 ms | 1 decision |
| Cortex 25×40 | 2.761 ms | 3.592 ms | 6.352 ms | 6.378 ms | 1 decision |
| P2P-150M | 1.258 ms | 39.190 ms | 40.448 ms | 40.909 ms | 1 decision |
| NitroGen | 5.100 ms | 46.124 ms | 51.224 ms | 51.335 ms | 18 actions |

Compact Cortex takes 4.11 ms at p50. Consuming all 25×40 DINO patches raises its policy segment by only 2.23 ms and total latency to 6.35 ms. Thus the full-grid experiment’s approximately 30-fold training-throughput penalty is primarily a training and packed-data cost, not a comparable deployment penalty at batch one. P2P takes 40.45 ms per 20 Hz decision in this eager steady-state implementation. NitroGen takes 51.22 ms per 18-action chunk, or 2.85 ms per queued 60 Hz action when amortized, although chunked and single-decision policies are not equivalent control interfaces. All p95 values fit their native simulated-time coverage (100 ms Cortex, 50 ms P2P, 300 ms NitroGen), although time control imposes no wall-clock deadline.

## 7. Results

### 7.1 E1M1 from a fresh spawn

*Generate the waypoint-survival figure with
[`make_waypoint_survival_figure.py`](../scripts/make_waypoint_survival_figure.py),
or run `make paper` for the complete rendered manuscript.*

| Metric | Cortex N=20 | Replication N=20 | P2P-150M N=5 | NitroGen N=5 |
| --- | ---: | ---: | ---: | ---: |
| Trainable parameters | 10.98M (39.7M total) | — | ~150M | ~493M |
| Training data | 474.7 h Quake | — | 8,300+ h multi-game | 40,000 h multi-game |
| Completion | 0/20 | 0/20 | 0/5 | 0/5 |
| Door/room/descent pose proxies | **20/20 each** | **20/20 each** | 2/5 / 1/5 / 0/5 | 2/5 / 0/5 / 0/5 |
| Route median (maximum) | 5 (9) | **6 (7)** | 1 (3) | 1 (2) |
| Episodes with a kill | 19/20 | 19/20 | 2/5 | 0/5 |
| Total kills | 32 | 28 | 2 | 0 |
| Deaths | 15/20 | 18/20 | 0/5 | 2/5 |

For completion, 0/20 has a Wilson 95% interval of [0%, 16.1%], while 0/5 has [0%, 43.4%]. The opening-region rate 20/20 has [83.9%, 100%]. Pooling the two fresh-seed Cortex batches descriptively gives 0/40 completions; this does not convert the waypoint heuristic into success.

In production, median maximum displacement is 1,481 units (maximum 2,526), median path length is 10,520, and 19/20 episodes record a kill. The replication again records a kill in 19/20. Reaching the gate descent strongly suggests that the opening interaction succeeded, but telemetry has no direct button-press event, so we do not claim 20/20 observed button presses.

| System, episode | Route | Max displacement | Path | Longest stationary | Kills | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| P2P 0 | 1 | 914 | 2,190 | 11.4 s | 0 | time limit |
| P2P 1 | 1 | 1,045 | 5,127 | 21.9 s | 0 | time limit |
| P2P 2 | 3 | 1,063 | 5,172 | 14.9 s | 1 | time limit |
| P2P 3 | 1 | 919 | 4,673 | 6.8 s | 0 | time limit |
| P2P 4 | 2 | 1,045 | 4,986 | 34.5 s | 1 | time limit |
| NitroGen 0 | 2 | 1,044 | 4,232 | 4.3 s | 0 | died at 59 s |
| NitroGen 1 | 1 | 1,043 | 9,170 | 4.6 s | 0 | time limit |
| NitroGen 2 | 2 | 1,047 | 10,983 | 5.6 s | 0 | died at 89 s |
| NitroGen 3 | 0 | 299 | 4,216 | 19.3 s | 0 | time limit |
| NitroGen 4 | 1 | 759 | 11,416 | 3.9 s | 0 | time limit |

In these ten episodes, extending the earlier 60-second screens does not yield deeper progress. This rules out the shorter cutoff for these traces, not an intrinsic ceiling for either model.

### 7.2 Exploratory additional maps

| Map | System | Chord median (best) | Kills | Died | Median survival |
| --- | --- | ---: | ---: | ---: | ---: |
| E1M2 | Cortex | **791 (2,008)** | **6** | 5/5 | 25 s |
| E1M2 | P2P-150M | 704 (791) | 4 | 3/5 | 37 s |
| E1M2 | NitroGen | 679 (797) | 1 | 5/5 | 26 s |
| E1M3 | Cortex | **706 (1,126)** | 12 | 5/5 | 14 s |
| E1M3 | P2P-150M | 658 (706) | **15** | 3/5 | 46 s |
| E1M3 | NitroGen | 390 (529) | 0 | 4/5 | 56 s |

One P2P episode on each map and one NitroGen episode on E1M3 ended in an environment truncation; all remain in the N=5 batches. Chord summaries use every episode with a valid pose sample, while survival uses observed duration up to death, time limit, or truncation. Cortex has the largest median displacement on both maps and more kills on E1M2; P2P has more kills and longer survival on E1M3. No evaluated episode completes either map. The small batches and three environment truncations make these mixed outcomes especially preliminary; they do not establish map generalization.

### 7.3 Exploratory shared mid-map starts

| Start | System | Sorted route indices | Chord median (best) | New kills | Died | Median survival |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| A | Cortex | [5,5,6,6] | **919 (950)** | 1 | 3/4 | 55 s |
| A | P2P-150M | [0,4,5,6,7] | 484 (**1,375**) | 4 | 2/5 | **120 s** |
| A | NitroGen | [5,5,5,6,6] | 661 (731) | 2 | 5/5 | 25 s |
| B | Cortex | [5,5,6,6,7] | 628 (716) | 2 | 5/5 | 32 s |
| B | P2P-150M | **[6,6,7,7,7]** | 652 (966) | **5** | 3/5 | 26 s |
| B | NitroGen | [5,5,5,5,6] | 230 (406) | 1 | 5/5 | 11 s |

Each save already contains one kill, so the table subtracts one per episode and reports only new kills. Cortex/A has only four episodes after a harness incident. Waypoint indices are order-independent and include the starting region. These results are diagnostic, not a robustness claim.

## 8. Controlled ablations and failure analysis

Only comparisons with retained checkpoints, manifests, and rollout summaries are included. N=4 screens are elimination tests, not precise effect estimates.

| Change | Live sample | Result relative to compact production |
| --- | ---: | --- |
| 5×8 → 10×16 patches | N=20 | Offline imitation improves; route median 5→4, descent proxy 20/20→14/20, kill incidence 19/20→14/20. |
| 5×8 → full 25×40 patches | two paired N=20 batches | Across N=40, kills 60→77 and deaths 33→17, but route mean 5.475→5.225 and descent proxy 40/40→35/40; no completions. |
| 100 ms → 50 ms | N=4 | Best route [4,3,3,2], two kills, versus production [6,4,4,7], seven kills. |
| Same K=4 sample for 4 passes | N=4 | Held F1 0.588→0.600; selected routes [1,4,6,7], final [3,4,4,4], versus [6,4,4,7]. |
| K=4 → K=16 distinct windows | N=4 | Held F1 0.588→0.598; best routes [4,5,4,2]. No live improvement demonstrated. |
| 4 → 8 visual frames | N=4 | Held F1 0.588→0.597; selected routes [2,4,4,7], final [5,4,2,4]. |
| Previous action, 50% dropout | N=20 | Route median 5.5, descent 19/20, kill incidence 17/20, deaths 16; production is 5, 20/20, 19/20, and 15. |
| Failure-similarity sampling | paired N=20 | Median route 5→6 and deaths 15→10, but mean route 5.45→5.25, descent 20/20→18/20, kill incidence 19/20→16/20. |

**Spatial detail changes the trade-off.** Full 25×40 features substantially improve combat and survival while slightly reducing route reliability. They also reduce measured training throughput from 2,601 to 87.7 examples/s and required an approximately 0.84 TB cache. In contrast, batch-one deployed inference rises only from 4.11 to 6.35 ms, so the large penalty is chiefly in training and packed-data handling. The result supports a more efficient multiscale design; it does not support saying that spatial detail “does not help.”

**Offline metrics are insufficient selectors.** Longer optimization, more distinct windows, longer passive visual context, and denser patches all improve at least one held-out metric without consistently improving route behavior. This shows that the measured offline metrics are unreliable selectors here. It is not evidence of a general statistical anti-correlation.

**Naive action history is not a free gain.** A previous held-state token makes next-state prediction easy to shortcut. Context dropout reduces self-lock and yields a competitive route median, but the N=20 variant does not improve the joint opening, combat, death, and collision profile. This result concerns our simple injection and does not contradict P2P’s richer causal training.

**Recovery-oriented sampling moves behavior.** Weighting windows whose starting features resemble observed failures reduces deaths in a paired batch but introduces corridor-sprint failures and weakens opening/combat reliability. It is a trade-off, not a promoted model.

Across the two production batches, stationary stretches never exceed 3.9 seconds, but 33/40 episodes end in death. Visual audits repeatedly show close-wall and water fixation. Human data contains nearby-looking recovery frames, but demonstrations can disagree about the correct turn. These observations are **consistent with** covariate shift and ambiguous recovery actions; they do not prove a causal mechanism or that DAgger is strictly necessary.

**Exploratory visual odometry did not replace engine pose.** Before the retained evaluations, we tested two-frame DINOv3 odometry distilled from a monocular visual-odometry teacher as a game-agnostic progress signal. Successive prototypes mostly learned near-stationary predictions: translation improved only marginally over the stationary prior and rotation remained near it. The teacher labels were noisy; we suspect DINO’s semantic invariances and coarse patch grid were a poor match for frame-to-frame correspondence. The resulting signal was not reliable enough for route evaluation, none of the present results uses it, and the exploration is not included as a controlled quantitative result.

## 9. Limitations and threats to validity

No evaluated Cortex episode completes E1M1. The supported task statement is reliable early-route progress and combat, not level solving. The route score is an order-independent pose heuristic; only engine intermission is success. Door, button-room, and descent regions are not direct interaction events.

The cross-system comparison is not architecture-controlled. Cortex is Quake-specialized; P2P and NitroGen are multi-game. They retain different resolutions, histories, action spaces, cadences, and sensitivities. NitroGen additionally depends on our unvalidated gamepad mapping. P2P shares source-data provenance with Cortex, but its distribution, correction data, and objectives differ. Reference batches contain only five episodes, leaving a 43.4% upper Wilson bound after observing 0/5. These results describe released checkpoints under one harness; they do not show that compact specialists generally outperform foundation-scale agents.

Additional-map and save-state experiments are exploratory N=4–5 studies with mixed winners. Many ablations are N=4 screens. Production was selected after iterative experimentation in this environment. Some raw videos were removed after inspection; summaries, telemetry, contact sheets, and manifests remain, but a permanent public artifact archive is not yet available.

Total system cost includes DINOv3 pretraining and a projected approximately 23 GPU-hours of feature extraction at the measured selective-encoder rate, neither of which appears in the 3.3-minute optimization headline.

The latency comparison likewise describes one RTX 5080 and the eager code paths used by this evaluator. Component boundaries follow each native implementation, preprocessing outside the model is excluded, and NitroGen returns a chunk while the other systems return one decision. The figure therefore compares measured deployed calls, not optimized kernels, equal-horizon control work, energy use, or theoretical compute.

## 10. Next steps

1. **Corrective recovery data.** Deploy frozen BC and collect human takeovers in wall, corner, water, and post-combat failure states. P2P reports that less than 1% correction data mitigates deployment shift. Compare this against similarity-weighted sampling.
2. **Efficient full-grid features.** Use a multiscale or pooling stem so all 25×40 patches contribute without flat attention’s ~30× throughput loss.
3. **Coherent actions and useful memory.** Test a joint or persistent action decoder separately from memory. Add longer causal memory only with an objective that consumes it, such as non-progress recognition or goal conditioning.
4. **Universal progress metrics.** Replace the engine pose ladder with a game-agnostic measure of progress and stagnation based on visual place recognition, geometric motion, and scene change. The failed DINO odometry prototypes suggest that semantic features alone are insufficient for correspondence; validate a dedicated geometric representation against engine state offline, while keeping it outside the policy inputs.
5. **Broader evaluation.** Add more maps, unseen content, other games, validated native adapters, and larger reference batches. Completion remains primary; pose progress remains diagnostic.

## Reproducibility and artifact availability

The evaluated compact policy implementation and action schema are frozen at exact [cortex-actor revision `b4de4f66420df2c408ec42b5c01c91a088d8b63d`](https://github.com/kvark/cortex-actor/tree/b4de4f66420df2c408ec42b5c01c91a088d8b63d). The audited results, latency data, and figure generators for this revision are at [revision `78f78760c3a4b4f145664dd2aa9dd89b553e4d8d`](https://github.com/kvark/cortex-actor/tree/78f78760c3a4b4f145664dd2aa9dd89b553e4d8d); the source-only revised manuscript is frozen by the [`paper-v2.3` Git tag](https://github.com/kvark/cortex-actor/tree/paper-v2.3). The exact evaluated checkpoint is on [Hugging Face](https://huggingface.co/mad-bot/cortex), SHA-256 `29c0e453fdfe7255bc6d8e64a0024fe9b617ed79917f5cd71b41f1173f1aa14b`. Its 86 tensors exactly match the selected step-30,000 training checkpoint; only optimizer/RNG state and local paths are removed. DINOv3 is obtained separately under Meta’s license. The public P2P corpus and Quake shareware episode supply the demonstrations and game content.

The public release does not contain the full feature-extraction, training, or game-execution pipeline. Evaluation used [vkQuake revision `def85227e6089231f23c1fbc2ba5e9c454add833`](https://github.com/kvark/vkQuake/tree/def85227e6089231f23c1fbc2ba5e9c454add833), whose read-only instrumentation exposes pose, health, kills, and intermission state to the evaluator; a separate runtime controlled simulated time. These signals never entered the policy. Cortex targets game-agnostic screen-and-input control, so this Quake adapter is an intermediate evaluation fixture rather than a policy dependency. The evaluator source is archived, but the complete runtime and retained evaluator artifact bundle are not; third-party pixel-to-game reproduction is therefore not yet turnkey.

## Broader impact

This work studies agents in a 1996 single-player video game. The main transferable risks are those of increasingly capable computer-control agents and of overclaiming results from visually plausible rollouts. Engine-authoritative endpoints, explicit adapter disclosures, and retained visual audits are intended to reduce the latter risk.

The experiment also shows that an individual researcher can run a useful closed-loop game-agent research program on one consumer GPU rather than a server-GPU cluster. This is an accessibility observation, not a claim of compute parity or general state-of-the-art performance: encoder pretraining, public demonstration collection, and one-time feature extraction externalize substantial cost.

## Acknowledgements

We thank Elefant AI for releasing the Pixels2Play corpus, inference code, and checkpoints; NVIDIA for releasing NitroGen; Axel Gneiting, the author of vkQuake, and the project’s contributors; id Software for the Quake engine and shareware episode; and Meta AI for DINOv3.

## AI assistance disclosure

The model implementation, training and evaluation tooling, experiments, and manuscript were developed with substantial assistance from Anthropic’s Claude and OpenAI’s Codex under the direction of the author. The author set the goals, reviewed the outputs, and takes responsibility for all claims.

## References

1. L. Magne et al. [“NitroGen: An Open Foundation Model for Generalist Gaming Agents.”](https://arxiv.org/abs/2601.02427) 2026.
2. Y. Yue, I. Salia, S. Hunt, C. Green, W. Shi, and J. J. Hunt. [“Scaling Behavior Cloning Improves Causal Reasoning: An Open Model for Real-Time Video Game Playing.”](https://arxiv.org/abs/2601.04575) 2026.
3. SIMA Team. [“SIMA 2: A Generalist Embodied Agent for Virtual Worlds.”](https://arxiv.org/abs/2512.04797) 2025.
4. O. Siméoni et al. [“DINOv3.”](https://arxiv.org/abs/2508.10104) 2025.
5. M. Oquab et al. [“DINOv2: Learning Robust Visual Features without Supervision.”](https://arxiv.org/abs/2304.07193) TMLR, 2024.
6. M. Caron et al. [“Emerging Properties in Self-Supervised Vision Transformers.”](https://arxiv.org/abs/2104.14294) ICCV, 2021.
7. M. Tschannen et al. [“SigLIP 2.”](https://arxiv.org/abs/2502.14786) 2025.
8. M. Tan and Q. V. Le. [“EfficientNet.”](https://arxiv.org/abs/1905.11946) ICML, 2019.
9. S. Ross, G. Gordon, and D. Bagnell. [“A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.”](https://proceedings.mlr.press/v15/ross11a.html) AISTATS, 2011.
10. D. A. Pomerleau. “ALVINN: An Autonomous Land Vehicle in a Neural Network.” NeurIPS 1, 1989.
11. T. Pearce and J. Zhu. [“Counter-Strike Deathmatch with Large-Scale Behavioural Cloning.”](https://arxiv.org/abs/2104.04258) IEEE CoG, 2022.
12. M. Kempka et al. [“ViZDoom.”](https://arxiv.org/abs/1605.02097) IEEE CIG, 2016.
13. M. Jaderberg et al. [“Human-level performance in 3D multiplayer games with population-based reinforcement learning.”](https://arxiv.org/abs/1807.01281) *Science*, 2019.
14. T. Zhao, V. Kumar, S. Levine, and C. Finn. [“Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.”](https://arxiv.org/abs/2304.13705) RSS, 2023.
15. C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song. [“Diffusion Policy.”](https://arxiv.org/abs/2303.04137) RSS, 2023.
16. S. Parisi et al. [“The Unsurprising Effectiveness of Pre-Trained Vision Models for Control.”](https://arxiv.org/abs/2203.03580) ICML, 2022.
17. S. Nair et al. [“R3M.”](https://arxiv.org/abs/2203.12601) CoRL, 2022.
18. A. Majumdar et al. [“Where are we in the search for an Artificial Visual Cortex for Embodied Intelligence?”](https://arxiv.org/abs/2303.18240) NeurIPS, 2023.
19. A. Mandlekar et al. [“What Matters in Learning from Offline Human Demonstrations for Robot Manipulation.”](https://arxiv.org/abs/2108.03298) CoRL, 2021.

## Appendix A: E1M1 route-waypoint heuristic

For every waypoint independently, the evaluator asks whether the player ever comes within 128 world units. The episode score is the largest index satisfying this test. It does **not** scan in order or require earlier waypoints. It is a progress diagnostic, never completion.

| # | Landmark | x | y | z |
| ---: | --- | ---: | ---: | ---: |
| 0 | spawn | 480 | -352 | 88 |
| 1 | first hall | 480 | 200 | 40 |
| 2 | automatic double doors | 232 | 576 | 64 |
| 3 | button room | -64 | 576 | 24 |
| 4 | gate descent | 0 | 576 | -120 |
| 5 | dark room | 0 | 900 | -200 |
| 6 | bridge foot | 128 | 1060 | -200 |
| 7 | bridge north end | 128 | 1400 | -212 |
| 8 | y=1800 double doors | 128 | 1800 | -170 |
| 9 | lift to pool ledge | 544 | 2048 | -168 |
| 10 | spiral descent | 1000 | 2300 | -220 |
| 11 | bottom corridor | 1316 | 1128 | -208 |
| 12 | side door | 1100 | 1024 | -216 |
| 13 | nailgun bridge | 1312 | 800 | -204 |
| 14 | exit slipgate | 1312 | 544 | -204 |

## Appendix B: Training details

| Setting | Value |
| --- | --- |
| Optimizer | AdamW, β₁=0.9, β₂=0.95, weight decay 0.01 |
| Learning rate | 3×10⁻⁴ cosine-annealed to 3×10⁻⁵ |
| Gradient clipping | global norm 1.0 |
| Precision | bf16 autocast; cached frozen features in fp8 |
| Maximum batch / steps | 16 / 32,320 |
| Loss | active-channel held-state BCE + 2× smooth-L1 mouse loss |
| Sampling | 4 deterministic windows per 128-frame chunk, sampler seed 0 |
| Validation split | 342 held-out recordings |
| Examples / coverage | 517,048 / 3.27% of 15,828,466 valid train windows |
| Throughput | 2,601 examples/s; 198.8 s optimization time |

The iterable loader assigns chunks to eight workers. Its eight final worker batches are partial (six of size 8 and two of size 4); all other batches contain 16 examples. This accounts for the otherwise non-integral ratio between 517,048 examples and 32,320 optimizer steps.

## Appendix C: Fresh-spawn route arrays

- Cortex production: `[7,4,5,4,5,6,4,4,5,6,5,7,6,6,9,6,5,7,4,4]`
- Cortex replication: `[4,4,6,6,4,7,4,6,7,7,6,4,6,5,6,4,7,6,6,5]`
- P2P-150M: `[1,1,3,1,2]`
- NitroGen: `[2,1,2,0,1]`

All 50 episodes have zero engine-observed completions.

## Appendix D: NitroGen gamepad-to-Quake mapping

This is our adapter, not an official NitroGen Quake configuration. Left-stick Y maps to held `w`/`s` and X to `d`/`a` beyond a ±0.3 threshold. Right-stick X/Y maps to mouse dx/dy at 20 counts per unit of deflection, SOUTH maps to held fire, and every other button—including jump—is unbound.
