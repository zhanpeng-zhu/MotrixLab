# Go2 Flat Walk v1 (2026-11-22)

## Artifact

- Name: `go2-flatwalk-v1-20261122`
- Artifact: `artifacts/go2-flatwalk-v1-20261122.deploy`
- Environment: `go2-walk-flat`
- Task: `go2-walk-flat/rslrl.ppo`
- Selected checkpoint: `it2900`

## Reproducibility

Training command:

```bash
mxlab train task=go2-walk-flat/rslrl.ppo \
  num_envs=4096 \
  seed=1 \
  algo.algorithm.num_mini_batches=16 \
  algo.max_iterations=3000
```

Training run:

```text
runs/go2-walk-flat/rslrl/torch/ppo/26-11-24_15-51-02-880350
```

The complete training completed `it0` through `it2999`.

## Required training runtime

This artifact was trained and reproduced with the default PyPI Torch resolution, not the repository CUDA profile:

| Component | Required version |
|---|---|
| Python | 3.10 |
| Torch | `2.11.0` |
| CUDA runtime bundled with Torch | `13.0` |
| cuDNN | `9.19.0.56` |
| Triton | `3.6.0` |
| RSL-RL | `4.0.1` |

On an NVIDIA host, `sh install.sh --rslrl` selects the repository CUDA profile and installs `torch 2.7.0+cu128`. That environment is suitable for general development, but it does **not** reproduce this checkpoint: training the same source, task configuration, and seed with Torch 2.7 produces measurably different posture and forward-shift metrics.

Keep the default `.venv` unchanged and create an isolated environment for this artifact instead:

```bash
export UV_PROJECT_ENVIRONMENT="${PWD}/../MotrixLab-go2-flatwalk-env"
uv sync --frozen --no-default-groups --extra rslrl
uv sync --frozen --inexact --group test --extra rslrl
source "${UV_PROJECT_ENVIRONMENT}/bin/activate"
```

The first sync deliberately omits the `cuda` extra so the locked default PyPI resolution selects Torch 2.11.0. The second sync adds the test tools with `--inexact`, preserving that Torch installation. Do not replace this environment with a bare `uv sync`, `uv run`, or another GPU profile.

Train from the repository root only after activating this environment. A verification run with Torch 2.11.0/CUDA 13.0 produced byte-identical `model_2400.pt` through `model_2999.pt` checkpoints and byte-identical seven-point audit JSON files.

## Integrity

| File | SHA-256 |
|---|---|
| `manifest.json` | `586a3a62a83885fb646f05f318eba267d5e6d412f2d76d007e975cbba23cf482` |
| `policy/model.onnx` | `31474ba36a7373454915a748c87c887251f8e2919743fc7c91f084dab75f22de` |
| selected `model_2900.pt` | `91904c728bbab1a4d3fae2c64bbb61496e736385a2e5e15134ae52ba6742b0fa` |
| resolved `task_config.yaml` | `8515efc0b13513d5263410ac40b9fcd571ff226e2ec010611e8467cb9b4b47b9` |

The tracked source tree used for this reproduction has a SHA-256 file-manifest digest of:

```text
b3e4380b2c4608f926b8905b1e3aa7a2eb4c7fc8cb6df64d39dfe6a126a0f65a
```

## Validation

- Source test suite: **59/59 passed**
- ONNX parity: **32 samples**, max absolute error `1.19209e-6`
- Selected checkpoint fixed audit: **16/16 passed**
- Selected checkpoint zero-command hold: **passed**
- Seven-point range audit: fixed **16/16**, speed-friction matrix **220/220**, zero-command hold **passed** at each audited checkpoint

### Selected `it2900` medians

| Audit | Result |
|---|---:|
| Fixed scenarios | 16/16 |
| Speed-friction matrix | 220/220 |
| Fixed roll / pitch | 0.1280 / 0.0878 rad |
| Matrix roll / pitch | 0.1244 / 0.0800 rad |
| Matrix swing height | 5.29 cm |
| Fixed forward-shift p95 | 4.28 cm |
| Matrix forward-shift p95 | 3.53 cm |
| Zero-command hold | pass |
