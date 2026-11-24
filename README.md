# Learning-based legged locomotion in simulation

Training a velocity-tracking locomotion policy for the Unitree Go1 quadruped with PPO,
in MuJoCo MJX with thousands of GPU-parallel environments (JAX / Brax).

**Goal:** a policy that follows joystick commands (v_x, v_y, yaw rate) and stays robust to
unmodelled dynamics and rough terrain, i.e. sim-to-real-ready.

## Status

| Stage | What | Status |
|---|---|---|
| 1 | Baseline: flat terrain, default reward | in progress |
| 2 | Reward design + ablations (gait shaping terms) | todo |
| 3 | Domain randomization (friction, mass, motor strength, pushes) + robustness eval | todo |
| 4 | Rough terrain + difficulty curriculum | todo |

## Results

_Videos, training curves and the robustness table go here as each stage is finished._

## Setup

Training needs an NVIDIA GPU (a free Colab T4 is enough for the flat-terrain baseline).
Open `colab.ipynb` in Google Colab, or locally:

```bash
pip install -r requirements.txt          # on GPU: pip install "jax[cuda12]==0.8.3"
python train.py --smoke                   # pipeline check, runs on CPU
python train.py --env Go1JoystickFlatTerrain --run flat_baseline
MUJOCO_GL=egl python rollout.py --run flat_baseline --vx 1.0
```

Each run writes `runs/<name>/` with the policy parameters, the full config and the eval log,
plus `rollout.mp4` and `tracking.png` after `rollout.py`.

## Notes

- Built on MuJoCo Playground 0.2.0 (Zakka et al., RSS 2025), the last release with the
  quadruped environments. Versions are pinned in `requirements.txt`.
- Observations: 48-dim proprioceptive state for the actor; 123-dim privileged state for the
  critic (asymmetric actor-critic). Actions: 12 joint position targets at 50 Hz.
