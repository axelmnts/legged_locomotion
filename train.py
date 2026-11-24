"""Train a quadruped (Unitree Go1) locomotion policy with PPO in MuJoCo MJX.

Usage (GPU, e.g. Colab):
    python train.py --env Go1JoystickFlatTerrain --run flat_baseline
Smoke test (CPU, a few seconds of training):
    python train.py --smoke
"""
import argparse
import functools
import json
import os
import pickle
import time

import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Go1JoystickFlatTerrain")
    p.add_argument("--run", default="baseline", help="run name, used for the output folder")
    p.add_argument("--timesteps", type=int, default=None, help="override total env steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="tiny CPU run to check the pipeline")
    args = p.parse_args()

    env_cfg = registry.get_default_config(args.env)
    env = registry.load(args.env, config=env_cfg)
    ppo_cfg = locomotion_params.brax_ppo_config(args.env)

    if args.smoke:
        ppo_cfg.num_timesteps = 20_000
        ppo_cfg.num_envs = 32
        ppo_cfg.batch_size = 32
        ppo_cfg.num_minibatches = 4
        ppo_cfg.unroll_length = 10
        ppo_cfg.num_updates_per_batch = 2
        ppo_cfg.num_evals = 2
        ppo_cfg.num_eval_envs = 4 if "num_eval_envs" in ppo_cfg else None
    if args.timesteps:
        ppo_cfg.num_timesteps = args.timesteps

    out_dir = os.path.join("runs", args.run)
    os.makedirs(out_dir, exist_ok=True)
    log = []
    t0 = time.time()

    def progress(step, metrics):
        row = {"step": int(step), "time_s": round(time.time() - t0, 1),
               "reward": float(metrics["eval/episode_reward"]),
               "reward_std": float(metrics.get("eval/episode_reward_std", 0.0))}
        log.append(row)
        print(row, flush=True)

    params = dict(ppo_cfg)
    params.pop("network_factory", None)
    params = {k: v for k, v in params.items() if v is not None}
    network_factory = functools.partial(ppo_networks.make_ppo_networks,
                                        **ppo_cfg.network_factory)

    train_fn = functools.partial(
        ppo.train, **params, network_factory=network_factory,
        progress_fn=progress, seed=args.seed,
        randomization_fn=registry.get_domain_randomizer(args.env),
    )
    make_policy, policy_params, _ = train_fn(
        environment=env, eval_env=registry.load(args.env, config=env_cfg),
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )

    with open(os.path.join(out_dir, "params.pkl"), "wb") as f:
        pickle.dump(jax.device_get(policy_params), f)
    with open(os.path.join(out_dir, "log.json"), "w") as f:
        json.dump({"env": args.env, "ppo": {k: str(v) for k, v in params.items()},
                   "env_cfg": env_cfg.to_dict(), "log": log}, f, indent=1, default=str)
    print(f"done in {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
