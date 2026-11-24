"""Roll out a trained policy and save a video + a velocity-tracking plot.

    python rollout.py --run flat_baseline --vx 1.0 --vy 0.0 --wz 0.0
"""
import argparse
import functools
import json
import os
import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mediapy
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--vx", type=float, default=1.0)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--wz", type=float, default=0.0)
    p.add_argument("--seconds", type=float, default=10.0)
    args = p.parse_args()

    out_dir = os.path.join("runs", args.run)
    env_name = json.load(open(os.path.join(out_dir, "log.json")))["env"]
    env = registry.load(env_name)
    ppo_cfg = locomotion_params.brax_ppo_config(env_name)
    params = pickle.load(open(os.path.join(out_dir, "params.pkl"), "rb"))

    # Brax normalises observations inside the policy: rebuild it with the saved stats.
    nets = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        preprocess_observations_fn=running_statistics.normalize, **ppo_cfg.network_factory)
    policy = jax.jit(ppo_networks.make_inference_fn(nets)(params, deterministic=True))

    reset, step = jax.jit(env.reset), jax.jit(env.step)
    state = reset(jax.random.PRNGKey(0))
    cmd = jnp.array([args.vx, args.vy, args.wz])
    rng = jax.random.PRNGKey(1)
    traj, vel = [state], []
    for _ in range(int(args.seconds / env.dt)):
        state.info["command"] = cmd
        rng, k = jax.random.split(rng)
        action, _ = policy(state.obs, k)
        state = step(state, action)
        traj.append(state)
        vel.append(env.get_local_linvel(state.data)[:2].tolist())

    frames = env.render(traj[::2], height=360, width=640)
    mediapy.write_video(os.path.join(out_dir, "rollout.mp4"), frames, fps=1.0 / env.dt / 2)

    t = [i * env.dt for i in range(len(vel))]
    plt.plot(t, [v[0] for v in vel], label="v_x measured")
    plt.plot(t, [v[1] for v in vel], label="v_y measured")
    plt.axhline(args.vx, ls="--", c="C0", label="v_x command")
    plt.axhline(args.vy, ls="--", c="C1", label="v_y command")
    plt.xlabel("time (s)"); plt.ylabel("m/s"); plt.legend(); plt.title(f"{args.run}: velocity tracking")
    plt.savefig(os.path.join(out_dir, "tracking.png"), dpi=150)
    print("saved", out_dir)


if __name__ == "__main__":
    main()
