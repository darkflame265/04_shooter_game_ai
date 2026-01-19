from __future__ import annotations

import argparse
import os
import time
from collections import deque
from typing import Any, Dict, Tuple

import numpy as np
import torch

from env import ShooterEnv, EnvConfig
from agent import PPOAgent, PPOConfig
from esc_exit import esc_pressed


def parse_args():
    p = argparse.ArgumentParser("shooter_game_ai - PPO trainer")
    # NOTE: episodes = "how many episodes to run from now" (not absolute episode index)
    p.add_argument("--episodes", type=int, default=2000, help="number of episodes to run from now")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    p.add_argument("--rollout-steps", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.01)

    p.add_argument("--no-render", action="store_true", help="disable rendering (default recommended)")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save", type=str, default="checkpoints/ppo.pt")
    return p.parse_args()


def _save_checkpoint(agent: PPOAgent, ppo_cfg: PPOConfig, env_cfg: EnvConfig, ep: int, global_step: int, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: Dict[str, Any] = {
        "net": agent.net.state_dict(),
        "cfg": vars(ppo_cfg),
        "env_cfg": vars(env_cfg),
        "episode": int(ep),
        "global_step": int(global_step),
    }

    # Optional: also save optimizer if agent exposes it (keeps learning momentum on resume)
    opt = getattr(agent, "opt", None) or getattr(agent, "optimizer", None)
    if opt is not None:
        try:
            payload["opt"] = opt.state_dict()
        except Exception:
            pass

    torch.save(payload, path)
    print(f"[SAVE] {path}")


def _try_load_checkpoint(agent: PPOAgent, path: str, device: str) -> Tuple[int, int]:
    """
    Returns (start_ep, global_step) after loading.
    If no checkpoint exists, returns (1, 0).
    """
    if not os.path.isfile(path):
        print(f"[CKPT] no checkpoint found at {path} -> starting fresh")
        return 1, 0

    try:
        ckpt = torch.load(path, map_location=device)
    except Exception as e:
        print(f"[CKPT] failed to load {path} -> starting fresh ({e})")
        return 1, 0

    if not (isinstance(ckpt, dict) and "net" in ckpt):
        print(f"[CKPT] invalid checkpoint format -> starting fresh")
        return 1, 0

    # load net
    agent.net.load_state_dict(ckpt["net"])

    # load optimizer if present and agent exposes it
    opt_state = ckpt.get("opt")
    if opt_state is not None:
        opt = getattr(agent, "opt", None) or getattr(agent, "optimizer", None)
        if opt is not None:
            try:
                opt.load_state_dict(opt_state)
            except Exception:
                pass

    last_ep = int(ckpt.get("episode", 0))
    global_step = int(ckpt.get("global_step", 0))
    start_ep = last_ep + 1

    print(f"[CKPT] loaded {path} -> resume from episode {start_ep}, global_step {global_step}")
    return start_ep, global_step


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_cfg = EnvConfig(seed=args.seed)
    env = ShooterEnv(env_cfg)

    ppo_cfg = PPOConfig(
        device=args.device,
        rollout_steps=args.rollout_steps,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip,
        ent_coef=args.ent,
    )
    agent = PPOAgent(obs_dim=env.obs_dim, act_dim=env.action_dim, cfg=ppo_cfg)

    # ✅ auto-resume: if checkpoint exists, continue from it
    start_ep, global_step = _try_load_checkpoint(agent, args.save, args.device)

    # ✅ FIX: interpret --episodes as "how many to run from now"
    end_ep = start_ep + max(1, int(args.episodes)) - 1

    recent_surv = deque(maxlen=50)
    recent_hit = deque(maxlen=50)

    t0 = time.time()
    obs, _ = env.reset(seed=args.seed)

    stopped_by_esc = False

    # ---- realtime throttle (only used in render mode)
    target_dt = float(env_cfg.dt)          # 1/60 sec
    next_tick = time.perf_counter()        # monotonic clock for stable pacing

    for ep in range(start_ep, end_ep + 1):
        done = False
        ep_rew = 0.0
        ep_steps = 0
        ep_t = 0.0
        ep_hit = 0

        while not done:
            # ✅ ESC to stop (minimal intrusion)
            if esc_pressed():
                print("[EXIT] ESC pressed -> stopping training")
                stopped_by_esc = True
                done = True
                break

            a, logp, v = agent.act(obs)
            next_obs, r, done, info = env.step(a)

            if not args.no_render:
                env.render()

                # throttle sim loop to real-time 60Hz (or env_cfg.dt)
                next_tick += target_dt
                now = time.perf_counter()
                sleep_sec = next_tick - now
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
                else:
                    next_tick = now

            agent.store(obs, a, logp, r, done, v)

            obs = next_obs
            ep_rew += r
            ep_steps += 1
            ep_t = float(info["t"])
            ep_hit = 1 if bool(info["hit"]) else 0

            global_step += 1

            # PPO update when rollout buffer is full
            if agent.buf.ptr >= ppo_cfg.rollout_steps:
                agent.finish_and_update(last_obs=obs, last_done=done)

        if stopped_by_esc:
            break

        # reset env for next episode
        recent_surv.append(ep_t)
        recent_hit.append(ep_hit)

        obs, _ = env.reset()

        # reset pacing per-episode
        if not args.no_render:
            next_tick = time.perf_counter()

        if ep % args.log_every == 0:
            dt = max(1e-6, time.time() - t0)
            sps = global_step / dt
            avg_surv = float(np.mean(recent_surv)) if recent_surv else 0.0
            hit_rate = float(np.mean(recent_hit)) if recent_hit else 0.0
            print(
                f"[EP {ep:6d}/{end_ep}] "
                f"survival_avg50={avg_surv:6.2f}s hit_rate50={hit_rate:4.2f} "
                f"steps={global_step} SPS={sps:7.1f}"
            )
            _save_checkpoint(agent, ppo_cfg, env_cfg, ep, global_step, args.save)

    # flush partial rollout (optional)
    if agent.buf.ptr > 0:
        agent.finish_and_update(last_obs=obs, last_done=False)

    # save once on ESC stop (so you don't lose progress)
    if stopped_by_esc:
        _save_checkpoint(agent, ppo_cfg, env_cfg, ep, global_step, args.save)

    print("done.")


if __name__ == "__main__":
    main()
