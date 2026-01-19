from __future__ import annotations

import argparse
import os
import time
from collections import deque
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch

from env_vec_torch import VecShooterEnvTorch, EnvConfig
from agent_vec import PPOAgentVec, PPOConfig
from esc_exit import esc_pressed


CKPT_LATEST = "checkpoints/ppo.pt"
CKPT_BEST = "checkpoints/ppo_best.pt"

LOG_EVERY = 20
ROLLING_N = 50
BEST_MIN_EPISODES = 200


def parse_args():
    p = argparse.ArgumentParser("shooter_game_ai - PPO trainer/eval (vec-gpu)")
    p.add_argument("--episodes", type=int, default=20000, help="how many episode terminations to process")
    p.add_argument("--no-render", action="store_true", help="training mode (vec env). if omitted => render-only viewer mode")
    p.add_argument("--eval", action="store_true", help="eval mode uses ppo_best.pt (runs vec env, no tkinter)")
    p.add_argument("--n-envs", type=int, default=256, help="number of parallel envs on GPU (training/eval only)")
    return p.parse_args()


def _pack_checkpoint(agent: PPOAgentVec, ppo_cfg: PPOConfig, env_cfg: EnvConfig, ep: int, global_step: int) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "net": agent.net.state_dict(),
        "cfg": vars(ppo_cfg),
        "env_cfg": vars(env_cfg),
        "episode": int(ep),
        "global_step": int(global_step),
    }
    try:
        payload["opt"] = agent.opt.state_dict()
    except Exception:
        pass
    return payload


def _save_checkpoint(payload: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    print(f"[SAVE] {path}")


def _try_load_checkpoint(agent: PPOAgentVec, path: str, device: str) -> Tuple[int, int]:
    if not os.path.isfile(path):
        print(f"[CKPT] no checkpoint found at {path} -> starting fresh")
        return 1, 0

    ckpt = torch.load(path, map_location=device)
    if not (isinstance(ckpt, dict) and "net" in ckpt):
        print("[CKPT] invalid checkpoint format -> starting fresh")
        return 1, 0

    agent.net.load_state_dict(ckpt["net"])
    agent.net.to(device)

    opt_state = ckpt.get("opt")
    if opt_state is not None:
        try:
            agent.opt.load_state_dict(opt_state)
            for st in agent.opt.state.values():
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.to(device)
        except Exception:
            pass

    last_ep = int(ckpt.get("episode", 0))
    global_step = int(ckpt.get("global_step", 0))
    start_ep = last_ep + 1
    print(f"[CKPT] loaded {path} -> resume from episode {start_ep}, global_step {global_step}")
    return start_ep, global_step


def _try_load_best_metrics(best_path: str, device: str) -> Tuple[Optional[float], Optional[int]]:
    if not os.path.isfile(best_path):
        return None, None
    try:
        ckpt = torch.load(best_path, map_location=device)
    except Exception:
        return None, None
    if not isinstance(ckpt, dict):
        return None, None

    bhr = ckpt.get("best_hit_rate50", None)
    bns = ckpt.get("best_nohit_streak", None)
    try:
        bhr = float(bhr) if bhr is not None else None
    except Exception:
        bhr = None
    try:
        bns = int(bns) if bns is not None else None
    except Exception:
        bns = None
    return bhr, bns


@torch.no_grad()
def _eval_vec(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = EnvConfig(seed=0)
    env = VecShooterEnvTorch(env_cfg, n_envs=args.n_envs, device=device)

    ppo_cfg = PPOConfig(device=device)
    agent = PPOAgentVec(obs_dim=env.obs_dim, act_dim=env.action_dim, n_envs=args.n_envs, cfg=ppo_cfg)

    if not os.path.isfile(CKPT_BEST):
        print(f"[EVAL] best checkpoint not found: {CKPT_BEST}")
        return

    ckpt = torch.load(CKPT_BEST, map_location=device)
    agent.net.load_state_dict(ckpt["net"])
    agent.net.to(device)
    agent.net.eval()

    obs, _ = env.reset()

    done_eps = 0
    survs = []
    hits = []
    ep_hit = torch.zeros((args.n_envs,), device=device, dtype=torch.bool)

    while done_eps < int(args.episodes):
        if esc_pressed():
            print("[EXIT] ESC pressed -> stopping eval")
            break

        # NOTE: PPOAgentVec.act() should already be torch-native. (No numpy copies)
        a, _, _ = agent.act(obs)
        obs, rew, done, info = env.step(a)

        ep_hit = ep_hit | info["hit"]

        if torch.any(done):
            d = done
            t_done = info["t"][d].detach().float().cpu().numpy()
            h_done = ep_hit[d].detach().float().cpu().numpy()
            survs.extend(t_done.tolist())
            hits.extend(h_done.tolist())
            done_eps += int(d.sum().item())

            obs_reset, _ = env.reset(mask=d)
            obs[d] = obs_reset[d]
            ep_hit[d] = False

        if done_eps > 0 and (done_eps % LOG_EVERY == 0):
            avg_surv = float(np.mean(survs[-ROLLING_N:])) if survs else 0.0
            hit_rate = float(np.mean(hits[-ROLLING_N:])) if hits else 0.0
            print(f"[EVAL EP {done_eps:7d}] survival_avg{ROLLING_N}={avg_surv:6.2f}s hit_rate{ROLLING_N}={hit_rate:4.2f}")

    avg_surv = float(np.mean(survs)) if survs else 0.0
    hit_rate = float(np.mean(hits)) if hits else 0.0
    print(f"[EVAL DONE] episodes={len(survs)} avg_survival={avg_surv:.3f}s hit_rate={hit_rate:.3f}")


@torch.no_grad()
def _render_viewer(args):
    """
    Render-only mode:
      - env = 1
      - no training/update
      - load CKPT_BEST (preferred), else CKPT_LATEST
      - show tkinter window via env.render()
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[RENDER] device:", device)

    env_cfg = EnvConfig(seed=0)
    env = VecShooterEnvTorch(env_cfg, n_envs=1, device=device)

    ppo_cfg = PPOConfig(device=device)
    agent = PPOAgentVec(obs_dim=env.obs_dim, act_dim=env.action_dim, n_envs=1, cfg=ppo_cfg)

    ckpt_path = CKPT_BEST if os.path.isfile(CKPT_BEST) else CKPT_LATEST
    if not os.path.isfile(ckpt_path):
        print(f"[RENDER] checkpoint not found: {CKPT_BEST} or {CKPT_LATEST}")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    if not (isinstance(ckpt, dict) and "net" in ckpt):
        print("[RENDER] invalid checkpoint format")
        return

    agent.net.load_state_dict(ckpt["net"])
    agent.net.to(device)
    agent.net.eval()

    print(f"[RENDER] loaded: {ckpt_path}")

    obs, _ = env.reset()

    # viewer loop: run until episodes terminations processed (args.episodes)
    done_eps = 0
    ep_hit = torch.zeros((1,), device=device, dtype=torch.bool)

    # render pacing: let env.render() internally cap, but we still avoid a busy-loop
    while done_eps < int(args.episodes):
        if esc_pressed():
            print("[EXIT] ESC pressed -> stopping render")
            break

        # action
        a, _, _ = agent.act(obs)
        obs, rew, done, info = env.step(a)

        ep_hit = ep_hit | info["hit"]

        # draw
        env.render()

        if bool(done.item()):
            done_eps += 1
            t_done = float(info["t"][0].item())
            h_done = int(ep_hit[0].item())
            print(f"[RENDER EP {done_eps:6d}] survival={t_done:6.2f}s hit={h_done}")
            obs, _ = env.reset(mask=done)
            ep_hit[0] = False

        # light sleep to reduce CPU spin (render() already caps visuals)
        time.sleep(0.001)

    try:
        env.close()
    except Exception:
        pass

    print("done.")


def main():
    args = parse_args()
    os.makedirs("checkpoints", exist_ok=True)

    # If user did NOT pass --no-render:
    # treat as "viewer" mode (single env, tkinter render, no training).
    if not args.no_render and not args.eval:
        _render_viewer(args)
        return

    if args.eval:
        _eval_vec(args)
        return

    # -------------------------
    # TRAIN MODE (vec)
    # -------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[DEV]", device)

    np.random.seed(0)
    torch.manual_seed(0)

    env_cfg = EnvConfig(seed=0)
    env = VecShooterEnvTorch(env_cfg, n_envs=args.n_envs, device=device)

    ppo_cfg = PPOConfig(
        device=device,
        rollout_steps=256,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        ent_coef=0.01,
        minibatch_size=4096,
        update_epochs=6,
    )
    agent = PPOAgentVec(obs_dim=env.obs_dim, act_dim=env.action_dim, n_envs=args.n_envs, cfg=ppo_cfg)

    start_ep, global_step = _try_load_checkpoint(agent, CKPT_LATEST, device)

    recent_surv = deque(maxlen=ROLLING_N)
    recent_hit = deque(maxlen=ROLLING_N)

    best_hit_rate, best_nohit_streak = _try_load_best_metrics(CKPT_BEST, device)
    if best_hit_rate is None:
        best_hit_rate = 1e9
    if best_nohit_streak is None:
        best_nohit_streak = 0

    print(f"[BEST] best_hit_rate{ROLLING_N}(current)={best_hit_rate:.4f} best_nohit_streak={best_nohit_streak} best_path={CKPT_BEST}")

    obs, _ = env.reset()
    ep_hit = torch.zeros((args.n_envs,), device=device, dtype=torch.bool)

    nohit_streak = 0
    done_eps = 0

    run_step0 = global_step
    t0 = time.time()

    while done_eps < int(args.episodes):
        if esc_pressed():
            print("[EXIT] ESC pressed -> stopping training")
            break

        agent.buf.reset()
        for _ in range(ppo_cfg.rollout_steps):
            a, logp, v = agent.act(obs)
            next_obs, rew, done, info = env.step(a)

            agent.store(obs, a, logp, rew, done, v)
            ep_hit = ep_hit | info["hit"]

            if torch.any(done):
                d = done

                t_done = info["t"][d].detach().float().cpu().numpy()
                h_done = ep_hit[d].detach().float().cpu().numpy()
                for tt, hh in zip(t_done.tolist(), h_done.tolist()):
                    recent_surv.append(float(tt))
                    recent_hit.append(float(hh))
                    done_eps += 1

                obs_reset, _ = env.reset(mask=d)
                next_obs[d] = obs_reset[d]
                ep_hit[d] = False

            obs = next_obs
            global_step += int(args.n_envs)

        agent.finish_and_update(
            last_obs=obs,
            last_done=torch.zeros((args.n_envs,), device=device, dtype=torch.bool),
        )

        if done_eps >= LOG_EVERY and (done_eps % LOG_EVERY == 0):
            dt = max(1e-6, time.time() - t0)
            run_steps = global_step - run_step0
            sps = run_steps / dt

            avg_surv = float(np.mean(recent_surv)) if recent_surv else 0.0
            hit_rate = float(np.mean(recent_hit)) if recent_hit else 0.0
            n = len(recent_hit)

            print(
                f"[EP_DONE {done_eps:7d}/{args.episodes}] "
                f"survival_avg{ROLLING_N}={avg_surv:6.2f}s hit_rate{ROLLING_N}={hit_rate:4.2f} "
                f"(n={n:2d}) steps={global_step} SPS={sps:7.1f}"
            )

            payload = _pack_checkpoint(agent, ppo_cfg, env_cfg, done_eps, global_step)
            _save_checkpoint(payload, CKPT_LATEST)

            if len(recent_hit) > 0:
                if recent_hit[-1] == 0.0:
                    nohit_streak += 1
                else:
                    nohit_streak = 0

            improved = (done_eps >= BEST_MIN_EPISODES) and (hit_rate < best_hit_rate - 1e-9)
            tied = abs(hit_rate - best_hit_rate) <= 1e-12
            tie_improved = (done_eps >= BEST_MIN_EPISODES) and tied and (nohit_streak > best_nohit_streak)

            if improved or tie_improved:
                if improved:
                    best_hit_rate = hit_rate
                best_nohit_streak = int(nohit_streak)

                best_payload = dict(payload)
                best_payload["best_hit_rate50"] = float(best_hit_rate)
                best_payload["best_avg50"] = float(avg_surv)
                best_payload["best_nohit_streak"] = int(best_nohit_streak)

                _save_checkpoint(best_payload, CKPT_BEST)

                if improved:
                    print(f"[BEST] updated best_hit_rate{ROLLING_N}={best_hit_rate:.4f} (best_nohit_streak={best_nohit_streak})")
                else:
                    print(f"[BEST] overwritten (tie) hit_rate{ROLLING_N}={best_hit_rate:.4f} with improved stability: best_nohit_streak={best_nohit_streak}")

    print("done.")


if __name__ == "__main__":
    main()
