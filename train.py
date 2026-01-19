from __future__ import annotations

import argparse
import os
import time
from collections import deque
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch

from env import ShooterEnv, EnvConfig
from agent import PPOAgent, PPOConfig
from esc_exit import esc_pressed


# =========================
# Fixed paths / knobs
# =========================
CKPT_LATEST = "checkpoints/ppo.pt"
CKPT_BEST = "checkpoints/ppo_best.pt"

LOG_EVERY = 20
ROLLING_N = 50

BEST_MIN_EPISODES = 200          # don't update best too early


def parse_args():
    p = argparse.ArgumentParser("shooter_game_ai - PPO trainer/eval")
    # NOTE: episodes = "how many episodes to run from now" (not absolute episode index)
    p.add_argument("--episodes", type=int, default=2000, help="number of episodes to run from now")
    p.add_argument("--no-render", action="store_true", help="disable rendering")
    p.add_argument("--eval", action="store_true", help="evaluation mode (no training), uses ppo_best.pt")
    return p.parse_args()


def _pack_checkpoint(agent: PPOAgent, ppo_cfg: PPOConfig, env_cfg: EnvConfig, ep: int, global_step: int) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "net": agent.net.state_dict(),
        "cfg": vars(ppo_cfg),
        "env_cfg": vars(env_cfg),
        "episode": int(ep),
        "global_step": int(global_step),
    }

    opt = getattr(agent, "opt", None) or getattr(agent, "optimizer", None)
    if opt is not None:
        try:
            payload["opt"] = opt.state_dict()
        except Exception:
            pass
    return payload


def _save_checkpoint(payload: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    print(f"[SAVE] {path}")


def _try_load_checkpoint(agent: PPOAgent, path: str, device: str) -> Tuple[int, int]:
    """
    Returns (start_ep, global_step) after loading.
    If no checkpoint exists, returns (1, 0).
    """
    # --- quick device sanity print (before load)
    try:
        print("net device(before):", next(agent.net.parameters()).device)
    except Exception:
        print("net device(before): <unknown>")
    print("cuda available:", torch.cuda.is_available(), "| requested map_location:", device)

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

    agent.net.load_state_dict(ckpt["net"])

    # IMPORTANT: loading a state_dict does NOT move the model to device.
    # So we explicitly ensure it stays/moves to the intended device.
    try:
        agent.net.to(device)
    except Exception:
        pass

    opt_state = ckpt.get("opt")
    if opt_state is not None:
        opt = getattr(agent, "opt", None) or getattr(agent, "optimizer", None)
        if opt is not None:
            try:
                opt.load_state_dict(opt_state)
                # Also move optimizer state tensors to the same device (important for CUDA resume)
                for st in opt.state.values():
                    for k, v in st.items():
                        if torch.is_tensor(v):
                            st[k] = v.to(device)
            except Exception:
                pass

    last_ep = int(ckpt.get("episode", 0))
    global_step = int(ckpt.get("global_step", 0))
    start_ep = last_ep + 1

    # --- quick device sanity print (after load)
    try:
        print("net device(after):", next(agent.net.parameters()).device)
    except Exception:
        print("net device(after): <unknown>")

    print(f"[CKPT] loaded {path} -> resume from episode {start_ep}, global_step {global_step}")
    return start_ep, global_step



def _try_load_best_metrics(best_path: str, device: str) -> Tuple[Optional[float], Optional[int]]:
    """
    Loads:
      - best_hit_rate50 (lower is better)
      - best_nohit_streak (tie-break stability)
    """
    if not os.path.isfile(best_path):
        return None, None
    try:
        ckpt = torch.load(best_path, map_location=device)
    except Exception:
        return None, None
    if not isinstance(ckpt, dict):
        return None, None

    bhr = None
    bns = None
    if "best_hit_rate50" in ckpt:
        try:
            bhr = float(ckpt["best_hit_rate50"])
        except Exception:
            bhr = None
    if "best_nohit_streak" in ckpt:
        try:
            bns = int(ckpt["best_nohit_streak"])
        except Exception:
            bns = None
    return bhr, bns


def _load_ckpt_into_agent(agent: PPOAgent, path: str, device: str) -> Tuple[int, int]:
    """
    Load checkpoint (net + optimizer if exists). Returns (episode, global_step) from ckpt.
    """
    ckpt = torch.load(path, map_location=device)
    if not (isinstance(ckpt, dict) and "net" in ckpt):
        raise RuntimeError("invalid checkpoint format")

    agent.net.load_state_dict(ckpt["net"])

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
    return last_ep, global_step


def _agent_act_eval(agent: PPOAgent, obs: np.ndarray):
    """
    Eval action helper:
    - If agent.act supports deterministic=..., use it.
    - Else fall back to agent.act(obs).
    Expected return in training: (a, logp, v). For eval we only need action.
    """
    try:
        out = agent.act(obs, deterministic=True)  # type: ignore[arg-type]
    except TypeError:
        out = agent.act(obs)

    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def main():
    args = parse_args()
    os.makedirs("checkpoints", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(0)
    torch.manual_seed(0)

    env_cfg = EnvConfig(seed=0)
    env = ShooterEnv(env_cfg)

    ppo_cfg = PPOConfig(
        device=device,
        rollout_steps=4096,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        ent_coef=0.01,
    )
    agent = PPOAgent(obs_dim=env.obs_dim, act_dim=env.action_dim, cfg=ppo_cfg)

    # -------------------------
    # EVAL MODE (no training)
    # -------------------------
    if args.eval:
        if not os.path.isfile(CKPT_BEST):
            print(f"[EVAL] best checkpoint not found: {CKPT_BEST}")
            print("       run training first to create ppo_best.pt")
            return

        _load_ckpt_into_agent(agent, CKPT_BEST, device)
        print(f"[EVAL] loaded best checkpoint: {CKPT_BEST}")

        obs, _ = env.reset(seed=0)

        target_dt = float(env_cfg.dt)
        next_tick = time.perf_counter()

        survs = []
        hits = []

        for ep in range(1, max(1, int(args.episodes)) + 1):
            done = False
            ep_t = 0.0
            ep_hit = 0

            while not done:
                if esc_pressed():
                    print("[EXIT] ESC pressed -> stopping eval")
                    done = True
                    ep_hit = 1 if bool(getattr(env, "last_hit", False)) else 0
                    break

                a = _agent_act_eval(agent, obs)
                obs, r, done, info = env.step(int(a))

                if not args.no_render:
                    env.render()
                    next_tick += target_dt
                    now = time.perf_counter()
                    sleep_sec = next_tick - now
                    if sleep_sec > 0:
                        time.sleep(sleep_sec)
                    else:
                        next_tick = now

                ep_t = float(info.get("t", ep_t))
                ep_hit = 1 if bool(info.get("hit", False)) else 0

            survs.append(ep_t)
            hits.append(ep_hit)
            obs, _ = env.reset()

            if not args.no_render:
                next_tick = time.perf_counter()

            if ep % LOG_EVERY == 0:
                avg_surv = float(np.mean(survs[-ROLLING_N:])) if survs else 0.0
                hit_rate = float(np.mean(hits[-ROLLING_N:])) if hits else 0.0
                print(f"[EVAL EP {ep:6d}/{args.episodes}] survival_avg{ROLLING_N}={avg_surv:6.2f}s hit_rate{ROLLING_N}={hit_rate:4.2f}")

        avg_surv = float(np.mean(survs)) if survs else 0.0
        hit_rate = float(np.mean(hits)) if hits else 0.0
        print(f"[EVAL DONE] episodes={len(survs)} avg_survival={avg_surv:.3f}s hit_rate={hit_rate:.3f}")
        return

    # -------------------------
    # TRAIN MODE
    # -------------------------
    start_ep, global_step = _try_load_checkpoint(agent, CKPT_LATEST, device)

    run_step0 = global_step
    t0 = time.time()

    end_ep = start_ep + max(1, int(args.episodes)) - 1

    recent_surv = deque(maxlen=ROLLING_N)
    recent_hit = deque(maxlen=ROLLING_N)

    obs, _ = env.reset(seed=0)
    stopped_by_esc = False

    target_dt = float(env_cfg.dt)
    next_tick = time.perf_counter()

    best_hit_rate, best_nohit_streak = _try_load_best_metrics(CKPT_BEST, device)
    if best_hit_rate is None:
        best_hit_rate = 1e9
    if best_nohit_streak is None:
        best_nohit_streak = 0

    print(
        f"[BEST] best_hit_rate{ROLLING_N}(current)={best_hit_rate:.4f} "
        f"best_nohit_streak={best_nohit_streak}  best_path={CKPT_BEST}"
    )

    nohit_streak = 0

    for ep in range(start_ep, end_ep + 1):
        done = False
        ep_t = 0.0
        ep_hit = 0

        while not done:
            if esc_pressed():
                print("[EXIT] ESC pressed -> stopping training")
                stopped_by_esc = True
                done = True
                break

            a, logp, v = agent.act(obs)
            next_obs, r, done, info = env.step(int(a))

            if not args.no_render:
                env.render()

                next_tick += target_dt
                now = time.perf_counter()
                sleep_sec = next_tick - now
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
                else:
                    next_tick = now

            agent.store(obs, int(a), logp, r, done, v)

            obs = next_obs
            ep_t = float(info.get("t", ep_t))
            ep_hit = 1 if bool(info.get("hit", False)) else 0
            global_step += 1

            if agent.buf.ptr >= ppo_cfg.rollout_steps:
                agent.finish_and_update(last_obs=obs, last_done=done)

        if stopped_by_esc:
            break

        recent_surv.append(ep_t)
        recent_hit.append(ep_hit)

        if ep_hit == 0:
            nohit_streak += 1
        else:
            nohit_streak = 0

        obs, _ = env.reset()
        if not args.no_render:
            next_tick = time.perf_counter()

        if ep % LOG_EVERY == 0:
            dt = max(1e-6, time.time() - t0)
            run_steps = global_step - run_step0
            sps = run_steps / dt

            avg_surv = float(np.mean(recent_surv)) if recent_surv else 0.0
            hit_rate = float(np.mean(recent_hit)) if recent_hit else 0.0
            n = len(recent_hit)

            print(
                f"[EP {ep:6d}/{end_ep}] "
                f"survival_avg{ROLLING_N}={avg_surv:6.2f}s hit_rate{ROLLING_N}={hit_rate:4.2f} "
                f"(n={n:2d}) nohit_streak={nohit_streak:5d} "
                f"steps={global_step} SPS={sps:7.1f}"
            )

            # save latest always (resume)
            payload = _pack_checkpoint(agent, ppo_cfg, env_cfg, ep, global_step)
            _save_checkpoint(payload, CKPT_LATEST)

            # best update only when more stable / better
            improved = (ep >= BEST_MIN_EPISODES) and (hit_rate < best_hit_rate - 1e-9)

            tied = (hit_rate <= best_hit_rate + 1e-12) and (hit_rate >= best_hit_rate - 1e-12)
            tie_improved = (ep >= BEST_MIN_EPISODES) and tied and (nohit_streak > best_nohit_streak)

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
                    print(
                        f"[BEST] updated best_hit_rate{ROLLING_N}={best_hit_rate:.4f} "
                        f"(avg_surv{ROLLING_N}={avg_surv:.3f}s, best_nohit_streak={best_nohit_streak})"
                    )
                else:
                    print(
                        f"[BEST] overwritten (tie) hit_rate{ROLLING_N}={best_hit_rate:.4f} "
                        f"with improved stability: best_nohit_streak={best_nohit_streak}"
                    )

    # flush partial rollout
    if getattr(agent, "buf", None) is not None and getattr(agent.buf, "ptr", 0) > 0:
        try:
            agent.finish_and_update(last_obs=obs, last_done=False)
        except Exception:
            pass

    # save once on ESC stop
    if stopped_by_esc:
        payload = _pack_checkpoint(agent, ppo_cfg, env_cfg, ep, global_step)
        _save_checkpoint(payload, CKPT_LATEST)

    print("done.")


if __name__ == "__main__":
    main()
