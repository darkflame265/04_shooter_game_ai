from __future__ import annotations

import argparse
import os
import time
import random
from collections import deque
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch

from env_vec_torch import VecShooterEnvTorch, EnvConfig
from agent_vec import PPOAgentVec, PPOConfig
from esc_exit import esc_pressed


CKPT_LATEST = "checkpoints/ppo.pt"
CKPT_BEST = "checkpoints/ppo_best.pt"

LOG_EVERY = 50
ROLLING_N = 50
BEST_MIN_EPISODES = 200

# -------------------------
# Auto difficulty (spawn-rate)  ✅ spawn을 +5.0/s씩 올림
# -------------------------
AUTO_DIFF_ENABLED = True
AUTO_DIFF_TARGET_HIT = 0.20
AUTO_DIFF_STEP_SPAWN = 5.0
AUTO_DIFF_COOLDOWN_LOGS = 1
AUTO_DIFF_MIN_READY = ROLLING_N

# -------------------------
# Auto ENT tuning only
# -------------------------
AUTO_ENT_ENABLED = True

PLATEAU_WINDOW_LOGS = 6
PLATEAU_MIN_SURV_IMPROVE = 1.0

HIT_IMPROVE_EPS = 0.03
HIT_WORSEN_EPS = 0.03

ENT_UP = 1.01
ENT_DOWN = 0.995
ENT_MIN = 0.003
ENT_MAX = 0.100

EPS_TIE = 1e-12

# -------------------------
# BEST rule
# -------------------------
BEST_SPAWN_EPS = 1e-6        # spawn 동률 판단 오차
BEST_HIT_MARGIN = 1e-6       # hit_rate 개선 최소 폭(너무 작은 흔들림 방지)


def parse_args():
    p = argparse.ArgumentParser("shooter_game_ai - PPO trainer (vec-gpu)")
    p.add_argument("--episodes", type=int, default=20000, help="how many episode terminations to process / render")
    p.add_argument(
        "--no-render",
        action="store_true",
        help="training mode (vec env). if omitted => render-only performance test mode",
    )
    p.add_argument("--n-envs", type=int, default=256, help="number of parallel envs on GPU (training only)")
    return p.parse_args()


# -------------------------
# RNG helpers
# -------------------------
def _capture_rng_state(device: str, env: Optional[VecShooterEnvTorch] = None) -> Dict[str, Any]:
    st: Dict[str, Any] = {}
    st["rng_python"] = random.getstate()
    st["rng_numpy"] = np.random.get_state()
    st["rng_torch_cpu"] = torch.get_rng_state()

    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            st["rng_torch_cuda_all"] = torch.cuda.get_rng_state_all()
        except Exception:
            st["rng_torch_cuda_all"] = None
    else:
        st["rng_torch_cuda_all"] = None

    if env is not None:
        try:
            st["rng_env_gen_state"] = env.get_rng_state()
        except Exception:
            st["rng_env_gen_state"] = None
    else:
        st["rng_env_gen_state"] = None

    return st


def _restore_rng_state(st: Dict[str, Any], device: str, env: Optional[VecShooterEnvTorch] = None) -> None:
    if not isinstance(st, dict):
        return

    try:
        if "rng_python" in st and st["rng_python"] is not None:
            random.setstate(st["rng_python"])
    except Exception:
        pass

    try:
        if "rng_numpy" in st and st["rng_numpy"] is not None:
            np.random.set_state(st["rng_numpy"])
    except Exception:
        pass

    try:
        if "rng_torch_cpu" in st and st["rng_torch_cpu"] is not None:
            torch.set_rng_state(st["rng_torch_cpu"])
    except Exception:
        pass

    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            cuda_all = st.get("rng_torch_cuda_all", None)
            if cuda_all is not None:
                torch.cuda.set_rng_state_all(cuda_all)
        except Exception:
            pass

    if env is not None:
        try:
            env_state = st.get("rng_env_gen_state", None)
            if env_state is not None:
                env.set_rng_state(env_state)
        except Exception:
            pass


def _pack_checkpoint(
    agent: PPOAgentVec,
    ppo_cfg: PPOConfig,
    env_cfg: EnvConfig,
    ep: int,
    global_step: int,
    manual_spawn: float,
    device: str,
    env: Optional[VecShooterEnvTorch] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "net": agent.net.state_dict(),
        "cfg": vars(ppo_cfg),
        "env_cfg": vars(env_cfg),
        "episode": int(ep),
        "global_step": int(global_step),
        "manual_spawn": float(manual_spawn),
        "rng_state": _capture_rng_state(device=device, env=env),
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


def _try_load_checkpoint(
    agent: PPOAgentVec,
    path: str,
    device: str,
    env: Optional[VecShooterEnvTorch] = None,
) -> Tuple[int, int, Optional[float], Optional[Dict[str, Any]]]:
    if not os.path.isfile(path):
        print(f"[CKPT] no checkpoint found at {path} -> starting fresh")
        return 1, 0, None, None

    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not (isinstance(ckpt, dict) and "net" in ckpt):
        print("[CKPT] invalid checkpoint format -> starting fresh")
        return 1, 0, None, None

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

    manual_spawn = ckpt.get("manual_spawn", None)
    try:
        manual_spawn = float(manual_spawn) if manual_spawn is not None else None
    except Exception:
        manual_spawn = None

    rng_state = ckpt.get("rng_state", None)
    if isinstance(rng_state, dict):
        _restore_rng_state(rng_state, device=device, env=env)

    start_ep = last_ep + 1
    print(f"[CKPT] loaded {path} -> resume from episode {start_ep}, global_step {global_step}")
    if manual_spawn is not None:
        print(f"[CKPT] loaded manual_spawn={manual_spawn:.3f}/s")
    if isinstance(rng_state, dict):
        print("[CKPT] restored RNG state")
    else:
        print("[CKPT] no RNG state in checkpoint (old ckpt)")

    return start_ep, global_step, manual_spawn, rng_state


def _try_load_best_metrics(best_path: str, device: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns: (best_spawn, best_hit_rate50, best_avg50)
    """
    if not os.path.isfile(best_path):
        return None, None, None
    try:
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
    except Exception:
        return None, None, None
    if not isinstance(ckpt, dict):
        return None, None, None

    bs = ckpt.get("best_spawn", ckpt.get("manual_spawn", None))
    bhr = ckpt.get("best_hit_rate50", None)
    bav = ckpt.get("best_avg50", None)

    try:
        bs = float(bs) if bs is not None else None
    except Exception:
        bs = None
    try:
        bhr = float(bhr) if bhr is not None else None
    except Exception:
        bhr = None
    try:
        bav = float(bav) if bav is not None else None
    except Exception:
        bav = None

    return bs, bhr, bav


@torch.no_grad()
def _render_viewer(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[RENDER] device:", device)

    env_cfg = EnvConfig(seed=0)
    env_cfg.curriculum = False
    env = VecShooterEnvTorch(env_cfg, n_envs=1, device=device)

    ppo_cfg = PPOConfig(device=device)
    agent = PPOAgentVec(obs_dim=env.obs_dim, act_dim=env.action_dim, n_envs=1, cfg=ppo_cfg)

    ckpt_path = CKPT_BEST if os.path.isfile(CKPT_BEST) else CKPT_LATEST
    if not os.path.isfile(ckpt_path):
        print(f"[RENDER] checkpoint not found: {CKPT_BEST} or {CKPT_LATEST}")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not (isinstance(ckpt, dict) and "net" in ckpt):
        print("[RENDER] invalid checkpoint format")
        return

    agent.net.load_state_dict(ckpt["net"])
    agent.net.to(device)
    agent.net.eval()

    rng_state = ckpt.get("rng_state", None)
    if isinstance(rng_state, dict):
        _restore_rng_state(rng_state, device=device, env=env)
        print("[RENDER] restored RNG state from checkpoint")

    # ✅ manual_spawn 우선
    spawn = ckpt.get("manual_spawn", None)
    if spawn is None:
        spawn = float(env_cfg.spawn_rate_start)
        print(f"[RENDER] WARNING: manual_spawn missing in ckpt -> using spawn_rate_start={spawn:.2f}/s")
    else:
        try:
            spawn = float(spawn)
        except Exception:
            spawn = float(env_cfg.spawn_rate_start)
            print(f"[RENDER] WARNING: manual_spawn invalid -> using spawn_rate_start={spawn:.2f}/s")

    spawn = float(np.clip(spawn, 0.0, float(env_cfg.spawn_rate)))
    env.set_manual_difficulty(spawn, enabled=True)

    print(f"[RENDER] loaded: {ckpt_path}")
    print(f"[RENDER] fixed spawn from ckpt: spawn={env.get_spawn_rate_s():.2f}/s")

    obs, _ = env.reset()
    done_eps = 0
    ep_hit = torch.zeros((1,), device=device, dtype=torch.bool)

    while done_eps < int(args.episodes):
        if esc_pressed():
            print("[EXIT] ESC pressed -> stopping render")
            break

        a, _, _ = agent.act(obs)
        obs, rew, done, info = env.step(a)
        ep_hit = ep_hit | info["hit"]

        env.render()

        if bool(done.item()):
            done_eps += 1
            t_done = float(info["t"][0].item())
            h_done = int(ep_hit[0].item())
            print(f"[RENDER EP {done_eps:6d}] survival={t_done:6.2f}s hit={h_done}")
            obs, _ = env.reset(mask=done)
            ep_hit[0] = False

        time.sleep(0.001)

    try:
        env.close()
    except Exception:
        pass

    print("done.")


def _get_ent_coef(agent: PPOAgentVec, ppo_cfg: PPOConfig) -> float:
    if hasattr(agent, "cfg") and hasattr(agent.cfg, "ent_coef"):
        try:
            return float(agent.cfg.ent_coef)
        except Exception:
            pass
    try:
        return float(ppo_cfg.ent_coef)
    except Exception:
        return float("nan")


def _set_ent_coef(agent: PPOAgentVec, ppo_cfg: PPOConfig, ent: float) -> None:
    ent = float(ent)
    if hasattr(agent, "cfg") and hasattr(agent.cfg, "ent_coef"):
        try:
            agent.cfg.ent_coef = ent
        except Exception:
            pass
    try:
        ppo_cfg.ent_coef = ent
    except Exception:
        pass


def main():
    args = parse_args()
    os.makedirs("checkpoints", exist_ok=True)

    if not args.no_render:
        _render_viewer(args)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[DEV]", device)

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    env_cfg = EnvConfig(seed=0)
    env_cfg.curriculum = False
    env = VecShooterEnvTorch(env_cfg, n_envs=args.n_envs, device=device)

    ppo_cfg = PPOConfig(
        device=device,
        rollout_steps=1024,
        lr=1e-4,
        gamma=0.995,
        gae_lambda=0.97,
        clip_ratio=0.12,
        ent_coef=0.003,
        minibatch_size=4096,
        update_epochs=4,
    )
    agent = PPOAgentVec(obs_dim=env.obs_dim, act_dim=env.action_dim, n_envs=args.n_envs, cfg=ppo_cfg)

    start_ep, global_step, ckpt_spawn, _rng = _try_load_checkpoint(agent, CKPT_LATEST, device, env=env)
    base_global_episode = int(max(0, start_ep - 1))

    recent_surv = deque(maxlen=ROLLING_N)
    recent_hit = deque(maxlen=ROLLING_N)

    hist_avg_surv = deque(maxlen=PLATEAU_WINDOW_LOGS)
    hist_hit_rate = deque(maxlen=PLATEAU_WINDOW_LOGS)

    # BEST metrics load
    best_spawn, best_hit_rate, best_avg50 = _try_load_best_metrics(CKPT_BEST, device)
    if best_spawn is None:
        best_spawn = -1.0
    if best_hit_rate is None:
        best_hit_rate = 1e9
    if best_avg50 is None:
        best_avg50 = 0.0

    print(
        f"[BEST] best_spawn={best_spawn:.2f}/s "
        f"best_hit_rate{ROLLING_N}={best_hit_rate:.4f} best_avg{ROLLING_N}={best_avg50:.2f}s "
        f"best_path={CKPT_BEST}"
    )

    manual_spawn = float(ckpt_spawn) if ckpt_spawn is not None else float(env_cfg.spawn_rate_start)
    manual_spawn = float(np.clip(manual_spawn, 0.0, float(env_cfg.spawn_rate)))
    env.set_manual_difficulty(manual_spawn, enabled=True)

    diff_cooldown = 0
    print(f"[DIFF] start spawn={env.get_spawn_rate_s():.2f}/s (step={AUTO_DIFF_STEP_SPAWN:.2f}/s)")

    obs, _ = env.reset()
    ep_hit = torch.zeros((args.n_envs,), device=device, dtype=torch.bool)

    done_eps = 0
    next_log_ep = LOG_EVERY

    run_step0 = global_step
    t0 = time.time()

    last_ent_msg = ""

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

        # PPO update
        agent.finish_and_update(
            last_obs=obs,
            last_done=torch.zeros((args.n_envs,), device=device, dtype=torch.bool),
        )

        if done_eps >= next_log_ep:
            dt = max(1e-6, time.time() - t0)
            run_steps = global_step - run_step0
            sps = run_steps / dt

            avg_surv = float(np.mean(recent_surv)) if recent_surv else 0.0
            hit_rate = float(np.mean(recent_hit)) if recent_hit else 0.0
            n = len(recent_hit)

            spawn_s = float(env.get_spawn_rate_s())
            global_episode = base_global_episode + int(done_eps)

            hist_avg_surv.append(avg_surv)
            hist_hit_rate.append(hit_rate)

            # -------------------------
            # ENT auto-tuning only
            # -------------------------
            ent_msg = ""
            if AUTO_ENT_ENABLED and (n >= AUTO_DIFF_MIN_READY):
                plateau = False
                improved = False
                worsened = False

                if len(hist_hit_rate) >= PLATEAU_WINDOW_LOGS:
                    prev = list(hist_hit_rate)[:-1]
                    cur = float(hist_hit_rate[-1])
                    prev_best = float(min(prev))  # 낮을수록 좋음

                    improved = (cur <= prev_best - HIT_IMPROVE_EPS)
                    worsened = (cur >= prev_best + HIT_WORSEN_EPS)

                    if len(hist_avg_surv) >= PLATEAU_WINDOW_LOGS:
                        surv_span = float(max(hist_avg_surv) - min(hist_avg_surv))
                        if (not improved) and (not worsened) and (surv_span < PLATEAU_MIN_SURV_IMPROVE):
                            plateau = True

                ent0 = _get_ent_coef(agent, ppo_cfg)
                ent1 = ent0

                if worsened:
                    ent1 = min(ENT_MAX, ent0 * ENT_UP)
                elif plateau:
                    ent1 = min(ENT_MAX, ent0 * ENT_UP)
                elif improved:
                    ent1 = max(ENT_MIN, ent0 * ENT_DOWN)

                if abs(ent1 - ent0) > 1e-12:
                    _set_ent_coef(agent, ppo_cfg, ent1)
                    tag = "worsen" if worsened else ("plateau" if plateau else "improve")
                    ent_msg = f" ent{ent0:.4f}->{ent1:.4f}({tag})"

            if ent_msg:
                last_ent_msg = ent_msg

            extra = f" |TUNE:{last_ent_msg}" if last_ent_msg else ""

            print(
                f"[EP_DONE {next_log_ep:7d}/{args.episodes}] "
                f"surv_avg{ROLLING_N}={avg_surv:5.2f}s hit_rate{ROLLING_N}={hit_rate:5.3f} "
                f"spawn={spawn_s:7.2f}/s "
                f"global_episode={global_episode} "
                f"(n={n:2d}) steps={global_step} SPS={sps:7.1f}"
                f"{extra}"
            )

            # save latest
            payload = _pack_checkpoint(
                agent=agent,
                ppo_cfg=ppo_cfg,
                env_cfg=env_cfg,
                ep=next_log_ep,
                global_step=global_step,
                manual_spawn=manual_spawn,
                device=device,
                env=env,
            )
            _save_checkpoint(payload, CKPT_LATEST)

            # -------------------------
            # ✅ BEST update
            # 1) spawn 증가가 최우선
            # 2) spawn 같으면 hit_rate 낮아지면(best)
            # 3) 위 둘 다 tie면 avg_surv가 높으면(best)
            # -------------------------
            window_ready = (n >= AUTO_DIFF_MIN_READY)
            eligible_best = window_ready and (next_log_ep >= BEST_MIN_EPISODES)

            if eligible_best:
                spawn_now = float(spawn_s)

                better_spawn = (spawn_now > float(best_spawn) + BEST_SPAWN_EPS)
                same_spawn = (abs(spawn_now - float(best_spawn)) <= BEST_SPAWN_EPS)

                # hit_rate는 낮을수록 좋음
                better_hit = (hit_rate < float(best_hit_rate) - BEST_HIT_MARGIN)
                tied_hit = (abs(hit_rate - float(best_hit_rate)) <= EPS_TIE)

                better_surv = (avg_surv > float(best_avg50) + 1e-9)

                should_update_best = False
                reason = ""

                if better_spawn:
                    should_update_best = True
                    reason = "spawn increased"
                elif same_spawn and better_hit:
                    should_update_best = True
                    reason = "hit_rate improved"
                elif same_spawn and tied_hit and better_surv:
                    should_update_best = True
                    reason = "tie: survival improved"

                if should_update_best:
                    best_spawn = float(spawn_now)
                    best_hit_rate = float(hit_rate)
                    best_avg50 = float(avg_surv)

                    best_payload = dict(payload)
                    best_payload["best_spawn"] = float(best_spawn)
                    best_payload["best_hit_rate50"] = float(best_hit_rate)
                    best_payload["best_avg50"] = float(best_avg50)
                    best_payload["manual_spawn"] = float(manual_spawn)

                    _save_checkpoint(best_payload, CKPT_BEST)

                    print(
                        f"[BEST] updated ({reason}) "
                        f"best_spawn={best_spawn:.2f}/s "
                        f"best_hit_rate{ROLLING_N}={best_hit_rate:.4f} "
                        f"best_avg{ROLLING_N}={best_avg50:.2f}s "
                        f"best_path={CKPT_BEST}"
                    )

            # -------------------------
            # Auto difficulty update (spawn +5.0)
            # -------------------------
            if diff_cooldown > 0:
                diff_cooldown -= 1

            eligible_for_diff = AUTO_DIFF_ENABLED and window_ready and (diff_cooldown == 0)
            if eligible_for_diff and (hit_rate <= AUTO_DIFF_TARGET_HIT + 1e-12) and (manual_spawn < float(env_cfg.spawn_rate) - 1e-9):
                old_spawn = float(manual_spawn)
                manual_spawn = float(np.clip(old_spawn + float(AUTO_DIFF_STEP_SPAWN), 0.0, float(env_cfg.spawn_rate)))
                env.set_manual_difficulty(manual_spawn, enabled=True)

                diff_cooldown = int(AUTO_DIFF_COOLDOWN_LOGS)

                print(
                    f"[DIFF] up(+{AUTO_DIFF_STEP_SPAWN:.2f}/s): "
                    f"spawn {old_spawn:.2f} -> {manual_spawn:.2f}/s  "
                    f"(hit_rate{ROLLING_N}={hit_rate:.3f} <= {AUTO_DIFF_TARGET_HIT:.3f})  "
                    f"cooldown_logs={diff_cooldown}"
                )

            next_log_ep = ((done_eps // LOG_EVERY) + 1) * LOG_EVERY

    print("done.")


if __name__ == "__main__":
    main()
