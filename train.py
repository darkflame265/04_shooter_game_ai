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

# Best 갱신 보수적으로: hit_rate 개선 최소 폭 (낮을수록 좋음)
BEST_HIT_MARGIN = 1e-3
EPS_TIE = 1e-12

# -------------------------
# Auto difficulty (manual diff01)
# -------------------------
AUTO_DIFF_ENABLED = True
AUTO_DIFF_TARGET_HIT = 0.20     # hit_rate50 <= 0.20 이면 난이도 업
AUTO_DIFF_STEP = 0.05           # diff01 증가폭 (0.0~1.0)
AUTO_DIFF_COOLDOWN_LOGS = 10    # 난이도 업 후, 로그 N번은 대기
AUTO_DIFF_MIN_READY = ROLLING_N # 최소 window 꽉 찬 뒤에만 판단


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

    # python / numpy
    st["rng_python"] = random.getstate()
    st["rng_numpy"] = np.random.get_state()

    # torch cpu
    st["rng_torch_cpu"] = torch.get_rng_state()

    # torch cuda
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            st["rng_torch_cuda_all"] = torch.cuda.get_rng_state_all()
        except Exception:
            st["rng_torch_cuda_all"] = None
    else:
        st["rng_torch_cuda_all"] = None

    # env internal generator
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

    # python / numpy
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

    # torch cpu
    try:
        if "rng_torch_cpu" in st and st["rng_torch_cpu"] is not None:
            torch.set_rng_state(st["rng_torch_cpu"])
    except Exception:
        pass

    # torch cuda
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            cuda_all = st.get("rng_torch_cuda_all", None)
            if cuda_all is not None:
                torch.cuda.set_rng_state_all(cuda_all)
        except Exception:
            pass

    # env internal generator
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
    manual_diff01: float,
    device: str,
    env: Optional[VecShooterEnvTorch] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "net": agent.net.state_dict(),
        "cfg": vars(ppo_cfg),
        "env_cfg": vars(env_cfg),
        "episode": int(ep),
        "global_step": int(global_step),
        "manual_diff01": float(manual_diff01),
        # ✅ RNG snapshot
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

    manual_diff01 = ckpt.get("manual_diff01", None)
    try:
        manual_diff01 = float(manual_diff01) if manual_diff01 is not None else None
    except Exception:
        manual_diff01 = None

    # ✅ RNG restore (if exists)
    rng_state = ckpt.get("rng_state", None)
    if isinstance(rng_state, dict):
        _restore_rng_state(rng_state, device=device, env=env)

    start_ep = last_ep + 1
    print(f"[CKPT] loaded {path} -> resume from episode {start_ep}, global_step {global_step}")
    if manual_diff01 is not None:
        print(f"[CKPT] loaded manual_diff01={manual_diff01:.3f}")
    if isinstance(rng_state, dict):
        print("[CKPT] restored RNG state")
    else:
        print("[CKPT] no RNG state in checkpoint (old ckpt)")

    return start_ep, global_step, manual_diff01, rng_state


def _try_load_best_metrics(
    best_path: str, device: str
) -> Tuple[Optional[float], Optional[int], Optional[float], Optional[float]]:
    """
    returns: (best_hit_rate50, best_nohit_streak, best_avg50, best_spawn)
      - hit_rate50: 낮을수록 좋음
      - best_spawn: 높을수록 좋음 (난이도)
    """
    if not os.path.isfile(best_path):
        return None, None, None, None
    try:
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
    except Exception:
        return None, None, None, None
    if not isinstance(ckpt, dict):
        return None, None, None, None

    bhr = ckpt.get("best_hit_rate50", None)
    bns = ckpt.get("best_nohit_streak", None)
    bav = ckpt.get("best_avg50", None)
    bsp = ckpt.get("best_spawn", None)

    try:
        bhr = float(bhr) if bhr is not None else None
    except Exception:
        bhr = None
    try:
        bns = int(bns) if bns is not None else None
    except Exception:
        bns = None
    try:
        bav = float(bav) if bav is not None else None
    except Exception:
        bav = None
    try:
        bsp = float(bsp) if bsp is not None else None
    except Exception:
        bsp = None

    return bhr, bns, bav, bsp


@torch.no_grad()
def _render_viewer(args):
    """
    Performance test mode:
      - no training
      - loads ppo_best.pt first (best), fallback to ppo.pt
      - uses the checkpoint's manual_diff01 if available to reproduce "best" difficulty
      - (optional) restores rng_state so that "same sequence" replay is possible
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[RENDER] device:", device)

    env_cfg = EnvConfig(seed=0)
    env_cfg.curriculum = False  # render는 manual difficulty로 고정

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

    # (optional) restore RNG for deterministic-ish replay
    rng_state = ckpt.get("rng_state", None)
    if isinstance(rng_state, dict):
        _restore_rng_state(rng_state, device=device, env=env)
        print("[RENDER] restored RNG state from checkpoint")

    diff01 = ckpt.get("manual_diff01", None)
    try:
        diff01 = float(diff01) if diff01 is not None else 1.0
    except Exception:
        diff01 = 1.0
    diff01 = float(np.clip(diff01, 0.0, 1.0))
    env.set_manual_difficulty(diff01)

    print(f"[RENDER] loaded: {ckpt_path}")
    print(f"[RENDER] fixed difficulty: diff01={diff01:.2f} spawn={env.get_spawn_rate_s():.2f}/s")

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


def main():
    args = parse_args()
    os.makedirs("checkpoints", exist_ok=True)

    # render-only mode
    if not args.no_render:
        _render_viewer(args)
        return

    # -------------------------
    # TRAIN MODE (vec)
    # -------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[DEV]", device)

    # NOTE: initial seeding. If ckpt has rng_state, it will override these.
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    env_cfg = EnvConfig(seed=0)
    env_cfg.curriculum = False  # trainer가 manual diff01로 제어
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

    start_ep, global_step, ckpt_diff, _rng = _try_load_checkpoint(agent, CKPT_LATEST, device, env=env)

    # ✅ 누적(글로벌) 에피소드 카운터: ckpt episode + 이번 런에서 완료한 done_eps
    base_global_episode = int(max(0, start_ep - 1))

    recent_surv = deque(maxlen=ROLLING_N)
    recent_hit = deque(maxlen=ROLLING_N)

    best_hit_rate, best_nohit_streak, best_avg50, best_spawn = _try_load_best_metrics(CKPT_BEST, device)

    if best_hit_rate is None:
        best_hit_rate = 1e9
    if best_nohit_streak is None:
        best_nohit_streak = 0
    if best_avg50 is None:
        best_avg50 = 0.0
    if best_spawn is None:
        best_spawn = -1.0

    print(
        f"[BEST] best_spawn={best_spawn:.2f}/s "
        f"best_hit_rate{ROLLING_N}={best_hit_rate:.4f} best_avg{ROLLING_N}={best_avg50:.2f}s "
        f"best_nohit_streak={best_nohit_streak} best_path={CKPT_BEST}"
    )

    manual_diff01 = float(ckpt_diff) if ckpt_diff is not None else 0.0
    manual_diff01 = float(np.clip(manual_diff01, 0.0, 1.0))
    env.set_manual_difficulty(manual_diff01)

    diff_cooldown = 0

    print(f"[DIFF] start manual_diff01={manual_diff01:.2f} spawn={env.get_spawn_rate_s():.2f}/s")

    obs, _ = env.reset()
    ep_hit = torch.zeros((args.n_envs,), device=device, dtype=torch.bool)

    nohit_streak = 0
    done_eps = 0
    next_log_ep = LOG_EVERY

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

        while done_eps >= next_log_ep:
            dt = max(1e-6, time.time() - t0)
            run_steps = global_step - run_step0
            sps = run_steps / dt

            avg_surv = float(np.mean(recent_surv)) if recent_surv else 0.0
            hit_rate = float(np.mean(recent_hit)) if recent_hit else 0.0
            n = len(recent_hit)

            spawn_s = env.get_spawn_rate_s()

            global_episode = base_global_episode + int(done_eps)

            print(
                f"[EP_DONE {next_log_ep:7d}/{args.episodes}] "
                f"surv_avg{ROLLING_N}={avg_surv:5.2f}s hit_rate{ROLLING_N}={hit_rate:5.3f} "
                f"spawn={spawn_s:5.2f}/s "
                f"global_episode={global_episode} "
                f"(n={n:2d}) steps={global_step} SPS={sps:7.1f}"
            )

            payload = _pack_checkpoint(
                agent=agent,
                ppo_cfg=ppo_cfg,
                env_cfg=env_cfg,
                ep=next_log_ep,
                global_step=global_step,
                manual_diff01=manual_diff01,
                device=device,
                env=env,
            )
            _save_checkpoint(payload, CKPT_LATEST)

            if len(recent_hit) > 0:
                if recent_hit[-1] == 0.0:
                    nohit_streak += 1
                else:
                    nohit_streak = 0

            # -------------------------
            # Auto difficulty update
            # -------------------------
            if diff_cooldown > 0:
                diff_cooldown -= 1

            window_ready = (n >= AUTO_DIFF_MIN_READY)
            eligible_for_diff = AUTO_DIFF_ENABLED and window_ready and (diff_cooldown == 0)

            if eligible_for_diff and (hit_rate <= AUTO_DIFF_TARGET_HIT + 1e-12) and (manual_diff01 < 1.0 - 1e-9):
                old = manual_diff01
                manual_diff01 = float(np.clip(manual_diff01 + AUTO_DIFF_STEP, 0.0, 1.0))
                env.set_manual_difficulty(manual_diff01)
                diff_cooldown = int(AUTO_DIFF_COOLDOWN_LOGS)

                print(
                    f"[DIFF] up: {old:.2f} -> {manual_diff01:.2f}  "
                    f"(hit_rate{ROLLING_N}={hit_rate:.3f} <= {AUTO_DIFF_TARGET_HIT:.3f})  "
                    f"spawn={env.get_spawn_rate_s():.2f}/s"
                )

            # -------------------------
            # BEST selection rule (난이도 우선)
            # -------------------------
            eligible = (next_log_ep >= BEST_MIN_EPISODES) and window_ready
            if eligible:
                SPAWN_EPS = 1e-9

                better_spawn = (spawn_s > best_spawn + SPAWN_EPS)
                same_spawn = abs(spawn_s - best_spawn) <= SPAWN_EPS

                better_hit = (hit_rate < (best_hit_rate - BEST_HIT_MARGIN))
                tied_hit = abs(hit_rate - best_hit_rate) <= EPS_TIE

                tie_surv_improved = same_spawn and tied_hit and (avg_surv > best_avg50 + 1e-9)
                tie_streak_improved = same_spawn and tied_hit and (nohit_streak > best_nohit_streak)

                should_update_best = False
                reason = ""

                if better_spawn:
                    should_update_best = True
                    reason = "spawn increased"
                elif same_spawn and better_hit:
                    should_update_best = True
                    reason = "hit_rate improved (lower)"
                elif tie_surv_improved:
                    should_update_best = True
                    reason = "tie: avg_survival improved"
                elif tie_streak_improved:
                    should_update_best = True
                    reason = "tie: nohit_streak improved"

                if should_update_best:
                    best_spawn = float(spawn_s)
                    if (better_spawn or (same_spawn and better_hit)):
                        best_hit_rate = float(hit_rate)
                        best_avg50 = float(avg_surv)
                        best_nohit_streak = int(nohit_streak)
                    else:
                        best_avg50 = max(best_avg50, float(avg_surv))
                        best_nohit_streak = max(best_nohit_streak, int(nohit_streak))

                    best_payload = dict(payload)
                    best_payload["best_spawn"] = float(best_spawn)
                    best_payload["best_hit_rate50"] = float(best_hit_rate)
                    best_payload["best_avg50"] = float(best_avg50)
                    best_payload["best_nohit_streak"] = int(best_nohit_streak)
                    best_payload["manual_diff01"] = float(manual_diff01)

                    _save_checkpoint(best_payload, CKPT_BEST)

                    print(
                        f"[BEST] updated ({reason}) "
                        f"best_spawn={best_spawn:.2f}/s "
                        f"best_hit_rate{ROLLING_N}={best_hit_rate:.4f} "
                        f"best_avg{ROLLING_N}={best_avg50:.2f}s "
                        f"best_nohit_streak={best_nohit_streak}"
                    )

            next_log_ep += LOG_EVERY

    print("done.")


if __name__ == "__main__":
    main()
