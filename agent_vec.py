from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from agent.models import ActorCritic


@dataclass
class PPOConfig:
    device: str = "cuda"
    gamma: float = 0.99
    gae_lambda: float = 0.95

    lr: float = 3e-4
    clip_ratio: float = 0.2

    update_epochs: int = 6
    minibatch_size: int = 2048  # vec에서는 크게 가는 게 유리

    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    rollout_steps: int = 256  # vec에선 짧게(T) + N 크게 추천

    # -------------------------
    # ✅ Adaptive LR by KL (NEW)
    # -------------------------
    lr_adapt: bool = True
    target_kl: float = 0.01          # 목표 KL (PPO에서 흔한 값)
    kl_high_mult: float = 2.0        # KL > 2*target => lr down
    kl_low_mult: float = 0.5         # KL < 0.5*target => lr up
    lr_down_factor: float = 0.5
    lr_up_factor: float = 1.10
    lr_min: float = 1e-5
    lr_max: float = 3e-4

    kl_ema_beta: float = 0.90        # KL EMA smoothing

    # 너무 큰 업데이트면 epoch 조기 중단(선택)
    early_stop_kl: bool = True
    early_stop_kl_mult: float = 4.0  # KL > 4*target이면 epoch loop break


class RolloutBufferVec:
    """
    Stores (T, N, ...) on device, then flattens to (T*N, ...).
    """
    def __init__(self, obs_dim: int, n_envs: int, T: int, device: str):
        self.obs_dim = int(obs_dim)
        self.n = int(n_envs)
        self.T = int(T)
        self.device = torch.device(device)
        self.reset()

    def reset(self):
        T, N = self.T, self.n
        dev = self.device
        self.ptr = 0

        self.obs = torch.zeros((T, N, self.obs_dim), device=dev, dtype=torch.float32)
        self.act = torch.zeros((T, N), device=dev, dtype=torch.int64)
        self.logp = torch.zeros((T, N), device=dev, dtype=torch.float32)
        self.rew = torch.zeros((T, N), device=dev, dtype=torch.float32)
        self.done = torch.zeros((T, N), device=dev, dtype=torch.float32)
        self.val = torch.zeros((T, N), device=dev, dtype=torch.float32)

    def add(self, obs: torch.Tensor, act: torch.Tensor, logp: torch.Tensor, rew: torch.Tensor, done: torch.Tensor, val: torch.Tensor):
        t = self.ptr
        if t >= self.T:
            raise RuntimeError("RolloutBuffer overflow")
        self.obs[t] = obs
        self.act[t] = act
        self.logp[t] = logp
        self.rew[t] = rew
        self.done[t] = done.float()
        self.val[t] = val
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.T

    def flatten(self):
        # (T,N,...) -> (T*N,...)
        T, N = self.T, self.n
        obs = self.obs.reshape(T * N, self.obs_dim)
        act = self.act.reshape(T * N)
        logp = self.logp.reshape(T * N)
        rew = self.rew.reshape(T * N)
        done = self.done.reshape(T * N)
        val = self.val.reshape(T * N)
        return obs, act, logp, rew, done, val


class PPOAgentVec:
    def __init__(self, obs_dim: int, act_dim: int, n_envs: int, cfg: PPOConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.net = ActorCritic(obs_dim, act_dim).to(self.device)
        self.opt = Adam(self.net.parameters(), lr=cfg.lr)

        self.act_dim = int(act_dim)
        self.n_envs = int(n_envs)

        self.buf = RolloutBufferVec(obs_dim, n_envs, cfg.rollout_steps, cfg.device)

        # ✅ KL EMA for stable lr tuning
        self._kl_ema = 0.0

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        obs: (N, obs_dim) torch on device
        returns:
          act: (N,) int64
          logp: (N,) float
          val: (N,) float
        """
        logits, v = self.net(obs)
        dist = Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return a, logp, v.squeeze(-1)

    def store(self, obs, act, logp, rew, done, val):
        self.buf.add(obs, act, logp, rew, done, val)

    def _get_lr(self) -> float:
        return float(self.opt.param_groups[0]["lr"])

    def _set_lr(self, lr: float) -> None:
        lr = float(np.clip(lr, self.cfg.lr_min, self.cfg.lr_max))
        for pg in self.opt.param_groups:
            pg["lr"] = lr

    @torch.no_grad()
    def _update_lr_by_kl(self, kl_value: float) -> float:
        """
        KL 기반 lr 자동조절.
        - kl이 너무 크면 lr down (업데이트 과격)
        - kl이 너무 작으면 lr up (업데이트 약함)
        """
        cfg = self.cfg
        if not cfg.lr_adapt:
            return self._get_lr()

        beta = float(cfg.kl_ema_beta)
        self._kl_ema = beta * float(self._kl_ema) + (1.0 - beta) * float(kl_value)
        kl = float(self._kl_ema)

        lr = self._get_lr()
        tkl = float(cfg.target_kl)

        if tkl > 0.0:
            if kl > cfg.kl_high_mult * tkl:
                lr *= float(cfg.lr_down_factor)
            elif kl < cfg.kl_low_mult * tkl:
                lr *= float(cfg.lr_up_factor)

        self._set_lr(lr)
        return self._get_lr()

    def finish_and_update(self, last_obs: torch.Tensor, last_done: torch.Tensor) -> Dict[str, float]:
        """
        last_obs: (N, obs_dim)
        last_done: (N,) bool
        Returns stats for logging:
          lr, approx_kl, entropy, pi_loss, v_loss, loss
        """
        cfg = self.cfg

        with torch.no_grad():
            _, last_v = self.net(last_obs)
            last_v = last_v.squeeze(-1)  # (N,)
            next_value = torch.where(last_done, torch.zeros_like(last_v), last_v)

        adv, ret = self._compute_gae(next_value)
        stats = self._update(adv, ret)
        self.buf.reset()
        return stats

    def _compute_gae(self, next_value: torch.Tensor):
        """
        next_value: (N,)
        Uses buffer tensors (T,N).
        Returns:
          adv: (T,N)
          ret: (T,N)
        """
        cfg = self.cfg
        T, N = self.buf.T, self.buf.n

        adv = torch.zeros((T, N), device=self.device, dtype=torch.float32)
        last_gae = torch.zeros((N,), device=self.device, dtype=torch.float32)

        for t in reversed(range(T)):
            not_done = 1.0 - self.buf.done[t]  # (N,)
            next_v = next_value if t == T - 1 else self.buf.val[t + 1]
            delta = self.buf.rew[t] + cfg.gamma * next_v * not_done - self.buf.val[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * not_done * last_gae
            adv[t] = last_gae

        ret = adv + self.buf.val
        # normalize advantage over all (T*N)
        adv_f = adv.reshape(-1)
        adv = (adv - adv_f.mean()) / (adv_f.std(unbiased=False) + 1e-8)
        return adv, ret

    def _update(self, adv: torch.Tensor, ret: torch.Tensor) -> Dict[str, float]:
        cfg = self.cfg

        obs, act, logp_old, _, _, _ = self.buf.flatten()
        adv_f = adv.reshape(-1)
        ret_f = ret.reshape(-1)

        n = obs.shape[0]
        idxs = np.arange(n)

        kl_sum = 0.0
        ent_sum = 0.0
        mb_count = 0

        last_pi_loss = 0.0
        last_v_loss = 0.0
        last_loss = 0.0

        for _epoch in range(int(cfg.update_epochs)):
            np.random.shuffle(idxs)

            early_stop = False

            for start in range(0, n, int(cfg.minibatch_size)):
                mb = idxs[start : start + int(cfg.minibatch_size)]
                mb_t = torch.as_tensor(mb, device=self.device, dtype=torch.long)

                obs_b = obs[mb_t]
                act_b = act[mb_t]
                logp_old_b = logp_old[mb_t]
                adv_b = adv_f[mb_t]
                ret_b = ret_f[mb_t]

                logits, v = self.net(obs_b)
                v = v.squeeze(-1)
                dist = Categorical(logits=logits)

                logp = dist.log_prob(act_b)
                ratio = torch.exp(logp - logp_old_b)

                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv_b
                pi_loss = -torch.min(surr1, surr2).mean()

                v_loss = F.mse_loss(v, ret_b)

                ent = dist.entropy().mean()

                loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    approx_kl = (logp_old_b - logp).mean()  # E[logp_old - logp_new]

                kl_sum += float(approx_kl.item())
                ent_sum += float(ent.item())
                mb_count += 1

                last_pi_loss = float(pi_loss.item())
                last_v_loss = float(v_loss.item())
                last_loss = float(loss.item())

                # optional early stop if KL too large
                if cfg.early_stop_kl and cfg.target_kl > 0.0:
                    if float(approx_kl.item()) > float(cfg.early_stop_kl_mult) * float(cfg.target_kl):
                        early_stop = True
                        break

            # epoch 끝날 때마다 mean KL 기반 lr 조절 (너무 자주 바꾸지 않음)
            if mb_count > 0:
                mean_kl = kl_sum / float(mb_count)
                self._update_lr_by_kl(mean_kl)

            if early_stop:
                break

        mean_kl = (kl_sum / float(mb_count)) if mb_count > 0 else 0.0
        mean_ent = (ent_sum / float(mb_count)) if mb_count > 0 else 0.0

        return {
            "lr": self._get_lr(),
            "approx_kl": float(mean_kl),
            "entropy": float(mean_ent),
            "pi_loss": float(last_pi_loss),
            "v_loss": float(last_v_loss),
            "loss": float(last_loss),
        }
