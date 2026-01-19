from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

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

    def finish_and_update(self, last_obs: torch.Tensor, last_done: torch.Tensor):
        """
        last_obs: (N, obs_dim)
        last_done: (N,) bool
        """
        cfg = self.cfg

        with torch.no_grad():
            _, last_v = self.net(last_obs)
            last_v = last_v.squeeze(-1)  # (N,)
            # if done, bootstrap=0
            next_value = torch.where(last_done, torch.zeros_like(last_v), last_v)

        adv, ret = self._compute_gae(next_value)
        self._update(adv, ret)
        self.buf.reset()

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

    def _update(self, adv: torch.Tensor, ret: torch.Tensor):
        cfg = self.cfg

        obs, act, logp_old, _, _, _ = self.buf.flatten()
        adv_f = adv.reshape(-1)
        ret_f = ret.reshape(-1)

        n = obs.shape[0]
        idxs = np.arange(n)

        for _ in range(cfg.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, cfg.minibatch_size):
                mb = idxs[start : start + cfg.minibatch_size]
                mb_t = torch.tensor(mb, device=self.device, dtype=torch.long)

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
