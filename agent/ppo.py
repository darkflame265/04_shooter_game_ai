from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from .models import ActorCritic
from .buffer import RolloutBuffer, Rollout


@dataclass
class PPOConfig:
    device: str = "cpu"

    gamma: float = 0.99
    gae_lambda: float = 0.95

    lr: float = 3e-4
    clip_ratio: float = 0.2

    update_epochs: int = 6
    minibatch_size: int = 256

    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    rollout_steps: int = 4096


class PPOAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig):
        self.cfg = cfg
        self.device = cfg.device

        self.net = ActorCritic(obs_dim, act_dim).to(self.device)
        self.opt = Adam(self.net.parameters(), lr=cfg.lr)

        self.buf = RolloutBuffer(obs_dim, cfg.rollout_steps, self.device)

        self.act_dim = act_dim

    @torch.no_grad()
    def act(self, obs_np: np.ndarray) -> Tuple[int, float, float]:
        obs = torch.tensor(obs_np, device=self.device).unsqueeze(0)
        logits, v = self.net(obs)
        dist = Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return int(a.item()), float(logp.item()), float(v.item())

    def store(self, obs, act, logp, rew, done, val):
        self.buf.add(obs, act, logp, rew, done, val)

    def finish_and_update(self, last_obs: np.ndarray, last_done: bool):
        roll = self.buf.to_torch()
        with torch.no_grad():
            last_obs_t = torch.tensor(last_obs, device=self.device).unsqueeze(0)
            _, last_v = self.net(last_obs_t)
            last_v = last_v.squeeze(0)

        adv, ret = self._compute_gae(roll, last_v, last_done)
        self._update(roll, adv, ret)
        self.buf.reset()

    def _compute_gae(self, roll: Rollout, last_v: torch.Tensor, last_done: bool):
        cfg = self.cfg
        n = roll.rew.shape[0]

        adv = torch.zeros(n, device=self.device)
        last_gae = 0.0

        # bootstrap value: 0 if done else V(s_T)
        next_value = torch.tensor(0.0, device=self.device) if last_done else last_v

        for t in reversed(range(n)):
            not_done = 1.0 - roll.done[t]
            next_v = next_value if t == n - 1 else roll.val[t + 1]
            delta = roll.rew[t] + cfg.gamma * next_v * not_done - roll.val[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * not_done * last_gae
            adv[t] = last_gae

        ret = adv + roll.val
        # normalize advantage
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        return adv, ret

    def _update(self, roll: Rollout, adv: torch.Tensor, ret: torch.Tensor):
        cfg = self.cfg
        n = roll.obs.shape[0]
        idxs = np.arange(n)

        for _ in range(cfg.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, cfg.minibatch_size):
                mb = idxs[start : start + cfg.minibatch_size]
                obs_b = roll.obs[mb]
                act_b = roll.act[mb]
                logp_old = roll.logp[mb]
                adv_b = adv[mb]
                ret_b = ret[mb]

                logits, v = self.net(obs_b)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(act_b)
                ratio = torch.exp(logp - logp_old)

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
