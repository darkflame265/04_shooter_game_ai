from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class Rollout:
    obs: torch.Tensor
    act: torch.Tensor
    logp: torch.Tensor
    rew: torch.Tensor
    done: torch.Tensor
    val: torch.Tensor


class RolloutBuffer:
    def __init__(self, obs_dim: int, size: int, device: str):
        self.obs_dim = obs_dim
        self.size = size
        self.device = device
        self.reset()

    def reset(self):
        self.ptr = 0
        self.obs = np.zeros((self.size, self.obs_dim), dtype=np.float32)
        self.act = np.zeros((self.size,), dtype=np.int64)
        self.logp = np.zeros((self.size,), dtype=np.float32)
        self.rew = np.zeros((self.size,), dtype=np.float32)
        self.done = np.zeros((self.size,), dtype=np.float32)
        self.val = np.zeros((self.size,), dtype=np.float32)

    def add(self, obs, act, logp, rew, done, val):
        if self.ptr >= self.size:
            return
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.logp[self.ptr] = logp
        self.rew[self.ptr] = rew
        self.done[self.ptr] = float(done)
        self.val[self.ptr] = val
        self.ptr += 1

    def to_torch(self) -> Rollout:
        n = self.ptr
        return Rollout(
            obs=torch.tensor(self.obs[:n], device=self.device),
            act=torch.tensor(self.act[:n], device=self.device),
            logp=torch.tensor(self.logp[:n], device=self.device),
            rew=torch.tensor(self.rew[:n], device=self.device),
            done=torch.tensor(self.done[:n], device=self.device),
            val=torch.tensor(self.val[:n], device=self.device),
        )
