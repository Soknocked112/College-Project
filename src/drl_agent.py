"""DRL response policy over TCN embeddings (spec §6c).

* :class:`PolicyEnv` - offline environment over the chronologically ordered
  training windows. State = frozen TCN embedding of the window, actions =
  ``{pass, monitor, block}``, reward from :data:`config.REWARDS`, next state =
  next window of the same flow group (terminal at the group's last window).
* :class:`DoubleDQNAgent` (default) - replay buffer, target-network sync every
  ``TARGET_UPDATE`` gradient steps, Huber loss, epsilon-greedy 1.0 -> 0.05 over
  ``EPS_DECAY_STEPS`` environment steps.
* :class:`PPOAgent` - clipped-objective actor-critic with GAE.

Full-pass guarantee: every DRL epoch first walks the **entire** 80 % training
set (all windows, inner-val included) to populate the buffer / rollout, then
runs one full pass of updates over it. The replay buffer stores transitions as
index arrays into a single embedding table, so its capacity grows to
``max(REPLAY_BUFFER, N)`` without duplicating embeddings in memory.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C
from .utils import autocast, get_logger, grad_scaler

log = get_logger()


def reward_matrix() -> np.ndarray:
    """Return the ``(2, 3)`` reward table ``R[label, action]``."""
    return np.array([[C.REWARDS[lab][a] for a in C.ACTIONS] for lab in (0, 1)], dtype=np.float32)


def epsilon_at(step: np.ndarray | int) -> np.ndarray | float:
    """Linear epsilon schedule ``EPS_START -> EPS_END`` over ``EPS_DECAY_STEPS``."""
    frac = np.minimum(np.asarray(step, dtype=np.float64) / max(C.EPS_DECAY_STEPS, 1), 1.0)
    return C.EPS_START + frac * (C.EPS_END - C.EPS_START)


def _mlp(inp: int, hidden: List[int], out: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    for h in hidden:
        layers += [nn.Linear(inp, h), nn.ReLU()]
        inp = h
    layers.append(nn.Linear(inp, out))
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """MLP Q-function ``Q(s, .)`` over embeddings."""

    def __init__(self, state_dim: int, n_actions: int = len(C.ACTIONS), hidden: List[int] = C.DRL_HIDDEN) -> None:
        super().__init__()
        self.net = _mlp(state_dim, hidden, n_actions)

    def forward(self, s: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(s)


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic for PPO."""

    def __init__(self, state_dim: int, n_actions: int = len(C.ACTIONS), hidden: List[int] = C.DRL_HIDDEN) -> None:
        super().__init__()
        self.trunk = _mlp(state_dim, hidden[:-1], hidden[-1]) if len(hidden) > 1 else nn.Linear(state_dim, hidden[0])
        self.act = nn.ReLU()
        self.pi = nn.Linear(hidden[-1], n_actions)
        self.v = nn.Linear(hidden[-1], 1)

    def forward(self, s: torch.Tensor):  # noqa: D102
        h = self.act(self.trunk(s))
        return self.pi(h), self.v(h).squeeze(-1)


class PolicyEnv:
    """Offline environment over training-window embeddings.

    Args:
        embeddings: ``(N, E)`` TCN embeddings (float16/32), chronological per group.
        labels: ``(N,)`` window labels.
        groups: ``(N,)`` group id per window (contiguous runs).
        device: Device holding the embedding table.
    """

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, groups: np.ndarray, device: str) -> None:
        self.n = int(len(labels))
        self.device = device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.states = torch.as_tensor(np.asarray(embeddings), dtype=self.dtype, device=device)
        self.labels_np = np.asarray(labels).astype(np.int64)
        g = np.asarray(groups)
        same_next = np.zeros(self.n, dtype=bool)
        same_next[:-1] = g[1:] == g[:-1]
        self.next_idx_np = np.where(same_next, np.arange(self.n) + 1, np.arange(self.n))
        self.done_np = (~same_next).astype(np.float32)
        self.R = torch.as_tensor(reward_matrix(), device=device)
        self.labels = torch.as_tensor(self.labels_np, device=device)
        self.next_idx = torch.as_tensor(self.next_idx_np, device=device)
        self.done = torch.as_tensor(self.done_np, device=device)
        self.state_dim = int(self.states.shape[1])

    def rewards(self, idx: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Reward for taking ``actions`` in windows ``idx``."""
        return self.R[self.labels[idx], actions]

    def state(self, idx: torch.Tensor) -> torch.Tensor:
        """Float32 states for ``idx``."""
        return self.states[idx].float()


class DoubleDQNAgent:
    """Double-DQN with Huber loss, target network and AMP on CUDA.

    Args:
        state_dim: Embedding size.
        device: Device string.
        lr: Learning rate.
    """

    algo = "double_dqn"

    def __init__(self, state_dim: int, device: str, lr: float = C.LR) -> None:
        self.device = device
        self.online = QNetwork(state_dim).to(device)
        self.target = QNetwork(state_dim).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt = torch.optim.AdamW(self.online.parameters(), lr=lr, weight_decay=C.WEIGHT_DECAY)
        self.amp = C.USE_AMP and device.startswith("cuda")
        self.scaler = grad_scaler(device, self.amp)
        self.state_dim = state_dim
        self.env_steps = 0
        self.updates = 0

    @torch.no_grad()
    def greedy(self, states: torch.Tensor) -> torch.Tensor:
        """Greedy actions for float states ``(B, E)``."""
        self.online.eval()
        return self.online(states.float()).argmax(dim=1)

    @torch.no_grad()
    def _collect(self, env: PolicyEnv, batch: int, step_log: List[Dict[str, float]]) -> torch.Tensor:
        """One full epsilon-greedy pass over every training window."""
        actions = torch.empty(env.n, dtype=torch.long, device=self.device)
        gen = torch.Generator(device="cpu").manual_seed(C.SEED + self.env_steps)
        for s in range(0, env.n, batch):
            idx = torch.arange(s, min(s + batch, env.n), device=self.device)
            eps = torch.as_tensor(epsilon_at(self.env_steps + np.arange(len(idx))), dtype=torch.float32)
            greedy = self.greedy(env.state(idx)).cpu()
            rand_a = torch.randint(0, len(C.ACTIONS), (len(idx),), generator=gen)
            explore = torch.rand(len(idx), generator=gen) < eps
            a = torch.where(explore, rand_a, greedy).to(self.device)
            actions[idx] = a
            self.env_steps += len(idx)
            step_log.append({"env_step": self.env_steps, "epsilon": float(eps[-1]),
                             "mean_reward": float(env.rewards(idx, a).float().mean())})
        return actions

    def train(self, env: PolicyEnv, epochs: int = C.DRL_EPOCHS, batch_size: int = C.BATCH_SIZE,
              on_epoch: Optional[Callable[[Dict[str, float]], None]] = None) -> Dict[str, List[Dict[str, float]]]:
        """Train for ``epochs`` full passes.

        Args:
            env: Offline environment.
            epochs: Number of DRL epochs.
            batch_size: Minibatch size.
            on_epoch: Callback receiving the epoch log row.

        Returns:
            ``{"epochs": [...], "steps": [...]}`` logs.
        """
        epoch_log: List[Dict[str, float]] = []
        step_log: List[Dict[str, float]] = []
        n_act = len(C.ACTIONS)
        for ep in range(1, epochs + 1):
            # 1) populate the buffer with the full training set
            if C.DRL_BUFFER_MODE == "exhaustive":
                b_idx = torch.arange(env.n, device=self.device).repeat_interleave(n_act)
                b_act = torch.arange(n_act, device=self.device).repeat(env.n)
                self.env_steps += env.n
                step_log.append({"env_step": self.env_steps, "epsilon": float(epsilon_at(self.env_steps)),
                                 "mean_reward": float(env.rewards(b_idx, b_act).float().mean())})
            else:
                b_act = self._collect(env, C.EVAL_BATCH_SIZE, step_log)
                b_idx = torch.arange(env.n, device=self.device)
            capacity = max(C.REPLAY_BUFFER, len(b_idx))
            assert len(b_idx) <= capacity
            b_rew = env.rewards(b_idx, b_act).float()
            behaviour_reward = float(b_rew.mean())
            # 2) one full pass of Double-DQN updates over the buffer
            perm = torch.randperm(len(b_idx), device=self.device)
            losses = []
            self.online.train()
            n_batches = math.ceil(len(perm) / batch_size)
            for bi in range(n_batches):
                sel = perm[bi * batch_size:(bi + 1) * batch_size]
                i, a, r = b_idx[sel], b_act[sel], b_rew[sel]
                s, ns = env.state(i), env.state(env.next_idx[i])
                done = env.done[i]
                with torch.no_grad(), autocast(self.device, self.amp):
                    na = self.online(ns).argmax(dim=1, keepdim=True)
                    tq = self.target(ns).gather(1, na).squeeze(1).float()
                    y = r + C.GAMMA * (1.0 - done) * tq
                with autocast(self.device, self.amp):
                    q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = F.smooth_l1_loss(q.float(), y, beta=C.HUBER_DELTA)
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.online.parameters(), C.GRAD_CLIP_NORM)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.updates += 1
                if self.updates % C.TARGET_UPDATE == 0:
                    self.target.load_state_dict(self.online.state_dict())
                losses.append(float(loss.detach()))
            greedy_r = self.evaluate(env)
            row = {"epoch": ep, "updates": self.updates, "env_steps": self.env_steps, "buffer_size": len(b_idx),
                   "buffer_capacity": capacity, "epsilon": float(epsilon_at(self.env_steps)),
                   "loss": float(np.mean(losses)) if losses else float("nan"),
                   "behaviour_reward": behaviour_reward, **greedy_r}
            epoch_log.append(row)
            log.info(f"[DRL/double_dqn] epoch {ep}/{epochs} loss={row['loss']:.4f} eps={row['epsilon']:.3f} "
                     f"behaviour_R={behaviour_reward:.3f} greedy_R={greedy_r['greedy_reward']:.3f}")
            if on_epoch:
                on_epoch(row)
        return {"epochs": epoch_log, "steps": step_log}

    @torch.no_grad()
    def evaluate(self, env: PolicyEnv, batch: int = C.EVAL_BATCH_SIZE) -> Dict[str, float]:
        """Mean reward and action distribution of the greedy policy over all windows."""
        tot, counts = 0.0, np.zeros(len(C.ACTIONS))
        for s in range(0, env.n, batch):
            idx = torch.arange(s, min(s + batch, env.n), device=self.device)
            a = self.greedy(env.state(idx))
            tot += float(env.rewards(idx, a).float().sum())
            counts += np.bincount(a.cpu().numpy(), minlength=len(C.ACTIONS))
        out = {"greedy_reward": tot / max(env.n, 1)}
        out.update({f"frac_{n}": c / max(env.n, 1) for n, c in zip(C.ACTIONS, counts)})
        return out

    def policy_module(self) -> nn.Module:
        """The network used at inference (online Q-net)."""
        return self.online


class PPOAgent:
    """PPO actor-critic (clipped objective, GAE) over the offline env.

    Args:
        state_dim: Embedding size.
        device: Device string.
        lr: Learning rate.
    """

    algo = "ppo"

    def __init__(self, state_dim: int, device: str, lr: float = C.LR) -> None:
        self.device = device
        self.net = ActorCritic(state_dim).to(device)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=C.WEIGHT_DECAY)
        self.amp = C.USE_AMP and device.startswith("cuda")
        self.scaler = grad_scaler(device, self.amp)
        self.state_dim = state_dim
        self.env_steps = 0
        self.updates = 0

    @torch.no_grad()
    def greedy(self, states: torch.Tensor) -> torch.Tensor:
        """Most probable action for float states."""
        self.net.eval()
        logits, _ = self.net(states.float())
        return logits.argmax(dim=1)

    @torch.no_grad()
    def _rollout(self, env: PolicyEnv, batch: int):
        logp = torch.empty(env.n, device=self.device)
        vals = torch.empty(env.n, device=self.device)
        acts = torch.empty(env.n, dtype=torch.long, device=self.device)
        self.net.eval()
        for s in range(0, env.n, batch):
            idx = torch.arange(s, min(s + batch, env.n), device=self.device)
            logits, v = self.net(env.state(idx))
            dist = torch.distributions.Categorical(logits=logits.float())
            a = dist.sample()
            acts[idx], logp[idx], vals[idx] = a, dist.log_prob(a), v.float()
        self.env_steps += env.n
        rew = env.rewards(torch.arange(env.n, device=self.device), acts).float()
        # GAE over chronologically ordered windows, reset at group ends
        r, v, d = rew.cpu().numpy(), vals.cpu().numpy(), env.done_np
        nv = np.where(d > 0, 0.0, v[env.next_idx_np])
        delta = r + C.GAMMA * nv - v
        adv = np.zeros(env.n, dtype=np.float32)
        last = 0.0
        for t in range(env.n - 1, -1, -1):
            last = delta[t] + C.GAMMA * C.PPO_GAE_LAMBDA * (1.0 - d[t]) * last
            adv[t] = last
        ret = adv + v
        adv_t = torch.as_tensor(adv, device=self.device)
        return acts, logp, (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8), torch.as_tensor(ret, device=self.device), rew

    def train(self, env: PolicyEnv, epochs: int = C.DRL_EPOCHS, batch_size: int = C.BATCH_SIZE,
              on_epoch: Optional[Callable[[Dict[str, float]], None]] = None) -> Dict[str, List[Dict[str, float]]]:
        """Train for ``epochs`` full rollouts, each followed by ``PPO_UPDATE_EPOCHS`` passes."""
        epoch_log: List[Dict[str, float]] = []
        step_log: List[Dict[str, float]] = []
        for ep in range(1, epochs + 1):
            acts, old_logp, adv, ret, rew = self._rollout(env, C.EVAL_BATCH_SIZE)
            step_log.append({"env_step": self.env_steps, "epsilon": float("nan"), "mean_reward": float(rew.mean())})
            losses = []
            self.net.train()
            for _ in range(C.PPO_UPDATE_EPOCHS):
                perm = torch.randperm(env.n, device=self.device)
                for bi in range(math.ceil(env.n / batch_size)):
                    i = perm[bi * batch_size:(bi + 1) * batch_size]
                    with autocast(self.device, self.amp):
                        logits, v = self.net(env.state(i))
                    dist = torch.distributions.Categorical(logits=logits.float())
                    lp = dist.log_prob(acts[i])
                    ratio = torch.exp(lp - old_logp[i])
                    s1 = ratio * adv[i]
                    s2 = torch.clamp(ratio, 1 - C.PPO_CLIP, 1 + C.PPO_CLIP) * adv[i]
                    loss = (-torch.min(s1, s2).mean() + C.PPO_VALUE_COEF * F.mse_loss(v.float(), ret[i])
                            - C.PPO_ENTROPY_COEF * dist.entropy().mean())
                    self.opt.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(self.net.parameters(), C.GRAD_CLIP_NORM)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.updates += 1
                    losses.append(float(loss.detach()))
            greedy_r = DoubleDQNAgent.evaluate(self, env)  # same greedy evaluation routine
            row = {"epoch": ep, "updates": self.updates, "env_steps": self.env_steps, "buffer_size": env.n,
                   "buffer_capacity": env.n, "epsilon": float("nan"), "loss": float(np.mean(losses)),
                   "behaviour_reward": float(rew.mean()), **greedy_r}
            epoch_log.append(row)
            log.info(f"[DRL/ppo] epoch {ep}/{epochs} loss={row['loss']:.4f} "
                     f"behaviour_R={row['behaviour_reward']:.3f} greedy_R={greedy_r['greedy_reward']:.3f}")
            if on_epoch:
                on_epoch(row)
        return {"epochs": epoch_log, "steps": step_log}

    def policy_module(self) -> nn.Module:
        """The network used at inference."""
        return self.net


def make_agent(algo: str, state_dim: int, device: str):
    """Factory for ``"double_dqn"`` / ``"ppo"`` agents."""
    if algo == "double_dqn":
        return DoubleDQNAgent(state_dim, device)
    if algo == "ppo":
        return PPOAgent(state_dim, device)
    raise ValueError(f"Unknown DRL algo {algo!r}; choose from {C.DRL_ALGOS}")


def save_agent(agent, path: Path, extra: Optional[Dict[str, Any]] = None) -> None:
    """Persist the inference network + metadata."""
    payload = {"algo": agent.algo, "state_dim": agent.state_dim, "actions": list(C.ACTIONS),
               "hidden": list(C.DRL_HIDDEN),
               "state_dict": {k: v.detach().cpu() for k, v in agent.policy_module().state_dict().items()}}
    payload.update(extra or {})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


class DRLPolicy(nn.Module):
    """Inference-only wrapper returning greedy action indices.

    Args:
        payload: Dict written by :func:`save_agent`.
    """

    def __init__(self, payload: Dict[str, Any]) -> None:
        super().__init__()
        self.algo = payload["algo"]
        self.actions = list(payload["actions"])
        if self.algo == "double_dqn":
            self.net: nn.Module = QNetwork(payload["state_dim"], len(self.actions), payload["hidden"])
        else:
            self.net = ActorCritic(payload["state_dim"], len(self.actions), payload["hidden"])
        self.net.load_state_dict(payload["state_dict"])
        self.eval()

    @torch.no_grad()
    def act(self, states: torch.Tensor) -> torch.Tensor:
        """Greedy action indices for ``(B, E)`` float states."""
        out = self.net(states.float())
        logits = out[0] if isinstance(out, tuple) else out
        return logits.argmax(dim=1)


def load_policy(path: Path, device: str = "cpu") -> DRLPolicy:
    """Load ``drl_agent.pt`` as a :class:`DRLPolicy`."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return DRLPolicy(payload).to(device)
