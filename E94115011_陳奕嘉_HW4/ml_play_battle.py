import os
import time
from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli
import datetime

"""
A COMPLETE PPO-based MLPlay agent for the Proly MLGame3D environment.

Usage
-----
• To **train** set `TRAINING_MODE = True` (default) and run with MLGame3D.
• To **play with a pre-trained model** set `TRAINING_MODE = False` and
  place your checkpoint at ``MODEL_LOAD_PATH``.

This file replaces the original `ppo_mlplay_template.py` and is ready for
battle as `ml_play_battle.py`.
"""

# ──────────────────────────────────────────────────────────────────────────────
# ───── Global configuration ──────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

# Device ­— automatically fallback to CPU on TA / CI machines
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#本人已經把訓練模式改爲False，因爲懶惰改code了qq
TRAINING_MODE: bool = False              # ⇧ flip to False when submitting
RESUME_TRAINING = False

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))   # ml_play_battle.py 所在資料夾
model_path    = os.path.join(BASE_DIR, "model.pt")    # ← ✔️  single source of truth
MODEL_SAVE_DIR = BASE_DIR        # save next to the script
MODEL_LOAD_PATH = model_path     # use the variable the way you asked

# PPO hyper‑parameters
LEARNING_RATE   = 1e-3
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_RATIO      = 0.15
VALUE_COEF      = 0.5
ENTROPY_COEF    = 0.005
MAX_GRAD_NORM   = 0.5
UPDATE_EPOCHS   = 6
BUFFER_SIZE     = 3072
BATCH_SIZE      = 512
SAVE_FREQUENCY  = 10

# Network
HIDDEN_SIZE = 128

# Reward shaping weights (feel free to retune)
REWARD_WEIGHTS = {
    "checkpoint":  5.0,
    "progress":    0.3,   # dense progress reward
    "health":      0.01,  # change in health
    "completion":  5.0,    # finished the race
    "mud": 0.0
}

# ──────────────────────────────────────────────────────────────────────────────
# ───── Helper: flatten observations to 1‑D list ──────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def _flatten(obj: Any, out: List[float]):
    """Recursively flatten lists / dicts / scalars into *out* list."""
    if obj is None:
        return
    if isinstance(obj, (int, float, np.floating, np.integer)):
        out.append(float(obj))
    elif isinstance(obj, (list, tuple, np.ndarray)):
        for v in obj:
            _flatten(v, out)
    elif isinstance(obj, dict):
        # sort keys for consistent order
        for k in sorted(obj.keys()):
            _flatten(obj[k], out)
    else:
        # unknown type → ignore
        pass

# ──────────────────────────────────────────────────────────────────────────────
# ───── ObservationProcessor ──────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class ObservationProcessor:
    """Turn the raw Proly observation dict into a 1-D float tensor."""

    def __init__(self):
        self.observation_size: int | None = None  # lazily filled

    # public ------------------------------------------------------------------
    def process(self, obs: Dict[str, Any]) -> torch.Tensor:
        flat: List[float] = []
        _flatten(obs, flat)
        # lazily record size (first call)
        if self.observation_size is None:
            self.observation_size = len(flat)
            #print(f"[ObsProc] observation_size = {self.observation_size}")
        # pad / truncate to fixed size if obs changes slightly
        if len(flat) < self.observation_size:
            flat.extend([0.0] * (self.observation_size - len(flat)))
        elif len(flat) > self.observation_size:
            flat = flat[: self.observation_size]
        return torch.tensor(flat, dtype=torch.float32, device=DEVICE)

    def get_size(self) -> int:
        if self.observation_size is None:
            # default large upper‑bound until first obs arrives
            return 256
        return self.observation_size

# ──────────────────────────────────────────────────────────────────────────────
# ───── ActionProcessor ───────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class ActionProcessor:
    """Convert network tensors → game action tuple (cont, disc)."""

    def __init__(self, action_space_info):  # action_space_info not used (yet)
        # Proly uses 2 continuous + 2 discrete binaries
        self.cont_size = 2
        self.disc_size = 2  # binary (select, use)
        self.action_size = (self.cont_size, self.disc_size)

    # public ------------------------------------------------------------------
    def create_action(self, cont: torch.Tensor, disc: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        cont_np = cont.detach().cpu().numpy().astype(np.float32)
        disc_np = disc.detach().cpu().numpy().astype(np.int32)
        return cont_np, disc_np

    def get_size(self):
        return self.action_size

# ──────────────────────────────────────────────────────────────────────────────
# ───── Reward shaping ────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class RewardCalculator:
    """Lightweight dense reward helper (optional)."""

    def __init__(self):
        self.w = REWARD_WEIGHTS
        self.reset()

    # public ------------------------------------------------------------------
    def reset(self):
        self.prev_checkpoint = -1
        self.prev_pos: np.ndarray | None = None
        self.prev_health = 0.0

    def __call__(self, obs: Dict[str, Any], raw_reward: float, done: bool) -> float:
        shaped = 0.0

        # checkpoint passed
        cp = int(obs.get("last_checkpoint_index", -1))
        if cp > self.prev_checkpoint:
            shaped += self.w["checkpoint"] * (cp - self.prev_checkpoint)
            self.prev_checkpoint = cp

        # progress (distance to target decreases)
        pos = np.array(obs.get("agent_position", [0, 0, 0]), dtype=np.float32)
        target = np.array(obs.get("target_position", [0, 0, 0]), dtype=np.float32)
        dist = np.linalg.norm(target - pos)
        if self.prev_pos is not None:
            prev_dist = np.linalg.norm(target - self.prev_pos)
            shaped += self.w["progress"] * (prev_dist - dist)  # positive if we moved closer
        self.prev_pos = pos

        # health change
        health = float(obs.get("agent_health", 0.0))
        shaped += self.w["health"] * (health - self.prev_health)
        self.prev_health = health
        
        # --- mud penalty -------------------------------------------------
        mud_near = False
        for obj in obs.get("nearby_map_objects", []):
            if int(obj.get("object_type", 0)) == 1:  # 1 = MudPit :contentReference[oaicite:1]{index=1}
                    dx, dz = obj["relative_position"]
                    dist_sq = dx*dx + dz*dz
                    if dist_sq < 1.5**2:                # within 1.5 m of centre
                        mud_near = True
                        break
        if mud_near:
            shaped += self.w["mud"] 

        # add raw reward and shaped
        total = raw_reward + shaped

        # completion bonus
        if done and cp >= 0:
            total += self.w["completion"]
        return total

# ──────────────────────────────────────────────────────────────────────────────
# ───── Experience buffer ─────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class ExperienceBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.clear()

    def add(self, *entry):
        self.storage.append(entry)

    def clear(self):
        self.storage: list = []

    def mini_batch_indices(self, batch_size: int):
        idx = np.random.permutation(len(self.storage))
        for start in range(0, len(idx), batch_size):
            yield idx[start : start + batch_size]

# ──────────────────────────────────────────────────────────────────────────────
# ───── Neural network (actor‑critic) ─────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class PPOModel(nn.Module):
    def __init__(self, obs_size: int, cont_size: int, disc_size: int):
        super().__init__()
        self.obs_size, self.cont_size, self.disc_size = obs_size, cont_size, disc_size

        # shared feature extractor
        self.fc = nn.Sequential(
            nn.Linear(obs_size, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.Tanh()
        )

        # continuous actions – mean + log_std (state‑independent log_std)
        self.mu = nn.Linear(HIDDEN_SIZE, cont_size)
        self.log_std = nn.Parameter(torch.zeros(cont_size))

        # discrete actions – logits for Bernoulli (multi‑binary)
        self.disc_logits = nn.Linear(HIDDEN_SIZE, disc_size)

        # value head
        self.v = nn.Linear(HIDDEN_SIZE, 1)

    # helper --------------------------------------------------------------
    def _dist(self, x):
        feat = self.fc(x)
        # continuous Normal distribution
        mu = torch.tanh(self.mu(feat))  # tanh because we want in (‑1,1)
        std = torch.exp(self.log_std)
        cont_dist = Normal(mu, std)
        # discrete Bernoulli distribution (independent)
        logits = self.disc_logits(feat)
        disc_dist = Bernoulli(logits=logits)
        value = self.v(feat).squeeze(-1)
        return cont_dist, disc_dist, value

    # public ---------------------------------------------------------------
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cont_d, disc_d, value = self._dist(obs)
        cont_action = torch.clamp(cont_d.sample(), -1.0, 1.0)
        disc_action = disc_d.sample()
        return cont_action, disc_action, value

    def act_and_logprob(self, obs: torch.Tensor):
        cont_d, disc_d, value = self._dist(obs)
        cont = torch.clamp(cont_d.sample(), -1.0, 1.0)
        disc = disc_d.sample()
        logp = cont_d.log_prob(cont).sum(-1) + disc_d.log_prob(disc).sum(-1)
        return cont, disc, logp, value

    def evaluate(self, obs: torch.Tensor, cont_a: torch.Tensor, disc_a: torch.Tensor):
        cont_d, disc_d, value = self._dist(obs)
        logp = cont_d.log_prob(cont_a).sum(-1) + disc_d.log_prob(disc_a).sum(-1)
        entropy = cont_d.entropy().sum(-1) + disc_d.entropy().sum(-1)
        return logp, entropy, value

# ──────────────────────────────────────────────────────────────────────────────
# ───── Main MLPlay class ─────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class MLPlay:
    """PPO agent compatible with the MLGame3D callback interface."""

    def __init__(self, action_space_info=None):
        super().__init__()
        self.name = "PPO_Agent"
        self.episode_return = 0.0
        self.episode_step   = 0

        # processors
        self.obs_proc = ObservationProcessor()
        self.act_proc = ActionProcessor(action_space_info)
        self.rew_calc = RewardCalculator()

        # actor-critic model (created lazily)
        self.model: PPOModel | None = None
        self.optimizer: torch.optim.Optimizer | None = None

        # experience
        self.buffer = ExperienceBuffer(BUFFER_SIZE)
        self.step_count = 0
        self.episode = 0

        # flags
        self.need_resume = TRAINING_MODE and RESUME_TRAINING
        self.defer_model_load = (
            not TRAINING_MODE and os.path.exists(MODEL_LOAD_PATH)
        )   # ← delay loading until obs_size is known

        # create ./models/ folder (or whatever MODEL_SAVE_DIR is)
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # ─── MLGame3D hooks ────────────────────────────────────────────────────
    def reset(self):
        #if self.episode_step > 0:
        #    print(f"[Episode] reward_sum = {self.episode_return:.3f}   steps = {self.episode_step}")

        self.rew_calc.reset()
        self.step_count = 0
        self.episode += 1
        self.episode_return = 0.0
        self.episode_step = 0
        
        if self.episode % SAVE_FREQUENCY == 0 and self.model is not None:
            ckpt = {
                "model": self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "episode": self.episode,
            }
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"{MODEL_SAVE_DIR}/model_ep{self.episode}_{timestamp}.pt"
            torch.save(ckpt, save_path)
            print(f"[Checkpoint] Saved at episode {self.episode} → {save_path}")
           
        if TRAINING_MODE and len(self.buffer.storage) > 0:
            print(f"[Forced Update] buffer = {len(self.buffer.storage)} (below full size)")
            self._update_policy()
            self.buffer.clear()   
            
    def update(self, observations: Dict[str, Any], done: bool = False, info: Dict[str, Any] | None = None):
        obs_t = self.obs_proc.process(observations)
        
        if self.model is None:
            obs_size = self.obs_proc.get_size()
            self._build_model(obs_size)

            if hasattr(self, "defer_model_load") and self.defer_model_load:
                ckpt = torch.load(MODEL_LOAD_PATH, map_location=DEVICE)
                if isinstance(ckpt, dict) and "model" in ckpt:
                    self.model.load_state_dict(ckpt["model"])
                else:
                    self.model.load_state_dict(ckpt)
                #print(f"[MLPlay] Loaded model from {MODEL_LOAD_PATH}")
                self.defer_model_load = False

        # raw reward from environment if provided
        raw_r = 0.0
        if info and "reward" in info:
            raw_r = float(info["reward"])
        shaped_r = self.rew_calc(observations, raw_r, done)
        
        self.episode_return += shaped_r
        self.episode_step   += 1

        # --------------- choose action ---------------
        if TRAINING_MODE:
            cont_a, disc_a, logp, value = self.model.act_and_logprob(obs_t)
        else:
            cont_a, disc_a, value = self.model.act(obs_t)
            logp = torch.tensor(0.0, device=DEVICE)

        # --------------- send to game ---------------
        cont_np, disc_np = self.act_proc.create_action(cont_a, disc_a)
        cont_np = self._avoid_mud(observations, cont_np)   # ← NEW
        game_action = (cont_np, disc_np)

        # --------------- store experience ---------------
        if TRAINING_MODE:
            self.buffer.add(
                obs_t.detach(), cont_a.detach(), disc_a.detach(),
                logp.detach(), value.detach(), shaped_r, done
            )

        # --------------- update ---------------
        self.step_count += 1
        #if self.step_count % 100 == 0:
        #    if "agent_position" in observations and "target_position" in observations:
        #        pos    = np.array(observations["agent_position"])
        #        target = np.array(observations["target_position"])
        #        print(f"[Step {self.step_count}] Distance to checkpoint: {np.linalg.norm(target - pos):.2f}")

        return game_action

    def _avoid_mud(self, obs: Dict[str, Any], cont_vec: np.ndarray) -> np.ndarray:
        # cont_vec: [ax, az] in agent local coords
        mud_ahead = None
        for obj in obs.get("nearby_map_objects", []):
            if int(obj["object_type"]) == 1:
                dx, dz = obj["relative_position"]
                # look only within a 120° cone in front and < 3 m away
                if dz > 0 and dx*dx + dz*dz < 3.0**2:
                    mud_ahead = (dx, dz)
                    break
        if mud_ahead is None:
            return cont_vec         # keep original

        # simple side-step: steer away from mud centre
        dx, dz = mud_ahead
        side = -np.sign(dx) if dx != 0 else np.random.choice([-1, 1])
        avoid_vec = np.array([side, 0.0], dtype=np.float32)
        # blend: 70 % original intent, 30 % avoidance
        new_vec = 0.7*cont_vec + 0.3*avoid_vec
        # clip to [-1,1]
        return np.clip(new_vec, -1.0, 1.0)

    # ─── internal helpers ────────────────────────────────────────────────
    def _build_model(self, obs_size: int):
        if self.model is not None:
            return
        cont, disc = self.act_proc.cont_size, self.act_proc.disc_size
        self.model = PPOModel(obs_size, cont, disc).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        #print(f"[MLPlay] Model built - obs {obs_size}, cont {cont}, disc {disc}")
        
        # ---- Lazy resume once the model shape is known -----------
        if self.need_resume and os.path.exists(MODEL_LOAD_PATH):
            ckpt = torch.load(MODEL_LOAD_PATH, map_location=DEVICE)

            # Back-compat: support both pure state_dict and full dict
            if isinstance(ckpt, dict) and "model" in ckpt:
                self.model.load_state_dict(ckpt["model"])
                self.optimizer.load_state_dict(ckpt["optim"])
                self.episode = ckpt.get("episode", 0)
            else:
                self.model.load_state_dict(ckpt)

            print(f"[MLPlay] Resumed from {MODEL_LOAD_PATH}")
            self.need_resume = False

    def _update_policy(self):
        if self.model is None or len(self.buffer.storage) == 0:
            return

        # ------ stack once, then index with the arrays we produce ------
        obs      = torch.stack([e[0] for e in self.buffer.storage]).to(DEVICE)
        cont_act = torch.stack([e[1] for e in self.buffer.storage]).to(DEVICE)
        disc_act = torch.stack([e[2] for e in self.buffer.storage]).to(DEVICE)
        old_logp = torch.stack([e[3] for e in self.buffer.storage]).to(DEVICE)
        old_val  = torch.stack([e[4] for e in self.buffer.storage]).to(DEVICE)
        rewards  = torch.tensor([e[5] for e in self.buffer.storage], dtype=torch.float32, device=DEVICE)
        dones    = torch.tensor([e[6] for e in self.buffer.storage], dtype=torch.float32, device=DEVICE)

        # ------- proper GAE with value bootstrap -------
        with torch.no_grad():
            last_val = self.model.v(self.model.fc(obs[-1])).squeeze(0)
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            next_value = last_val if t == len(rewards) - 1 else old_val[t + 1]
            delta = rewards[t] + GAMMA * next_value * mask - old_val[t]
            gae = delta + GAMMA * GAE_LAMBDA * mask * gae
            advantages[t] = gae
        returns = advantages + old_val
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        print(f"  avg R = {rewards.mean():.3f}  avg |adv| = {advantages.abs().mean():.3f}")

        # --------------- PPO epochs ---------------
        for _ in range(UPDATE_EPOCHS):
            for mb_idx in self.buffer.mini_batch_indices(BATCH_SIZE):
                b_obs      = obs[mb_idx]
                b_cont_act = cont_act[mb_idx]
                b_disc_act = disc_act[mb_idx]
                b_old_logp = old_logp[mb_idx]
                b_returns  = returns[mb_idx]
                b_adv      = advantages[mb_idx]

                logp, entropy, value = self.model.evaluate(b_obs, b_cont_act, b_disc_act)
                ratio = torch.exp(logp - b_old_logp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO) * b_adv
                policy_loss  = -torch.min(surr1, surr2).mean()
                value_loss   = F.mse_loss(value, b_returns)
                entropy_loss = -entropy.mean()

                loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

        print(f"[PPO] update ▶ steps {len(rewards)}  loss {loss.item():.3f}")
        
        # ─── Update learning rate scheduler ───
        if hasattr(self, "scheduler"):
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"[LR] Scheduler stepped. Current LR: {current_lr:.5e}")
        
        ckpt = {
            "model": self.model.state_dict(),
            "optim": self.optimizer.state_dict(),
            "episode": self.episode,
        }
        save_path = f"{MODEL_SAVE_DIR}/model_latest.pt"
        torch.save(ckpt, save_path)
        print(f"[AutoSave] Saved to {save_path} at episode {self.episode}")
