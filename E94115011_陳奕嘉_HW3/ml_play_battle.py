# ml_play_battle.py  (burst‑free)

import os
import pickle
import random
from typing import Dict, Tuple, List, Optional

# ── run‑time hyper‑parameters ─────────────────────────────────────────
ACTIONS        = ["UP", "DOWN", "LEFT", "RIGHT", "NONE"]
STEP_PENALTY   = -0.001
ALPHA_MIN      = 0.08
GAMMA          = 0.9
ONLINE_EPSILON = 0.01          # exploration during play
SAVE_EVERY     = 0             # 0 ⇒ never auto‑save
DANGER_RADIUS  = 35.0          # reflex layer
# ---------------------------------------------------------------------

# =====================================================================
# 1.  State encoder  (unchanged from training)
# =====================================================================
def encode_state(scene: Dict) -> Tuple[int, ...]:
    sx, sy   = scene["self_x"], scene["self_y"]
    vel      = scene["self_vel"]

    look_near   = vel * 5
    corridor_hw = vel * 3

    dirs = {"UP": (0, -1), "DOWN": (0, 1),
            "LEFT": (-1, 0), "RIGHT": (1, 0)}
    near, far = {d: 0.0 for d in dirs}, {d: 0.0 for d in dirs}

    # --------------------------------------------------------------
    # accumulate corridor sums
    for itm in scene.get("foods", []):
        rx, ry, score = itm["x"] - sx, itm["y"] - sy, itm.get("score", 0)
        for d, (dx, dy) in dirs.items():
            proj = rx*dx + ry*dy
            perp = abs(-dy*rx + dx*ry)
            if perp > corridor_hw:
                continue
            w = min(1.0 / (proj + 1e-6), 2.0)
            if 0 < proj <= look_near:
                near[d] += score * w * 80
            elif look_near < proj:
                far[d]  += score * w * 50

    # --------------------------------------------------------------
    # radial cue – encourage moving **downward only**
    # trigger only when UP & DOWN near‑corridor sums are zero
    if near["UP"] == near["DOWN"] == 0:
        best_below_d2 = 40.0**2            # search radius squared
        for itm in scene.get("foods", []):
            if itm.get("score", 0) <= 0:   # skip garbage
                continue
            dy = itm["y"] - sy
            if dy <= 0:                    # we only care about food below us
                continue
            dx  = itm["x"] - sx
            d2  = dx*dx + dy*dy
            if d2 < best_below_d2:         # choose closest food BELOW
                best_below_d2 = d2
        if best_below_d2 < 40.0**2:        # found something within range
            near["DOWN"] = 140              # strong positive cue downward

    # --------------------------------------------------------------
    clamp = lambda v: int(round(max(-100, min(100, v))))
    return tuple(clamp(near[d]) for d in ("UP","DOWN","LEFT","RIGHT")) + \
           tuple(clamp(far[d])  for d in ("UP","DOWN","LEFT","RIGHT"))

# =====================================================================
# 2.  Q‑learning helper
# =====================================================================
def update_q(table, s, a, r, s_next, alpha, gamma):
    if s not in table:
        table[s] = dict.fromkeys(ACTIONS, 0.0)
    if s_next not in table:
        table[s_next] = dict.fromkeys(ACTIONS, 0.0)
    old = table[s][a]
    nxt = max(table[s_next].values())
    table[s][a] = old + alpha * (r + gamma*nxt - old)

# =====================================================================
# 3.  Reflex layer (garbage avoidance)
# =====================================================================
def imminent_garbage(scene: Dict, action: str,
                     radius: float = DANGER_RADIUS) -> bool:
    x, y, vel = scene["self_x"], scene["self_y"], scene["self_vel"]

    if action == "UP":   y -= vel
    elif action == "DOWN": y += vel
    elif action == "LEFT": x -= vel
    elif action == "RIGHT": x += vel

    w = scene.get("env", {}).get("width",  1200)
    h = scene.get("env", {}).get("height", 650)
    x = max(0, min(x, w))
    y = max(0, min(y, h))

    for itm in scene.get("foods", []):
        if itm.get("score", 0) >= 0:
            continue
        if (itm["x"] - x)**2 + (itm["y"] - y)**2 < radius**2:
            return True
    return False

# =====================================================================
# 4.  MLPlay
# =====================================================================
class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.model_path = os.path.join(os.path.dirname(__file__),
                                       "model.pickle")
        with open(self.model_path, "rb") as f:
            self.q_table: Dict[Tuple[int, ...], Dict[str, float]] = pickle.load(f)

        self.prev_state: Optional[Tuple[int, ...]] = None
        self.prev_action: Optional[str] = None
        self.prev_game_score = 0
        self.round_count = 0

    # ---------------------------- helpers ----------------------------
    def _choose(self, state) -> str:
        if random.random() < ONLINE_EPSILON or state not in self.q_table:
            return random.choice(ACTIONS)
        return max(self.q_table[state], key=self.q_table[state].get)

    @staticmethod
    def _reward(prev_score, new_score) -> float:
        return new_score - prev_score + STEP_PENALTY

    # ------------------------------ core -----------------------------
    def update(self, scene_info: Dict, *args, **kwargs) -> List[str]:
        if scene_info.get("status") != "GAME_ALIVE":
            return ["NONE"]

        state = encode_state(scene_info)

        # online Q‑update for the last step
        if self.prev_state is not None and self.prev_action is not None:
            r = self._reward(self.prev_game_score,
                             scene_info.get("game_score", 0))
            update_q(self.q_table, self.prev_state, self.prev_action,
                     r, state, alpha=ALPHA_MIN, gamma=GAMMA)

        # pick action
        action = self._choose(state)

        # reflex: dodge imminent garbage
        if imminent_garbage(scene_info, action):
            for alt in ("LEFT", "RIGHT", "UP", "DOWN", "NONE"):
                if alt != action and not imminent_garbage(scene_info, alt):
                    action = alt
                    break

        # remember for next frame
        self.prev_state      = state
        self.prev_action     = action
        self.prev_game_score = scene_info.get("game_score", 0)

        return [action]

    # ---------------------------- reset ------------------------------
    def reset(self):
        self.prev_state = None
        self.prev_action = None
        self.prev_game_score = 0
        self.round_count += 1

        if SAVE_EVERY > 0 and self.round_count % SAVE_EVERY == 0:
            with open(self.model_path, "wb") as f:
                pickle.dump(self.q_table, f)
            print(f"[MLPlay] Q‑table saved after {self.round_count} rounds.")
