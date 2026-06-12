"""
DroneReactionLearner — learns how drone2 should react to drone1.

Learning paradigm: Cross-Entropy Method (CEM)
  Why not RL?  RL (PPO, DDPG) works here too, but adds a value function,
  per-step reward signals, and careful hyperparameter tuning.  CEM is
  simpler: sample a population of policy parameter vectors, score each
  with the fitness function over a full episode, keep the top fraction,
  and shift the search distribution toward them.  For this problem size
  it converges in minutes with no gradient computation.

Usage:
  python drone_learner.py train   # run CEM, save policy
  python drone_learner.py eval    # load policy, visualise in GUI

Integration with pybulletsim.py:
  from drone_learner import DroneReactionLearner
  learner = DroneReactionLearner(); learner.load()
  # replace react(...) call with:
  force = learner.get_action(follower_pos, follower_vel, leader_pos)
"""

import sys
import time
import numpy as np
import pybullet as p
import pybullet_data

# ── must match pybulletsim.py ─────────────────────────────────────────────────
DAMPING = 0.8
BARRIER_RADIUS = 0.5
BOUNDS_RADIUS = 3.0
BOUNDS_GAIN = 8.0
LEADER_FORCE_SCALE = 8.0
SIM_HZ = 240
# ─────────────────────────────────────────────────────────────────────────────


class DroneReactionLearner:
    STATE_DIM = 6   # [leader_pos - follower_pos,  follower_vel]
    ACTION_DIM = 3  # force vector applied to drone2
    MAX_FORCE = 15.0

    def __init__(self, hidden_size: int = 16):
        h = hidden_size
        n, a = self.STATE_DIM, self.ACTION_DIM
        self._shapes = [(n, h), (h,), (h, a), (a,)]
        self._sizes  = [n * h,   h,   h * a,   a ]
        self.n_params = sum(self._sizes)
        self.params = self._init_params()

    # ── policy ────────────────────────────────────────────────────────────────

    def _init_params(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        parts = []
        for shape in self._shapes:
            scale = 1.0 / np.sqrt(shape[0]) if len(shape) > 1 else 1.0
            parts.append(rng.normal(0.0, scale, size=int(np.prod(shape))))
        return np.concatenate(parts)

    def _unpack(self, flat: np.ndarray):
        out, offset = [], 0
        for shape, size in zip(self._shapes, self._sizes):
            out.append(flat[offset: offset + size].reshape(shape))
            offset += size
        return out  # W1, b1, W2, b2

    def _forward(self, flat: np.ndarray, state: np.ndarray) -> np.ndarray:
        W1, b1, W2, b2 = self._unpack(flat)
        h = np.tanh(state @ W1 + b1)
        return np.tanh(h @ W2 + b2) * self.MAX_FORCE

    def get_action(self,
                   follower_pos, follower_vel, leader_pos) -> np.ndarray:
        """Execution mode: return force vector for drone2."""
        state = np.concatenate([
            np.asarray(leader_pos) - np.asarray(follower_pos),
            np.asarray(follower_vel),
        ])
        return self._forward(self.params, state)

    # ── fitness ───────────────────────────────────────────────────────────────

    @staticmethod
    def fitness(distances: np.ndarray) -> float:
        """
        Higher is better.  Current objective: minimise mean distance to drone1.
        Change this function to alter the learned behaviour.
        """
        return -float(np.mean(distances))

    # ── simulation episode ────────────────────────────────────────────────────

    def _run_episode(self,
                     flat: np.ndarray,
                     n_steps: int = 400,
                     render: bool = False,
                     playback_speed: float = 1.0) -> float:
        """Run one physics episode; return fitness score."""
        client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=client)
        p.setGravity(0, 0, 0, physicsClientId=client)
        p.loadURDF("plane.urdf", physicsClientId=client)

        def spawn(pos, color):
            r, g, b, _ = color
            drone_vis = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.12,
                rgbaColor=color, physicsClientId=client)
            barrier_vis = p.createVisualShape(
                p.GEOM_SPHERE, radius=BARRIER_RADIUS,
                rgbaColor=[r, g, b, 0.15], physicsClientId=client)
            barrier_col = p.createCollisionShape(
                p.GEOM_SPHERE, radius=BARRIER_RADIUS, physicsClientId=client)
            body = p.createMultiBody(
                baseMass=1,
                baseCollisionShapeIndex=barrier_col,
                baseVisualShapeIndex=barrier_vis,
                basePosition=pos,
                linkMasses=[0],
                linkCollisionShapeIndices=[-1],
                linkVisualShapeIndices=[drone_vis],
                linkPositions=[[0, 0, 0]],
                linkOrientations=[[0, 0, 0, 1]],
                linkInertialFramePositions=[[0, 0, 0]],
                linkInertialFrameOrientations=[[0, 0, 0, 1]],
                linkParentIndices=[0],
                linkJointTypes=[p.JOINT_FIXED],
                linkJointAxis=[[0, 0, 1]],
                physicsClientId=client,
            )
            p.changeDynamics(body, -1, linearDamping=0.0, angularDamping=0.0,
                             restitution=1.0, physicsClientId=client)
            return body

        rng = np.random.default_rng()
        d1 = spawn([0.0, 0.0, 1.0], [1, 0, 0, 1])
        start = rng.uniform(-2.0, 2.0, 3)
        start[2] = float(np.abs(start[2])) + 0.5
        d2 = spawn(start.tolist(), [0, 0, 1, 1])

        distances = np.empty(n_steps)
        t = 0.0

        for step in range(n_steps):
            t += 1.0 / SIM_HZ

            pos1, _ = p.getBasePositionAndOrientation(d1, physicsClientId=client)
            vel1, _ = p.getBaseVelocity(d1, physicsClientId=client)
            pos1, vel1 = np.array(pos1), np.array(vel1)

            f1 = (np.array([
                      np.sin(t * 2.5) + 0.5 * np.sin(t * 4.1),
                      np.cos(t * 1.9) + 0.5 * np.cos(t * 3.3),
                      np.sin(t * 1.4) * 0.6,
                  ]) * LEADER_FORCE_SCALE - DAMPING * vel1)
            d_origin = np.linalg.norm(pos1)
            if d_origin > BOUNDS_RADIUS:
                f1 -= BOUNDS_GAIN * (d_origin - BOUNDS_RADIUS) * (pos1 / d_origin)
            p.applyExternalForce(d1, -1, f1.tolist(), pos1.tolist(),
                                 p.WORLD_FRAME, physicsClientId=client)

            pos2, _ = p.getBasePositionAndOrientation(d2, physicsClientId=client)
            vel2, _ = p.getBaseVelocity(d2, physicsClientId=client)
            pos2, vel2 = np.array(pos2), np.array(vel2)

            state = np.concatenate([pos1 - pos2, vel2])
            force = self._forward(flat, state)
            p.applyExternalForce(d2, -1, force.tolist(), pos2.tolist(),
                                 p.WORLD_FRAME, physicsClientId=client)

            p.stepSimulation(physicsClientId=client)
            distances[step] = np.linalg.norm(pos1 - pos2)
            if render:
                time.sleep(1.0 / (SIM_HZ * playback_speed))

        p.disconnect(client)
        return self.fitness(distances)

    # ── training (CEM) ────────────────────────────────────────────────────────

    def train(self,
              n_generations: int = 60,
              pop_size: int = 24,
              elite_frac: float = 0.25,
              sigma: float = 0.5):
        """
        Cross-Entropy Method:
          1. Sample pop_size candidates around the current mean (self.params)
          2. Score each with _run_episode / fitness()
          3. Keep the top elite_frac; update the mean to their centroid
          4. Slowly shrink sigma so the search converges
        """
        mu = self.params.copy()
        n_elite = max(2, int(pop_size * elite_frac))

        print(f"CEM  generations={n_generations}  pop={pop_size}  elite={n_elite}")
        print(f"     params={self.n_params}  sigma0={sigma}")

        for gen in range(n_generations):
            candidates = mu + np.random.randn(pop_size, self.n_params) * sigma
            scores = np.array([self._run_episode(c) for c in candidates])

            elite_idx = np.argsort(scores)[-n_elite:]
            mu = candidates[elite_idx].mean(axis=0)
            sigma = max(0.02, sigma * 0.97)

            print(f"  gen {gen + 1:3d}  "
                  f"mean_dist={-scores.mean():.3f}  "
                  f"best_dist={-scores[elite_idx[-1]]:.3f}  "
                  f"sigma={sigma:.4f}")

        self.params = mu
        print("Training complete.")

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = "drone_policy.npy"):
        np.save(path, self.params)
        print(f"Saved policy → {path}")

    def load(self, path: str = "drone_policy.npy"):
        self.params = np.load(path)
        print(f"Loaded policy ← {path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    learner = DroneReactionLearner()

    if mode == "train":
        learner.train()
        learner.save()

    elif mode == "eval":
        learner.load()
        episode = 0
        while True:
            score = learner._run_episode(learner.params, n_steps=600, render=True, playback_speed=2.0)
            episode += 1
            print(f"Episode {episode}  fitness={score:.4f}  mean_dist={-score:.3f}")

    else:
        print("Usage: python drone_learner.py [train|eval]")
