"""
DroneReactionLearner — learns how drone2 should react to drone1.

The learning algorithm is modular: pass any PolicyOptimizer to train().
Two implementations are provided:
  - CEMOptimizer    (Cross-Entropy Method, default)
  - GAOptimizer     (Genetic Algorithm)

Both treat the task as a black-box optimisation problem: sample a
population of policy parameter vectors, score each by running a full
simulation episode, and shift the population toward higher-scoring
individuals.  No gradients are needed.

Usage:
  python drone_learner.py train cem   # train with CEM (default)
  python drone_learner.py train ga    # train with genetic algorithm
  python drone_learner.py eval        # load policy and visualise

Integration with pybulletsim.py:
  from drone_learner import DroneReactionLearner
  learner = DroneReactionLearner(); learner.load()
  # replace react(...) call with:
  force = learner.get_action(follower_pos, follower_vel, leader_pos)
"""

import sys
import time
from abc import ABC, abstractmethod

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


# ══ optimizers ════════════════════════════════════════════════════════════════

class PolicyOptimizer(ABC):
    """
    Base class for population-based policy optimizers.

    Subclasses implement optimize(), which takes a fitness function and an
    initial parameter vector and returns the best parameter vector found.
    """

    @abstractmethod
    def optimize(self, fitness_fn, initial_params: np.ndarray) -> np.ndarray:
        """
        fitness_fn : callable(np.ndarray) -> float   higher = better
        initial_params : starting point for the search
        returns : best parameter vector found
        """


class CEMOptimizer(PolicyOptimizer):
    """
    Cross-Entropy Method.
    Maintains a Gaussian distribution over parameters.  Each generation:
      1. Sample pop_size candidates from N(mu, sigma^2 I)
      2. Score with fitness_fn
      3. Update mu to the mean of the top elite_frac candidates
      4. Decay sigma toward a floor
    """

    def __init__(self,
                 n_generations: int = 60,
                 pop_size: int = 24,
                 elite_frac: float = 0.25,
                 sigma: float = 0.5,
                 sigma_decay: float = 0.97,
                 sigma_floor: float = 0.02):
        self.n_generations = n_generations
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_floor = sigma_floor

    def optimize(self, fitness_fn, initial_params: np.ndarray) -> np.ndarray:
        mu = initial_params.copy()
        n = len(mu)
        n_elite = max(2, int(self.pop_size * self.elite_frac))
        sigma = self.sigma

        print(f"CEM  generations={self.n_generations}  "
              f"pop={self.pop_size}  elite={n_elite}  sigma0={sigma}")

        for gen in range(self.n_generations):
            candidates = mu + np.random.randn(self.pop_size, n) * sigma
            scores = np.array([fitness_fn(c) for c in candidates])

            elite_idx = np.argsort(scores)[-n_elite:]
            mu = candidates[elite_idx].mean(axis=0)
            sigma = max(self.sigma_floor, sigma * self.sigma_decay)

            print(f"  gen {gen + 1:3d}  "
                  f"mean_dist={-scores.mean():.3f}  "
                  f"best_dist={-scores[elite_idx[-1]]:.3f}  "
                  f"sigma={sigma:.4f}")

        return mu


class GAOptimizer(PolicyOptimizer):
    """
    Genetic Algorithm.
    Each generation:
      1. Score all individuals with fitness_fn
      2. Carry the top elite_frac directly to the next generation (elitism)
      3. Fill the rest via tournament selection + uniform crossover + mutation
    """

    def __init__(self,
                 n_generations: int = 60,
                 pop_size: int = 30,
                 elite_frac: float = 0.1,
                 mutation_sigma: float = 0.2,
                 mutation_decay: float = 0.98,
                 mutation_floor: float = 0.01,
                 crossover_rate: float = 0.7,
                 tournament_k: int = 3):
        self.n_generations = n_generations
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.mutation_sigma = mutation_sigma
        self.mutation_decay = mutation_decay
        self.mutation_floor = mutation_floor
        self.crossover_rate = crossover_rate
        self.tournament_k = tournament_k

    def _tournament(self, pop: np.ndarray, scores: np.ndarray) -> np.ndarray:
        idx = np.random.choice(len(pop), self.tournament_k, replace=False)
        return pop[idx[np.argmax(scores[idx])]].copy()

    def optimize(self, fitness_fn, initial_params: np.ndarray) -> np.ndarray:
        n = len(initial_params)
        n_elite = max(1, int(self.pop_size * self.elite_frac))
        mutation_sigma = self.mutation_sigma

        # Seed population around the initial params
        pop = initial_params + np.random.randn(self.pop_size, n) * 0.5

        best_params = initial_params.copy()
        best_score = -np.inf

        print(f"GA   generations={self.n_generations}  "
              f"pop={self.pop_size}  elite={n_elite}  "
              f"mutation_sigma0={mutation_sigma}")

        for gen in range(self.n_generations):
            scores = np.array([fitness_fn(ind) for ind in pop])

            # Track global best
            gen_best_idx = np.argmax(scores)
            if scores[gen_best_idx] > best_score:
                best_score = scores[gen_best_idx]
                best_params = pop[gen_best_idx].copy()

            # Elitism: carry top individuals unchanged
            elite_idx = np.argsort(scores)[-n_elite:]
            new_pop = [pop[i].copy() for i in elite_idx]

            # Fill rest with crossover + mutation
            while len(new_pop) < self.pop_size:
                parent_a = self._tournament(pop, scores)
                parent_b = self._tournament(pop, scores)

                if np.random.random() < self.crossover_rate:
                    mask = np.random.rand(n) < 0.5
                    child = np.where(mask, parent_a, parent_b)
                else:
                    child = parent_a.copy()

                child += np.random.randn(n) * mutation_sigma
                new_pop.append(child)

            pop = np.array(new_pop)
            mutation_sigma = max(self.mutation_floor,
                                 mutation_sigma * self.mutation_decay)

            print(f"  gen {gen + 1:3d}  "
                  f"mean_dist={-scores.mean():.3f}  "
                  f"best_dist={-best_score:.3f}  "
                  f"mutation={mutation_sigma:.4f}")

        return best_params


# ══ learner ═══════════════════════════════════════════════════════════════════

class DroneReactionLearner:
    STATE_DIM = 9   # [leader_pos - follower_pos,  follower_vel,  leader_vel]
    ACTION_DIM = 3  # force vector applied to drone2
    MAX_FORCE = 12.0  # matches leader peak (LEADER_FORCE_SCALE * 1.5)

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
                   follower_pos, follower_vel, leader_pos, leader_vel) -> np.ndarray:
        """Execution mode: return force vector for drone2."""
        state = np.concatenate([
            np.asarray(leader_pos) - np.asarray(follower_pos),
            np.asarray(follower_vel),
            np.asarray(leader_vel),
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
                     n_steps: int = 800,
                     render: bool = False,
                     playback_speed: float = 1.5) -> float:
        """Run one physics episode and return the fitness score."""
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

            state = np.concatenate([pos1 - pos2, vel2, vel1])
            force = self._forward(flat, state)
            p.applyExternalForce(d2, -1, force.tolist(), pos2.tolist(),
                                 p.WORLD_FRAME, physicsClientId=client)

            p.stepSimulation(physicsClientId=client)
            distances[step] = np.linalg.norm(pos1 - pos2)
            if render:
                time.sleep(1.0 / (SIM_HZ * playback_speed))

        p.disconnect(client)
        return self.fitness(distances)

    # ── training ──────────────────────────────────────────────────────────────

    def train(self, optimizer: PolicyOptimizer | None = None):
        """
        Train the policy using the given optimizer.
        Defaults to CEMOptimizer if none is provided.
        """
        if optimizer is None:
            optimizer = CEMOptimizer()
        fitness_fn = lambda params: self._run_episode(params)
        self.params = optimizer.optimize(fitness_fn, self.params)
        print("Training complete.")

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = "drone_policy.npy"):
        np.save(path, self.params)
        print(f"Saved policy → {path}")

    def load(self, path: str = "drone_policy.npy"):
        self.params = np.load(path)
        print(f"Loaded policy ← {path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

OPTIMIZERS = {
    "cem": CEMOptimizer,
    "ga":  GAOptimizer,
}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    learner = DroneReactionLearner()

    if mode == "train":
        algo = sys.argv[2] if len(sys.argv) > 2 else "cem"
        if algo not in OPTIMIZERS:
            print(f"Unknown optimizer '{algo}'. Choose from: {list(OPTIMIZERS)}")
            sys.exit(1)
        n_gen = int(sys.argv[3]) if len(sys.argv) > 3 else None
        kwargs = {"n_generations": n_gen} if n_gen is not None else {}
        learner.train(optimizer=OPTIMIZERS[algo](**kwargs))
        learner.save()

    elif mode == "eval":
        learner.load()
        episode = 0
        while True:
            score = learner._run_episode(
                learner.params, n_steps=1000, render=True, playback_speed=2.0)
            episode += 1
            print(f"Episode {episode}  fitness={score:.4f}  mean_dist={-score:.3f}")

    else:
        print("Usage: python drone_learner.py [train [cem|ga]] [eval]")
