# sim-to-real-to-sim

PyBullet simulation of two drones: a leader that moves autonomously and a follower that learns to chase it.

## Files

| File | Purpose |
|---|---|
| `pybulletsim.py` | Real-time GUI simulation with a hardcoded follower |
| `drone_learner.py` | Train and evaluate a learned follower policy |

## Running

**Watch the simulation (hardcoded follower):**
```bash
python pybulletsim.py
```

**Train a follower policy:**
```bash
python drone_learner.py train cem   # Cross-Entropy Method (default)
python drone_learner.py train ga    # Genetic Algorithm
```
Runs 60 generations in headless mode and saves the policy to `drone_policy.npy`.

**Evaluate the trained policy:**
```bash
python drone_learner.py eval
```
Opens the GUI and loops indefinitely at 2× speed. Press Ctrl-C to stop.

---

## What it does

The **red drone (leader)** follows a scripted multi-frequency sinusoidal path and is softly constrained to stay within 3 m of the origin.

The **blue drone (follower)** tries to minimise its distance to the leader. In `pybulletsim.py` this is a hand-coded proportional spring force. In `drone_learner.py` it is a small neural network trained by one of the optimizers below.

Each drone is surrounded by a transparent barrier sphere (radius 0.5 m). The spheres collide elastically — if the follower catches the leader they bounce off rather than overlap.

---

## How the policy works

The follower policy is a two-layer neural network:

```
state (6 numbers) → [16 hidden neurons, tanh] → force (3 numbers)
```

**Input** (computed every timestep):
- `leader_pos − follower_pos` — where the leader is relative to the follower
- `follower_vel` — the follower's current velocity

**Output:** a 3D force vector applied directly to drone2 by the physics engine, clipped to ±15 N by a final `tanh`. The network has no memory — it reacts to the current snapshot only, which is sufficient since relative position and own velocity capture everything needed to chase.

Training finds values for the ~160 weights and biases such that the network produces good chasing behaviour as measured by the fitness function.

---

## Learning algorithms

Both optimizers treat the policy as a black box. Neither uses gradients. The loop is:

1. **Sample** a population of candidate parameter vectors
2. **Score** each by running a full simulation episode: `fitness = −mean(distance to leader)`
3. **Update** the population toward the higher-scoring candidates

### CEMOptimizer (Cross-Entropy Method)

Maintains a Gaussian distribution over parameters. Each generation it fits that distribution to the elite candidates and shrinks the search radius (`sigma`) over time. Fast to converge; can get stuck if `sigma` starts too small.

### GAOptimizer (Genetic Algorithm)

Maintains a population of individuals. Each generation:
- The top `elite_frac` are carried over unchanged
- The rest are bred by **tournament selection** (pick the best of a random subset), **uniform crossover** (mix two parents gene-by-gene), and **Gaussian mutation**

More exploratory than CEM — crossover can reach parts of the parameter space neither parent occupied.

### Adding your own optimizer

Subclass `PolicyOptimizer` and implement `optimize(fitness_fn, initial_params)`:

```python
from drone_learner import PolicyOptimizer

class MyOptimizer(PolicyOptimizer):
    def optimize(self, fitness_fn, initial_params):
        # fitness_fn(params) -> float, higher is better
        ...
        return best_params
```

Pass it to `train()`:
```python
learner.train(optimizer=MyOptimizer())
```

---

## Key parameters

Constants at the top of each file — **keep them in sync** between `pybulletsim.py` and `drone_learner.py`:

| Parameter | Default | Effect |
|---|---|---|
| `BARRIER_RADIUS` | 0.5 m | Radius of the collision sphere around each drone |
| `BOUNDS_RADIUS` | 3.0 m | How far the leader can stray from the origin |
| `DAMPING` | 0.8 | Drag applied to both drones |
| `LEADER_FORCE_SCALE` | 8.0 | Peak thrust of the leader |
| `CHASE_GAIN` | 5.0 | Spring stiffness of the hardcoded follower (pybulletsim.py only) |

---

## Extending

- **Change the follower objective** — edit `DroneReactionLearner.fitness()` in `drone_learner.py` and retrain. The optimizers are unaffected.
- **Use the learned policy in the main sim** — replace the `react()` call in `pybulletsim.py`:
  ```python
  from drone_learner import DroneReactionLearner
  learner = DroneReactionLearner()
  learner.load()
  # inside move():
  follower_force = learner.get_action(pos2, vel2, pos1)
  ```
- **Export follower trajectory for a real Crazyflie** — log `pos2` each step and stream as position setpoints via `cflib`. Requires a Crazyflie positioning system (Lighthouse, LPS, or motion capture).
