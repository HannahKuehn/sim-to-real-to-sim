# sim-to-real-to-sim

PyBullet simulation of two drones: a leader that moves autonomously and a follower that learns to chase it.

## Files

| File | Purpose |
|---|---|
| `pybulletsim.py` | Real-time GUI simulation |
| `drone_learner.py` | Train and evaluate a learned follower policy |

## Running

**Watch the simulation (hardcoded follower):**
```bash
python pybulletsim.py
```

**Train a follower policy:**
```bash
python drone_learner.py train
```
Runs ~60 generations of CEM in headless mode and saves the policy to `drone_policy.npy`.

**Evaluate the trained policy:**
```bash
python drone_learner.py eval
```
Opens the GUI and loops indefinitely. Press Ctrl-C to stop.

## What it does

The **red drone (leader)** moves along a scripted multi-frequency path and stays within a 3 m radius of the origin.

The **blue drone (follower)** tries to minimise its distance to the leader. In `pybulletsim.py` this is a hand-coded spring force. In `drone_learner.py` it is a small neural network trained with the **Cross-Entropy Method (CEM)**: candidate policies are sampled, scored by mean distance to the leader over an episode, and the best fraction is used to update the next generation.

Each drone is surrounded by a transparent barrier sphere (radius 0.5 m). The spheres collide elastically — if the follower catches the leader they bounce off rather than overlap.

## Dependencies

```
pip install pybullet numpy
```

## Key parameters

All in the top of each file, kept in sync:

| Parameter | Default | Effect |
|---|---|---|
| `BARRIER_RADIUS` | 0.5 m | Size of the collision sphere around each drone |
| `BOUNDS_RADIUS` | 3.0 m | How far the leader can stray from the origin |
| `DAMPING` | 0.8 | Drag applied to both drones |
| `CHASE_GAIN` | 5.0 | Spring stiffness of the hardcoded follower |
| `LEADER_FORCE_SCALE` | 8.0 | Peak thrust of the leader |

## Extending

- **Change the follower objective** — edit `DroneReactionLearner.fitness()` in `drone_learner.py` and retrain.
- **Use the learned policy in the main sim** — import the class and replace the `react()` call in `pybulletsim.py`:
  ```python
  from drone_learner import DroneReactionLearner
  learner = DroneReactionLearner()
  learner.load()
  # inside move():
  follower_force = learner.get_action(pos2, vel2, pos1)
  ```
- **Export follower trajectory for a real Crazyflie** — log `pos2` each step and stream as position setpoints via `cflib`. Requires a Crazyflie positioning system (Lighthouse, LPS, or motion capture).
