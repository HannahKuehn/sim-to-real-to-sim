import pybullet as p
import pybullet_data
import time
import numpy as np

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)
p.loadURDF("plane.urdf")

DAMPING = 0.8
CHASE_GAIN = 5.0
BARRIER_RADIUS = 0.5
BOUNDS_RADIUS = 3.0
BOUNDS_GAIN = 8.0


def make_drone(color, pos):
    # Inner drone marker (visual only, no collision)
    drone_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.12, rgbaColor=color)
    # Outer barrier sphere (collision + transparent visual)
    barrier_visual = p.createVisualShape(
        p.GEOM_SPHERE, radius=BARRIER_RADIUS,
        rgbaColor=[color[0], color[1], color[2], 0.15],
    )
    barrier_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=BARRIER_RADIUS)

    # Base = barrier sphere; fixed child link = small drone marker at center
    body = p.createMultiBody(
        baseMass=1,
        baseCollisionShapeIndex=barrier_collision,
        baseVisualShapeIndex=barrier_visual,
        basePosition=pos,
        linkMasses=[0],
        linkCollisionShapeIndices=[-1],
        linkVisualShapeIndices=[drone_visual],
        linkPositions=[[0, 0, 0]],
        linkOrientations=[[0, 0, 0, 1]],
        linkInertialFramePositions=[[0, 0, 0]],
        linkInertialFrameOrientations=[[0, 0, 0, 1]],
        linkParentIndices=[0],
        linkJointTypes=[p.JOINT_FIXED],
        linkJointAxis=[[0, 0, 1]],
    )
    p.changeDynamics(body, -1, linearDamping=0.0, angularDamping=0.0, restitution=1.0)
    return body


def react(follower_pos, follower_vel, leader_pos):
    """Compute force that drives the follower toward the leader."""
    to_leader = leader_pos - follower_pos
    return CHASE_GAIN * to_leader - DAMPING * follower_vel


def move(drone1, drone2, t):
    pos1, _ = p.getBasePositionAndOrientation(drone1)
    vel1, _ = p.getBaseVelocity(drone1)
    pos1 = np.array(pos1)
    vel1 = np.array(vel1)

    leader_force = np.array([
        np.sin(t * 1.0),
        np.cos(t * 0.7),
        np.sin(t * 0.5) * 0.3,
    ]) * 5.0 - DAMPING * vel1

    # Soft restoring force to keep drone1 within BOUNDS_RADIUS of origin
    dist_from_origin = np.linalg.norm(pos1)
    if dist_from_origin > BOUNDS_RADIUS:
        overshoot = dist_from_origin - BOUNDS_RADIUS
        leader_force -= BOUNDS_GAIN * overshoot * (pos1 / dist_from_origin)

    p.applyExternalForce(drone1, -1, leader_force.tolist(), pos1.tolist(), p.WORLD_FRAME)

    pos2, _ = p.getBasePositionAndOrientation(drone2)
    vel2, _ = p.getBaseVelocity(drone2)
    pos2 = np.array(pos2)
    vel2 = np.array(vel2)

    follower_force = react(pos2, vel2, pos1)
    p.applyExternalForce(drone2, -1, follower_force.tolist(), pos2.tolist(), p.WORLD_FRAME)


drone1 = make_drone([1, 0, 0, 1], [0, 0, 1])
drone2 = make_drone([0, 0, 1, 1], [3, 3, 1])

t = 0
while True:
    t += 1 / 240
    move(drone1, drone2, t)
    p.stepSimulation()
    time.sleep(1 / 240)