"""Commissioning references: terminal native idle and one bounded elbow round trip.

Uses SONIC's original 1762-dimensional encoder layout.
No IK, VLA, root translation, hand motion or low-level motor command is generated.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .state import NAMES, JOINTS, checked_state

RATE = 50
LOOKAHEAD = 55
MAX_DELTA = .05
DURATION = 7.


def from_native_idle(row):
    measured = checked_state(row)
    if row['motion'] != 'planner_motion' or not row['encoder_source']:
        raise ValueError('Expected measured native planner idle')
    if measured['tilt_rad'] > .25 or max(abs(v) for v in measured['dq']) > 1:
        raise ValueError('Native standing not settled')
    obs = np.asarray(row['encoder_obs'], dtype=float)
    if obs.shape != (1762,) or not np.isfinite(obs).all() or np.any(obs[:4] != 0):
        raise ValueError('Only finite original-release G1 encoder observations are supported')
    q, dq = obs[4:294].reshape(10, 29), obs[294:584].reshape(10, 29)
    orientations = obs[601:661].reshape(10, 6)
    if max(np.ptp(x, axis=0).max() for x in (q, dq, orientations)) > 1e-5:
        raise ValueError('Reference is not a terminal idle window')
    columns = orientations[0].reshape(3, 2)
    x = columns[:, 0]
    if np.linalg.norm(x) < 1e-6:
        raise ValueError('Degenerate encoder orientation')
    x = x / np.linalg.norm(x)
    y = columns[:, 1] - x * (x @ columns[:, 1])
    if np.linalg.norm(y) < 1e-6:
        raise ValueError('Degenerate encoder orientation')
    y = y / np.linalg.norm(y)
    relative = Rotation.from_matrix(np.column_stack((x, y, np.cross(x, y))))
    base = Rotation.from_quat(measured['base_quat_wxyz'][1:] + measured['base_quat_wxyz'][:1])
    quat = (base * relative).as_quat()[[3, 0, 1, 2]]
    return dict(joint_pos=np.repeat(q[:1], LOOKAHEAD, axis=0).astype(np.float32),
                joint_vel=np.repeat(dq[:1], LOOKAHEAD, axis=0).astype(np.float32),
                body_quat=np.repeat(quat[None], LOOKAHEAD, axis=0).astype(np.float32))


def check_pulse(side, delta_rad):
    if side not in ('left', 'right'):
        raise ValueError('side must be left or right')
    if type(delta_rad) not in (int, float) or not np.isfinite(delta_rad) or not 0 < abs(delta_rad) <= MAX_DELTA:
        raise ValueError(f'delta_rad must be finite, nonzero, and at most {MAX_DELTA} rad')


def pulse_envelope(t):
    """One second entry, two second rise, one hold, two return, one settle."""
    t = np.asarray(t, dtype=float)
    u = np.where(t < 3, (t-1)/2, (6-t)/2)
    u = np.clip(u, 0, 1)
    s = u**3*(10+u*(-15+6*u))
    derivative = 30*u*u*(1-u)**2/2
    derivative *= np.where(t < 3, 1, -1)
    return s, derivative


def pulse_window(reference, side, delta_rad, elapsed, frame):
    check_pulse(side, delta_rad)
    if not np.isfinite(elapsed) or elapsed < 0 or type(frame) is not int or frame < 0:
        raise ValueError('Invalid trajectory clock/frame')
    index = NAMES.index(side+'_elbow_joint')
    q = np.repeat(reference['joint_pos'][:1], LOOKAHEAD, axis=0)
    dq = np.repeat(reference['joint_vel'][:1], LOOKAHEAD, axis=0)
    amplitude, velocity = pulse_envelope(elapsed+np.arange(LOOKAHEAD)/RATE)
    q[:, index] += delta_rad*amplitude
    dq[:, index] += delta_rad*velocity
    joint = JOINTS[index]
    if np.any(q[:, index] < joint['lower']+.1) or np.any(q[:, index] > joint['upper']-.1):
        raise ValueError('Elbow reference too close to model limit')
    return dict(joint_pos=q.astype(np.float32), joint_vel=dq.astype(np.float32),
                body_quat=np.repeat(reference['body_quat'][:1], LOOKAHEAD, axis=0).astype(np.float32),
                frame_index=np.arange(frame, frame+LOOKAHEAD, dtype=np.int64))


def hold_window(reference, frame):
    return {key: np.repeat(reference[key][:1], LOOKAHEAD, axis=0).astype(np.float32)
            for key in ('joint_pos', 'joint_vel', 'body_quat')} | {
                'frame_index': np.arange(frame, frame+LOOKAHEAD, dtype=np.int64)}
