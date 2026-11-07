"""Offline CoMotion-style local rotation deltas; no hardware communication."""
import numpy as np
from scipy.spatial.transform import Rotation as R

LEGS = {'left': (1, 4, 7, 10), 'right': (2, 5, 8, 11)}


def amplify_windows(poses, side, gain, max_angle=.35, rate=.6, dt=.02):
    """Apply bounded extra rotation relative to frame-zero, without accumulation.

    Same local convention as CoMotion: R_new = R_ref @ Exp(delta).
    Rate limits apply to the extra SMPL rotation, not the resulting robot speed.
    """
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 4 or poses.shape[1:] != (10, 21, 3) or not np.isfinite(poses).all():
        raise ValueError('Expected finite [N,10,21,3] reference windows')
    if side not in LEGS or not 1 <= gain <= 2 or not 0 < max_angle <= .4 or not 0 < rate <= 1 or dt != .02:
        raise ValueError('Invalid leg delta settings')
    ids = np.array(LEGS[side]) - 1
    anchor = R.from_rotvec(poses[0][:, ids].reshape(-1, 3))
    result = poses.copy()
    deltas = np.zeros((len(poses), 10, 4, 3))
    previous = np.zeros((40, 3))
    for i, window in enumerate(poses):
        reference = R.from_rotvec(window[:, ids].reshape(-1, 3))
        target = (anchor.inv() * reference).as_rotvec() * (gain - 1)
        target *= np.minimum(1., max_angle / np.maximum(np.linalg.norm(target, axis=1), 1e-12))[:, None]
        step = (R.from_rotvec(previous).inv() * R.from_rotvec(target)).as_rotvec()
        step *= np.minimum(1., rate * dt / np.maximum(np.linalg.norm(step, axis=1), 1e-12))[:, None]
        delta = (R.from_rotvec(previous) * R.from_rotvec(step)).as_rotvec()
        if i == 0 or gain == 1:
            delta[:] = 0.
        deltas[i] = delta.reshape(10, 4, 3)
        if np.any(delta):
            result[i][:, ids] = (reference * R.from_rotvec(delta)).as_rotvec().reshape(10, 4, 3)
        previous = delta
    return result, deltas
