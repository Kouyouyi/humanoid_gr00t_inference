"""Bounded SONIC slow-walk request, not a distance or odometry controller.

Planner mode 1 is SLOW_WALK. Keep cleared upper-body targets while native SONIC
generates the leg motion. This module has no sockets or hardware side effects.
"""
import numpy as np

from .state import HARDWARE_TO_ISAAC

UPPER_HARDWARE=[12,13,14,15,22,16,23,17,24,18,25,19,26,20,27,21,28]
UPPER_IDS=np.array([HARDWARE_TO_ISAAC.index(i) for i in UPPER_HARDWARE])
RAMP_SECONDS=.5


def validate_walk(speed, plateau_seconds):
    if (isinstance(speed,bool) or isinstance(plateau_seconds,bool) or
        not np.isfinite([speed,plateau_seconds]).all() or
        not .1<=speed<=.15 or not .25<=plateau_seconds<=1.):
        raise ValueError('Walk requires speed 0.10..0.15 m/s and plateau 0.25..1.0 s')


def pulse(elapsed, speed=.1, plateau_seconds=.5):
    validate_walk(speed,plateau_seconds)
    if not np.isfinite(elapsed):raise ValueError('Invalid walk elapsed time')
    total=2*RAMP_SECONDS+plateau_seconds
    if elapsed<=0 or elapsed>=total:return dict(mode=0,movement=[0.,0.,0.],facing=[1.,0.,0.],speed=0.)
    u=min(elapsed/RAMP_SECONDS,(total-elapsed)/RAMP_SECONDS,1.)
    amount=u*u*(3-2*u)
    return dict(mode=1,movement=[1.,0.,0.],facing=[1.,0.,0.],speed=float(speed*amount))


def upper_fields(q,dq):
    q,dq=np.asarray(q),np.asarray(dq)
    if q.shape!=(29,) or dq.shape!=(29,) or not np.isfinite([q,dq]).all():
        raise ValueError('Invalid body reference for planner upper-body hold')
    return dict(upper_body_position=q[UPPER_IDS].tolist(),upper_body_velocity=dq[UPPER_IDS].tolist())
