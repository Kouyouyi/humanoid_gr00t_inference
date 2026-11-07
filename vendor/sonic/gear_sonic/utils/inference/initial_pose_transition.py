"""Nonblocking smoothstep transition from the controller's current latent."""

import math
import time

import numpy as np


class InitialPoseTransition:
    def __init__(self, target, duration_s, publish_rate):
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("initial_pose_blend_duration must be finite and positive")
        if not math.isfinite(publish_rate) or publish_rate <= 0:
            raise ValueError("action_publish_rate must be finite and positive")
        self.target = self._token(target)
        self.steps = max(1, math.ceil(duration_s * publish_rate))
        self.active = False
        self.step = 0
        self.origin = None

    @staticmethod
    def _token(value):
        token = np.asarray(value, dtype=np.float32)
        if token.shape != (64,) or not np.isfinite(token).all():
            raise ValueError("Expected a finite 64D controller token_state")
        return token.copy()

    def start(self, state, now_ns=None):
        """Require fresh telemetry from the C++ process on this workstation.

        Never substitute zero or a policy prediction for missing controller state.
        A failed restart leaves the existing transition unchanged.
        """
        if state is None:
            raise ValueError("No controller state; wait for 'k' startup and retry 'i'")
        stamp = state.get("state_monotonic_ns")
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        if not isinstance(stamp, int) or not 0 <= now_ns - stamp <= 250_000_000:
            raise ValueError("Missing/stale controller timestamp; initial pose was not sent")
        origin = self._token(state.get("token_state"))
        self.origin = origin
        self.step = 0
        self.active = True

    def next_token(self):
        if not self.active:
            raise RuntimeError("No active initial pose transition")
        # One sample per publish tick: scheduling delays slow the transition,
        # rather than skipping ahead and making a large latent jump.
        t = self.step / self.steps
        alpha = t * t * (3.0 - 2.0 * t)
        token = ((1.0 - alpha) * self.origin + alpha * self.target).astype(np.float32)
        if self.step == self.steps:
            self.active = False
            return self.target.copy(), True
        self.step += 1
        return token, False

    def cancel(self):
        self.active = False
