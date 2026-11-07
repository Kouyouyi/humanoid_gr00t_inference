"""BrainCo binary teleop over the same DDS protocol as xr_teleoperate.

Six normalized positions per hand selected from configurable open/closed poses, dq=1. This module
never sends a startup/shutdown pose and never creates a body publisher.
"""

from collections import deque
import json
import math
import threading
import time


from gear_sonic.utils.teleop.brainco_poses import SIDES, load_poses, pose_targets


class BinaryHandInput:
    """Require fresh XR frames and released controls before arming."""

    def __init__(self, close_threshold=0.55, open_threshold=0.45, timeout=0.2):
        if not 0 <= open_threshold < close_threshold <= 1:
            raise ValueError("Require 0 <= open threshold < close threshold <= 1")
        self.close_threshold = close_threshold
        self.open_threshold = open_threshold
        self.timeout = timeout
        self.stamp = None
        self.received = None
        self.armed = False
        self.closed = [False, False]
        self.status = "waiting for fresh PICO frames"

    def reset(self, reason):
        self.armed = False
        self.closed = [False, False]
        self.status = reason

    def update(self, stamp, left, right, enabled=True, now=None):
        now = time.monotonic() if now is None else now
        values = (left, right)
        if not enabled:
            self.reset("paused")
            self.stamp = None
            return None
        if stamp is None or stamp <= 0 or not all(
            math.isfinite(v) and 0 <= v <= 1 for v in values
        ):
            self.reset("invalid PICO input")
            self.stamp = None
            return None
        if self.stamp is None or stamp < self.stamp:
            self.reset("waiting for advancing PICO timestamp")
            self.stamp, self.received = stamp, now
            return None
        if now - self.received > self.timeout:
            self.reset("PICO timeout; release both controls to re-arm")
        if stamp == self.stamp:
            return None
        self.stamp, self.received = stamp, now
        if not self.armed:
            if any(v > self.open_threshold for v in values):
                self.status = "release both controls to arm"
                return None
            self.armed = True
        for i, value in enumerate(values):
            if value >= self.close_threshold:
                self.closed[i] = True
            elif value <= self.open_threshold:
                self.closed[i] = False
        self.status = "active"
        return tuple(int(v) for v in self.closed)


class BraincoHands:
    """DDS output with fresh feedback gating; call update from the input loop.

    dry_run subscribes to feedback but creates no command publishers. It can
    preview PICO actions even when the robot-side service is unavailable.
    """

    def __init__(self, interface, *, dry_run=False, log_path=None, pose_config=""):
        self.poses = load_poses(pose_config)
        self.last_q_command = None
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
        )
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

        if not interface:
            raise ValueError("An explicit BrainCo network interface is required")
        self.dry_run = dry_run
        self.input = BinaryHandInput()
        self.lock = threading.Lock()
        self.states = {}
        self.publishers = {}
        self.subscribers = []
        self.last_write = float("-inf")
        self.last_report = None
        self.last_action = None
        self.command_history = deque(maxlen=128)
        self.command_session_id = time.time_ns()
        self.command_sequence = 0
        self.last_command_timestamp_ns = 0
        self.last_pico_timestamp_ns = 0
        self.command_enabled = False
        self.log = None
        self._command_type = MotorCmds_
        self._motor_type = unitree_go_msg_dds__MotorCmd_
        ChannelFactoryInitialize(0, interface)
        try:
            if log_path:
                self.log = open(log_path, "a", buffering=1)
            for side in SIDES:
                sub = ChannelSubscriber(f"rt/brainco/{side}/state", MotorStates_)
                self.subscribers.append(sub)
                sub.Init(lambda msg, side=side: self._on_state(side, msg), 1)
                if not dry_run:
                    pub = ChannelPublisher(f"rt/brainco/{side}/cmd", MotorCmds_)
                    self.publishers[side] = pub
                    pub.Init()
        except BaseException:
            self.close()
            raise
        print(f"[BrainCo] {'DRY RUN' if dry_run else 'DDS'} on {interface}; "
              "0=open, 1=closed. Release both controls to arm.")

    def _on_state(self, side, message):
        q = [float(m.q) for m in message.states]
        dq = [float(m.dq) for m in message.states]
        valid = len(q) == 6 and all(math.isfinite(x) and 0 <= x <= 1 for x in q)
        valid = valid and len(dq) == 6 and all(math.isfinite(x) for x in dq)
        with self.lock:
            self.states[side] = (time.monotonic(), q, dq, valid, time.time_ns())

    def feedback(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            states = dict(self.states)
        ready = all(
            side in states and states[side][3] and 0 <= now - states[side][0] <= 0.2
            for side in SIDES
        )
        return ready, {
            side: {"q": state[1], "dq": state[2], "age_s": now - state[0], "valid": state[3]}
            for side, state in states.items()
        }

    def update(self, stamp, left, right, *, enabled=True):
        self.command_enabled = False
        now = time.monotonic()
        ready, states = self.feedback(now)
        if not ready and not self.dry_run:
            self.input.reset("waiting for fresh six-motor feedback from BOTH hands")
            targets = None
        else:
            targets = self.input.update(stamp, left, right, enabled, now)
        status = self.input.status
        report = (status, targets if targets is not None else self.last_action)
        if report != self.last_report:
            print(f"[BrainCo] {status}; left/right={report[1]}")
            self.last_report = report
        if targets is None or now - self.last_write < 0.01:
            return False
        q_command = pose_targets(self.poses, targets)
        if not self.dry_run:
            for side in SIDES:
                message = self._command_type()
                message.cmds = [self._motor_type() for _ in range(6)]
                for motor, q in zip(message.cmds, q_command[side]):
                    motor.q, motor.dq = q, 1.0
                if not self.publishers[side].Write(message):
                    self.input.reset("DDS write failed")
                    raise RuntimeError(f"BrainCo {side} DDS write failed")
        self.last_write, self.last_action = now, targets
        self.last_q_command = q_command
        self.last_command_timestamp_ns = time.time_ns()
        self.command_sequence += 1
        if not self.dry_run:
            self.command_history.append({
                'sequence': self.command_sequence, 'monotonic_ns': time.monotonic_ns(),
                'timestamp_ns': self.last_command_timestamp_ns, 'binary': list(targets),
                'q_command': q_command,
            })
        self.last_pico_timestamp_ns = int(stamp)
        self.command_enabled = not self.dry_run
        if self.log:
            self.log.write(json.dumps({
                "timestamp_ns": time.time_ns(), "monotonic_s": now,
                "pico_timestamp_ns": int(stamp), "dry_run": self.dry_run,
                "binary_action": dict(zip(SIDES, targets)),
                "q_command": q_command,
                "pose_config": self.poses,
                "feedback_ready": ready, "feedback": states,
            }) + "\n")
        return True

    def recording_snapshot(self):
        """Read-only telemetry; command validity means a DDS write succeeded.

        q is the service's normalized actuator position, not an angle in radians.
        Feedback timestamps are workstation DDS receipt times, not device clocks.
        """
        now = time.monotonic()
        with self.lock:
            states = dict(self.states)
        return {
            "version": 3,
            "pose_config": self.poses,
            "q_command": self.last_q_command,
            "monotonic_ns": time.monotonic_ns(),
            "command_session_id": self.command_session_id,
            "command_history": list(self.command_history),
            "timestamp_ns": time.time_ns(),
            "feedback": {
                side: {"q": state[1], "dq": state[2],
                       "valid": state[3] and 0 <= now - state[0] <= 0.2,
                       "received_timestamp_ns": state[4]}
                for side, state in states.items()
            },
            "binary_command": list(self.last_action) if self.last_action is not None else None,
            "command_timestamp_ns": self.last_command_timestamp_ns,
            "pico_timestamp_ns": self.last_pico_timestamp_ns,
            "command_valid": self.last_action is not None and not self.dry_run,
            "command_enabled": self.command_enabled,
            "dry_run": self.dry_run,
        }

    def close(self):
        # Closing DDS does not command open or cancel motion already accepted.
        for sub in self.subscribers:
            sub.Close()
        self.subscribers.clear()
        for pub in self.publishers.values():
            pub.Close()
        self.publishers.clear()
        if self.log:
            self.log.close()
            self.log = None
