"""Control-clock RTC queue. No hardware I/O; the caller owns the control loop."""

import math

import numpy as np


def latency_steps(latencies_s, control_hz=50.0, max_delay_steps=10, margin_steps=1):
    values = np.asarray(latencies_s, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Provide finite nonnegative round-trip latency samples")
    if not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("control_hz must be positive")
    if type(margin_steps) is not int or margin_steps < 0:
        raise ValueError("margin_steps must be a nonnegative integer")
    delay = math.ceil(float(np.quantile(values, 0.95)) * control_hz) + margin_steps
    if delay > max_delay_steps:
        raise ValueError(
            f"Measured delay needs {delay} control steps; trained maximum is {max_delay_steps}. Do not silently clamp."
        )
    return delay


class RTCActionQueue:
    """Single-stream queue indexed by absolute control ticks, batch size one.

    At request tick s capture observation and prefix(s,d) atomically, then keep
    executing the old queue while inference runs on a worker. On response at tick
    t, install(s,result,t,d): elapsed output steps are discarded, never replayed.
    The caller must serialize queue access and allow only one request in flight.
    """

    def __init__(self):
        self.start_tick = 0
        self.actions = None

    def _offset(self, tick):
        if type(tick) is not int or tick < 0:
            raise ValueError("tick must be a nonnegative integer")
        offset = tick - self.start_tick
        if self.actions is None or not 0 <= offset < len(next(iter(self.actions.values()))):
            raise RuntimeError("Action queue exhausted; caller must enter its configured fallback")
        return offset

    def prefix(self, tick, delay_steps):
        if type(delay_steps) is not int or delay_steps < 0:
            raise ValueError("delay_steps must be a nonnegative integer")
        offset = self._offset(tick)
        if offset + delay_steps > len(next(iter(self.actions.values()))):
            raise RuntimeError("Insufficient queued actions to cover inference latency")
        return {k: v[offset : offset + delay_steps][None].copy() for k, v in self.actions.items()}

    def action_at(self, tick):
        offset = self._offset(tick)
        return {k: v[offset].copy() for k, v in self.actions.items()}

    def install(self, request_tick, actions, now_tick, delay_steps):
        if any(type(v) is not int or v < 0 for v in (request_tick, now_tick, delay_steps)):
            raise ValueError("ticks and delay must be nonnegative integers")
        elapsed = now_tick - request_tick
        if elapsed < 0 or elapsed > delay_steps:
            raise RuntimeError(
                "Late/stale RTC response: retain queued actions and recalibrate; do not replay elapsed actions"
            )
        if not actions:
            raise ValueError("Empty action chunk")
        horizon = None
        for v in actions.values():
            if (
                not isinstance(v, np.ndarray)
                or v.ndim != 3
                or v.shape[0] != 1
                or not np.isfinite(v).all()
            ):
                raise ValueError("Expected finite (1, horizon, dim) arrays")
            horizon = v.shape[1] if horizon is None else horizon
            if v.shape[1] != horizon or elapsed >= horizon:
                raise ValueError("Invalid action horizons")
        self.actions = {k: v[0, elapsed:].copy() for k, v in actions.items()}
        self.start_tick = now_tick


class AsyncRTCController:
    """Nonblocking control-tick adapter; inference client is owned by one worker.

    Call start(observation) before enabling execution. Then call step(observation,
    tick) once per actual control tick, on the same control thread. Observations
    must correspond to that tick. Exceptions require the caller's fallback.
    Omit delay_steps to discover the server's pinned delay during start().
    """

    def __init__(self, client_factory, delay_steps=None, request_interval=10):
        from concurrent.futures import ThreadPoolExecutor

        if delay_steps is not None and (type(delay_steps) is not int or not 1 <= delay_steps <= 10):
            raise ValueError("Async RTC delay must be in [1, 10]")
        if type(request_interval) is not int or request_interval < (delay_steps or 1):
            raise ValueError("request_interval must be an integer >= delay_steps")
        self.delay = delay_steps
        self.interval = request_interval
        self.queue = RTCActionQueue()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.factory = client_factory
        self.client = None
        self.pending = None
        self.request_tick = None
        self.next_request = 0
        self.last_tick = -1

    def _infer(self, observation, options):
        if self.client is None:
            self.client = self.factory()
        return self.client.get_action(observation, options)

    def _bootstrap(self, observation):
        # Resolve server configuration on the worker that owns the ZMQ socket.
        if self.client is None:
            self.client = self.factory()
        if self.delay is None:
            config = self.client.get_rtc_config()
            delay = config.get("rtc_delay_steps")
            if type(delay) is not int or not 1 <= delay <= 10:
                raise ValueError(
                    "Server has no valid RTC deployment delay; set delay_steps explicitly"
                )
            if self.interval < delay:
                raise ValueError("request_interval must be >= server RTC delay")
            self.delay = delay
        return self._infer(observation, None)

    def start(self, observation):
        if self.queue.actions is not None:
            raise RuntimeError("Controller already started")
        actions, _ = self.executor.submit(self._bootstrap, observation).result()
        horizon = next(iter(actions.values())).shape[1]
        if self.interval + self.delay >= horizon:
            raise ValueError("Request interval + delay must be smaller than action horizon")
        self.queue.install(0, actions, 0, 0)

    def step(self, observation, tick):
        from copy import deepcopy

        if type(tick) is not int or tick != self.last_tick + 1:
            raise RuntimeError("Control ticks must be consecutive; do not replay missed ticks")
        self.last_tick = tick
        if self.pending is not None:
            if self.pending.done():
                actions, _ = self.pending.result()
                self.queue.install(self.request_tick, actions, tick, self.delay)
                self.pending = None
            elif tick - self.request_tick >= self.delay:
                raise RuntimeError("RTC inference missed its deadline; enter caller fallback")
        if self.pending is None and tick >= self.next_request:
            prefix = self.queue.prefix(tick, self.delay)
            self.pending = self.executor.submit(
                self._infer,
                deepcopy(observation),
                {"rtc_delay_steps": self.delay, "rtc_prefix": prefix},
            )
            self.request_tick = tick
            self.next_request = tick + self.interval
        return self.queue.action_at(tick)

    def close(self):
        # Worker owns the socket throughout its lifetime, including closing it.
        def close_client():
            if self.client is not None and hasattr(self.client, "close"):
                self.client.close()

        self.executor.submit(close_client).result()
        self.executor.shutdown(wait=True)
