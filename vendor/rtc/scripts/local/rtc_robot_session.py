"""RTC lifecycle and wall-clock scheduling, independent of robot transports."""

from concurrent.futures import ThreadPoolExecutor
import math
import time


class RobotRTCSession:
    """Keep keyboard handling responsive through bootstrap, pause and shutdown.

    The main thread calls poll() and owns observation capture and publication.
    Controller start/close run on a lifecycle worker; model RPCs run on the
    controller's own worker. Pausing detaches the entire queue immediately.
    """

    def __init__(self, factory, read_observation, publish, hz=50, clock=time.monotonic):
        if not math.isfinite(hz) or hz <= 0:
            raise ValueError("Publish frequency must be positive")
        self.factory = factory
        self.read_observation = read_observation
        self.publish = publish
        self.hz = hz
        self.clock = clock
        self.lifecycle = ThreadPoolExecutor(max_workers=1)
        self.controller = None
        self.bootstrap = None
        self.cleanup = None
        self.epoch = None
        self.last_tick = -1

    @property
    def active(self):
        return self.controller is not None

    @property
    def running(self):
        return self.active and self.bootstrap is None

    def resume(self):
        if self.active:
            raise RuntimeError("RTC session is already active")
        if self.cleanup is not None:
            if not self.cleanup.done():
                raise RuntimeError("Previous request is closing; wait, then press p again")
            self.cleanup.result()
        observation = self.read_observation()
        if observation is None:
            raise RuntimeError("No fresh observation for RTC bootstrap")
        self.controller = self.factory()
        self.bootstrap = self.lifecycle.submit(self.controller.start, observation)
        self.epoch = None
        self.last_tick = -1

    def pause(self):
        controller = self.controller
        self.controller = None
        self.bootstrap = None
        self.epoch = None
        if controller is not None:
            # Queued after any bootstrap task. No discarded result can reattach
            # its queue, and no socket is closed from a different worker.
            self.cleanup = self.lifecycle.submit(controller.close)

    def poll(self):
        if not self.active:
            return
        try:
            if self.bootstrap is not None:
                if not self.bootstrap.done():
                    return
                self.bootstrap.result()
                self.bootstrap = None
                self.epoch = self.clock()
                print(f"RTC ready: delay={self.controller.delay}, rate={self.hz} Hz", flush=True)
            tick = math.floor((self.clock() - self.epoch) * self.hz + 1e-7)
            if tick == self.last_tick:
                return
            if tick != self.last_tick + 1:
                raise RuntimeError(
                    "Missed RTC control tick; paused without replaying stale actions"
                )
            observation = self.read_observation()
            if observation is None:
                raise RuntimeError("Missing/stale observation during RTC execution")
            action = self.controller.step(observation, tick)
            if self.clock() >= self.epoch + (tick + 1) / self.hz:
                raise RuntimeError("Observation/RTC processing exceeded one control tick")
            self.publish(action)
            self.last_tick = tick
        except Exception:
            self.pause()
            raise

    def sleep_seconds(self):
        if not self.running:
            return 0.005
        return max(0, self.epoch + (self.last_tick + 1) / self.hz - self.clock())

    def close(self):
        self.pause()
        try:
            if self.cleanup is not None:
                self.cleanup.result()
        finally:
            self.lifecycle.shutdown(wait=True)
