"""Sequential standard-GR00T chunks; no RTC prefix or stale response replay."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import time
import numpy as np


def validate_plain_server(client, adapter):
    remote = client.get_modality_config()
    for mode, expected in adapter.modalities.items():
        for field in ('modality_keys', 'delta_indices'):
            actual = remote[mode]
            value = actual[field] if isinstance(actual, dict) else getattr(actual, field)
            if list(value) != list(expected[field]):
                raise ValueError('Server/client layouts differ')
    config = client.get_rtc_config()
    if config.get('rtc_delay_steps') is not None or config.get('action_horizon') != 40:
        raise ValueError('Expected standard GR00T service without RTC delay')
    return config


class PlainChunkController:
    def __init__(self, factory):
        self.factory = factory
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.client = None
        self.pending = None
        self.actions = None
        self.cursor = 0
        self.last_tick = -1
        self.delay = None
        self.wait_started = None
        self.last_action = None

    def _infer(self, observation):
        if self.client is None:
            self.client = self.factory()
        return self.client.get_action(observation, None)

    def install(self, actions):
        if set(actions) != {'motion_token', 'left_hand_joints', 'right_hand_joints'}:
            raise ValueError('Unexpected standard GR00T outputs')
        for key, dim in (('motion_token', 64), ('left_hand_joints', 6), ('right_hand_joints', 6)):
            a = actions[key]
            if not isinstance(a, np.ndarray) or a.shape != (1, 40, dim) or not np.isfinite(a).all():
                raise ValueError('Invalid standard GR00T chunk: ' + key)
        self.actions = {k:v[0].copy() for k,v in actions.items()}
        self.cursor = 0

    def start(self, observation):
        self.install(self.executor.submit(self._infer, observation).result()[0])

    def step(self, observation, tick):
        if type(tick) is not int or tick != self.last_tick + 1:
            raise RuntimeError('Control ticks must be consecutive; do not replay missed ticks')
        self.last_tick = tick
        if self.pending is not None:
            if self.pending.done():
                self.install(self.pending.result()[0])
                self.pending = None
                self.wait_started = None
            elif time.monotonic() - self.wait_started > 2:
                raise RuntimeError('Standard GR00T inference timeout; pause and hold')
        if self.cursor >= 40 and self.pending is None:
            # The last chunk has finished. Observe NOW, hold its endpoint while
            # inference is in flight, then begin the next chunk from step zero.
            self.pending = self.executor.submit(self._infer, deepcopy(observation))
            self.wait_started = time.monotonic()
        if self.cursor < 40:
            self.last_action = {k:v[self.cursor].copy() for k,v in self.actions.items()}
            self.cursor += 1
        if self.last_action is None:
            raise RuntimeError('No standard GR00T endpoint to hold')
        return {k:v.copy() for k,v in self.last_action.items()}

    def close(self):
        def close_client():
            if self.client is not None:
                self.client.close()
        self.executor.submit(close_client).result()
        self.executor.shutdown(wait=True)
        self.actions = None
        self.pending = None
        self.last_action = None
