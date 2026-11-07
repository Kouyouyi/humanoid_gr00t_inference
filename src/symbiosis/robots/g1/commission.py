"""Operator-only SONIC idle owner. Not an agent tool; publishes no walking commands.

The native controller owns DDS and the O-key stop. This process only requests
planner IDLE, with an explicit --start-policy option. SIGINT exits the upper owner;
the native controller remains running and its transport watchdog latches IDLE.
"""
import argparse
import json
from pathlib import Path
import signal
import time

import numpy as np
import zmq

from .camera_service import atomic_json
from .state import TraceReader, checked_state
from .vendor import wire_module


def run(args):
    args.run.mkdir(parents=True, exist_ok=True)
    wire = wire_module()
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    publisher.setsockopt(zmq.SNDHWM, 10)
    publisher.bind(f'tcp://127.0.0.1:{args.port}')  # Exclusive ownership; fail on occupied port.
    trace = TraceReader(args.run/'controller_trace.jsonl')
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    phase, fault = 'waiting_for_native_init', None
    started = time.monotonic()
    start_requested = False
    stable_since = None
    next_tick = started
    last_report = 0
    last_index = None
    events = (args.run/'commission_events.jsonl').open('a')

    def event(kind, **data):
        record = dict(event=kind, monotonic=time.monotonic(), **data)
        events.write(json.dumps(record)+'\n'); events.flush()
        print(json.dumps(record), flush=True)

    try:
        while not stop:
            now = time.monotonic()
            if now-next_tick > .1 and start_requested:
                fault = 'owner_tick_late'
                phase = 'fault_idle'
            next_tick = now+.02
            log = args.run/'controller.log'
            if phase == 'waiting_for_native_init' and log.exists() and 'Init Done' in log.read_text(errors='replace'):
                phase = 'ready_to_start'
                event('native_init_observed')
            pulse = False
            if phase == 'ready_to_start' and args.start_policy:
                start_requested = True
                pulse = True
                phase = 'starting_planner_idle'
                started = now
                event('operator_policy_start_requested')
            publisher.send(wire.build_planner_message(0, [0, 0, 0], [1, 0, 0], speed=0.))
            publisher.send(wire.build_command_message(start=pulse, stop=False, planner=True))
            measured = None
            if start_requested:
                try:
                    row = trace.poll()
                    measured = checked_state(row)
                    index = measured['state_monotonic_ns']
                    if last_index is not None and index < last_index:
                        raise ValueError('native_controller_restarted')
                    last_index = index
                    if measured['tilt_rad'] > .5:
                        raise ValueError('base_tilt_above_0.5_rad')
                    stable = (measured['tilt_rad'] < .25 and max(abs(v) for v in measured['dq']) < 1.
                              and np.linalg.norm(measured['base_ang_vel']) < .25
                              and measured['motion'] == 'planner_motion' and measured['encoder_source'])
                    if not stable:
                        stable_since = None
                    elif stable_since is None:
                        stable_since = time.monotonic()
                    if phase != 'fault_idle' and stable_since and time.monotonic()-stable_since >= 3:
                        if phase != 'standing_observed':
                            event('standing_observed', note='IMU/joints only; foot support not measured')
                        phase = 'standing_observed'
                except (OSError, KeyError, ValueError, RuntimeError) as exc:
                    if phase != 'starting_planner_idle' or time.monotonic()-started > 20:
                        if phase != 'fault_idle':
                            event('fault_idle', reason=str(exc))
                        phase, fault = 'fault_idle', str(exc)
            if now-last_report > .2:
                atomic_json(args.run/'commission_status.json', dict(phase=phase, fault=fault,
                    updated_monotonic=time.monotonic(), measured=measured,
                    motion_scope='planner_idle_only', foot_support_verified=False))
                last_report = now
            time.sleep(max(0., next_tick-time.monotonic()))
    finally:
        event('upper_owner_exiting', native_controller_stopped=False,
              note='Native 350ms watchdog latches planner idle; operator O ends native policy')
        trace.close(); publisher.close(0); context.term(); events.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--port', type=int, default=17586)
    p.add_argument('--start-policy', action='store_true', help='Operator authorizes actual SONIC standing')
    run(p.parse_args())


if __name__ == '__main__':
    main()
