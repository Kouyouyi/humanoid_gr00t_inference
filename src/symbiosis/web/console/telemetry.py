"""Opt-in DDS read-only collector. Publish atomic case/winding telemetry for the UI.

Run explicitly in the Unitree SDK environment. Starting the console never starts it.
"""
import argparse
import json
import os
from pathlib import Path
import threading
import time

from .adapters import BOOT_ID, JOINTS


def sample_message(msg):
    if len(msg.motor_state) < 29:
        raise ValueError('expected 29 body motor states')
    return dict(schema_version=1, source='live_lowstate', sampled_at=time.time(),
                monotonic=time.monotonic(), host_boot_id=BOOT_ID,
                motors=[dict(hardware_index=j['hardware_index'], name=j['name'],
                             temperature_case_winding_c=list(msg.motor_state[j['hardware_index']].temperature),
                             motorstate=int(msg.motor_state[j['hardware_index']].motorstate),
                             tau_est=float(msg.motor_state[j['hardware_index']].tau_est)) for j in JOINTS])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface', default='eno1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    latest = {}
    lock = threading.Lock()

    def receive(msg):
        if len(msg.motor_state) < 29:
            return
        sample = sample_message(msg)
        with lock:
            latest.clear()
            latest.update(sample)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ChannelFactoryInitialize(0, args.interface)
    subscriber = ChannelSubscriber('rt/lowstate', LowState_)
    subscriber.Init(receive, 1)
    tmp = args.output.with_name('.' + args.output.name + f'.{os.getpid()}.tmp')
    try:
        while True:
            with lock:
                sample = dict(latest)
            if sample:
                # Preserve callback timestamps. A stopped DDS stream must become stale.
                tmp.write_text(json.dumps(sample, allow_nan=False) + '\n')
                os.replace(tmp, args.output)
            time.sleep(.2)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.Close()
        tmp.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
