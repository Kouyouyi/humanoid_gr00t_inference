"""Read-only native controller trace reader. Same host/boot only; no DDS writes."""
import json
import os
from pathlib import Path
import time

import numpy as np

JOINTS = json.loads((Path(__file__).parent/'joints.json').read_text())['joints']
NAMES = [j['name'] for j in JOINTS]
HARDWARE_TO_ISAAC = [j['hardware_index'] for j in JOINTS]


def checked_state(row, now=None):
    now = time.monotonic() if now is None else now
    if not 0 <= now-row['monotonic'] <= .25:
        raise ValueError('native_trace_stale')
    # The native trace is sampled at 50 Hz.  Allow one half-second of
    # scheduler/bridge jitter before declaring the body feedback unusable.
    if not 0 <= now-row['lowstate_receive_monotonic_ns']/1e9 <= .5:
        raise ValueError('lowstate_stale')
    if row.get('transport_fault') or row.get('log_dropped', 0):
        raise ValueError('native_transport_fault_or_trace_dropped')
    values = {}
    for key, n in [('body_q', 29), ('body_dq', 29), ('base_quat', 4), ('base_ang_vel', 3)]:
        a = np.asarray(row[key], dtype=float)
        if a.shape != (n,) or not np.isfinite(a).all():
            raise ValueError('Invalid measured '+key)
        values[key] = a
    quat = values['base_quat']
    if not .99 <= np.linalg.norm(quat) <= 1.01:
        raise ValueError('Invalid base quaternion norm')
    quat = quat/np.linalg.norm(quat)
    tilt = float(np.arccos(np.clip(1-2*(quat[1]**2+quat[2]**2), -1, 1)))
    return dict(source='native_lowstate_trace', measured=True,
                joint_names=NAMES, joint_order='isaac',
                q=values['body_q'][HARDWARE_TO_ISAAC].tolist(),
                dq=values['body_dq'][HARDWARE_TO_ISAAC].tolist(),
                base_quat_wxyz=quat.tolist(), base_ang_vel=values['base_ang_vel'].tolist(),
                imu_source=row.get('imu_source','legacy_simulated_pose' if row.get('encoder_source')=='simulated_joint_and_imu_sensors' else 'native_lowstate_imu'),
                tilt_rad=tilt, lowstate_age_s=now-row['lowstate_receive_monotonic_ns']/1e9,
                state_monotonic_ns=row['state_monotonic_ns'],
                motion=row['motion'], encoder_source=row['encoder_source'])


class TraceReader:
    def __init__(self, path):
        self.path = Path(path) if path else None
        self.file = None
        self.latest = None
        self.last_index = None

    def poll(self):
        if self.path is None:
            raise RuntimeError('No native controller trace configured')
        if self.file is None:
            self.file = self.path.open('rb')
        # Native rows include large encoder/decoder arrays. Only read the tail;
        # parsing the entire history would make first observations stale.
        size = os.fstat(self.file.fileno()).st_size
        offset = max(0, size-262144)
        self.file.seek(offset)
        data = self.file.read(size-offset)
        if offset:
            data = data.partition(b'\n')[2]
        complete = data.split(b'\n')[:-1]
        if not complete:
            raise RuntimeError('No native controller feedback yet')
        self.latest = json.loads(complete[-1])
        index = self.latest['state_monotonic_ns']
        if self.last_index is not None and index < self.last_index:
            raise ValueError('Native controller restarted; create a new session')
        self.last_index = index
        return self.latest

    def status(self):
        # Read clock AFTER reading latest row; producer may update while reading.
        return checked_state(self.poll())

    def close(self):
        if self.file:
            self.file.close()
            self.file = None
