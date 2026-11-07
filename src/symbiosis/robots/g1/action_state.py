"""Bounded per-action measured feedback. The owner never waits for trace disk I/O."""
import hashlib
import json
import math
from pathlib import Path
import queue
import re
import threading
import time

from .state import NAMES

MAX_SAMPLES = 1024
MAX_BYTES = 8_000_000
TRACE_SAMPLE_HZ = 3
TRACE_INTERVAL_NS = (1_000_000_000 + TRACE_SAMPLE_HZ - 1) // TRACE_SAMPLE_HZ


def job_identifier(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{32}', value):
        raise ValueError('Expected a 32-character action job ID')
    return value


def measured_snapshot(value, now):
    if not isinstance(value, dict) or value.get('source') != 'native_lowstate_trace':
        raise ValueError('Measured native state unavailable')
    stamp = value.get('state_monotonic_ns')
    if type(stamp) is not int or not 0 <= now-stamp/1e9 <= .25:
        raise ValueError('Action body state stale or future')
    age = value.get('lowstate_age_s')
    if type(age) not in (int, float) or not 0 <= age <= .5:
        raise ValueError('Action lowstate stale')
    if value.get('joint_names') != NAMES or value.get('joint_order') != 'isaac':
        raise ValueError('Action joint order mismatch')
    result = dict(state_monotonic_ns=stamp, lowstate_age_s=age,
                  received_monotonic=now, motion=value.get('motion'),
                  encoder_source=value.get('encoder_source'))
    result['imu_source']=value.get('imu_source','legacy_simulated_pose' if value.get('encoder_source')=='simulated_joint_and_imu_sensors' else 'native_lowstate_imu')
    for key, size in (('q', 29), ('dq', 29), ('base_quat_wxyz', 4), ('base_ang_vel', 3)):
        row = value.get(key)
        if not isinstance(row, list) or len(row) != size or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in row):
            raise ValueError('Invalid measured '+key)
        result[key] = list(row)
    norm=math.sqrt(sum(v*v for v in result['base_quat_wxyz']))
    if not .99<=norm<=1.01:raise ValueError('Invalid measured IMU quaternion norm')
    tilt = value.get('tilt_rad')
    if type(tilt) not in (int, float) or not math.isfinite(tilt):
        raise ValueError('Invalid measured tilt')
    result['tilt_rad'] = tilt
    return result


class ActionStates:
    """Summary uses every unique valid owner sample; bounded trace targets 3 Hz.

    One instance is shared by movement and VLA jobs. Completed buffers transfer
    to a bounded worker queue; no disk calls occur in update/start/finish.
    """
    def __init__(self, root, session_id, *, max_samples=MAX_SAMPLES, queue_size=8):
        self.root, self.session_id = Path(root), session_id
        self.max_samples = max_samples
        self.active = {}
        self.latest = None
        self.error = 'not_sampled'
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopping = threading.Event()
        self.worker = threading.Thread(target=self._work, daemon=True, name='g1-action-state-writer')
        self.worker.start()

    def update(self, measured, now=None, error=None):
        now = time.monotonic() if now is None else now
        try:
            if error:
                raise ValueError(error)
            self.latest = measured_snapshot(measured, now)
            self.error = None
        except (ValueError, TypeError) as exc:
            self.latest = None
            self.error = str(exc)
        for record in self.active.values():
            self._sample(record)

    def start(self, job_id, kind, now):
        job_identifier(job_id)
        if job_id in self.active or len(self.active) >= 2:
            raise ValueError('Action feedback already active or capacity exceeded')
        record = dict(job_id=job_id, session_id=self.session_id, kind=kind,
                      started_monotonic=now, before=None, end=None, samples=[],
                      unique_samples=0, duplicate_samples=0, invalid_samples=0,
                      trace_dropped_samples=0, max_source_gap_s=0., last_error=None,
                      q_min=None, q_max=None, peak_abs_dq=None, peak_tilt_rad=0.)
        record['imu_yaw_delta_deg']=0.
        self.active[job_id] = record
        self._sample(record)

    def _sample(self, record):
        state = self.latest
        if state is None:
            record['invalid_samples'] += 1
            record['last_error'] = self.error
            return
        last = record['end']
        if last and state['state_monotonic_ns'] <= last['state_monotonic_ns']:
            if state['state_monotonic_ns'] == last['state_monotonic_ns']:
                record['duplicate_samples'] += 1
            else:
                record['invalid_samples'] += 1
                record['last_error'] = 'source_timestamp_regressed'
            return
        if last:
            from .motion_feedback import heading_increment
            record['imu_yaw_delta_deg']+=math.degrees(heading_increment(last,state))
            gap = (state['state_monotonic_ns']-last['state_monotonic_ns'])/1e9
            record['max_source_gap_s'] = max(record['max_source_gap_s'], gap)
        else:
            record['before'] = state
            record['q_min'] = state['q'][:]
            record['q_max'] = state['q'][:]
            record['peak_abs_dq'] = [0.]*29
        record['end'] = state
        record['unique_samples'] += 1
        record['q_min'] = [min(a, b) for a, b in zip(record['q_min'], state['q'])]
        record['q_max'] = [max(a, b) for a, b in zip(record['q_max'], state['q'])]
        record['peak_abs_dq'] = [max(a, abs(b)) for a, b in zip(record['peak_abs_dq'], state['dq'])]
        record['peak_tilt_rad'] = max(record['peak_tilt_rad'], state['tilt_rad'])
        samples = record['samples']
        if not samples or state['state_monotonic_ns']-samples[-1]['state_monotonic_ns'] >= TRACE_INTERVAL_NS:
            if len(samples) < self.max_samples:
                samples.append(state)
            else:
                record['trace_dropped_samples'] += 1

    def snapshot(self, job_id, now=None):
        if job_id not in self.active:
            return dict(available=False, reason='no_active_action_feedback')
        now = time.monotonic() if now is None else now
        r = self.active[job_id]
        value = {k: v for k, v in r.items() if k != 'samples'}
        age = now-r['end']['state_monotonic_ns']/1e9 if r['end'] else None
        value.update(schema='g1_action_state_v1', source='native_lowstate_trace', measured=True,
                     joint_names=NAMES, joint_order='isaac', units=dict(q='rad', dq='rad/s'),
                     available=r['end'] is not None, last_sample_age_s=age,
                     fresh=age is not None and 0 <= age <= .25 and self.error is None,
                     trajectory_sample_hz_target=TRACE_SAMPLE_HZ, hand_feedback='not_collected',
                     complete=False, robot_camera_hardware_synchronized=False,
                     trace=dict(path=f'action_states/{job_id}.jsonl', status='recording'))
        return value

    def finish(self, job_id, reason, now=None):
        now = time.monotonic() if now is None else now
        summary = self.snapshot(job_id, now)
        record = self.active.pop(job_id, None)
        if record is None:
            return summary
        samples = record['samples']
        if record['end'] and (not samples or samples[-1] != record['end']):
            if len(samples) >= self.max_samples:
                samples[-1] = record['end']
                summary['trace_dropped_samples'] += 1
            else:
                samples.append(record['end'])
        summary.update(ended_monotonic=now, end_reason=reason, complete=True,
                       valid_coverage=bool(summary['fresh'] and record['unique_samples'] >= 2
                                           and not record['invalid_samples']
                                           and record['max_source_gap_s'] <= .25))
        summary['fresh_at_end'] = summary.pop('fresh')
        summary['historical_action_state'] = True
        summary['endpoint_semantics'] = 'last_valid_measurement_at_job_termination_not_future_settling'
        summary['trace'] = dict(path=f'action_states/{job_id}.jsonl', status='pending_or_unavailable')
        # The worker owns separate metadata; the returned result remains immutable.
        try:
            self.queue.put_nowait((dict(summary), samples))
        except queue.Full:
            summary['trace'] = dict(status='dropped_writer_backpressure')
        return summary

    def _write(self, summary, samples):
        folder = self.root/'action_states'
        folder.mkdir(exist_ok=True)
        ident = summary['job_id']
        origin = summary['before']['state_monotonic_ns'] if summary['before'] else 0
        data = ''.join(json.dumps(dict(s, t_s=(s['state_monotonic_ns']-origin)/1e9),
                                  allow_nan=False)+'\n' for s in samples).encode()
        if len(data) > MAX_BYTES:
            raise ValueError('Action trace exceeds byte limit')
        path = folder/(ident+'.jsonl')
        temp = path.with_suffix('.tmp')
        temp.write_bytes(data)
        temp.replace(path)
        summary['trace'] = dict(path=f'action_states/{ident}.jsonl', status='ready',
                               sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), samples=len(samples))
        from .vla import atomic
        atomic(folder/(ident+'.json'), summary)

    def _work(self):
        while not self.stopping.is_set() or not self.queue.empty():
            try:
                summary, samples = self.queue.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                self._write(summary, samples)
            except Exception:
                # Result stays pending_or_unavailable. Never claim durable evidence
                # or propagate a disk failure into the real-time owner loop.
                pass
            finally:
                self.queue.task_done()

    def close(self):
        self.stopping.set()
        self.worker.join(timeout=2)


def load_record(root, job_id, session_id):
    """Canonical paths, exact session/job, bounded bytes and hash-checked trace."""
    job_identifier(job_id)
    root = Path(root).resolve()
    folder = root/'action_states'
    meta = folder/(job_id+'.json')
    path = folder/(job_id+'.jsonl')
    for p in (meta, path):
        if not p.resolve().is_relative_to(root) or p.is_symlink():
            raise ValueError('Action state path escapes run')
    if not meta.exists():
        raise ValueError('Action state archive pending or unavailable')
    if meta.stat().st_size > 100_000 or path.stat().st_size > MAX_BYTES:
        raise ValueError('Action state archive too large')
    summary = json.loads(meta.read_text())
    if summary.get('job_id') != job_id or summary.get('session_id') != session_id:
        raise ValueError('Action state job/session mismatch')
    data = path.read_bytes()
    trace = summary['trace']
    if trace.get('path') != f'action_states/{job_id}.jsonl' or hashlib.sha256(data).hexdigest() != trace.get('sha256'):
        raise ValueError('Action state hash/path mismatch')
    return summary, data


def enrich_result(root, result):
    """Resolve asynchronous persistence without altering the original result."""
    feedback = result.get('body_feedback')
    if not feedback or feedback.get('job_id') != result.get('job_id'):
        return result
    try:
        saved, _ = load_record(root, result['job_id'], result['session_id'])
        return dict(result, body_feedback=saved)
    except (OSError, ValueError, KeyError) as exc:
        return dict(result, body_feedback=dict(feedback, archive_error=str(exc)))


def review_summary(feedback):
    """Bounded preview with explicit quality and most-changing joints first."""
    value = {k: feedback.get(k) for k in ('schema', 'job_id', 'session_id', 'kind', 'available',
             'end_reason', 'valid_coverage', 'fresh_at_end', 'last_sample_age_s', 'unique_samples',
             'invalid_samples', 'duplicate_samples', 'trace_dropped_samples', 'max_source_gap_s',
             'last_error', 'peak_tilt_rad', 'hand_feedback', 'trace', 'archive_error',
             'trajectory_sample_hz_target')}
    if feedback.get('ended_monotonic') is not None:
        value['recorded_duration_s'] = feedback['ended_monotonic']-feedback['started_monotonic']
    if feedback.get('before') and feedback.get('end'):
        first, last = feedback['before'], feedback['end']
        changed = sorted(range(29), key=lambda i:feedback['q_max'][i]-feedback['q_min'][i], reverse=True)[:6]
        value['largest_joint_ranges'] = [dict(name=NAMES[i], before_rad=first['q'][i], end_rad=last['q'][i],
            min_rad=feedback['q_min'][i], max_rad=feedback['q_max'][i], peak_abs_dq_rad_s=feedback['peak_abs_dq'][i]) for i in changed]
        value['source_time_range_ns'] = [first['state_monotonic_ns'], last['state_monotonic_ns']]
        value['end_body_pose'] = dict(base_quat_wxyz=last['base_quat_wxyz'],
                                     base_ang_vel=last['base_ang_vel'], tilt_rad=last['tilt_rad'])
    value['limitation'] = 'Measured sampled body joints only; no translation, contact or grasp-success inference.'
    return value


def query(root, session_id, job_id, joints=None, start_s=0., end_s=None, offset=0, limit=20):
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 64:
        raise ValueError('Invalid action state page')
    for v in (start_s, end_s):
        if v is not None and (type(v) not in (int, float) or not math.isfinite(v) or v < 0):
            raise ValueError('Invalid action time range')
    if start_s is None or end_s is not None and end_s < start_s:
        raise ValueError('Invalid action time range')
    selected = NAMES if joints is None else joints
    if not isinstance(selected, list) or not 1 <= len(selected) <= 29 or any(j not in NAMES for j in selected):
        raise ValueError('Select known joint names')
    summary, data = load_record(root, job_id, session_id)
    indices = [NAMES.index(j) for j in selected]
    rows = [json.loads(line) for line in data.splitlines()]
    # Also limit historical higher-rate traces before filtering/pagination, so
    # page boundaries and time windows cannot change the sampling grid.
    sampled = []
    for row in rows:
        if not sampled or row['state_monotonic_ns']-sampled[-1]['state_monotonic_ns'] >= TRACE_INTERVAL_NS:
            sampled.append(row)
    if rows and sampled[-1]['state_monotonic_ns'] != rows[-1]['state_monotonic_ns']:
        sampled.append(rows[-1])  # Explicit endpoint exception, as in recording.
    rows = sampled
    rows = [r for r in rows if r['t_s'] >= start_s and (end_s is None or r['t_s'] <= end_s)]
    page = [dict(r, q=[r['q'][i] for i in indices], dq=[r['dq'][i] for i in indices]) for r in rows[offset:offset+limit]]
    return dict(job_id=job_id, session_id=session_id, historical_action_state=True,
                joint_names=selected, joint_order='selected_names', units=summary['units'],
                trace=summary['trace'], samples=page, sample_hz_target=TRACE_SAMPLE_HZ,
                endpoint_sample_retained=True,
                next_offset=offset+limit if offset+limit < len(rows) else None)
