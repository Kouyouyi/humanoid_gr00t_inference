"""Operator-owned reference executor; RAX submits bounded jobs through a private spool.

Only this owner publishes SONIC high-level references. It never creates DDS publishers.
It has no agent-facing start, gains or recovery operation. Native O remains available.
"""
import argparse
import json
from pathlib import Path
import signal
import time
import uuid

import numpy as np
import zmq

from .camera_service import atomic_json
from .reference import from_native_idle, hold_window, pulse_window, check_pulse, DURATION
from .state import TraceReader, checked_state, NAMES
from .vendor import wire_module


def validate_job(job, baseline_id, now, allow_reach=False, allow_grasp=False, allow_gait=False):
    gait=job.get('action')=='walk_step'
    grasp=job.get('action')=='grasp_step'
    reach = job.get('action') in ('clear_arms','reach_return') or grasp or gait
    expected = {'id', 'action', 'baseline_id', 'created_monotonic'} | ({'delta_m','rotation_delta'} if grasp else set() if reach else {'side','delta_rad'})
    if set(job) != expected:
        raise ValueError('Invalid reference job fields')
    if (gait and not allow_gait) or (grasp and not allow_grasp) or (reach and not allow_reach) or (not reach and job['action'] != 'elbow_pulse') or job['baseline_id'] != baseline_id:
        raise ValueError('Wrong action or changed standing baseline')
    if not isinstance(job['id'], str) or len(job['id']) != 32 or any(c not in '0123456789abcdef' for c in job['id']):
        raise ValueError('Invalid job id')
    if not np.isfinite(job['created_monotonic']) or not 0 <= now-job['created_monotonic'] <= 2:
        raise ValueError('Expired/future reference job')
    if not reach:
        check_pulse(job['side'], job['delta_rad'])
    if grasp:
        from .reach import validate_step
        validate_step(job['delta_m'],job['rotation_delta'])


def run(args):
    reach_enabled = getattr(args, 'reach_profile', False)
    kin = None
    if reach_enabled:
        from .reach import Kinematics, ReachFlow, check_site, PROFILE
        kin = Kinematics()  # Load before activating any reference publisher.
    args.run.mkdir(parents=True, exist_ok=True)
    requests = args.run/'reference_requests'
    requests.mkdir(exist_ok=True)
    results = args.run/'reference_results'
    results.mkdir(exist_ok=True)
    wire = wire_module()
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.setsockopt(zmq.SNDHWM, 10)
    pub.bind(f'tcp://127.0.0.1:{args.port}')
    trace = TraceReader(args.run/'controller_trace.jsonl')
    events = (args.run/'executor_events.jsonl').open('a')
    stop = False

    def stopping(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, stopping)
    signal.signal(signal.SIGTERM, stopping)
    phase, fault = 'waiting_for_native_init', None
    started = None
    stable_since = None
    reference = None
    baseline_id = None
    anchor_q = None
    frame = 0
    switch_at = None
    active = None
    last_report = 0
    previous_tick = time.monotonic()
    last_completed = None
    execution_log = (args.run/'reference_execution_trace.jsonl').open('a')
    # Authority is operator CLI --start-policy/--reference-hold, not a tool argument.
    authorized_until = float('inf')
    posture = 'standing'
    grasp_site_verified=False
    walk_attempted=False
    last_holding_motion_notice=0.

    def event(kind, **data):
        record = dict(event=kind, monotonic=time.monotonic(), **data)
        events.write(json.dumps(record, allow_nan=False)+'\n'); events.flush()
        print(json.dumps(record), flush=True)

    def finish_job(reason, measured=None):
        nonlocal active, last_completed, reference, posture, anchor_q, grasp_site_verified
        if active is None:
            return
        job = active['job']
        if 'flow' in active:
            flow = active['flow']
            if not flow.done:
                flow.fail(reason)
            result = dict(flow.result, id=job['id'])
            # Hold the last continuous target on review, never snap to old idle.
            if reference is not None:
                reference = {k:v.copy() for k,v in reference.items()}
                reference['joint_pos'][:] = flow.target
            if result['completed'] and job['action']=='clear_arms':
                posture='clear_arms'
            if result['completed'] and job['action']=='walk_step':
                anchor_q=np.asarray(measured['q']).copy()
                grasp_site_verified=False
            flow.close()
            atomic_json(results/(job['id']+'.json'), result)
            last_completed=result
            event('job_finished', **result)
            active=None
            return
        i = NAMES.index(job['side']+'_elbow_joint')
        samples = active['samples']
        before = active['q_before']
        end = np.asarray(measured['q']) if measured else None
        response = [float(s['q'][i]-before[i]) for s in samples]
        directional = float(max((v*np.sign(job['delta_rad']) for v in response), default=0.))
        return_error = float(abs(end[i]-before[i])) if end is not None else None
        body_return_error = float(np.max(abs(end-before))) if end is not None else None
        result = dict(id=job['id'], reason=reason, completed=reason == 'trajectory_finished',
            reference_delta_rad=job['delta_rad'], side=job['side'], measured_samples=len(samples),
            directional_joint_response_rad=directional, return_error_rad=return_error,
            measured_response_detected=directional >= .005,
            measured_return_within_0_05_rad=return_error is not None and return_error <= .05,
            max_body_return_error_rad=body_return_error,
            measured_body_return_within_0_05_rad=body_return_error is not None and body_return_error <= .05,
            max_tilt_rad=max((s['tilt_rad'] for s in samples), default=None),
            q_before=before.tolist(), q_after=None if end is None else end.tolist(),
            actual_tracking_accuracy_characterized=False, physical_grasp_tested=False)
        result.update(native_peak_reference_observed=active['peak_observed'],
                      native_return_reference_observed=active['return_observed'],
                      elapsed_s=time.monotonic()-active['at'])
        atomic_json(results/(job['id']+'.json'), result)
        last_completed = result
        event('job_finished', **result)
        active = None

    try:
        while not stop:
            now = time.monotonic()
            if started is not None and now-previous_tick > .15:
                if phase != 'fault_idle':
                    event('fault_idle', reason='owner_tick_late', gap_s=now-previous_tick)
                phase, fault = 'fault_idle', 'owner_tick_late'
            previous_tick = now
            pulse = False
            log = args.run/'controller.log'
            if phase == 'waiting_for_native_init' and log.exists() and 'Init Done' in log.read_text(errors='replace'):
                phase = 'ready_to_start'
                event('native_init_observed')
            if phase == 'ready_to_start' and args.start_policy:
                started = now
                pulse = True
                phase = 'starting_planner_idle'
                event('operator_policy_start_requested')
            measured = None
            row = None
            if started is not None:
                try:
                    row = trace.poll()
                    measured = checked_state(row)
                    if measured['tilt_rad'] > .25:
                        raise ValueError('base_tilt_above_0.25_rad')
                    if phase in ('reference_handoff', 'reference_ready', 'executing', 'reference_review_required'):
                        obs = np.asarray(row['encoder_obs'], dtype=float)
                        if obs.shape != (1762,) or not np.isfinite(obs).all():
                            raise ValueError('Invalid native encoder readback')
                        if phase != 'reference_handoff' and (row['motion'] != 'streamed' or not row['encoder_source']):
                            raise ValueError('Native reference source changed during ownership')
                        walking=active is not None and active['job']['action']=='walk_step'
                        if walking:
                            active['flow'].guard(measured)
                        elif phase in ('reference_handoff','executing') and max(abs(v) for v in measured['dq']) > 1.5:
                            raise ValueError('joint_speed_above_1.5_rad_s')
                        elif phase in ('reference_ready','reference_review_required') and max(abs(v) for v in measured['dq']) > 1.5:
                            # Operator may reposition a foot while holding. Record it,
                            # keep the held reference, and gate new jobs on settling.
                            # This is not an override of native motor/thermal checks.
                            if now-last_holding_motion_notice>1.:
                                event('motion_observed_during_hold',max_joint_speed_rad_s=max(abs(v) for v in measured['dq']),
                                      cause='not_inferred',new_jobs_require_settling=True)
                                last_holding_motion_notice=now
                        lower = [i for i, name in enumerate(NAMES) if 'elbow' not in name and
                                 not any(w in name for w in ('shoulder', 'wrist'))]
                        if not walking and max(abs(np.asarray(measured['q'])[lower]-anchor_q[lower])) > .15:
                            raise ValueError('standing_body_departed_from_anchor')
                    stable = (max(abs(v) for v in measured['dq']) < .5 and
                              np.linalg.norm(measured['base_ang_vel']) < .25 and measured['encoder_source'])
                    if stable:
                        stable_since = stable_since or time.monotonic()
                    else:
                        stable_since = None
                    if phase == 'starting_planner_idle' and stable_since and time.monotonic()-stable_since >= 3:
                        phase = 'standing_observed'
                        event('standing_observed')
                    if phase == 'standing_observed' and args.reference_hold:
                        reference = from_native_idle(row)
                        anchor_q = np.asarray(measured['q'])
                        baseline_id = uuid.uuid4().hex
                        np.savez_compressed(args.run/'standing_reference.npz', **reference)
                        atomic_json(args.run/'handoff_anchor.json', row)
                        switch_at = time.monotonic()
                        authorized_until = switch_at+args.lease_seconds
                        phase = 'reference_handoff'
                        event('reference_handoff_started', baseline_id=baseline_id)
                    if phase == 'reference_handoff' and time.monotonic()-switch_at > 3:
                        if row['motion'] != 'streamed' or not row['encoder_source']:
                            raise ValueError('Native stream handoff not observed')
                        obs = np.asarray(row['encoder_obs'])
                        if obs.shape != (1762,) or np.max(np.abs(obs[4:294].reshape(10, 29)-reference['joint_pos'][0])) > 1e-4:
                            raise ValueError('Native encoder did not receive the intended held joint reference')
                        if np.max(np.abs(obs[294:584].reshape(10, 29)-reference['joint_vel'][0])) > 1e-4:
                            raise ValueError('Native reference velocity mismatch')
                        phase = 'reference_ready'
                        event('reference_handoff_verified', baseline_id=baseline_id)
                except (OSError, KeyError, ValueError, RuntimeError) as exc:
                    if phase != 'starting_planner_idle' or time.monotonic()-started > 20:
                        if phase != 'fault_idle':
                            event('fault_idle', reason=str(exc))
                        phase, fault = 'fault_idle', str(exc)
            if time.monotonic() > authorized_until and phase != 'fault_idle':
                phase, fault = 'fault_idle', 'operator_reference_lease_expired'
                event('fault_idle', reason=fault)
            if phase == 'fault_idle':
                finish_job('fault: '+str(fault), measured)
                reference = None
            # A single bounded request slot, outside the agent workspace. Only the backend writes it.
            request_path = requests/'pending.json'
            if request_path.exists():
                claimed = requests/'claimed.json'
                request_path.replace(claimed)
                job = None
                try:
                    job = json.loads(claimed.read_text())
                    validate_job(job, baseline_id, time.monotonic(), reach_enabled, getattr(args,'grasp_profile',False),getattr(args,'gait_profile',False))
                    if phase != 'reference_ready' or measured is None:
                        raise ValueError('Reference executor not ready or busy')
                    if (stable_since is None or time.monotonic()-stable_since<1. or
                        max(abs(v) for v in measured['dq'])>=.3):
                        raise ValueError('Robot is moving or has not settled after adjustment; do not start a new job')
                    is_reach = job['action'] in ('clear_arms','reach_return','grasp_step','walk_step')
                    if authorized_until-time.monotonic() < (60 if is_reach else DURATION+13):
                        raise ValueError('Operator reference lease too short for verification')
                    if job['action']=='walk_step':
                        if posture!='clear_arms' or walk_attempted:
                            raise ValueError('Walk requires cleared arms and is limited to one step per commissioning run')
                        from .gait_flow import GaitFlow,check_site as check_walk_site
                        check_walk_site(args.run/'operator_walk_clearance.json',baseline_id,measured,time.monotonic())
                        flow=GaitFlow(reference,measured,time.monotonic())
                        np.savez_compressed(args.run/'walk_reference.npz',**flow.profile)
                        active=dict(job=job,flow=flow)
                        walk_attempted=True
                    elif is_reach:
                        if job['action']=='clear_arms' and posture!='standing':
                            raise ValueError('Arms already cleared; do not accumulate offsets')
                        if job['action']=='reach_return':
                            if posture!='clear_arms':
                                raise ValueError('Clear-arms stage required first')
                            check_site(args.run/'operator_clearance.json',baseline_id,measured,time.monotonic())
                        if job['action']=='grasp_step':
                            if posture!='clear_arms':
                                raise ValueError('Clear arms before approaching the object')
                            if not grasp_site_verified:
                                check_site(args.run/'operator_clearance.json',baseline_id,measured,time.monotonic())
                                grasp_site_verified=True
                        active=dict(job=job,flow=ReachFlow(reference,measured,job['action'],kin,time.monotonic(),
                            delta_m=job.get('delta_m'),rotation_delta=job.get('rotation_delta')))
                    else:
                        if posture!='standing':
                            raise ValueError('Legacy pulse requires original standing posture')
                        pulse_window(reference, job['side'], job['delta_rad'], 3., frame)
                        active = dict(job=job, at=time.monotonic(), q_before=np.asarray(measured['q']), samples=[],
                                      peak_observed=False, return_observed=False, response_observed=False, settled_since=None)
                    phase = 'executing'
                    event('job_started', **job)
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    event('job_rejected', reason=str(exc))
                    if isinstance(job, dict) and isinstance(job.get('id'), str) and len(job['id']) == 32 and all(c in '0123456789abcdef' for c in job['id']):
                        atomic_json(results/(job['id']+'.json'), dict(id=job['id'], completed=False, reason=str(exc)))
                finally:
                    claimed.unlink(missing_ok=True)
            fields = None
            if active is not None and 'flow' in active and measured is not None:
                flow=active['flow']; now_reach=time.monotonic()
                flow.step(measured,row,now_reach)
                fields=flow.fields(now_reach,frame)
                execution_log.write(json.dumps(dict(monotonic=now_reach,job_id=active['job']['id'],stage=flow.stage,
                    sent_q=fields['joint_pos'][0].tolist(),sent_dq=fields['joint_vel'][0].tolist(),
                    native_q=np.asarray(row['encoder_obs'])[4:33].tolist(),
                    measured_q=measured['q'],measured_dq=measured['dq'],wrists=flow.latest),allow_nan=False)+'\n')
                execution_log.flush()
                if flow.done:
                    success=flow.result['completed']
                    planning_only=flow.result.get('planning_only_rejection',False)
                    finish_job(flow.result['reason'],measured)
                    phase='reference_ready' if success or planning_only else 'reference_review_required'
                    event(phase,reason='reach profile result available')
            elif active is not None and measured is not None:
                active['samples'].append(measured)
                elapsed = time.monotonic()-active['at']
                job = active['job']
                i = NAMES.index(job['side']+'_elbow_joint')
                obs = np.asarray(row['encoder_obs'])
                observed_q = obs[4:294].reshape(10, 29)
                observed_dq = obs[294:584].reshape(10, 29)
                current_delta = float(observed_q[0, i]-reference['joint_pos'][0, i])
                active['peak_observed'] = bool(active['peak_observed'] or
                    current_delta*np.sign(job['delta_rad']) >= abs(job['delta_rad'])*.8)
                at_baseline = (np.max(abs(observed_q-reference['joint_pos'][0])) < 1e-4 and
                               np.max(abs(observed_dq-reference['joint_vel'][0])) < 1e-4)
                active['return_observed'] = bool(active['peak_observed'] and elapsed >= DURATION and at_baseline)
                active['response_observed'] = bool(active['response_observed'] or
                    (measured['q'][i]-active['q_before'][i])*np.sign(job['delta_rad']) >= .005)
                actual_return = np.max(abs(np.asarray(measured['q'])-active['q_before'])) <= .05
                settled = (active['return_observed'] and active['response_observed'] and actual_return and
                           max(abs(v) for v in measured['dq']) < .15)
                active['settled_since'] = (active['settled_since'] or time.monotonic()) if settled else None
                fields = pulse_window(reference, job['side'], job['delta_rad'], elapsed, frame)
                execution_log.write(json.dumps(dict(monotonic=time.monotonic(), job_id=job['id'],
                    elapsed_s=elapsed, frame_index=frame,
                    sent_q=fields['joint_pos'][0].tolist(), sent_dq=fields['joint_vel'][0].tolist(),
                    native_q=observed_q[0].tolist(), native_dq=observed_dq[0].tolist(),
                    measured_q=measured['q'], measured_dq=measured['dq']), allow_nan=False)+'\n')
                execution_log.flush()
                if active['settled_since'] and time.monotonic()-active['settled_since'] >= .5:
                    finish_job('trajectory_finished', measured)
                    phase = 'reference_ready'
                elif elapsed >= DURATION+10:
                    finish_job('reference_or_measured_return_not_verified', measured)
                    phase = 'reference_review_required'  # No new motion after failed verification.
                    event('reference_review_required', reason='Readback or measured return failed; operator review required')
            if reference is not None:
                fields = fields if fields is not None else hold_window(reference, frame)
                pub.send(wire.pack_pose_message(fields, version=1))
                pub.send(wire.build_command_message(start=False, stop=False, planner=False))
                frame += 1
            else:
                pub.send(wire.build_planner_message(0, [0, 0, 0], [1, 0, 0], speed=0.))
                pub.send(wire.build_command_message(start=pulse, stop=False, planner=True))
            if now-last_report >= .05:
                atomic_json(args.run/'executor_status.json', dict(phase=phase, fault=fault,
                    baseline_id=baseline_id, updated_monotonic=time.monotonic(), measured=measured,
                    active_job=None if active is None else active['job']['id'], last_completed=last_completed,
                    reference_lease_remaining_s=max(0., authorized_until-now) if np.isfinite(authorized_until) else None,
                    scope='clear_reach_v1_and_legacy_pulse' if reach_enabled else 'single_elbow_roundtrip_at_most_0.05_rad',
                    reach_profile_enabled=reach_enabled,posture=posture,
                    grasp_profile_enabled=getattr(args,'grasp_profile',False),grasp_site_verified=grasp_site_verified,
                    gait_profile_enabled=getattr(args,'gait_profile',False),walk_attempted=walk_attempted,
                    reach_stage=active['flow'].stage if active and 'flow' in active else None,
                    wrists=None if kin is None or measured is None else kin.fk(measured['q']),hands_commanded=False))
                last_report = now
            time.sleep(max(0., now+.02-time.monotonic()))
    finally:
        try:
            finish_job('operator_owner_exit')
            # Do not kill the native process or request damping from an agent/service exit.
            event('upper_owner_exiting', native_controller_stopped=False)
            atomic_json(args.run/'executor_status.json', dict(phase='owner_exited', fault=fault,
                        updated_monotonic=time.monotonic(), baseline_id=baseline_id, active_job=None))
        finally:
            trace.close(); pub.close(0); ctx.term(); events.close(); execution_log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--port', type=int, default=17586)
    p.add_argument('--start-policy', action='store_true')
    p.add_argument('--reference-hold', action='store_true')
    p.add_argument('--reach-profile', action='store_true', help='Operator enable for fixed clear/reach stages; no arbitrary IK goals')
    p.add_argument('--grasp-profile',action='store_true',help='Operator enable bounded right-wrist approach steps with observed endpoints')
    p.add_argument('--gait-profile',action='store_true',help='Enable one fixed left leading step; requires current walking-site evidence')
    p.add_argument('--lease-seconds', type=float, default=600.)
    args = p.parse_args()
    if args.grasp_profile and not (args.reach_profile and args.reference_hold):
        p.error('grasp-profile requires reach-profile and reference-hold')
    if args.gait_profile and not (args.reach_profile and args.reference_hold):
        p.error('gait-profile requires reach-profile and reference-hold')
    if not 10 <= args.lease_seconds <= 600:
        p.error('reference lease must be 10..600 seconds')
    run(args)


if __name__ == '__main__':
    main()
