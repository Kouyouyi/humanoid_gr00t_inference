"""Operator-owned SONIC v4 demo-token executor with live ankle alarms.

One high-level ZMQ publisher; DDS is read-only. Native O/watchdog remain native.
Manual tests and explicitly armed temperature dispatch share one serialized action.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import signal
import threading
import time
import uuid

import numpy as np
import zmq

from .ankle_demo import load_demos,validate_request,guard,DemoFlow,announce
from .ankle_thermal import AlarmDetector,ANKLES
from .state import TraceReader,checked_state
from .vendor import wire_module
from .lease import SessionLease, EXPIRED, duration
from .action_state import ActionStates


def atomic(path,value):
    path=Path(path);tmp=path.with_name('.'+path.name)
    tmp.write_text(json.dumps(value,allow_nan=False,indent=2)+'\n');tmp.replace(path)


def run(args,feedback_source=None):
    thermal_rule=getattr(args,'thermal_rule','strict91')
    repeat_seconds=float(getattr(args,'thermal_repeat_seconds',0.))
    if not np.isfinite(repeat_seconds) or repeat_seconds!=0 and not 10<=repeat_seconds<=600:
        raise ValueError('Thermal repeat interval must be 0 or 10..600 seconds')
    configured_sides=set(getattr(args,'thermal_auto_sides',[]) or [])
    if not configured_sides<={'left','right'}:raise ValueError('Invalid automatic ankle side')
    demos=load_demos(args.run/'ankle_demos')
    for name in ('ankle_requests','ankle_results','ankle_audio'):(args.run/name).mkdir(exist_ok=True)
    session_id=uuid.uuid4().hex
    lease=SessionLease(args.run,session_id,getattr(args,'lease_seconds',0))
    feedback=ActionStates(args.run,session_id)
    moves=None
    if getattr(args,'locomotion',False):
        from .locomotion import LocomotionJobs
        moves=LocomotionJobs(args.run,session_id,feedback=feedback)
    vla=None
    if getattr(args,'vla',False):
        from .vla import VLAJobs
        from .vla_runtime import Runtime
        vla=VLAJobs(args.run,session_id,Runtime(),feedback=feedback)
    latest={};lock=threading.Lock();sub=None;health_bridge=None
    if vla and feedback_source is None:
        from .health_bridge import HealthBridge
        health_bridge=HealthBridge();feedback_source=health_bridge.poll
    def receive(msg):
        row=dict(monotonic=time.monotonic(),motors=[dict(hardware_index=i,name=name,
            temperature_winding_driver_c=list(msg.motor_state[i].temperature),
            motorstate=int(msg.motor_state[i].motorstate),tau_est=float(msg.motor_state[i].tau_est)) for i,name in ANKLES.items()])
        with lock:latest.clear();latest.update(row)
    if feedback_source is None:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize,ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        ChannelFactoryInitialize(0,'eno1');sub=ChannelSubscriber('rt/lowstate',LowState_);sub.Init(receive,1)
    wire=wire_module();ctx=zmq.Context();pub=ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER,0);pub.setsockopt(zmq.SNDHWM,10)
    pub.bind(f'tcp://127.0.0.1:{args.port}')
    trace=TraceReader(args.run/'controller_trace.jsonl')
    events=(args.run/'ankle_events.jsonl').open('a');samples=(args.run/'ankle_execution.jsonl').open('a')
    thermal_trace=(args.run/'ankle_temperatures.jsonl').open('a')
    pool=ThreadPoolExecutor(max_workers=1);stop=False
    def stopping(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGINT,stopping);signal.signal(signal.SIGTERM,stopping)
    def event(kind,**data):
        row=dict(event=kind,monotonic=time.monotonic(),**data)
        events.write(json.dumps(row,allow_nan=False)+'\n');events.flush();print(json.dumps(row),flush=True)
    detector=AlarmDetector(thermal_rule);last_health=None;armed=set(configured_sides);dispatched_alarms={};repeat_ready_at={}
    phase='waiting_for_native_init';fault=None;started=None;stable_since=None
    held=None;flow=None;job=None;future=None;audio_done=None;audio_wait_at=None
    recent=deque(maxlen=40);frame=0;last_status=0.;previous=time.monotonic();source_bad_since=None;handoff_at=None
    completed_sides=set();last_result=None;ready_at=None
    # These faults mean that the feedback path was temporarily late.  Once a
    # fresh, settled trace and ankle health are observed again, the owner may
    # return to READY by itself.  Hardware faults, tilt faults, lease expiry,
    # and invalid measurements remain latched for an explicit operator reset.
    recoverable_feedback_faults = {
        'Stale ankle health', 'lowstate_stale', 'native_trace_stale',
        'owner_tick_late', 'No fresh ankle feedback before start',
    }
    def finish(reason,ok=False):
        nonlocal job,flow,last_result,future,audio_done,audio_wait_at
        if job:
            last_result=dict(id=job['id'],side=job['side'],trigger=job['trigger'],completed=ok,reason=reason,
                physical_heel_lift_verified=False,thermal_relief_validated=False,
                source_frames_submitted=flow.index if flow else 0,session_id=session_id)
            atomic(args.run/'ankle_results'/(job['id']+'.json'),last_result);event('job_finished',**last_result)
            if ok:
                completed_sides.add(job['side'])
                if job['trigger']=='temperature' and repeat_seconds:
                    repeat_ready_at[job['side']]=time.monotonic()+repeat_seconds
        job=None;flow=None;future=None;audio_done=None;audio_wait_at=None
    def begin(side,trigger,request_id=None):
        nonlocal job,future,phase,audio_wait_at
        if moves and moves.busy:raise ValueError('Movement must settle before ankle motion')
        if vla:vla.pause('ankle_relief_'+side,invalidate=True)
        job=dict(id=request_id or uuid.uuid4().hex,side=side,trigger=trigger)
        if trigger=='temperature':
            repeated=dispatched_alarms.get(side)==detector.episodes[side]
            dispatched_alarms[side]=detector.episodes[side];repeat_ready_at.pop(side,None)
            if repeated:event('thermal_repeat_requested',side=side,episode=detector.episodes[side],interval_seconds=repeat_seconds)
        future=pool.submit(announce,side,args.run/'ankle_audio',job['id'],trigger=='temperature')
        audio_wait_at=time.monotonic();phase='announcing';event('announcement_requested',**job)
    try:
        event('owner_ready',session_id=session_id,thermal_auto_sides=sorted(armed),thermal_rule=thermal_rule,
              thermal_repeat_seconds=repeat_seconds,source='Pico episodes 13/15',dds_publishers=0)
        while not stop:
            now=time.monotonic();pulse=False;measured=None;native=None;health=None;move_command=None
            if started and now-previous>.15 and phase!='fault_idle':
                phase='fault_idle';fault='owner_tick_late'
                event('fault_idle',reason=fault,gap_s=now-previous)
            previous=now
            try:
                if feedback_source is not None:health=feedback_source(now)
                else:
                    with lock:health=dict(latest)
                if health and health['monotonic']!=last_health:
                    alarms=detector.update(health,time.monotonic());last_health=health['monotonic']
                    thermal_trace.write(json.dumps(health)+'\n');thermal_trace.flush()
                    for alarm in alarms:
                        alarm=dict(alarm);kind=alarm.pop('event')
                        alarm.pop('dispatch_block_reason',None);alarm['motion_enabled']=alarm.get('side') in armed
                        event(kind,**alarm)
                if phase=='waiting_for_native_init':
                    log=args.run/'controller.log'
                    if log.exists() and 'Init Done' in log.read_text(errors='replace'):
                        if not args.start_policy:raise ValueError('Policy start not authorized')
                        if not health or not 0<=time.monotonic()-health['monotonic']<=.2:raise ValueError('No fresh ankle feedback before start')
                        if any(m['motorstate'] for m in health['motors']):raise ValueError('Ankle motor fault before start')
                        phase='starting_planner_idle';started=now;lease.start(now);pulse=True;event('policy_start_requested')
                # A startup race used to leave the owner in fault_idle forever
                # when the first lowstate packet arrived just after the check.
                # Retry only this narrowly-scoped feedback fault; the native
                # controller and all motion gates remain unchanged.
                if (phase == 'fault_idle' and fault == 'No fresh ankle feedback before start'
                        and health and 0 <= now-health['monotonic'] <= .2
                        and not any(m['motorstate'] for m in health['motors'])):
                    phase='waiting_for_native_init';fault=None
                    event('body_feedback_recovered', automatic_motion_resume=False,
                          recovery='startup_lowstate')
                if started is not None:
                    native=trace.poll();measured=checked_state(native)
                    feedback.update(measured)
                    if not health or not 0<=time.monotonic()-health['monotonic']<=.2:raise ValueError('Stale ankle health')
                    if any(m['motorstate'] for m in health['motors']):raise ValueError('Ankle motor fault reported')
                    vla_moving = vla is not None and vla.moving
                    # Fixed demo speed bounds do not apply to the VLA deployment,
                    # including its thermal handoff. Standalone demo mode retains them.
                    guard(measured,moving=(flow is not None and not flow.done) or vla_moving or bool(moves and moves.busy),
                          check_speed=vla is None and not (moves and moves.busy))
                    settled=max(map(abs,measured['dq']))<.3 and np.linalg.norm(measured['base_ang_vel'])<.25
                    stable_since=(stable_since if stable_since is not None else now) if settled else None
                    if (phase == 'fault_idle' and fault in recoverable_feedback_faults
                            and stable_since is not None and now-stable_since >= 1
                            and health and 0 <= now-health['monotonic'] <= .2):
                        phase='ready';fault=None;ready_at=now;armed.clear()
                        event('body_feedback_recovered', automatic_motion_resume=False,
                              recovery='fresh_settled_feedback')
                    if lease.consume(now,phase,fault,
                            stable_since is not None and now-stable_since>=1,
                            job is not None or bool(moves and moves.busy) or vla_moving):
                        phase='ready';fault=None;ready_at=now
                        event('session_lease_recovered',automatic_motion_resume=False)
                    if lease.expired(now):raise ValueError(EXPIRED)
                    if phase=='starting_planner_idle' and native['motion']=='planner_motion' and native['encoder_source'] and stable_since is not None and now-stable_since>=3:
                        phase='ready';ready_at=now;event('standing_ready')
                    if held is not None and phase!='fault_idle':
                        good=(native['motion']=='temporary_motion' and any(np.max(abs(np.asarray(native['token'])-v))<1e-5 for v in recent))
                        # Native HarnessBlend uses a fixed 0.6 s encoder/token
                        # source transition. During it the logged token is a blend,
                        # not a packet echo. Bound this allowance to initial entry.
                        alpha=native.get('blend_alpha',1.)
                        if (handoff_at is not None and now-handoff_at<1. and
                            native['motion'] in ('planner_motion','temporary_motion') and
                            isinstance(alpha,(int,float)) and np.isfinite(alpha) and 0<=alpha<=1):
                            good=True
                        source_bad_since=None if good else (source_bad_since if source_bad_since is not None else now)
                        if source_bad_since is not None and now-source_bad_since>.5:raise ValueError('Native token readback mismatch')
                    arm_path=args.run/'thermal_arm.json'
                    if arm_path.exists() and phase!='fault_idle':
                        arm=json.loads(arm_path.read_text());wanted=set(arm['sides'])
                        if arm['session_id']!=session_id or not wanted<=(configured_sides|completed_sides):raise ValueError('Thermal arming requires configured sides or this-session completed tests')
                        if wanted!=armed:
                            armed=wanted;event('thermal_arming_changed',sides=sorted(armed))
                    path=args.run/'ankle_requests/pending.json'
                    if path.exists():
                        claimed=path.with_name('claimed.json');path.replace(claimed);request=None
                        try:
                            request=json.loads(claimed.read_text());validate_request(request,session_id,time.monotonic())
                            if phase!='ready' or job is not None or (moves and moves.busy) or stable_since is None or now-stable_since<1:raise ValueError('Owner busy or robot not settled')
                            begin(request['side'],'manual_test',request['id'])
                        except (ValueError,KeyError,TypeError) as exc:
                            event('job_rejected',reason=str(exc))
                            if isinstance(request,dict) and isinstance(request.get('id'),str) and len(request['id'])==32 and all(c in '0123456789abcdef' for c in request['id']):
                                atomic(args.run/'ankle_results'/(request['id']+'.json'),dict(id=request['id'],completed=False,reason=str(exc)))
                        finally:claimed.unlink(missing_ok=True)
                    thermal_pending=[]
                    if phase=='ready' and job is None:
                        for side in ('left','right'):
                            hot=(detector.active[side] if thermal_rule=='native90' else
                                 any(max(m['temperature_winding_driver_c'])>91 for m in health['motors'] if m['name'].startswith(side)))
                            eligible=(dispatched_alarms.get(side)!=detector.episodes[side] or
                                      side in repeat_ready_at and now>=repeat_ready_at[side])
                            if hot and detector.active[side] and side in armed and eligible:thermal_pending.append(side)
                    if thermal_pending and vla and (vla.moving or vla.initialized):
                        vla.pause('ankle_temperature_priority',invalidate=True)
                    if phase=='ready' and job is None and not (moves and moves.busy) and stable_since is not None and now-stable_since>=1:
                        if thermal_pending:begin(thermal_pending[0],'temperature')
                    if future is not None and future.done() and audio_done is None:
                        try:
                            audio_done=future.result();event('audio_completed',**job,**audio_done)
                        except Exception as exc:
                            finish('audio_failed: '+str(exc));phase='ready'
                    if job is not None and audio_done is not None and flow is None:
                        still_hot=(detector.active[job['side']] if thermal_rule=='native90' else
                            any(max(m['temperature_winding_driver_c'])>91 for m in health['motors'] if m['name'].startswith(job['side'])))
                        if job['trigger']=='temperature' and not still_hot:
                            finish('cooled_before_motion');phase='ready'
                        elif stable_since is not None and now-stable_since>=1:
                            anchor=held if held is not None else np.asarray(native['token'],dtype=np.float32)
                            if held is None:handoff_at=now
                            flow=DemoFlow(demos[job['side']],anchor,now,audio_completed=True)
                            phase='executing';event('motion_started',**job)
                        elif now-audio_done['playback_completed_monotonic']>10:
                            finish('not_settled_after_audio');phase='ready'
                    if flow is not None:
                        old_stage=flow.stage;held=flow.sample(now,settled)
                        if flow.stage!=old_stage:event('motion_stage',stage=flow.stage,**job)
                        samples.write(json.dumps(dict(monotonic=now,job=job,stage=flow.stage,source_frame=flow.source_frame,
                            sent_token=held.tolist(),native_token=native['token'],q=measured['q'],dq=measured['dq'],tilt=measured['tilt_rad']))+'\n');samples.flush()
                        if flow.done:
                            ok=flow.return_verified;finish('returned_and_settled' if ok else 'return_not_settled',ok)
                            phase='ready' if ok else 'review_required';ready_at=now
                    if moves:
                        move_command=moves.tick(now,native,
                            stable_since is not None and now-stable_since>=1,
                            phase=='ready' and not thermal_pending,
                            lease.remaining(now),vla)
                        if move_command is not None:
                            held=None;recent.clear();handoff_at=None;source_bad_since=None
                    if vla:
                        action=vla.tick(now,phase=='ready' and not thermal_pending and move_command is None and not (moves and moves.busy),
                            stable_since is not None and now-stable_since>=1,
                            held if held is not None else native['token'],lease.remaining(now))
                        if action is not None:
                            # The same owner applies paired BrainCo targets and body token.
                            if held is None:handoff_at=now
                            held=np.asarray(action['motion_token'],dtype=np.float32)
            except (OSError,KeyError,ValueError,RuntimeError,TypeError) as exc:
                feedback.update(None,error=str(exc))
                if phase=='starting_planner_idle' and native is None and now-started<20:pass
                else:
                    if phase!='fault_idle':
                        at=time.monotonic()
                        event('fault_idle',reason=str(exc),
                              trace_age_s=at-native['monotonic'] if native else None,
                              lowstate_age_s=at-native['lowstate_receive_monotonic_ns']/1e9 if native else None,
                              health_age_s=at-health['monotonic'] if health else None,
                              loop_elapsed_s=at-now)
                    # A later expiry must never mask an actual fault and make it recoverable.
                    if phase!='fault_idle' or fault==EXPIRED:fault=str(exc)
                    phase='fault_idle'
            if phase=='fault_idle':
                if moves:moves.abort('owner_fault: '+str(fault))
                move_command=None
                if vla:vla.pause('owner_fault: '+str(fault),invalidate=True)
                finish('fault: '+str(fault));held=None;armed.clear()
            if held is None:
                command=move_command or (moves.idle_command if moves else dict(locomotion=0,movement=[0,0,0],facing=[1,0,0],speed=0.))
                pub.send(moves.packet(wire,command) if moves else wire.build_planner_message(command['locomotion'],command['movement'],command['facing'],speed=command['speed']))
                pub.send(wire.build_command_message(start=pulse,stop=False,planner=True))
            else:
                pub.send(wire.pack_pose_message(dict(token_state=held[None].astype(np.float32),frame_index=np.array([frame],dtype=np.int64)),version=4))
                pub.send(wire.build_command_message(start=False,stop=False,planner=False));frame+=1;recent.append(held.copy())
            if now-last_status>.05:
                atomic(args.run/'ankle_status.json',dict(phase=phase,fault=fault,session_id=session_id,
                    updated_monotonic=time.monotonic(),measured=measured,active_job=job,
                    stage=flow.stage if flow else None,last_result=last_result,
                    software_completed_sides=sorted(completed_sides),thermal_auto_sides=sorted(armed),
                    configured_auto_sides=sorted(configured_sides),
                    thermal_rule=thermal_rule,thermal_repeat_seconds=repeat_seconds,
                    repeat_ready_at=repeat_ready_at.copy(),
                    **lease.snapshot(now),
                    health=health,holding_demonstration_pose=held is not None,
                    vla=vla.snapshot() if vla else None,
                    locomotion=moves.snapshot() if moves else None))
                last_status=now
            delay=max(0.,now+.02-time.monotonic())
            if vla and vla.runtime.session.running:delay=vla.runtime.session.sleep_seconds()
            time.sleep(delay)
    finally:
        feedback.update(None,error='owner_exit_no_new_measurement')
        if moves:
            if moves.busy:
                command=moves.idle_command
                pub.send(moves.packet(wire,command))
                pub.send(wire.build_command_message(start=False,stop=False,planner=True))
            moves.abort('owner_exit')
        if vla:
            vla.pause('owner_exit',invalidate=True)
            vla.runtime.close()
        finish('owner_exit');pool.shutdown(wait=True)
        feedback.close()
        if sub:sub.Close()
        if health_bridge:health_bridge.close()
        event('owner_exited',native_controller_stopped=False)
        atomic(args.run/'ankle_status.json',dict(phase='owner_exited',session_id=session_id,updated_monotonic=time.monotonic()))
        trace.close();pub.close(0);ctx.term();events.close();samples.close();thermal_trace.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path)
    p.add_argument('--port',type=int,default=17586);p.add_argument('--start-policy',action='store_true')
    p.add_argument('--vla',action='store_true',help='Enable bounded agent RTC tasks inside this sole body owner')
    p.add_argument('--locomotion',action='store_true',help='Enable bounded g1_move planner jobs in this sole owner')
    p.add_argument('--lease-seconds',type=float,default=0,help='0: unlimited operator session (default); otherwise seconds >=30')
    p.add_argument('--thermal-auto-sides',choices=['left','right'],nargs='+',default=[],
                   help='Operator-configured sides: automatically announce and lift the hot side after standing is ready')
    p.add_argument('--thermal-rule',choices=['strict91','native90'],default='strict91')
    p.add_argument('--thermal-repeat-seconds',type=float,default=0.,help='0: once per hot episode; otherwise repeat only after a successful return and this interval (10..600s)')
    a=p.parse_args()
    try:duration(a.lease_seconds)
    except ValueError as exc:p.error(str(exc))
    run(a)


if __name__=='__main__':main()
