"""Bounded SONIC planner jobs, consumed only by the existing body owner.

Direction/heading conventions follow HumanoidHarnessDeploy sonic_actions/contract
and HumanoidHarnessDeploy929 obstacle/turning. No publisher or hardware imports.
"""
from .lease import remaining as lease_remaining
from .mailbox import publish
import json
import math
from pathlib import Path
import re
import time
import uuid

from .vla import atomic
from .walk import upper_fields, UPPER_IDS, UPPER_HARDWARE
from .state import HARDWARE_TO_ISAAC

# Body frame at segment start: x forward, y left. Diagonals have unit length.
DIRECTIONS = dict(forward=(1, 0), backward=(-1, 0), left=(0, 1), right=(0, -1),
                  forward_left=(1, 1), forward_right=(1, -1),
                  backward_left=(-1, 1), backward_right=(-1, -1))
TURNS = ('rotate_left', 'rotate_right')
MOTIONS = (*DIRECTIONS, *TURNS, 'step_in_place')
MIN_SPEED = 0.4
MAX_SPEED = 0.5
DEFAULT_SPEED = 0.4
SPEED_POLICY = 'planner_reference_0_4_to_0_5_v1'
HANDOFF_TIMEOUT = 3.
HANDOFF_SETTLE_TIMEOUT = 6.
SETTLE_TIMEOUT = 10.
MOVE_OVERHEAD = HANDOFF_SETTLE_TIMEOUT + SETTLE_TIMEOUT + 2.
LIMITS = dict(min_speed=MIN_SPEED, max_speed=MAX_SPEED, default_speed=DEFAULT_SPEED, speed_policy=SPEED_POLICY,
              duration_policy='caller_selected_positive_seconds',
              max_yaw_rate_deg_s=30.,
              handoff_timeout_s=HANDOFF_TIMEOUT, handoff_settle_timeout_s=HANDOFF_SETTLE_TIMEOUT)


def number(value, name, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be finite in [{low}, {high}]')
    return float(value)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{32}', value):
        raise ValueError('Expected a 32-character frame/job ID')
    return value


def validate(action, **args):
    if action in ('status', 'stop'):
        if args:raise ValueError(f'{action} takes only action')
        return {}
    if action == 'result':
        if set(args) != {'job_id'}:raise ValueError('result requires only job_id')
        return dict(job_id=identifier(args['job_id']))
    if action != 'run':raise ValueError('action must be status, run, stop or result')
    allowed = {'direction', 'speed', 'duration_s', 'mode', 'yaw_rate_deg_s', 'frame_id', 'evidence'}
    if set(args)-allowed or not {'direction','frame_id','evidence'} <= set(args):
        raise ValueError('run requires direction, frame_id and evidence; unknown fields rejected')
    direction = args['direction']
    if direction not in MOTIONS:
        raise ValueError(f"Unknown movement direction {direction!r}; use one of: "
                         f"{', '.join(MOTIONS)}. Turns are rotate_left/rotate_right.")
    identifier(args['frame_id'])
    evidence = args['evidence']
    if not isinstance(evidence, str) or not 1 <= len(evidence.strip()) <= 2000:
        raise ValueError('Describe the current observation and intended motion')
    mode = args.get('mode', 'walk' if direction in DIRECTIONS else 'slowWalk')
    if mode not in ('slowWalk', 'walk'):raise ValueError('mode must be slowWalk or walk')
    if direction not in DIRECTIONS and mode != 'slowWalk':
        raise ValueError('Rotation/stepping requires slowWalk')
    duration = duration_seconds(args.get('duration_s'))
    speed = number(args.get('speed', DEFAULT_SPEED), 'speed', MIN_SPEED, MAX_SPEED)
    command = dict(direction=direction, mode=mode, speed=speed, duration_s=duration,
                   frame_id=args['frame_id'], evidence=evidence.strip())
    if direction in TURNS:
        rate = number(args.get('yaw_rate_deg_s', 15.), 'yaw_rate_deg_s', 1., 30.)
        command['yaw_rate_deg_s'] = rate
    elif 'yaw_rate_deg_s' in args:
        raise ValueError('yaw_rate_deg_s is only for rotation')
    return command


def duration_seconds(value):
    """The caller selects the duration; there is no fixed pulse or upper bound."""
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('duration_s is required and must be finite and greater than zero')
    return float(value)


def yaw(quat):
    w, x, y, z = quat
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def idle(heading=0.):
    return dict(locomotion=0, movement=[0.,0.,0.], facing=[math.cos(heading),math.sin(heading),0.], speed=0.)


def planner_command(command, body_yaw, reference_yaw, elapsed):
    heading = body_yaw-reference_yaw
    x, y = DIRECTIONS.get(command['direction'], (0.,0.))
    norm = math.hypot(x,y) or 1.
    movement = [(math.cos(heading)*x-math.sin(heading)*y)/norm,
                (math.sin(heading)*x+math.cos(heading)*y)/norm, 0.]
    if command['direction'] in TURNS:
        sign = 1 if command['direction']=='rotate_left' else -1
        heading += math.radians(sign*command['yaw_rate_deg_s']*min(max(elapsed,0.),command['duration_s']))
    return dict(locomotion=1 if command['mode']=='slowWalk' else 2, movement=movement,
                facing=[math.cos(heading),math.sin(heading),0.], speed=command['speed'])


def status(directory):
    if directory is None:raise RuntimeError('No movement owner configured')
    state = json.loads((Path(directory)/'ankle_status.json').read_text())
    if not 0 <= time.monotonic()-state['updated_monotonic'] <= .25:
        raise RuntimeError('Movement owner status stale')
    if not state.get('locomotion'):raise RuntimeError('Owner was not started with --locomotion')
    return state


def submit(directory, action, command=None, observed_at=None):
    state = status(directory)
    if action not in ('run','stop'):raise ValueError('Only run/stop can be submitted')
    if action == 'run':
        command = validate('run', **command)
        from .freshness import initial_observation_allowed
        initial=initial_observation_allowed(directory,command['frame_id'])
        number(observed_at, 'observed_at', 0, time.monotonic())
        if state['phase']!='ready' or state.get('fault') or state['locomotion']['active_job']:
            raise RuntimeError('Movement owner busy or unavailable')
        if (state.get('vla') or {}).get('active_job'):raise RuntimeError('Pause VLA and observe before moving')
        if lease_remaining(state) < command['duration_s']+MOVE_OVERHEAD:
            raise RuntimeError('Insufficient owner lease')
    request = dict(id=uuid.uuid4().hex, session_id=state['session_id'],
                   created_monotonic=time.monotonic(), action=action)
    if action == 'run':request.update(command=command, observed_at=observed_at)
    if action=='run' and initial:
        marker=json.loads((Path(directory)/'execution-memory.json').read_text())
        from .vla import atomic
        atomic(Path(directory)/'initial-motion-admission.json',dict(episode=marker['episode'],
            job_id=request['id'],frame_id=command['frame_id']))
    folder = Path(directory)/'move_requests'
    temp = folder/(request['id']+'.json')
    try:
        temp.write_text(json.dumps(request, allow_nan=False))
        # A second caller must not replace an unconsumed command.
        publish(temp, folder/('stop.json' if action=='stop' else 'pending.json'))
    finally:temp.unlink(missing_ok=True)
    return dict(job_id=request['id'], status='submitted', task_success_verified=False)


def result(directory, job_id):
    identifier(job_id)
    if directory is None:raise RuntimeError('No movement owner configured')
    path = Path(directory)/'move_results'/(job_id+'.json')
    if path.exists():
        from .action_state import enrich_result
        return enrich_result(directory,json.loads(path.read_text()))
    return dict(job_id=job_id,status='pending_or_unknown',owner=status(directory)['locomotion'])


class LocomotionJobs:
    def __init__(self, directory, session_id, feedback=None):
        self.root=Path(directory);self.session_id=session_id
        for name in ('move_requests','move_results'):(self.root/name).mkdir(exist_ok=True)
        self.job=None;self.phase='idle';self.last_result=None;self.stop_ids=[]
        self.idle_command=idle();self.started=None;self.at=None;self.quiet_at=None
        self.confirmed=False;self.measured_yaw=0.;self.last_yaw=None;self.stop_reason=None
        self.upper={}
        self.upper_error_peak=0.
        self.feedback=feedback
        self.handoff={}
        self.handoff_extended=False

    @property
    def busy(self):return self.job is not None

    def snapshot(self):
        return dict(phase=self.phase,active_job=self.job,last_result=self.last_result,
                    directions=list(MOTIONS),limits=LIMITS,hardware_validated=False,
                    upper_body_hold=self.upper,hand_policy='retain_existing_targets_no_hand_writes',
                    handoff=self.handoff,
                    body_feedback=self.feedback.snapshot(self.job['id']) if self.feedback and self.job else None)

    def packet(self, wire, command):
        return wire.build_planner_message(command['locomotion'],command['movement'],command['facing'],
                                          speed=command['speed'],**self.upper)

    def upper_confirmed(self, native):
        obs=native.get('encoder_obs',[])
        if len(obs)<584:return False
        return all(math.isfinite(obs[4+frame*29+index]) and
                   abs(obs[4+frame*29+index]-self.upper['upper_body_position'][i])<1e-4 and
                   math.isfinite(obs[294+frame*29+index]) and abs(obs[294+frame*29+index])<1e-4
                   for frame in range(10) for i,index in enumerate(UPPER_IDS))

    def write_result(self, ident, **data):
        from .motion_feedback import motion_feedback
        if 'body_feedback' in data:
            data['motion_feedback']=motion_feedback(data['body_feedback'])
            data['measured_yaw_delta_deg']=data['motion_feedback']['heading']['angle_deg']
        value=dict(job_id=ident,session_id=self.session_id,task_success_verified=False,
                   actual_distance_m=None,speed_policy=SPEED_POLICY,**data)
        atomic(self.root/'move_results'/(ident+'.json'),value)
        return value

    def finish(self, reason, completed=False, settled=False):
        data=dict(completed=completed,reason=reason,settled=settled,upper_body_hold=self.upper,
                  upper_tracking_max_error_rad=self.upper_error_peak,
                  source_confirmed=self.confirmed,measured_yaw_delta_deg=math.degrees(self.measured_yaw),
                  handoff=dict(self.handoff))
        if self.job:
            if self.feedback:
                data['body_feedback']=self.feedback.finish(self.job['id'],reason)
            self.last_result=self.write_result(self.job['id'],command=self.job['command'],**data)
        for ident in self.stop_ids:
            self.write_result(ident,**(data|dict(completed=settled, affected_job_id=self.job['id'] if self.job else None)))
        self.stop_ids=[];self.job=None;self.phase='idle'

    def abort(self, reason):
        self.finish(reason)

    def stop(self, now, native, reason):
        if not self.busy:return
        if self.phase != 'settling':
            # A streamed -> planner switch reinitializes the reference heading.
            self.idle_command=idle(0. if native['motion']!='planner_motion' else yaw(native['base_quat'])-native['reference_yaw'])
            self.phase='settling';self.at=now;self.quiet_at=None
        # A stop/preemption arriving during normal settling still interrupts the job.
        self.stop_reason=reason

    def request(self, filename):
        path=self.root/'move_requests'/filename
        if not path.exists():return None
        claimed=path.with_suffix('.claimed');path.replace(claimed);request=None
        try:
            request=json.loads(claimed.read_text());identifier(request['id'])
            if request['session_id']!=self.session_id or not 0<=time.monotonic()-request['created_monotonic']<=2:
                raise ValueError('Changed session or expired movement request')
            expected='stop' if filename=='stop.json' else 'run'
            if request['action']!=expected:raise ValueError('Invalid mailbox action')
            keys={'id','session_id','created_monotonic','action'}
            if expected=='run':
                keys|={'command','observed_at'}
                request['command']=validate('run',**request['command'])
                from .freshness import initial_observation_allowed
                initial=initial_observation_allowed(self.root,request['command']['frame_id'],request['id'])
                number(request['observed_at'],'observed_at',0,time.monotonic())
            if set(request)!=keys:raise ValueError('Unexpected movement request fields')
            return request
        except (ValueError,KeyError,TypeError) as exc:
            if isinstance(request,dict) and re.fullmatch('[0-9a-f]{32}',str(request.get('id',''))):
                self.write_result(request['id'],completed=False,reason=str(exc),settled=False)
            return None
        finally:claimed.unlink(missing_ok=True)

    def tick(self, now, native, settled, available, lease_remaining, vla=None):
        """Return planner command while owning movement, else None. Never publish here."""
        stop=self.request('stop.json');request=self.request('pending.json')
        if stop:
            if self.busy:
                self.stop_ids.append(stop['id']);self.stop(now,native,'agent_stop')
            else:self.write_result(stop['id'],completed=True,reason='no_active_movement',settled=settled)
        if request:
            reason=None
            if stop:reason='superseded_by_stop'
            elif self.busy or not available:reason='owner_busy'
            elif vla and vla.moving:reason='vla_busy'
            elif not settled:reason='robot_not_settled'
            elif lease_remaining<request['command']['duration_s']+MOVE_OVERHEAD:reason='insufficient_owner_lease'
            elif not isinstance(native.get('reference_yaw'),(int,float)) or not math.isfinite(native['reference_yaw']):
                reason='missing_native_reference_yaw'
            if reason:self.write_result(request['id'],completed=False,reason=reason,settled=False)
            else:
                # Preserve waist + both arms in the native encoder's 17-joint order.
                # Keep the original target across consecutive planner segments (no drift).
                if not self.upper or native['motion']!='planner_motion':
                    measured=[native['body_q'][i] for i in HARDWARE_TO_ISAAC]
                    self.upper=upper_fields(measured,[0.]*29)
                # Planner changes body pose. Do not automatically resume an old RTC queue.
                if vla:vla.pause('locomotion_handoff',invalidate=True)
                self.job=request;self.phase='handoff';self.at=now;self.started=None;self.quiet_at=None
                self.confirmed=False;self.measured_yaw=0.;self.last_yaw=None;self.stop_reason=None
                self.upper_error_peak=0.
                self.handoff={};self.handoff_extended=False
                if self.feedback:self.feedback.start(request['id'],'move',now)
                self.idle_command=idle(0. if native['motion']!='planner_motion' else yaw(native['base_quat'])-native['reference_yaw'])
        if not self.busy:return None
        self.upper_error_peak=max(self.upper_error_peak,
            max(abs(native['body_q'][j]-target) for j,target in zip(UPPER_HARDWARE,self.upper['upper_body_position'])))
        if not available:self.stop(now,native,'owner_preempted')
        current_yaw=yaw(native['base_quat'])
        if self.last_yaw is not None:self.measured_yaw+=wrap(current_yaw-self.last_yaw)
        self.last_yaw=current_yaw
        planner=native['motion']=='planner_motion' and native['encoder_source']
        quiet=planner and native.get('locomotion')==0 and settled
        self.quiet_at=(self.quiet_at if self.quiet_at is not None else now) if quiet else None
        if self.phase=='handoff':
            # Source/upper echo must arrive within the original three-second
            # window. Only a confirmed idle planner may get more time to settle.
            stamp=native.get('state_monotonic_ns')
            fresh=type(stamp) is int and 0 <= time.monotonic()-stamp/1e9 <= .25
            source_ready=bool(planner and native.get('locomotion')==0)
            upper_ready=self.upper_confirmed(native)
            quiet_for=now-self.quiet_at if self.quiet_at is not None else 0.
            blockers=[name for name,ok in (('fresh_feedback',fresh),('planner_idle',source_ready),
                      ('upper_reference_echo',upper_ready),('settled',settled),('quiet_dwell',quiet_for>=.4)) if not ok]
            self.handoff=dict(elapsed_s=now-self.at,planner_idle=source_ready,upper_reference_echo=upper_ready,
                              feedback_fresh=fresh,settled=settled,quiet_dwell_s=quiet_for,
                              blockers=blockers,extended_for_settling=self.handoff_extended)
            if not fresh:
                self.stop(now,native,'planner_handoff_stale_feedback')
            elif now-self.at>HANDOFF_TIMEOUT and not (source_ready and upper_ready):
                self.stop(now,native,'planner_handoff_timeout')
            elif now-self.at>HANDOFF_SETTLE_TIMEOUT:
                self.stop(now,native,'planner_handoff_settle_timeout')
            elif now-self.at>=.6 and not blockers:
                self.phase='running';self.started=now;self.origin=current_yaw
                self.reference=native['reference_yaw'];self.measured_yaw=0.
            elif now-self.at>HANDOFF_TIMEOUT:
                self.handoff_extended=True
                self.handoff['extended_for_settling']=True
        if self.phase=='running':
            self.idle_command=idle(current_yaw-native['reference_yaw'])
            elapsed=now-self.started
            expected=1 if self.job['command']['mode']=='slowWalk' else 2
            if planner and native.get('locomotion')==expected:self.confirmed=True
            if not self.upper_confirmed(native):
                self.stop(now,native,'upper_body_reference_lost')
            elif abs(wrap(native['reference_yaw']-self.reference))>.05:
                self.stop(now,native,'reference_heading_changed')
            elif elapsed>.6 and (not planner or native.get('locomotion')!=expected):
                self.stop(now,native,'planner_source_lost')
            elif elapsed>=self.job['command']['duration_s']:
                self.stop(now,native,'duration_complete' if self.confirmed else 'planner_source_unconfirmed')
            else:return planner_command(self.job['command'],self.origin,self.reference,elapsed)
        if self.phase=='settling':
            # Cancel residual turn targets at the measured heading, never turn back to zero.
            if self.quiet_at is not None and now-self.quiet_at>=.4 and now-self.at>=.4:
                self.finish(self.stop_reason,completed=self.stop_reason=='duration_complete',settled=True)
            elif now-self.at>SETTLE_TIMEOUT:
                self.finish('movement_settle_timeout')
                raise RuntimeError('Movement did not settle')
        return self.idle_command
