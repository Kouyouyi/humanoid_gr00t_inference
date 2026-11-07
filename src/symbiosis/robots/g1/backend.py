"""RAX G1 tools. Construction is read-only; motion requires an operator-owned executor."""
import base64
import json
import os
from pathlib import Path
import re
import shutil
import time
import uuid

import cv2
import numpy as np

from .state import TraceReader
from .reference import check_pulse, DURATION
from .mailbox import publish
# camera_max_age_s remains accepted for old deployment files but is ignored.
DEFAULTS = dict(camera_directory=None, trace_path=None, camera_max_age_s=None, control_directory=None,
                experimental_control_views="first_person", third_camera_directory=None, camera_profile=None)


def tool(name, description, properties=None, required=()):
    return dict(name=name, description=description, inputSchema=dict(type='object',
                properties=properties or {}, required=list(required), additionalProperties=False))


class G1SonicBackend:
    mutating = frozenset({'g1_execute_reference', 'g1_vla', 'g1_hand', 'g1_move', 'g1_session_lease', 'g1_smpl_execute'})

    def __init__(self, config):
        unknown = set(config.backend_options)-DEFAULTS.keys()
        if unknown:
            raise ValueError(f'Unknown G1 backend settings: {sorted(unknown)}')
        self.options = DEFAULTS | config.backend_options
        if self.options['experimental_control_views'] not in ('first_person','first_and_third'):
            raise ValueError('Unknown control view mode')
        if self.options['experimental_control_views']=='first_and_third' and not self.options['third_camera_directory']:
            raise ValueError('Dual-view real backend requires an external third_camera_directory')
        self.config = config
        from symbiosis.task_runtime import for_config, default_bundle
        self.task_bundle=for_config(config) or default_bundle()
        self.camera = Path(self.options['camera_directory']).resolve() if self.options['camera_directory'] else None
        self.trace = TraceReader(self.options['trace_path'])
        self.control = Path(self.options['control_directory']).resolve() if self.options['control_directory'] else None
        self.plans = {}
        self.move_frames = set()
        self.frames = config.root/'agent-workspace/runtime/frames'
        self.boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()

    def tools(self):
        from .vla import TASKS
        from .locomotion import MOTIONS, MIN_SPEED, MAX_SPEED, DEFAULT_SPEED
        definitions = [tool('g1_session_lease', 'Read/change the operator session duration: status; set seconds=0 for unlimited or seconds>=30 to renew from now; result with job_id. Only an expiry fault can recover, after fresh settled feedback. Never clears hardware/transport faults, restarts native, resumes old motion, opens hands or sends O. Default is unlimited; 15-second VLA segments still apply.',
                     dict(action=dict(type='string',enum=['status','set','result']),seconds=dict(type='number',minimum=0),job_id=dict(type='string')),('action',)),
                tool('g1_move', 'Bounded body-relative movement through the single SONIC owner. '
                     'action=status lists capabilities; run requires direction, a g1_observe frame_id from this session and evidence. '
                     'Image age alone does not expire a frame; each frame ID may authorize only one movement segment. '
                     'Eight walking directions, rotate_left/right and step_in_place. speed is 0.4–0.5, default 0.4. '
                     'This is a planner gait/velocity reference (m/s for translation), '
                     'not measured speed; yaw_rate_deg_s ramps a turn heading target. '
                     'For every run explicitly choose duration_s from the target and current observation; '
                     'it must be finite and positive, with no default, minimum pulse, fixed turn duration or upper limit. '
                     'Stop may interrupt a command at any time. '
                     'run returns job_id; result requires only job_id; stop/status take only action. '
                     'Stop means planner idle, not motor power off. Observe after completion. '
                     'VLA must be paused first; movement invalidates VLA preparation. '
                     'A fresh image ID alone is not progress. Use the returned image and measured feedback to decide whether to continue, change the viewpoint, or revise the subgoal. '
                     'Waist and arm references hold the captured posture, including after stopping; existing hand targets remain unchanged. '
                     'Physical tracking/grasp retention must be verified from observations.',
                     dict(action=dict(type='string',enum=['status','run','stop','result']),
                          direction=dict(type='string',enum=list(MOTIONS)),
                          speed=dict(type='number',minimum=MIN_SPEED,maximum=MAX_SPEED,default=DEFAULT_SPEED),
                          duration_s=dict(type='number',exclusiveMinimum=0,
                                          description='Required for run. Choose seconds from the goal and current image; no default or fixed duration range. Stop may interrupt sooner.'),
                          mode=dict(type='string',enum=['slowWalk','walk'],description='Default walk for translation; slowWalk for rotation/stepping.'),
                          yaw_rate_deg_s=dict(type='number',minimum=1,maximum=30,default=15),
                          frame_id=dict(type='string'),evidence=dict(type='string',minLength=1,maxLength=2000),
                          job_id=dict(type='string')),('action',)),
                tool('g1_operator_messages', 'Read UI operator instructions or acknowledge one with a public reply. Read before actions and at every 15-second VLA boundary. Acknowledgement is not execution.',
                     dict(action=dict(type='string',enum=['read','ack']),message_id=dict(type='string'),reply=dict(type='string')),('action',)),
                tool('g1_hand', 'Open or close left, right, or both BrainCo hands over 2 seconds through the sole owner. Keep body posture and any unselected hand target. Requires idle owner with known hand targets; works after locomotion without VLA re-prepare. Observe contact/release geometry first. Pause VLA or stop movement and wait for settling. Poll g1_hand_result then observe; completion is not grasp/release success.',
                     dict(side=dict(type='string',enum=['left','right','both']),action=dict(type='string',enum=['open','close']),evidence=dict(type='string',minLength=1,maxLength=2000)),('side','action','evidence')),
                tool('g1_hand_result','Read completion/interruption of g1_hand. Command completion is not measured grasp/release success.',dict(job_id=dict(type='string')),('job_id',)),
                tool('g1_vla_status','Read optional VLA owner, trained tasks, progress and model; never starts hardware.'),
                tool('g1_vla','Use VLA as a bounded tool: run starts from current observations and executes N steps at 50 Hz; it does not require prepare, including after movement. Optional prepare blends initial body pose and opens hands in 6 s; pause discards queue and holds endpoint. For prepare/pause send ONLY action; omit task, steps and evidence. All three extra fields are required only for run. Thermal relief preempts the current queue. Result is not grasp success.',
                     dict(action=dict(type='string',enum=['prepare','run','pause']),
                          task=dict(type='string',enum=list(TASKS)),steps=dict(type='integer',minimum=1,maximum=1500),
                          evidence=dict(type='string',minLength=1,maxLength=2000)),('action',)),
                tool('g1_vla_result','Read VLA job completion or interruption; observe actual object after completion.',
                     dict(job_id=dict(type='string')),('job_id',)),
                tool('g1_action_state', 'Read hash-checked measured joint samples for a completed action in this owner session. '
                     'Samples are downsampled to 3 Hz with the final endpoint retained. '
                     'Filter by joint names and seconds relative to the first source sample; paginate with offset/limit. '
                     'Historical action state, not a current observation. No robot actions. Hand feedback is not included.',
                     dict(job_id=dict(type='string'), joints=dict(type='array',minItems=1,maxItems=29,items=dict(type='string')),
                          start_s=dict(type='number',minimum=0),end_s=dict(type='number',minimum=0),
                          offset=dict(type='integer',minimum=0),limit=dict(type='integer',minimum=1,maximum=64)),('job_id',)),
                tool('g1_status', 'Read measured body state and sensor availability; never moves hardware.'),
                tool('g1_observe', 'Save fresh rectified RGB, depth, valid mask, K and measured state. '
                     'T_base_cam is uncalibrated. Receive timestamps are not exposure timestamps.'),
                tool('g1_pixel_to_position', 'Back-project a saved rectified frame pixel to left optical '
                     'coordinates in metres. Base coordinates are refused until extrinsics are calibrated.',
                     dict(frame_id=dict(type='string'), x=dict(type='integer', minimum=0, maximum=639),
                          y=dict(type='integer', minimum=0, maximum=359),
                          radius=dict(type='integer', minimum=0, maximum=5),
                          coordinate_frame=dict(type='string', enum=['camera', 'base'])),
                     ('frame_id', 'x', 'y')),
                tool('g1_plan_arm_motion', 'Commissioning preview only: one elbow round trip, at most '
                     '0.05 rad, seven seconds of reference plus up to ten seconds of measured verification. No IK or collision check. '
                     'Requires an operator-commissioned standing reference; preview moves nothing.',
                     dict(side=dict(type='string', enum=['left', 'right']),
                          delta_rad=dict(type='number', minimum=-.05, maximum=.05)), ('side', 'delta_rad')),
                tool('g1_plan_reach', 'Preview fixed commissioning stages: clear both arms, or right reach and return. '
                     'Requires operator-enabled clear_reach_v1; reach_return also requires recent on-site arm-clearance evidence. '
                     'Wrist FK is not calibrated BrainCo grasp geometry. No arbitrary goals or hand commands.',
                     dict(action=dict(type='string',enum=['clear_arms','reach_return'])),('action',)),
                tool('g1_execute_reference', 'Submit a previewed bounded reference asynchronously to SONIC. '
                     'Returns job_id, NOT measured success. Read g1_reference_result and observe. '
                     'Plan expires after 30 seconds and cannot be reused. No hand commands.',
                     dict(plan_id=dict(type='string')), ('plan_id',)),
                tool('g1_plan_grasp_step', 'Preview one right wrist displacement from measured position, then hold. '
                     'Translation <= 10 cm, rotation <= 0.35 rad. Observe the actual hand and object before each next step. '
                     'No collision or grasp success claim.',
                     dict(delta_m=dict(type='array',items=dict(type='number'),minItems=3,maxItems=3),
                          rotation_delta=dict(type='array',items=dict(type='number'),minItems=3,maxItems=3)),
                     ('delta_m','rotation_delta')),
                tool('g1_plan_walk', 'Preview one fixed left leading step, 4.4 s plus measured settling. '
                     'Requires patched native cursor and current walking-site evidence; one step per commissioning run. '
                     'Actual distance must be observed after stopping; no automatic repetition.'),
                tool('g1_reference_result', 'Read the measured result of a submitted reference job, or pending.',
                     dict(job_id=dict(type='string')), ('job_id',))]
        from .smpl import OWNED
        delta=dict(type='array',items=dict(type='number'),minItems=3,maxItems=3)
        keyframe=dict(type='object',additionalProperties=False,required=['step','joint_deltas'],
            properties=dict(step=dict(type='integer',minimum=0,maximum=149),
                joint_deltas=dict(type='object',additionalProperties=False,
                    properties={joint:delta for joint in OWNED})))
        program=dict(type='object',additionalProperties=False,required=['n','keyframes'],
            properties=dict(n=dict(type='integer',enum=[150]),
                keyframes=dict(type='array',minItems=3,maxItems=5,items=keyframe)))
        definitions.extend([
            tool('g1_smpl_reference','CoMotion adjustment reference and current public body state. Only for direct interaction; no motion.'),
            tool('g1_smpl_execute','CoMotion-style local SMPL arm adjustment. 150 steps at 50 Hz, 3..5 keyframes spanning 0..149. First keyframe zero delta; quintic interpolation, preserve hand targets. Returns job_id; host polls and enters judgment.',
                dict(program=program,evidence=dict(type='string',minLength=1,maxLength=2000)),('program','evidence')),
            tool('g1_smpl_result','Read direct CoMotion job result.',dict(job_id=dict(type='string')),('job_id',))])
        from symbiosis.task_runtime import for_config
        from symbiosis.task_runtime.registry import apply_overlays
        bundle=self.task_bundle
        return apply_overlays(definitions,bundle) if bundle else definitions

    def validate(self, name, arguments):
        schema = next((t['inputSchema'] for t in self.tools() if t['name'] == name), None)
        if schema is None:
            raise ValueError('Unknown G1 tool')
        if set(arguments)-schema['properties'].keys() or set(schema['required'])-arguments.keys():
            raise ValueError('Unknown or missing arguments')
        if name == 'g1_smpl_execute':
            from .smpl import validate as check_program
            check_program(arguments['program'])
        if name == 'g1_hand':
            from .vla import validate_hand
            validate_hand(**arguments)
        if name == 'g1_session_lease':
            from .lease import validate
            validate(**arguments)
        if name == 'g1_operator_messages':
            action = arguments.get('action')
            if action == 'read' and set(arguments) != {'action'}:
                raise ValueError('read takes only action')
            if action == 'ack' and (set(arguments) != {'action','message_id','reply'} or not isinstance(arguments['message_id'],str)):
                raise ValueError('ack takes message_id and reply')
            if action not in ('read','ack'):
                raise ValueError('Unknown message action')
        if name == 'g1_vla':
            if arguments.get('action') not in ('prepare','run','pause'):raise ValueError('Unknown VLA action')
            from .vla import validate
            validate(**arguments)
        if name == 'g1_move':
            from .locomotion import validate
            validate(**arguments)
        if name == 'g1_plan_arm_motion':
            check_pulse(arguments['side'], arguments['delta_rad'])
        if name == 'g1_plan_reach' and arguments['action'] not in ('clear_arms','reach_return'):
            raise ValueError('Unknown commissioning reach action')
        if name == 'g1_plan_grasp_step':
            from .reach import validate_step
            validate_step(**arguments)
        for key in ('plan_id', 'job_id'):
            if key in arguments and (not isinstance(arguments[key], str) or not re.fullmatch('[0-9a-f]{32}', arguments[key])):
                raise ValueError('Invalid '+key)
        if name == 'g1_pixel_to_position':
            if not isinstance(arguments['frame_id'], str) or not re.fullmatch('[0-9a-f]{32}', arguments['frame_id']):
                raise ValueError('Invalid frame_id')
            for key, lo, hi, default in [('x', 0, 639, None), ('y', 0, 359, None), ('radius', 0, 5, 2)]:
                v = arguments.get(key, default)
                if type(v) is not int or not lo <= v <= hi:
                    raise ValueError('Invalid pixel '+key)
            if arguments.get('coordinate_frame', 'camera') not in ('camera', 'base'):
                raise ValueError('Unknown coordinate_frame')

    def _camera_meta(self):
        if self.camera is None:
            raise RuntimeError('No RGB-D service configured')
        health = json.loads((self.camera/'health.json').read_text())
        meta = json.loads((self.camera/'latest.json').read_text())
        now = time.monotonic()
        if not health['running'] or health['errors'] or not 0 <= now-health['updated_monotonic'] <= 1:
            raise RuntimeError('RGB-D service stopped or unhealthy')
        if meta['host_boot_id'] != self.boot_id or meta['source'] != 'live_camera':
            raise RuntimeError('RGB-D must come from this workstation boot and a live camera')
        if not re.fullmatch('[0-9a-f]{32}', meta['frame_id']):
            raise ValueError('Invalid source frame id')
        return meta

    def status(self):
        out = dict(robot='g1_sonic', motion_available=False, hands_available=False,
                   motion_blocker='No live operator executor',
                   calibration=dict(metric_depth=False, camera_to_robot=False))
        try:
            out['executor'] = self._executor_status()
            out['motion_available'] = out['executor']['phase'] == 'reference_ready'
            out['motion_blocker'] = None if out['motion_available'] else out['executor']['phase']
        except (OSError, ValueError, KeyError, RuntimeError) as e:
            out['executor_error'] = str(e)
        try:
            out['body'] = self.trace.status()
        except (OSError, ValueError, KeyError, RuntimeError) as e:
            out['body_error'] = str(e)
        try:
            out['camera'] = self._camera_meta()
        except (OSError, ValueError, KeyError, RuntimeError) as e:
            out['camera_error'] = str(e)
        try:
            from .vla import status
            out['vla_owner']=status(self.control)
            out['motion_available']=out['vla_owner']['phase']=='ready'
            out['hands_available']=True
            out['motion_blocker']=None if out['motion_available'] else out['vla_owner']['phase']
        except (OSError,ValueError,KeyError,RuntimeError) as e:
            out['vla_error']=str(e)
        try:
            from .locomotion import status as movement_status
            owner=movement_status(self.control)
            out['locomotion']=owner['locomotion']
            out['locomotion_available']=owner['phase']=='ready' and not owner.get('fault') and not owner['locomotion']['active_job'] and not (owner.get('vla') or {}).get('active_job')
            if out['locomotion_available']:
                out['motion_available']=True;out['motion_blocker']=None
        except (OSError,ValueError,KeyError,RuntimeError) as e:
            out['locomotion_available']=False;out['locomotion_error']=str(e)
        return out

    def _executor_status(self):
        if self.control is None:
            raise RuntimeError('No operator reference executor configured')
        value = json.loads((self.control/'executor_status.json').read_text())
        if not 0 <= time.monotonic()-value['updated_monotonic'] <= .25:
            raise RuntimeError('Reference executor status stale')
        return value

    def plan(self, side, delta_rad):
        state = self._executor_status()
        if state['phase'] != 'reference_ready' or state['fault']:
            raise RuntimeError('Reference executor not ready')
        if state['reference_lease_remaining_s'] < DURATION+13:
            raise RuntimeError('Operator reference lease too short for a new motion')
        now = time.monotonic()
        self.plans = {k: v for k, v in self.plans.items() if now-v['created'] < 30}
        if len(self.plans) >= 64:
            raise RuntimeError('Too many pending plans')
        plan_id = uuid.uuid4().hex
        self.plans[plan_id] = dict(side=side, delta_rad=delta_rad, created=now,
            baseline_id=state['baseline_id'], q_before=state['measured']['q'])
        return dict(plan_id=plan_id, side=side, delta_rad=delta_rad, duration_s=DURATION,
                    expires_after_s=30, returns_to_standing_reference=True, collision_checked=False,
                    joint=side+'_elbow_joint', motion_sent=False)

    def execute(self, plan_id):
        plan = self.plans.get(plan_id)
        if plan is None or time.monotonic()-plan['created'] > 30:
            raise ValueError('Unknown, consumed or expired plan')
        state = self._executor_status()
        if state['phase'] != 'reference_ready' or state['fault'] or state['baseline_id'] != plan['baseline_id']:
            raise RuntimeError('Standing reference changed, busy or unavailable')
        if state['reference_lease_remaining_s'] < (60 if 'action' in plan else DURATION+13):
            raise RuntimeError('Operator reference lease too short')
        if np.max(np.abs(np.asarray(state['measured']['q'])-plan['q_before'])) > .05:
            raise RuntimeError('Robot moved since preview; plan again')
        job_id = uuid.uuid4().hex
        job = dict(id=job_id, action=plan.get('action','elbow_pulse'), baseline_id=plan['baseline_id'],
                   created_monotonic=time.monotonic())
        if 'action' not in plan:
            job.update(side=plan['side'],delta_rad=plan['delta_rad'])
        if plan.get('action')=='grasp_step':
            job.update(delta_m=plan['delta_m'],rotation_delta=plan['rotation_delta'])
        folder = self.control/'reference_requests'
        temp = folder/('.'+job_id)
        temp.write_text(json.dumps(job, allow_nan=False)+'\n')
        try:
            # Atomic exclusive submission: never overwrite another pending job.
            publish(temp, folder/'pending.json')
        finally:
            temp.unlink(missing_ok=True)
        del self.plans[plan_id]
        return dict(job_id=job_id, status='submitted', measured_success=None)

    def plan_reach(self, action):
        state=self._executor_status()
        if (not state.get('reach_profile_enabled') or state['phase']!='reference_ready' or state['fault'] or
            state['reference_lease_remaining_s']<60):
            raise RuntimeError('Operator reach profile is unavailable or lease too short')
        if state.get('posture') != ('standing' if action=='clear_arms' else 'clear_arms'):
            raise RuntimeError('Reach stage order does not match current posture')
        now=time.monotonic()
        self.plans={k:v for k,v in self.plans.items() if now-v['created']<30}
        if len(self.plans)>=64:
            raise RuntimeError('Too many pending plans')
        plan_id=uuid.uuid4().hex
        self.plans[plan_id]=dict(action=action,created=now,baseline_id=state['baseline_id'],q_before=state['measured']['q'])
        return dict(plan_id=plan_id,action=action,profile='clear_reach_v1',motion_sent=False,
            transition_seconds=6,verification_timeout_per_stage_s=24,
            brainco_collision_checked=False,requires_site_clearance=action=='reach_return')

    def reference_result(self, job_id):
        if self.control is None:
            raise RuntimeError('No operator reference executor configured')
        path = self.control/'reference_results'/(job_id+'.json')
        if path.exists():
            return json.loads(path.read_text())
        return dict(job_id=job_id, status='pending_or_unknown', executor=self._executor_status()['phase'])

    def plan_walk(self):
        state=self._executor_status()
        if not state.get('gait_profile_enabled') or state.get('walk_attempted'):
            raise RuntimeError('Walking profile unavailable or already used in this run')
        result=self.plan_reach('walk_step')
        result.update(profile='reference_leading_step_v1',transition_seconds=4.4,
                      requires_site_clearance=True,actual_distance_m=None,automatic_repeat_allowed=False)
        return result

    def plan_grasp_step(self, delta_m, rotation_delta):
        from .reach import validate_step
        validate_step(delta_m,rotation_delta)
        state=self._executor_status()
        if not state.get('grasp_profile_enabled'):
            raise RuntimeError('Operator grasp profile is not enabled')
        result=self.plan_reach('grasp_step')
        self.plans[result['plan_id']].update(delta_m=delta_m,rotation_delta=rotation_delta)
        result.update(profile='grasp_step_v1',delta_m=delta_m,rotation_delta=rotation_delta,
                      holds_endpoint=True,requires_site_clearance=not state.get('grasp_site_verified'))
        return result

    def observe(self):
        meta = self._camera_meta()
        source = self.camera/meta['frame_id']
        frame_id = uuid.uuid4().hex
        destination = self.frames/frame_id
        destination.mkdir(parents=True)
        try:
            for file in ('rgb.png', 'depth.png', 'valid.png'):
                shutil.copyfile(source/file, destination/file)
            state = self.status()
            observation = dict(meta, source_frame_id=meta['frame_id'], frame_id=frame_id,
                               saved_path='runtime/frames/'+frame_id,
                               robot_state=state.get('body'), robot_state_error=state.get('body_error'),
                               robot_camera_hardware_synchronized=False)
            model_image=destination/'rgb.png'
            if self.options['experimental_control_views']=='first_and_third':
                from .external_camera import pair_observation
                observation,model_image=pair_observation(observation,destination,self.options['third_camera_directory'],profile_path=self.options['camera_profile'])
            (destination/'meta.json').write_text(json.dumps(observation, indent=2)+'\n')
            image = dict(type='image', mimeType='image/png',
                         data=base64.b64encode(model_image.read_bytes()).decode())
            return observation, [image]
        except BaseException:
            shutil.rmtree(destination)
            raise

    def pixel(self, frame_id, x, y, radius=2, coordinate_frame='camera'):
        if coordinate_frame == 'base':
            raise ValueError('Camera-to-robot extrinsics not calibrated; base position unavailable')
        folder = self.frames/frame_id
        meta = json.loads((folder/'meta.json').read_text())
        depth = cv2.imread(str(folder/'depth.png'), cv2.IMREAD_UNCHANGED)
        valid = cv2.imread(str(folder/'valid.png'), cv2.IMREAD_GRAYSCALE)
        if depth is None or valid is None or depth.dtype != np.uint16 or depth.shape != (360, 640):
            raise ValueError('Invalid saved depth')
        region = np.s_[max(0, y-radius):min(360, y+radius+1), max(0, x-radius):min(640, x+radius+1)]
        samples = depth[region][(valid[region] > 0) & (depth[region] > 0)]
        if not samples.size:
            raise ValueError('No valid depth around pixel')
        z = float(np.median(samples))/1000
        k = np.asarray(meta['K'])
        return dict(frame_id=frame_id, coordinate_frame=meta['optical_frame'],
                    position_m=[(x-k[0, 2])*z/k[0, 0], (y-k[1, 2])*z/k[1, 1], z],
                    valid_samples=int(samples.size), metric_accuracy_validated=False)

    def call(self, name, arguments):
        self.validate(name, arguments)
        from .messages import Inbox
        inbox = Inbox(self.control) if self.control and (self.control/'operator-messages.sqlite3').exists() else None
        if name == 'g1_operator_messages':
            if self.control is None:
                raise RuntimeError('No control directory configured')
            inbox = inbox or Inbox(self.control)
            return (inbox.snapshot(read=True) if arguments['action']=='read' else
                    inbox.acknowledge(arguments['message_id'],arguments['reply'])), []
        pause = (name == 'g1_vla' and arguments.get('action') == 'pause') or (name == 'g1_move' and arguments.get('action') in ('stop','status','result')) or (name == 'g1_session_lease' and arguments.get('action') in ('status','result'))
        if inbox and name in self.mutating and not pause and inbox.snapshot()['pending']:
            raise RuntimeError('Pending UI instructions: read and acknowledge g1_operator_messages before new actions')
        value, images = self._call(name, arguments)
        if inbox:
            value = dict(value, operator_messages=inbox.snapshot(read=True))
        return value, images

    def _call(self, name, arguments):
        if name == 'g1_action_state':
            from .action_state import query
            if self.control is None:raise RuntimeError('No action state directory configured')
            owner=json.loads((self.control/'ankle_status.json').read_text())
            return query(self.control,owner['session_id'],**arguments),[]
        if name == 'g1_session_lease':
            from . import lease
            if arguments['action']=='status':return lease.status(self.control),[]
            if arguments['action']=='result':return lease.result(self.control,arguments['job_id']),[]
            return lease.submit(self.control,arguments['seconds']),[]
        if name == 'g1_move':
            from . import locomotion
            action=arguments['action']
            if action=='status':return locomotion.status(self.control),[]
            if action=='result':return locomotion.result(self.control,arguments['job_id']),[]
            if action=='stop':return locomotion.submit(self.control,'stop'),[]
            command=locomotion.validate(**arguments)
            from .freshness import reusable_episode_frame
            if command['frame_id'] in self.move_frames and not reusable_episode_frame(self.control,command['frame_id']):
                raise ValueError('Observe again before the next movement segment')
            meta=json.loads((self.frames/command['frame_id']/'meta.json').read_text())
            self._camera_meta()  # Camera must still be live when submitting a saved observation.
            if meta['host_boot_id']!=self.boot_id or meta['source']!='live_camera':
                raise ValueError('Movement requires a live observation from this boot')
            response=locomotion.submit(self.control,'run',command,meta['received_monotonic'])
            self.move_frames.add(command['frame_id'])
            return response,[]
        if name=='g1_smpl_reference':
            from .smpl import reference
            from .vla import status
            owner=status(self.control)
            if owner.get('fault') or (owner.get('vla') or {}).get('active_job'):
                raise RuntimeError('CoMotion reference requires idle healthy owner')
            return dict(**reference(),robot_state=self.trace.status()),[]
        if name=='g1_smpl_execute':
            from .vla import submit
            return submit(self.control,'smpl',**arguments),[]
        if name=='g1_smpl_result':
            from .vla import result
            return result(self.control,arguments['job_id']),[]
        if name == 'g1_hand':
            from .vla import submit_hand
            return submit_hand(self.control, **arguments), []
        if name in ('g1_vla','g1_vla_status','g1_vla_result','g1_hand_result'):
            from . import vla
            if name=='g1_vla':return vla.submit(self.control,**arguments),[]
            if name=='g1_vla_status':return vla.status(self.control),[]
            return vla.result(self.control,**arguments),[]
        if name == 'g1_status':
            return self.status(), []
        if name == 'g1_observe':
            return self.observe()
        if name == 'g1_pixel_to_position':
            return self.pixel(**arguments), []
        if name == 'g1_plan_arm_motion':
            return self.plan(**arguments), []
        if name == 'g1_plan_reach':
            return self.plan_reach(**arguments), []
        if name == 'g1_plan_grasp_step':
            return self.plan_grasp_step(**arguments), []
        if name == 'g1_plan_walk':
            return self.plan_walk(), []
        if name == 'g1_execute_reference':
            return self.execute(**arguments), []
        return self.reference_result(**arguments), []

    def close(self):
        self.trace.close()
