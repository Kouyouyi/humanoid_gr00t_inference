"""Bounded commissioning reach profile and measured wrist FK; no transport.

FK covers G1 body/wrist links only, not BrainCo collision or grasp geometry.
Planning runs outside the publisher thread. Site clearance is operator evidence.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .state import NAMES
from .reference import LOOKAHEAD, RATE

PROFILE = 'clear_reach_v1'
DURATION = 6.
TIMEOUT = 24.
ARM_IDS = {s:np.array([NAMES.index(s+'_'+p+'_joint') for p in
    ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]) for s in ('left','right')}
OTHER_IDS = np.array([i for i in range(29) if i not in np.r_[ARM_IDS['left'], ARM_IDS['right']]])


class Kinematics:
    def __init__(self):
        model = json.loads(Path(__file__).with_name('kinematics.json').read_text())
        self.bodies = model['bodies']
        self.local = [Rotation.from_quat(np.array(b['quat'])[[1,2,3,0]]).as_matrix() for b in self.bodies]
        self.limits = np.zeros((29,2))
        self.index = {name:i for i,name in enumerate(NAMES)}
        for b in self.bodies:
            for j in b['joints']:
                self.limits[self.index[j['name']]] = j['range']

    def fk(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (29,) or not np.isfinite(q).all():
            raise ValueError('Invalid measured q for wrist FK')
        rotations = [np.eye(3)]; positions = [np.zeros(3)]; wrists = {}
        for i,b in enumerate(self.bodies):
            parent = b['parent']
            if b['name'] == 'pelvis':
                rot, pos = np.eye(3), np.zeros(3)
            else:
                rot = rotations[parent]@self.local[i]
                pos = positions[parent]+rotations[parent]@b['pos']
                for j in b['joints']:
                    angle = q[self.index[j['name']]]-j['qpos0']
                    turn = Rotation.from_rotvec(np.asarray(j['axis'])*angle).as_matrix()
                    pos = pos+rot@(np.asarray(j['pos'])-turn@j['pos'])
                    rot = rot@turn
            rotations.append(rot); positions.append(pos)
            for side in ('left','right'):
                if b['name'] == side+'_wrist_yaw_link':
                    wrists[side] = dict(position_m=pos.tolist(), rotation_vector=Rotation.from_matrix(rot).as_rotvec().tolist())
        return wrists

    def solve(self, q, targets, rotations=None):
        q = np.asarray(q, dtype=float).copy(); details = {}
        for side,target in targets.items():
            ids = ARM_IDS[side]; anchor = q[ids].copy()
            rotation = Rotation.from_rotvec((rotations or {}).get(side,self.fk(q)[side]['rotation_vector']))
            lo = np.maximum(self.limits[ids,0]+.1,anchor-.8)
            hi = np.minimum(self.limits[ids,1]-.1,anchor+.8)
            def residual(values):
                candidate = q.copy(); candidate[ids] = values
                pose = self.fk(candidate)[side]
                return np.r_[np.asarray(pose['position_m'])-target,
                    .06*(rotation.inv()*Rotation.from_rotvec(pose['rotation_vector'])).as_rotvec(), .002*(values-anchor)]
            fit = least_squares(residual,np.clip(anchor,lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=120)
            error = float(np.linalg.norm(residual(fit.x)[:3]))
            if not fit.success or error > .005:
                raise ValueError(f'{side} IK unavailable within fixed bounds: {error:.4f} m')
            q[ids] = fit.x
            details[side] = dict(target_base_m=np.asarray(target).tolist(),nominal_error_m=error,
                                max_reference_change_rad=float(np.max(abs(fit.x-anchor))))
        return q,details


def window(reference, start, target, elapsed, frame):
    start,target = np.asarray(start),np.asarray(target)
    if start.shape != (29,) or target.shape != (29,) or not np.isfinite([start,target]).all():
        raise ValueError('Invalid reach reference')
    if np.max(abs(target[OTHER_IDS]-start[OTHER_IDS])) > 1e-8 or np.max(abs(target-start)) > .800001:
        raise ValueError('Reach changes non-arm references or exceeds joint displacement bound')
    u = np.clip((elapsed+np.arange(LOOKAHEAD)/RATE)/DURATION,0,1)
    s = u**3*(10+u*(-15+6*u)); ds=30*u*u*(1-u)**2/DURATION
    return dict(joint_pos=(start+s[:,None]*(target-start)).astype(np.float32),
        joint_vel=(reference['joint_vel'][0]+ds[:,None]*(target-start)).astype(np.float32),
        body_quat=np.repeat(reference['body_quat'][:1],LOOKAHEAD,axis=0).astype(np.float32),
        frame_index=np.arange(frame,frame+LOOKAHEAD,dtype=np.int64))


def check_site(path, baseline_id, measured, now):
    evidence = json.loads(Path(path).read_text())
    if (evidence.get('baseline_id') != baseline_id or evidence.get('phase') != 'clear_arms' or
        evidence.get('feet_bearing_weight') is not True or evidence.get('both_arms_clear') is not True or
        not 0 <= now-evidence['monotonic'] <= 120 or
        np.max(abs(np.asarray(evidence['q'])-measured['q'])) > .12):
        raise ValueError('Fresh operator observation of loaded feet and both arms clear is required')
    return evidence


class ReachFlow:
    def __init__(self, reference, measured, action, kin, now, delta_m=None, rotation_delta=None):
        self.reference=reference; self.kin=kin; self.action=action; self.at=now; self.stage_at=now
        self.start=np.asarray(reference['joint_pos'][0]).copy(); self.target=self.start.copy()
        self.q_before=np.asarray(measured['q']).copy(); self.before=kin.fk(self.q_before)
        nominal=kin.fk(self.start)
        self.goals = ({s:np.asarray(nominal[s]['position_m'])+[.08,.08 if s=='left' else -.08,.06] for s in ('left','right')}
                      if action=='clear_arms' else {'right':np.asarray(self.before['right']['position_m'])+[.10,0,.04]})
        rotations=None
        if action=='grasp_step':
            validate_step(delta_m,rotation_delta)
            goal=np.asarray(self.before['right']['position_m'])+delta_m
            if not (.04<=goal[0]<=.46 and -.38<=goal[1]<=.10 and -.10<=goal[2]<=.35):
                raise ValueError('Grasp step outside standing right-wrist workspace')
            self.goals={'right':goal}
            rotations={'right':(Rotation.from_rotvec(rotation_delta)*Rotation.from_rotvec(self.before['right']['rotation_vector'])).as_rotvec()}
        self.pool=ThreadPoolExecutor(max_workers=1)
        self.future=self.pool.submit(kin.solve,self.start.copy(),self.goals,rotations)
        self.stage='planning'; self.settled_since=None; self.done=False; self.result=None; self.latest=None
        self.peak=None; self.samples=[]; self.plan=None

    def fields(self, now, frame):
        return window(self.reference,self.start,self.target,max(0.,now-self.stage_at),frame)

    def fail(self, reason):
        self.done=True
        self.result=dict(completed=False,reason=reason,action=self.action,profile=PROFILE,
                         elapsed_s=time.monotonic()-self.at,measured=self.latest,plan=self.plan,
                         motion_started=self.stage!='planning',
                         planning_only_rejection=self.stage=='planning' and
                             (reason.startswith('planning_rejected:') or reason=='ik_planning_timeout'),
                         brainco_geometry_calibrated=False,physical_grasp_tested=False)

    def step(self, measured, row, now):
        poses=self.kin.fk(measured['q']); self.latest=poses
        self.samples.append(dict(monotonic=now,q=measured['q'],dq=measured['dq'],wrists=poses))
        if self.stage=='planning':
            if now-self.at > 5:
                self.fail('ik_planning_timeout'); return
            if not self.future.done(): return
            try:
                target,self.plan=self.future.result()
                window(self.reference,self.start,target,0.,0)
            except Exception as exc:
                self.fail('planning_rejected: '+str(exc)); return
            self.target=target; self.stage='clear' if self.action=='clear_arms' else 'step' if self.action=='grasp_step' else 'outward'
            self.stage_at=now; return
        obs=np.asarray(row['encoder_obs'])
        received=bool(np.max(abs(obs[4:294].reshape(10,29)-self.target))<1e-4 and
                      np.max(abs(obs[294:584].reshape(10,29)-self.reference['joint_vel'][0]))<1e-4)
        stable=received and max(abs(v) for v in measured['dq'])<.15 and now-self.stage_at>=DURATION
        self.settled_since=(self.settled_since or now) if stable else None
        if self.settled_since and now-self.settled_since>=1.:
            if self.stage=='outward':
                self.peak=poses
                self.start=self.target.copy(); self.target=self.reference['joint_pos'][0].copy()
                self.stage='return'; self.stage_at=now; self.settled_since=None; return
            before=self.before['right']; end=poses['right']
            if self.action in ('clear_arms','grasp_step'):
                errors={s:float(np.linalg.norm(np.asarray(poses[s]['position_m'])-self.goals[s])) for s in self.goals}
                success=max(errors.values())<=(.04 if self.action=='clear_arms' else .06)
                values=dict(target_errors_m=errors,site_clearance_verified=False,
                            awaits_site_observation=self.action=='clear_arms',holds_endpoint=True)
            else:
                pos=np.asarray(self.peak['right']['position_m']); origin=np.asarray(before['position_m'])
                goal_error=float(np.linalg.norm(pos-self.goals['right']))
                back_error=float(np.linalg.norm(np.asarray(end['position_m'])-origin))
                rot0=Rotation.from_rotvec(before['rotation_vector'])
                turn=float((rot0.inv()*Rotation.from_rotvec(self.peak['right']['rotation_vector'])).magnitude())
                back_turn=float((rot0.inv()*Rotation.from_rotvec(end['rotation_vector'])).magnitude())
                success=goal_error<=.03 and back_error<=.02 and pos[0]-origin[0]>=.07 and max(turn,back_turn)<=.20
                values=dict(reach_displacement_base_m=(pos-origin).tolist(),reach_goal_error_m=goal_error,
                    return_error_m=back_error,reach_orientation_change_rad=turn,return_orientation_error_rad=back_turn)
            self.done=True
            self.result=dict(completed=bool(success),reason='task_space_verified' if success else 'task_space_review_required',
                action=self.action,profile='grasp_step_v1' if self.action=='grasp_step' else PROFILE,
                elapsed_s=now-self.at,native_target_received=received,motion_started=True,
                before=self.before,after=poses,peak=self.peak,plan=self.plan,
                max_body_return_error_rad=float(np.max(abs(np.asarray(measured['q'])-self.q_before))),
                measurement='FK of actual native lowstate joints, not camera or calibrated BrainCo grasp frame',
                brainco_geometry_calibrated=False,physical_grasp_tested=False,**values)
        elif now-self.stage_at>TIMEOUT:
            self.fail('native_reference_or_measured_settling_timeout')

    def close(self):
        self.pool.shutdown(wait=False,cancel_futures=True)


def validate_step(delta_m, rotation_delta):
    for value,limit,label in [(delta_m,.10,'translation'),(rotation_delta,.35,'rotation')]:
        a=np.asarray(value,dtype=float)
        if a.shape!=(3,) or not np.isfinite(a).all() or np.linalg.norm(a)>limit+1e-9:
            raise ValueError('Invalid bounded grasp '+label)
    if np.linalg.norm(delta_m)<1e-5 and np.linalg.norm(rotation_delta)<1e-5:
        raise ValueError('Empty grasp step')
