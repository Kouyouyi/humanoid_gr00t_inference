"""One finite leading-step reference, then measured settling. No position command.

E054 validates this leading step; E049 validates the longer mirrored pair. Physical distance and
foot contacts require external observation; LowState alone cannot verify them.
"""
import json
import numpy as np

from .gait import make, fields, LEGS
from .state import NAMES

PROFILE='reference_leading_step_v1'
NONLEGS=np.array([i for i in range(29) if i not in LEGS])


def check_site(path, baseline_id, measured, now):
    site=json.loads(path.read_text())
    required=('feet_weight_bearing','harness_allows_translation','operator_at_stop',
              'arms_clear','forward_corridor_checked')
    if site.get('baseline_id')!=baseline_id or not 0<=now-site['monotonic']<=120:
        raise ValueError('Walking site evidence is stale or belongs to another run')
    if not all(site.get(k) is True for k in required):
        raise ValueError('Walking site conditions are not confirmed')
    if not .20<=float(site['minimum_forward_clearance_m']):
        raise ValueError('First leading step requires at least 20 cm assessed forward clearance')
    q=np.asarray(site['q'],dtype=float)
    if q.shape!=(29,) or not np.isfinite(q).all() or np.max(abs(q-measured['q']))>.12:
        raise ValueError('Robot moved after walking clearance check')


class GaitFlow:
    def __init__(self,reference,measured,now):
        self.profile=make(reference,amplitude=.5,mirrored_pair=False)
        self.reference=reference
        self.at=now;self.stage='leading_step';self.done=False;self.result=None
        self.target=reference['joint_pos'][0].copy()
        self.q_before=np.asarray(measured['q']).copy()
        self.settled_since=None;self.samples=[];self.latest={}
        self.peak_seen={side:False for side in ('left',)}
        self.low=np.min(self.profile['joint_pos'],axis=0)-.25
        self.high=np.max(self.profile['joint_pos'],axis=0)+.25

    def guard(self,measured):
        q=np.asarray(measured['q']);dq=np.asarray(measured['dq'])
        if max(abs(dq[LEGS]))>6. or max(abs(dq[NONLEGS]))>1.5:
            raise ValueError('gait_joint_speed_limit')
        if np.any(q[LEGS]<self.low[LEGS]) or np.any(q[LEGS]>self.high[LEGS]):
            raise ValueError('gait_leg_configuration_outside_profile_envelope')
        waist=[NAMES.index('waist_'+name+'_joint') for name in ('yaw','roll','pitch')]
        if max(abs(q[waist]-self.q_before[waist]))>.20:
            raise ValueError('gait_waist_departed_from_anchor')

    def step(self,measured,row,now):
        if self.done:return
        elapsed=now-self.at
        obs=np.asarray(row['encoder_obs'])
        native_q=obs[4:294].reshape(10,29);native_dq=obs[294:584].reshape(10,29)
        for side in self.peak_seen:
            i=NAMES.index(side+'_knee_joint')
            peak=float(max(self.profile['joint_pos'][:,i]))
            self.peak_seen[side] |= bool(native_q[0,i]>=peak-.015)
        self.samples.append(dict(elapsed_s=elapsed,q=measured['q'],dq=measured['dq'],tilt_rad=measured['tilt_rad']))
        terminal=(max(abs(native_q-self.reference['joint_pos'][0]).ravel())<1e-4 and
                  max(abs(native_dq-self.reference['joint_vel'][0]).ravel())<1e-4)
        if elapsed>=self.profile['duration_s']:
            self.stage='settling'
            settled=(terminal and all(self.peak_seen.values()) and max(abs(np.asarray(measured['dq'])))<.3
                     and np.linalg.norm(measured['base_ang_vel'])<.2)
            self.settled_since=(self.settled_since if self.settled_since is not None else now) if settled else None
            if self.settled_since is not None and now-self.settled_since>=1.:
                self._finish(True,'leading_reference_executed_and_measured_settled')
            elif elapsed>=self.profile['duration_s']+10:
                self.fail('gait_reference_or_settling_not_verified')

    def fields(self,now,frame):
        result=fields(self.profile,now-self.at,frame)
        self.target=result['joint_pos'][0].copy()
        return result

    def _finish(self,success,reason):
        self.done=True;self.stage='completed' if success else 'review_required'
        self.result=dict(profile=PROFILE,completed=success,reason=reason,
            native_knee_peaks_observed=self.peak_seen.copy(),measured_samples=len(self.samples),
            max_measured_joint_speed_rad_s=max((max(map(abs,s['dq'])) for s in self.samples),default=None),
            max_tilt_rad=max((s['tilt_rad'] for s in self.samples),default=None),
            q_before=self.q_before.tolist(),q_after=self.samples[-1]['q'] if self.samples else None,
            actual_distance_m=None,physical_footsteps_verified=False,requires_rgbd_and_site_observation=True,
            physical_grasp_tested=False,automatic_repeat_allowed=False)

    def fail(self,reason):
        if not self.done:self._finish(False,reason)

    def close(self):
        pass
