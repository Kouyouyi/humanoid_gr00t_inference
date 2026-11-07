"""Public-sensor CoMotion-style keyframes; sole owner applies compiled tokens."""
from pathlib import Path
import numpy as np
from .deployment import external_path
from .smpl_program import ProgramCompiler, JOINTS

OWNED=('left_collar','right_collar','left_shoulder','right_shoulder',
       'left_elbow','right_elbow','left_wrist','right_wrist')


def validate(program):
    if not isinstance(program,dict) or set(program)!={'n','keyframes'} or program['n']!=150:
        raise ValueError('CoMotion requires a 150-step program')
    keys=program['keyframes']
    if not isinstance(keys,list) or not 3<=len(keys)<=5:
        raise ValueError('Use 3..5 keyframes covering steps 0..149')
    times=[]
    for row in keys:
        if not isinstance(row,dict) or set(row)-{'step','joint_deltas'}:
            raise ValueError('Only step and joint_deltas; hands retain current targets')
        tick=row.get('step')
        if type(tick) is not int or not 0<=tick<150 or times and tick<=times[-1]:
            raise ValueError('Keyframes must advance within 0..149')
        times.append(tick)
        deltas=row.get('joint_deltas',{})
        if not isinstance(deltas,dict) or set(deltas)-set(OWNED):raise ValueError('Arm joints only')
        for delta in deltas.values():
            value=np.asarray(delta,float)
            if value.shape!=(3,) or not np.isfinite(value).all() or np.linalg.norm(value)>.5:
                raise ValueError('Finite local rotation delta <=0.5 rad required')
            if tick==0 and np.any(value):raise ValueError('First keyframe must preserve actual posture')
    if times[0]!=0 or times[-1]!=149:raise ValueError('Cover steps 0..149')


def reference():
    return dict(method='comotion_local_smpl_keyframes',n=150,rate_hz=50,
        owned_joints=list(OWNED),delta_rad=.5,first_keyframe='step 0: zero delta',
        interpolation='quintic smoothstep',hands='preserve current targets')


class Executor:
    def __init__(self):
        from .retry_comotion import SonicResidualConverter
        from .state import JOINTS as BODY
        self.converter=SonicResidualConverter(controller_assets=external_path('smpl_controller_assets'),
            kinematic_assets=external_path('smpl_kinematic_assets'),robot_scene=external_path('smpl_robot_model'))
        self.compiler=ProgramCompiler(self.converter,dict(owned_joints=list(OWNED),delta_rad=.5,
            angular_speed_rad_s=1.5,angular_acceleration_rad_s2=15.,
            hand_min=np.zeros((2,6)),hand_max=np.ones((2,6)),hand_speed_s=2.,
            body_min=[j['lower'] for j in BODY],body_max=[j['upper'] for j in BODY],
            body_speed_rad_s=np.full(29,15.)),dict(max_steps=150,max_segment_s=3.))

    def compile(self,anchor,program):
        validate(program)
        return self.compiler.compile(anchor,program)
