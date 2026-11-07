"""Simulation candidate from a native SONIC walking reference, never motor targets.

Only leg references and root tilt vary. Cleared arms and waist are retained.
Root translation from the source is NOT a robot position command. Amplitude does
not establish physical step length; execution must measure movement and stopping.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .state import NAMES
from .reference import LOOKAHEAD, RATE
from .reach import Kinematics
from .deployment import external_path

SOURCE=external_path('sonic_root')/'gear_sonic_deploy/reference/example/walking_quip_360_R_002__A428'
if Path(__file__).with_name('gait_data').is_dir():
    SOURCE=Path(__file__).with_name('gait_data')
LEGS=np.array([i for i,n in enumerate(NAMES) if any(s in n for s in ('hip','knee','ankle'))])


def make(reference, amplitude=.5, start=50, end=120, mirrored_pair=False):
    if not .1<=amplitude<=1 or not 0<=start<end<=455 or end-start<25:
        raise ValueError('Invalid simulation gait segment')
    source=np.loadtxt(SOURCE/'joint_pos.csv',delimiter=',',skiprows=1)[start:end]
    quat=np.loadtxt(SOURCE/'body_quat.csv',delimiter=',',skiprows=1)[start:end,:4]
    anchor=reference['joint_pos'][0].copy();velocity=reference['joint_vel'][0].copy()
    q=np.tile(anchor,(len(source),1));q[:,LEGS]+=amplitude*(source[:,LEGS]-source[0,LEGS])
    rotations=Rotation.from_quat(quat[:,[1,2,3,0]])
    relative=rotations[0].inv()*rotations
    # Retain observed tilt shape; source heading is not a commanded turn.
    tilt=relative.as_rotvec();tilt[:,2]=0
    root0=Rotation.from_quat(reference['body_quat'][0][[1,2,3,0]])
    turns=root0*Rotation.from_rotvec(amplitude*tilt)
    u=np.arange(1,101)/100.;s=u**3*(10-15*u+6*u*u)
    exit_q=q[-1]+s[:,None]*(anchor-q[-1])
    exit_rot=Slerp([0,1],Rotation.from_quat(np.array([turns[-1].as_quat(),root0.as_quat()])))(s)
    all_q=np.concatenate([np.tile(anchor,(50,1)),q,exit_q,np.tile(anchor,(61,1))])
    all_rot=Rotation.from_quat(np.concatenate([np.tile(root0.as_quat(),(50,1)),turns.as_quat(),
                                              exit_rot.as_quat(),np.tile(root0.as_quat(),(61,1))]))
    dq=np.tile(velocity,(len(all_q),1));dq[:,LEGS]=np.gradient(all_q[:,LEGS],1/RATE,axis=0)
    # Blend the native idle velocity convention at entry/exit, avoiding a jump.
    dq[:50,LEGS]+=(1-np.arange(50)[:,None]/50)*velocity[LEGS]
    exit_start=50+len(q)
    dq[exit_start:exit_start+100,LEGS]+=s[:,None]*velocity[LEGS]
    dq[-61:]=velocity
    duration=(50+len(q)+100)/RATE
    if mirrored_pair:
        count=len(all_q)-61
        mirrored_q=np.tile(anchor,(count,1));mirrored_dq=np.tile(velocity,(count,1))
        for side,other in [('left','right'),('right','left')]:
            for part in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll'):
                dest=NAMES.index(side+'_'+part+'_joint');src=NAMES.index(other+'_'+part+'_joint')
                sign=-1 if part in ('hip_roll','hip_yaw','ankle_roll') else 1
                mirrored_q[:,dest]+=sign*(all_q[:count,src]-anchor[src])
                mirrored_dq[:,dest]+=sign*(dq[:count,src]-velocity[src])
        rv=(root0.inv()*all_rot[:count]).as_rotvec();rv[:,[0,2]]*=-1
        mirrored_rot=root0*Rotation.from_rotvec(rv)
        all_q=np.concatenate([all_q[:count],mirrored_q,all_q[-61:]])
        dq=np.concatenate([dq[:count],mirrored_dq,dq[-61:]])
        all_rot=Rotation.from_quat(np.concatenate([all_rot[:count].as_quat(),mirrored_rot.as_quat(),all_rot[-61:].as_quat()]))
        duration*=2
    bounds=Kinematics().limits
    if np.any(all_q<bounds[:,0]+.03) or np.any(all_q>bounds[:,1]-.03):raise ValueError('Gait exceeds joint limits')
    if np.max(abs(dq))>6:raise ValueError('Simulation gait velocity exceeds 6 rad/s')
    return dict(joint_pos=all_q.astype(np.float32),joint_vel=dq.astype(np.float32),
                body_quat=all_rot.as_quat()[:,[3,0,1,2]].astype(np.float32),
                duration_s=duration,amplitude=amplitude,source_start=start,source_end=end,mirrored_pair=mirrored_pair)


def fields(profile,elapsed,frame):
    index=np.minimum(np.arange(LOOKAHEAD)+max(0,int(elapsed*RATE)),len(profile['joint_pos'])-1)
    return {k:profile[k][index] for k in ('joint_pos','joint_vel','body_quat')} | {
        'frame_index':np.arange(frame,frame+LOOKAHEAD,dtype=np.int64)}
