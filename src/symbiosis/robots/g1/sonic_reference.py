"""Harness-owned SONIC/SMPL reference conversion, with no CoMotion imports.

The calibrated reference/residual method follows the CoMotion experiment, but
only the required FK, encoder layout and reference interpolation live here.
There is no simulator stepping, learned controller, Pico input or agent runner.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.special import expit

ROOT_ROTATION = Rotation.from_quat([.5, .5, .5, .5]).as_matrix()
REFERENCE_LINKS = {
    1:'left_hip_yaw_link',2:'right_hip_yaw_link',4:'left_knee_link',5:'right_knee_link',
    7:'left_ankle_roll_link',8:'right_ankle_roll_link',9:'torso_link',
    16:'left_shoulder_yaw_link',17:'right_shoulder_yaw_link',18:'left_elbow_link',
    19:'right_elbow_link',20:'left_wrist_yaw_link',21:'right_wrist_yaw_link'}


def finite(value,shape):
    result=np.asarray(value,dtype=np.float64)
    if result.shape!=shape or not np.isfinite(result).all():
        raise ValueError(f'Expected finite array {shape}')
    return result


def rotation_from_columns(values):
    finite(values,(10,6))
    columns=np.asarray(values).reshape(10,3,2)
    x=columns[:,:,0];norm=np.linalg.norm(x,axis=1,keepdims=True)
    if np.any(norm<1e-6):raise ValueError('Degenerate rotation reference')
    x=x/norm
    y=columns[:,:,1]-x*np.sum(x*columns[:,:,1],axis=1,keepdims=True)
    norm=np.linalg.norm(y,axis=1,keepdims=True)
    if np.any(norm<1e-6):raise ValueError('Degenerate rotation reference')
    y=y/norm
    return Rotation.from_matrix(np.stack([x,y,np.cross(x,y)],axis=-1))


def wrist_reference(poses):
    """The SONIC encoder's established IsaacLab wrist order/sign convention."""
    result=np.empty((len(poses),6))
    for side,elbow,wrist in ((0,17,19),(1,18,20)):
        q=Rotation.from_rotvec(poses[:,elbow]).as_quat()
        twist=np.zeros_like(q);twist[:,[1,3]]=q[:,[1,3]]
        norm=np.linalg.norm(twist,axis=1)
        if np.any(norm<1e-7):raise ValueError('Singular elbow swing/twist')
        swing=(Rotation.from_quat(twist/norm[:,None]).inv()*Rotation.from_quat(q)).as_euler('XYZ')
        local=Rotation.from_rotvec(poses[:,wrist]).as_euler('XYZ')
        sign=1 if side==0 else -1
        result[:,side]=sign*(swing[:,0]+local[:,0])
        result[:,side+2]=sign*local[:,1]
        result[:,side+4]=swing[:,2]+local[:,2]
    return result


class MatchedKinematics:
    def __init__(self,assets):
        root=Path(assets);manifest=json.loads((root/'manifest.json').read_text())
        if manifest['status']!='exact_encoder_and_control_weight_match':
            raise ValueError('Unmatched auxiliary weights')
        for name,expected in manifest['outputs'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
                raise ValueError('Auxiliary asset changed: '+name)
        with np.load(root/'g1_kin_weights.npz',allow_pickle=False) as data:
            self.weights={k:data[k].copy() for k in data}
        self.layers=sorted(int(k.split('.')[0]) for k in self.weights if k.endswith('.weight'))

    def __call__(self,tokens):
        x=np.asarray(tokens,np.float32)
        if x.ndim!=2 or x.shape[1]!=64 or not np.isfinite(x).all():
            raise ValueError('Expected finite token batch [N,64]')
        for layer in self.layers:
            x=x@self.weights[f'{layer}.weight'].T+self.weights[f'{layer}.bias']
            if layer!=self.layers[-1]:x=x*expit(x)
        decoded=x.reshape(-1,10,64)
        command=decoded[:,:,:58].reshape(-1,20,29)
        return dict(q=command[:,:10].copy(),dq=command[:,10:].copy(),orientation6d=decoded[:,:,58:].copy())


class HumanReference:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as data:
            self.J=data['J'].copy();self.parents=data['parents'].copy()
        finite(self.J,(55,3))
        if self.parents.shape!=(55,) or any(not 0<=p<i for i,p in enumerate(self.parents) if i):
            raise ValueError('Invalid reference skeleton')
        self.offset=self.J.copy();self.offset[1:]-=self.J[self.parents[1:]]

    def windows(self,poses):
        poses=np.asarray(poses,np.float64)
        if poses.ndim!=3 or poses.shape[1:]!=(21,3) or not np.isfinite(poses).all():
            raise ValueError('Expected local SMPL rotations [N,21,3]')
        count=len(poses)
        local=Rotation.from_rotvec(poses.reshape(-1,3)).as_matrix().reshape(count,21,3,3)
        world=np.empty((count,55,3,3));position=np.empty((count,55,3))
        world[:,0]=ROOT_ROTATION;position[:,0]=self.J[0]
        for joint in range(1,55):
            parent=int(self.parents[joint])
            position[:,joint]=position[:,parent]+world[:,parent]@self.offset[joint]
            world[:,joint]=world[:,parent]@local[:,joint-1] if joint<22 else world[:,parent]
        return position,world

    def fk(self,pose):
        position,world=self.windows(finite(pose,(21,3))[None])
        return position[0],world[0]

    def encoder_joints(self,poses):
        return self.windows(finite(poses,(10,21,3)))[0][:,list(range(22))+[39,54]]


class RobotReference:
    """Private FK model; resetting this reference model never touches a robot."""
    def __init__(self,scene,mapping):
        import mujoco
        self.mj=mujoco
        self.model=mujoco.MjModel.from_xml_path(str(scene));self.data=mujoco.MjData(self.model)
        joints=sorted(json.loads(Path(mapping).read_text())['joints'],key=lambda j:j['isaaclab_index'])
        self.qids=np.array([self.model.jnt_qposadr[self.model.joint(j['joint']).id] for j in joints])
        self.ids={j:self.model.body(name).id for j,name in REFERENCE_LINKS.items()}

    def rotations(self,q):
        self.data.qpos[:]=0;self.data.qpos[3]=1
        self.data.qpos[self.qids]=finite(q,(29,))
        self.mj.mj_kinematics(self.model,self.data)
        return {j:self.data.xmat[index].reshape(3,3).copy() for j,index in self.ids.items()}


class ReferenceEncoder:
    def __init__(self,controller_assets,kinematic_assets,scene,mapping):
        import onnxruntime as ort
        self.kin=MatchedKinematics(kinematic_assets)
        self.human=HumanReference(Path(kinematic_assets)/'human_skeleton.npz')
        self.robot=RobotReference(scene,mapping)
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        self.encoder=ort.InferenceSession(str(Path(controller_assets)/'model_encoder.onnx'),
            sess_options=options,providers=['CPUExecutionProvider'])

    def encode(self,poses,orientation):
        poses=finite(poses,(10,21,3));orientation=finite(orientation,(10,6))
        rotation_from_columns(orientation)
        obs=np.zeros((1,1762),np.float32);obs[0,0]=2
        obs[0,922:1642]=self.human.encoder_joints(poses).ravel()
        obs[0,1642:1702]=orientation.ravel();obs[0,1702:1762]=wrist_reference(poses).ravel()
        return self.encoder.run(None,{'obs_dict':obs})[0][0],obs


class CalibratedTokenReference:
    def __init__(self,bridge,fixture):
        self.bridge=bridge;self.seed=finite(fixture['smpl'][0],(21,3)).copy()
        _,human=bridge.human.fk(self.seed)
        q=bridge.kin(np.asarray(fixture['token'],np.float32)[None])['q'][0,0]
        robot=bridge.robot.rotations(q)
        self.offsets={j:matrix.T@human[j] for j,matrix in robot.items()}

    def reference(self,token):
        bridge=self.bridge;decoded=bridge.kin(finite(token,(64,))[None].astype(np.float32))
        source_time=np.arange(10)*.1;target_time=np.arange(10)*.02
        q=np.stack([np.interp(target_time,source_time,decoded['q'][0,:,j]) for j in range(29)],axis=1)
        orientation=Slerp(source_time,rotation_from_columns(decoded['orientation6d'][0]))(target_time)
        poses=[]
        for frame in q:
            robot=bridge.robot.rotations(frame)
            local=Rotation.from_rotvec(self.seed).as_matrix();world={0:ROOT_ROTATION}
            for joint in range(1,22):
                parent=world[int(bridge.human.parents[joint])]
                if joint in robot:
                    world[joint]=robot[joint]@self.offsets[joint];local[joint-1]=parent.T@world[joint]
                else:world[joint]=parent@local[joint-1]
            poses.append(Rotation.from_matrix(local).as_rotvec())
        return np.stack(poses),orientation.as_matrix()[:,:,:2].reshape(10,6)


class ResidualOverlay:
    def __init__(self,bridge,reference):
        self.bridge,self.reference=bridge,reference;self.cached=None

    def apply(self,token,clutch):
        token=np.asarray(finite(token,(64,)),np.float32)
        if not np.any(clutch.delta):return token.copy()
        if self.cached is None or not np.array_equal(token,self.cached):
            self.poses,self.orientation=self.reference.reference(token)
            self.zero,_=self.bridge.encode(self.poses,self.orientation);self.cached=token.copy()
        encoded,_=self.bridge.encode(clutch.compose(self.poses),self.orientation)
        result=token+(encoded-self.zero)
        finite(result,(64,))
        return result.astype(np.float32)
