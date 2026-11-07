"""Portable RTC imports and the deployed six-channel BrainCo model layout.

Importing this module creates no robot transport. The sole owner explicitly
creates a bridge when it starts; preflight uses its read-only variant.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

import numpy as np

from .brainco_bridge import BraincoPolicyBridge, feedback_hands
from .deployment import external_path
from .native_deployment import optional_path


def robot_model(waist_location='lower_and_upper_body'):
    """Use a complete local URDF asset tree, keeping upstream joint semantics."""
    from gear_sonic.data.robot_model.robot_model import RobotModel
    from gear_sonic.data.robot_model.supplemental_info.g1.g1_supplemental_info import (
        ElbowPose, G1SupplementalInfo, WaistLocation)
    root = optional_path('robot_model_assets', lambda: external_path('sonic_root')/
                         'gear_sonic/data/robot_model/model_data/g1')
    waist = dict(lower_body=WaistLocation.LOWER_BODY, upper_body=WaistLocation.UPPER_BODY,
                 lower_and_upper_body=WaistLocation.LOWER_AND_UPPER_BODY)[waist_location]
    info = G1SupplementalInfo(waist_location=waist, elbow_pose=ElbowPose.LOW)
    return RobotModel(str(root/'g1_29dof_with_hand.urdf'), str(root), supplemental_info=info)


class BraincoModelAdapter:
    def __init__(self, model_path, bridge):
        path = Path(model_path)
        self.stats = json.loads((path/'statistics.json').read_text())['unitree_g1_sonic']
        self.modalities = json.loads((path/'processor_config.json').read_text())[
            'processor_kwargs']['modality_configs']['unitree_g1_sonic']
        self.bridge = bridge
        self.horizon = len(self.modalities['action']['delta_indices'])
        for mode, keys in (('state', ('left_hand', 'right_hand')),
                           ('action', ('left_hand_joints', 'right_hand_joints'))):
            for key in keys:
                if key not in self.modalities[mode]['modality_keys'] or len(self.stats[mode][key]['mean']) != 6:
                    raise ValueError('This entry point requires BrainCo 6D '+mode+'.'+key)
        if (self.modalities['action']['modality_keys'] != ['motion_token','left_hand_joints','right_hand_joints']
                or len(self.stats['action']['motion_token']['mean']) != 64):
            raise ValueError('Expected 64D SONIC token and two 6D hands')

    def observation(self, observation):
        if observation is None:return None
        hands = feedback_hands(self.bridge.feedback())
        result = {mode:{key:observation[mode][key] for key in cfg['modality_keys']}
                  for mode,cfg in self.modalities.items() if mode != 'action'}
        for key,q in hands.items():result['state'][key] = q[None,None].copy()
        for key,value in result['state'].items():
            expected = len(self.stats['state'][key]['mean'])
            if value.shape != (1,1,expected) or not np.isfinite(value).all():
                raise ValueError('Invalid state '+key)
        return result

    def action(self, robot_model, action):
        result = {k.removeprefix('action.'):np.asarray(v,np.float32) for k,v in action.items()}
        for key in self.modalities['action']['modality_keys']:
            value = result[key]
            dim = len(self.stats['action'][key]['mean'])
            if value.shape != (1,self.horizon,dim) or not np.isfinite(value).all():
                raise ValueError('Invalid model output '+key)
            if key.endswith('hand_joints'):
                if np.any((value<0)|(value>1)):
                    warnings.warn('Clipping BrainCo targets to service range [0,1]',stacklevel=2)
                result[key] = np.clip(value,0,1)
        return result


@dataclass
class Config:
    rtc_request_interval: int = 30
    action_horizon: int = 40


def validate_server(client, adapter, config):
    remote = client.get_modality_config()
    for mode,expected in adapter.modalities.items():
        for field in ('modality_keys','delta_indices'):
            actual = remote[mode]
            value = actual[field] if isinstance(actual,dict) else getattr(actual,field)
            if list(value) != expected[field]:raise ValueError('Server/client layouts differ')
    rtc = client.get_rtc_config()
    delay = rtc.get('rtc_delay_steps')
    if type(delay) is not int or not 1 <= delay <= min(10,rtc['rtc_max_delay_steps']):
        raise ValueError('Invalid trained RTC delay')
    if rtc['action_horizon'] != config.action_horizon:
        raise ValueError('Server/client action horizons differ')
    if not delay <= config.rtc_request_interval < config.action_horizon-delay:
        raise ValueError('Invalid RTC request interval')
    return rtc


def dependencies(rtc_repo):
    sys.path[:0] = [str(rtc_repo), str(Path(rtc_repo)/'scripts/local'), str(external_path('sonic_root'))]
    from gr00t.policy.rtc_client import AsyncRTCController
    from gr00t.policy.server_client import PolicyClient
    from rtc_robot_session import RobotRTCSession
    from gear_sonic.scripts import run_vla_inference as runner
    from gear_sonic.utils.teleop.brainco_poses import load_poses
    return SimpleNamespace(PolicyClient=PolicyClient, AsyncRTCController=AsyncRTCController,
        RobotRTCSession=RobotRTCSession, validate_server=validate_server, Config=Config,
        robot_model=robot_model,
        legacy=SimpleNamespace(BraincoModelAdapter=BraincoModelAdapter, BraincoPolicyBridge=BraincoPolicyBridge,
                               feedback_hands=feedback_hands, runner=runner, load_poses=load_poses))
