"""Harness-native live-token SMPL residual conversion, without actuation.

Execution must separately bind the current physical ACK, hand targets and live
decoder history. This converter never creates a controller-ready certificate.
"""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np


def digest(path):
    hasher=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):hasher.update(block)
    return hasher.hexdigest()


def verify_deployment_assets(options,sonic_root):
    """Bind residual encoding to the release used by the field launcher."""
    selected=Path(sonic_root).resolve()/'gear_sonic_deploy/policy/release'
    assets=Path(options['controller_assets']).resolve()
    binding={}
    for name in ('model_encoder.onnx','model_decoder.onnx'):
        actual=digest(selected/name)
        if actual!=digest(assets/name):
            raise ValueError('Retry assets differ from the selected field controller: '+name)
        binding[name]=actual
    return {'controller_release':str(selected),'model_files':binding}


def scene_files(scene):
    """Fingerprint the included model and its mesh assets, not just the wrapper."""
    scene=Path(scene).resolve();root=scene.parent;seen={}
    def visit(path):
        path=path.resolve()
        if str(path) in seen:return
        seen[str(path)]=digest(path)
        tree=ET.parse(path).getroot()
        compiler=tree.find('compiler')
        meshdir=compiler.get('meshdir','') if compiler is not None else ''
        for item in tree.iter('include'):visit(path.parent/item.attrib['file'])
        for item in tree.iter('mesh'):
            if item.get('file'):
                mesh=(path.parent/meshdir/item.attrib['file']).resolve()
                seen[str(mesh)]=digest(mesh)
    visit(scene)
    return {str(Path(name).relative_to(root)):value for name,value in sorted(seen.items())}


def action(value):
    value=np.asarray(value,dtype=np.float32)
    if value.shape!=(66,) or not np.isfinite(value).all() or not np.isin(value[64:],[0.,1.]).all():
        raise ValueError('actual_finite_latent64_plus_applied_binary_hands_required')
    return value


class SmplDeltaClutch:
    """Postmultiply local SMPL rotations; unedited joints are byte-preserved."""
    def __init__(self,delta):
        delta=np.asarray(delta,dtype=np.float64)
        if delta.shape==(21,3):delta=np.repeat(delta[None],10,axis=0)
        if delta.shape!=(10,21,3) or not np.isfinite(delta).all():
            raise ValueError('finite_21_local_SMPL_rotations_or_10_frame_window_required')
        self.delta=delta.copy()

    def compose(self,poses):
        from scipy.spatial.transform import Rotation
        result=np.asarray(poses,dtype=np.float64).copy()
        if result.shape!=(10,21,3):raise ValueError('ten_frame_reference_required')
        edited=np.any(self.delta!=0,axis=-1)
        result[edited]=(Rotation.from_rotvec(result[edited])*Rotation.from_rotvec(self.delta[edited])).as_rotvec()
        return result


class SonicResidualConverter:
    def __init__(self,*,controller_assets,kinematic_assets,robot_scene,mapping=None,fixture=None,comotion_root=None):
        # comotion_root is accepted only for old session compatibility; it is
        # never imported, opened or required. New sessions omit it entirely.
        local_assets=Path(__file__).parent/'assets'
        mapping=Path(mapping) if mapping else local_assets/'g1_reference_limits.json'
        fixture=Path(fixture) if fixture else local_assets/'smpl_standing_physical_v02.npz'
        controller_assets=Path(controller_assets);kinematic_assets=Path(kinematic_assets)
        auxiliary=json.loads((kinematic_assets/'manifest.json').read_text())
        if auxiliary.get('status')!='exact_encoder_and_control_weight_match':
            raise ValueError('matched_auxiliary_weights_required')
        for name in ('model_encoder.onnx','model_decoder.onnx'):
            if digest(controller_assets/name)!=auxiliary.get('controller_files',{}).get(name):
                raise ValueError('encoder_decoder_changed_since_auxiliary_extraction')
        from . import sonic_reference
        from . import reference_handoff as sonic_handoff
        self._handoff_class=sonic_handoff.OwnershipHandoff
        self.bridge=sonic_reference.ReferenceEncoder(controller_assets,kinematic_assets,robot_scene,mapping)
        with np.load(fixture,allow_pickle=False) as data:
            reference={k:data[k].copy() for k in ('token','smpl')}
        self.reference=sonic_reference.CalibratedTokenReference(self.bridge,reference)
        self.overlay=sonic_reference.ResidualOverlay(self.bridge,self.reference)
        self.fingerprint=dict(schema='harness_sonic_residual/v2',external_comotion_required=False,
            model_files={name:digest(controller_assets/name) for name in ('model_encoder.onnx','model_decoder.onnx')},
            auxiliary_manifest_sha256=digest(kinematic_assets/'manifest.json'),
            robot_scene_sha256=digest(robot_scene),mapping_sha256=digest(mapping),fixture_sha256=digest(fixture),
            robot_scene_files=scene_files(robot_scene),
            source_files={p.name:digest(p) for p in
                (Path(__file__),Path(sonic_reference.__file__),Path(sonic_handoff.__file__))},
            reference='approximate_matched_G1_token_to_SMPL; not a lossless inverse',
            hand_mapping='body residual only; native hand targets are independent; no BrainCo/Dex3 conversion',
            controller_ready=False,physical_grasp_validated=False)

    def new_handoff(self, initial, *, now=0., duration=.2):
        # Harness owns the transition; moving targets do not restart it.
        return self._handoff_class(initial, now, duration=duration)

    def convert(self,actual_action66,delta):
        original=action(actual_action66);clutch=SmplDeltaClutch(delta)
        corrected=original.copy()
        corrected[:64]=self.overlay.apply(original[:64],clutch)
        if not np.isfinite(corrected).all():raise ValueError('nonfinite_corrected_token')
        assert np.array_equal(corrected[64:],original[64:])
        if not np.any(clutch.delta):assert np.array_equal(corrected,original)
        return dict(action66=corrected,
            metadata=dict(zero_delta_identity=bool(not np.any(clutch.delta)),hand_binary_preserved=True,
                token_change_detected=bool(not np.array_equal(corrected[:64],original[:64])),
                nonzero_delta_quantized_to_identity=bool(np.any(clutch.delta)
                    and np.array_equal(corrected[:64],original[:64])),
                requested_smpl_delta_norm=float(np.linalg.norm(clutch.delta)),
                body_residual_norm=float(np.linalg.norm(corrected[:64]-original[:64])),
                source_actual_action66_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
                corrected_action66_sha256=hashlib.sha256(corrected.tobytes()).hexdigest(),
                decoder_history_used=False,controller_ready=False,authorizes_motion=False))

    def reference_pose(self,actual_action66):
        poses,orientation=self.reference.reference(action(actual_action66)[:64])
        return dict(local_smpl_axis_angle=poses,body_orientation6d=orientation,
                    approximate_reference=True,authorizes_motion=False)


# Import compatibility for existing scripts; implementation is entirely local.
CoMotionResidualConverter=SonicResidualConverter
