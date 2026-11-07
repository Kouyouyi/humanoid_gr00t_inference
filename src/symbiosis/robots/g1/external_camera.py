"""Pair public RGB images; depth/intrinsics remain HEAD-only. No robot SDK."""
import json
from pathlib import Path
import re
import shutil
from PIL import Image,ImageOps


def sdk_module():
    try:import pyrealsense2 as sdk
    except ModuleNotFoundError:
        import sys
        from symbiosis.common.deployment import external_path
        extra=Path(external_path('artifact_root'))/'cache/teacher-camera-python'
        if extra.is_dir():sys.path.insert(0,str(extra))
        import pyrealsense2 as sdk
    return sdk


def choose_serial(sdk,serial=None):
    serials=[d.get_info(sdk.camera_info.serial_number) for d in sdk.context().query_devices()]
    if serial is None:
        if len(serials)!=1:raise ValueError('Select --third-camera-serial; external RealSense devices: '+str(serials))
        return serials[0]
    if serial not in serials:raise ValueError('External RealSense not connected: '+serial)
    return serial


def latest(directory):
    root=Path(directory).resolve()
    meta=json.loads((root/'latest.json').read_text())
    ident=meta.get('frame_id','')
    if not isinstance(ident,str) or not re.fullmatch('[a-f0-9]{32}',ident):raise ValueError('Invalid external camera frame identity')
    source=(root/ident/'rgb.png').resolve()
    if not source.is_relative_to(root):raise ValueError('Invalid external camera path')
    return meta,source


def pair_observation(observation,destination,third_directory,*,profile_path=None):
    destination=Path(destination)
    from .camera_profile import load,fit
    profile=load(profile_path);width=profile['frame']['width'];height=profile['frame']['height']
    # Retry only a publication/pruning race; image age is not a rejection rule.
    for attempt in range(3):
        try:
            third,source=latest(third_directory)
            shutil.copyfile(source,destination/'third-rgb.png')
            break
        except FileNotFoundError:
            if attempt==2:raise RuntimeError('External third camera has no available RGB frame') from None
    (destination/'third-meta.json').write_text(json.dumps(third,indent=2))
    with Image.open(destination/'rgb.png') as head,Image.open(destination/'third-rgb.png') as side:
        if head.size!=(640,360):raise ValueError('Expected HEAD RGB 640x360')
        # Preserve all of the external view; never stretch its aspect ratio.
        head_scale=min(width/head.width,height/head.height)
        head_transform=dict(scale=head_scale,pad_x=(width-head.width*head_scale)/2,pad_y=(height-head.height*head_scale)/2,source_resolution=[head.width,head.height])
        head=fit(head,profile);side=fit(side,profile)
        combined=Image.new('RGB',(width*2,height));combined.paste(head,(0,0));combined.paste(side,(width,0))
        model_path=destination/'control-views.png';combined.save(model_path)
    stamp=observation.get('captured_unix_s',observation.get('published_unix'))
    third_stamp=third.get('captured_unix_s')
    result=dict(observation,experimental_control_views='first_and_third',third_person_is_control_input=True,
        control_image_layout=dict(split_x=width,width=width*2,height=height,head='left',third='right',depth_and_K='HEAD_only',head_depth_resolution=[640,360],head_source_to_model=head_transform),
        camera_output_profile=profile,
        external_third_view=dict(source='external_realsense',frame_id=third['frame_id'],captured_unix_s=third_stamp,
            head_third_time_delta_s=(third_stamp-stamp if isinstance(stamp,(int,float)) and isinstance(third_stamp,(int,float)) else None),
            synchronization='independent_sensor_clocks_not_hardware_synchronized',aspect_ratio='letterboxed'))
    return result,model_path
