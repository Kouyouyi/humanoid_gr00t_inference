"""Continuous external RGB capture/recording, with an independent preview stream.

Never imports a robot controller. Every received color frame is encoded with its
host-arrival clock; model/UI sampling does not determine which frames are saved.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import signal
import time
import uuid

from symbiosis.experience.execution_memory import atomic
from .external_camera import sdk_module,choose_serial


from .rgb_recording import RGBRecording


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify-on-exit',action='store_true',help='Full-decode a short commissioning recording before exit; normal service seals video immediately')
    parser.add_argument('--camera-profile',type=Path)
    parser.add_argument('--serial');parser.add_argument('--seconds',type=float,default=0)
    parser.add_argument('--stop-file',type=Path)
    args=parser.parse_args(argv);root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    from .camera_profile import load,fit
    profile=load(args.camera_profile)
    sdk=sdk_module();serial=choose_serial(sdk,args.serial)
    pipeline=sdk.pipeline();cfg=sdk.config();cfg.enable_device(serial)
    cfg.enable_stream(sdk.stream.color,profile['third_source']['width'],profile['third_source']['height'],sdk.format.rgb8,profile['frame']['fps'])
    running=[True]
    signal.signal(signal.SIGTERM,lambda *_:running.__setitem__(0,False))
    signal.signal(signal.SIGINT,lambda *_:running.__setitem__(0,False))
    recorder=None;started=False;began=time.monotonic();preview_at=0.;previews=[]
    from PIL import Image
    import numpy as np
    try:
        pipeline.start(cfg);started=True
        while running[0] and (not args.seconds or time.monotonic()-began<args.seconds):
            if args.stop_file and args.stop_file.exists():break
            frames=pipeline.poll_for_frames()
            if not frames:
                time.sleep(.005)
                continue
            frame=frames.get_color_frame()
            if not frame:continue
            received=time.monotonic();unix=time.time();native=np.asanyarray(frame.get_data()).copy()
            pixels=np.array(fit(Image.fromarray(native),profile))
            if recorder is None:recorder=RGBRecording(root/'recordings',pixels.shape[1],pixels.shape[0],profile=profile)
            recorder.append(pixels,received,unix,frame.get_timestamp())
            if received<preview_at:continue
            preview_at=received+1/profile['preview']['fps']
            ident=uuid.uuid4().hex;folder=root/ident;folder.mkdir()
            Image.fromarray(pixels).save(folder/'rgb.png')
            meta=dict(frame_id=ident,source='external_realsense',serial=serial,captured_unix_s=unix,
                received_monotonic=received,hardware_timestamp_ms=frame.get_timestamp(),width=pixels.shape[1],height=pixels.shape[0],
                timestamp_basis='host_receive_not_exposure',hardware_synchronized_with_head=False,native_width=native.shape[1],native_height=native.shape[0],normalization='shared_profile_letterbox',profile=profile)
            atomic(folder/'meta.json',meta);atomic(root/'latest.json',meta)
            atomic(root/'ready.json',dict(ready=True,serial=serial,recording=str(root/'recordings/third-person.mp4')))
            previews.append(folder)
            while len(previews)>profile['preview']['keep_frames']:
                import shutil
                shutil.rmtree(previews.pop(0),ignore_errors=True)
    except BaseException as exc:
        atomic(root/'error.json',dict(error=f'{type(exc).__name__}: {exc}'))
        raise
    finally:
        if started:pipeline.stop()
        if recorder:recorder.close(verify=args.verify_on_exit)
        atomic(root/'status.json',dict(running=False,serial=serial,frames=recorder.count if recorder else 0))


if __name__=='__main__':main()
