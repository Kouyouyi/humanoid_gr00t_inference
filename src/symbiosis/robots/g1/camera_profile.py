"""Shared RGB output/encoding profile, independent of camera device and control."""
import json
from pathlib import Path
from PIL import Image,ImageOps


def load(path=None):
    if path is None:
        from symbiosis.common.project import project_root
        path=project_root()/'configs/camera_profiles/real_dual.json'
    value=json.loads(Path(path).read_text())
    frame=value['frame'];preview=value['preview'];encoding=value['encoding']
    for name in ('width','height'):
        if type(frame[name]) is not int or frame[name]<2 or frame[name]%2:raise ValueError('Camera frame dimensions must be positive even integers')
    if not isinstance(frame['fps'],int) or frame['fps']<=0:raise ValueError('Camera fps must be positive')
    if not 0<preview['fps']<=frame['fps'] or preview['keep_frames']<2:raise ValueError('Invalid camera preview profile')
    if encoding['codec']!='libx264' or encoding['pixel_format']!='yuv420p' or not 0<=encoding['crf']<=51:raise ValueError('Unsupported recording encoding profile')
    # Keep the established stereo calibration at its authored native resolution.
    if value['head_source']['stereo_width']!=2560 or value['head_source']['height']!=720:raise ValueError('HEAD native stereo resolution is bound to its calibration')
    return value


def fit(image,profile):
    return ImageOps.pad(image.convert('RGB'),(profile['frame']['width'],profile['frame']['height']),color='black')
