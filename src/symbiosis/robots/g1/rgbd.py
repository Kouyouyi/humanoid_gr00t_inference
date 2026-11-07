"""ZED Mini rev2 factory fisheye rectification and host-side stereo depth.

No camera SDK or robot control is imported. Input is one raw 2560x720 SBS image.
"""
import configparser
import hashlib
from pathlib import Path

import cv2
import numpy as np


CALIBRATION = Path(__file__).parent/'calibration/SN12234465.conf'


class StereoDepth:
    def __init__(self, calibration=CALIBRATION):
        path = Path(calibration)
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        c = configparser.ConfigParser()
        c.read(path)

        def camera(side):
            s = c[side+'_CAM_HD']
            k = np.array([[float(s['fx']), 0, float(s['cx'])],
                          [0, float(s['fy']), float(s['cy'])], [0, 0, 1.]])
            return k, np.array([float(s[f'k{i}']) for i in range(1, 5)])

        kl, dl = camera('LEFT')
        kr, dr = camera('RIGHT')
        s = c['STEREO']
        rotation = cv2.Rodrigues(np.array([float(s[k]) for k in ('RX_HD', 'CV_HD', 'RZ_HD')]))[0]
        translation = np.array([-float(s['Baseline']), float(s['TY']), float(s['TZ'])])/1000
        r1, r2, p1, p2, _ = cv2.fisheye.stereoRectify(
            kl, dl, kr, dr, (1280, 720), rotation, translation,
            flags=cv2.CALIB_ZERO_DISPARITY, balance=0)
        self.maps = [cv2.fisheye.initUndistortRectifyMap(k, d, r, p, (1280, 720), cv2.CV_32FC1)
                     for k, d, r, p in ((kl, dl, r1, p1), (kr, dr, r2, p2))]
        self.K = p1[:3, :3].copy()
        self.K[:2] /= 2
        self.baseline_m = abs(float(p2[0, 3]/p2[0, 0]))
        options = dict(numDisparities=128, blockSize=5, P1=200, P2=800,
                       disp12MaxDiff=1, preFilterCap=31, uniquenessRatio=12,
                       speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.left = cv2.StereoSGBM_create(minDisparity=0, **options)
        self.right = cv2.StereoSGBM_create(minDisparity=-128, **options)

    def rectified_left(self, stereo):
        if stereo.shape != (720,2560,3) or stereo.dtype != np.uint8:
            raise ValueError('Calibration requires uint8 BGR raw SBS 2560x720')
        return cv2.resize(cv2.remap(stereo[:,:1280],*self.maps[0],cv2.INTER_LINEAR),(640,360),interpolation=cv2.INTER_AREA)

    def compute(self, stereo):
        if stereo.shape != (720, 2560, 3) or stereo.dtype != np.uint8:
            raise ValueError('Calibration requires uint8 BGR raw SBS 2560x720')
        images = [cv2.resize(cv2.remap(im, *maps, cv2.INTER_LINEAR), (640, 360),
                             interpolation=cv2.INTER_AREA)
                  for im, maps in zip(np.split(stereo, 2, axis=1), self.maps)]
        l, r = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images]
        dl = self.left.compute(l, r).astype(np.float32)/16
        dr = self.right.compute(r, l).astype(np.float32)/16
        ys, xs = np.indices(dl.shape)
        xr = np.rint(xs-dl).astype(int)
        rd = dr[ys, np.clip(xr, 0, 639)]
        valid = (dl > 0) & (rd > -129) & (xr >= 0) & (xr < 640) & (np.abs(dl+rd) <= 1.)
        depth = np.zeros(dl.shape, np.float32)
        depth[valid] = self.K[0, 0]*self.baseline_m/dl[valid]
        valid &= (depth >= .25) & (depth <= 5.)
        depth[~valid] = 0
        return images[0], np.rint(depth*1000).astype(np.uint16), valid.astype(np.uint8)*255

    def metadata(self):
        return dict(camera='zed_mini', serial=None, calibration_serial='12234465',
                    camera_serial_verified=False, width=640, height=360,
                    K=self.K.tolist(), depth_units='uint16_mm_zero_invalid', baseline_m=self.baseline_m,
                    calibration_sha256=self.sha256, lens_model='fisheye',
                    optical_frame='rectified_left_x_right_y_down_z_forward',
                    T_base_cam=None, metric_accuracy_validated=False,
                    camera_to_robot_extrinsics_validated=False)
