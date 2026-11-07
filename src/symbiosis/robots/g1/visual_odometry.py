"""Two-view RGB-D camera displacement. No world coordinates or model calls."""
import numpy as np


def rigid_fit(before,after):
    ca,cb=before.mean(axis=0),after.mean(axis=0)
    u,_,vt=np.linalg.svd((before-ca).T@(after-cb))
    rotation=vt.T@u.T
    if np.linalg.det(rotation)<0:
        vt[-1]*=-1;rotation=vt.T@u.T
    return rotation,cb-rotation@ca


def robust_displacement(before,after):
    before,after=np.asarray(before,float),np.asarray(after,float)
    if before.shape!=after.shape or before.ndim!=2 or before.shape[1]!=3 or len(before)<16:
        raise ValueError('insufficient_3d_correspondences')
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError('nonfinite_correspondences')
    best=np.zeros(len(before),bool);rng=np.random.default_rng(0)
    for _ in range(160):
        indices=rng.choice(len(before),3,replace=False)
        if np.linalg.norm(np.cross(before[indices[1]]-before[indices[0]],before[indices[2]]-before[indices[0]]))<1e-5:continue
        rotation,translation=rigid_fit(before[indices],after[indices])
        inliers=np.linalg.norm(before@rotation.T+translation-after,axis=1)<.04
        if inliers.sum()>best.sum():best=inliers
    if best.sum()<max(16,.4*len(best)):raise ValueError('insufficient_static_scene_inliers')
    rotation,translation=rigid_fit(before[best],after[best])
    error=float(np.sqrt(np.mean(np.sum((before[best]@rotation.T+translation-after[best])**2,axis=1))))
    if error>.03:raise ValueError('depth_registration_error_too_large')
    displacement=-rotation.T@translation
    return dict(available=True,source='first_person_rgbd',distance_m=float(np.linalg.norm(displacement)),
        displacement_m=displacement.tolist(),coordinate_frame='initial_camera_optical',
        measurement='camera_endpoint_displacement',path_length_available=False,
        inliers=int(best.sum()),correspondences=len(best),registration_rmse_m=error,
        note='第一视角相机净位移估计，受头部摆动、深度误差和动态物体影响；不是 IMU 或底盘里程计。')


def estimate_frames(directory,before_id,after_id):
    from pathlib import Path
    import json
    import re
    failure=dict(available=False,source='first_person_rgbd',distance_m=None)
    try:
        import cv2
        root=Path(directory).resolve()
        points=[];keys=[];descriptors=[]
        orb=cv2.ORB_create(nfeatures=2200)
        for ident in (before_id,after_id):
            if not isinstance(ident,str) or not re.fullmatch('[a-f0-9]{32}',ident):raise ValueError('frame_id_unavailable')
            folder=(root/ident).resolve()
            if not folder.is_relative_to(root):raise ValueError('evidence_path_invalid')
            meta=json.loads((folder/'observation.json').read_text())
            image=cv2.imread(str(folder/'rgb.png'),cv2.IMREAD_GRAYSCALE)
            depth=cv2.imread(str(folder/'depth.png'),cv2.IMREAD_UNCHANGED)
            if image is None or depth is None or depth.shape!=image.shape:raise ValueError('rgbd_unavailable')
            valid=cv2.imread(str(folder/'valid.png'),cv2.IMREAD_GRAYSCALE)
            mask=((depth>0)&(depth*meta.get('depth_scale_m',.001)<8))
            if valid is not None:mask&=valid>0
            key,descriptor=orb.detectAndCompute(image,mask.astype('uint8')*255)
            if descriptor is None or len(key)<20:raise ValueError('insufficient_visual_features')
            k=np.asarray(meta.get('K'),float)
            if k.shape!=(3,3) or not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0:raise ValueError('camera_intrinsics_unavailable')
            uv=np.asarray([x.pt for x in key]);pixels=np.rint(uv).astype(int)
            z=depth[pixels[:,1],pixels[:,0]]*meta.get('depth_scale_m',.001)
            points.append(np.column_stack(((uv[:,0]-k[0,2])*z/k[0,0],(uv[:,1]-k[1,2])*z/k[1,1],z)))
            keys.append(uv);descriptors.append(descriptor)
        matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(*descriptors,k=2)
        matches=[pair[0] for pair in matches if len(pair)==2 and pair[0].distance<.7*pair[1].distance]
        # Reject duplicate destination features and tiny textured patches.
        unique={}
        for match in sorted(matches,key=lambda m:m.distance):unique.setdefault(match.trainIdx,match)
        matches=list(unique.values())
        if len(matches)<16:raise ValueError('insufficient_frame_overlap')
        uv=keys[0][[m.queryIdx for m in matches]]
        if np.ptp(uv[:,0])<80 or np.ptp(uv[:,1])<50:raise ValueError('features_not_spatially_distributed')
        result=robust_displacement(points[0][[m.queryIdx for m in matches]],points[1][[m.trainIdx for m in matches]])
        result['evidence_frames']=[before_id,after_id]
        return result
    except (ValueError,OSError,ImportError,KeyError,IndexError,TypeError) as exc:
        return dict(failure,reason=str(exc))
