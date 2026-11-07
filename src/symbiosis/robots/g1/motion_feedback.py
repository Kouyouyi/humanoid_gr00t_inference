"""Relative motion from public IMU feedback; never accepts world positions."""
import math


def imu_yaw(state):
    values=state.get('base_quat_wxyz')
    if not isinstance(values,list) or len(values)!=4:
        raise ValueError('IMU quaternion unavailable')
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
        raise ValueError('IMU quaternion invalid')
    norm=math.sqrt(sum(v*v for v in values))
    if not .99 <= norm <= 1.01:raise ValueError('IMU quaternion not normalized')
    w,x,y,z=[v/norm for v in values]
    return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def heading_increment(previous,current):
    delta=imu_yaw(current)-imu_yaw(previous)
    return math.atan2(math.sin(delta),math.cos(delta))


def motion_feedback(feedback):
    result=dict(heading=dict(available=False,source='imu_orientation',angle_deg=None),
                translation=dict(available=False,source=None,distance_m=None,
                    reason='当前遥测没有线性里程计；仅凭姿态和角速度不能测量行走距离。'))
    before,end=feedback.get('before') or {},feedback.get('end') or {}
    legacy=before.get('imu_source')=='legacy_simulated_pose' or (before.get('encoder_source')=='simulated_joint_and_imu_sensors' and not before.get('imu_source'))
    if legacy:
        result['heading']['reason']='历史仿真记录未绑定真实模拟 IMU，不能作为 IMU 实测角度。'
        return result
    if not feedback.get('valid_coverage'):
        result['heading']['reason']='IMU 采样不完整或有中断，不能确认完整动作转角。'
        return result
    try:
        delta=feedback.get('imu_yaw_delta_deg')
        if not isinstance(delta,(int,float)) or not math.isfinite(delta):
            delta=math.degrees(heading_increment(before,end))
        result['heading'].update(available=True,angle_deg=delta,
            sensor_source=before.get('imu_source','native_lowstate_imu'),
            convention='左转为正、右转为负；相对本段起始方向。')
    except ValueError as exc:
        result['heading']['reason']=str(exc)
    return result
