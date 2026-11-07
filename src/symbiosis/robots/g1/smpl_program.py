"""Offline compilation of B/C; no publisher, controller reset or object writes."""
import hashlib
import numpy as np

def integer(value, name, minimum, maximum):
    if type(value) is not int or value < minimum or maximum is not None and value > maximum:
        raise ValueError('Invalid '+name)
    return value

JOINTS = ('left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
          'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot',
          'right_foot', 'neck', 'left_collar', 'right_collar', 'head',
          'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
          'left_wrist', 'right_wrist')


class ProgramCompiler:
    def __init__(self, converter, limits, profile):
        self.converter, self.limits, self.profile = converter, limits, profile

    def compile(self, anchor, program):
        limits=[x for x in (self.profile['max_steps'],
            None if self.profile['max_segment_s'] is None else int(self.profile['max_segment_s']*50)) if x is not None]
        n = integer(program['n'], 'retry program steps', 3, min(limits) if limits else None)
        keys = program['keyframes']
        if not isinstance(keys, list) or not 2 <= len(keys) <= 8:
            raise ValueError('B/C requires 2-8 keyframes')
        current_hands = np.asarray(anchor['hands'], np.float32)
        hands = current_hands.copy()
        times, deltas, handkeys = [], [], []
        for row in keys:
            if not isinstance(row, dict) or set(row)-{'step', 'joint_deltas', 'hands'}:
                raise ValueError('invalid program keyframe')
            step = integer(row['step'], 'keyframe step', 0, n-1)
            if times and step <= times[-1]:
                raise ValueError('keyframes must advance')
            delta = np.zeros((21, 3))
            sparse = row.get('joint_deltas', {})
            if not isinstance(sparse, dict) or set(sparse)-set(self.limits['owned_joints']):
                raise ValueError('SMPL joint is not site-authorized')
            for name, vector in sparse.items():
                v = np.asarray(vector, float)
                if v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) > self.limits['delta_rad']:
                    raise ValueError('SMPL delta exceeds reviewed bound')
                delta[JOINTS.index(name)] = v
            commands = row.get('hands', {})
            if not isinstance(commands, dict) or set(commands)-{'left', 'right'}:
                raise ValueError('invalid native hand program')
            for side, target in commands.items():
                if not hands.size:
                    raise ValueError('This deployment has no hand actuator contract')
                target = np.asarray(target, np.float32)
                j = ('left', 'right').index(side)
                if target.shape != hands[j].shape or not np.isfinite(target).all():
                    raise ValueError('native hand dimension mismatch')
                lower=np.asarray(self.limits['hand_min']);upper=np.asarray(self.limits['hand_max'])
                if lower.ndim==2:lower,upper=lower[j],upper[j]
                if np.any(target < lower) or np.any(target > upper):
                    raise ValueError('native hand joint bound')
                hands[j] = target
            if step == 0 and (np.any(delta) or not np.array_equal(hands, current_hands)):
                raise ValueError('first keyframe must preserve actual body and hand command')
            times.append(step); deltas.append(delta); handkeys.append(hands.copy())
        if times[0] != 0 or times[-1] != n-1:
            raise ValueError('program must cover exact bounded horizon')
        grid = np.arange(n+9)
        curve = np.empty((n+9, 21, 3)); handcurve = np.empty((n+9, *current_hands.shape))
        for i, tick in enumerate(grid):
            k = min(max(0, np.searchsorted(times, tick, side='right')-1), len(times)-2)
            u = np.clip((tick-times[k])/(times[k+1]-times[k]), 0, 1)
            alpha = u**3*(10-15*u+6*u*u)
            curve[i] = deltas[k]+alpha*(deltas[k+1]-deltas[k])
            handcurve[i] = handkeys[k]+alpha*(handkeys[k+1]-handkeys[k])
        speed = np.linalg.norm(np.diff(curve, axis=0)*50, axis=-1)
        if np.max(speed) > self.limits['angular_speed_rad_s']:
            raise ValueError('SMPL velocity limit')
        acceleration = np.linalg.norm(np.diff(curve, n=2, axis=0)*2500, axis=-1)
        if acceleration.max() > self.limits['angular_acceleration_rad_s2']:
            raise ValueError('SMPL acceleration limit')
        if handcurve.size and np.max(np.abs(np.diff(handcurve, axis=0))*50) > self.limits['hand_speed_s']:
            raise ValueError('hand velocity limit')
        original = np.r_[anchor['token'], [0., 0.]].astype(np.float32)
        tokens = []
        for i in range(n):
            # Exact entry; look-ahead must never perturb the first output.
            token = original[:64].copy() if i == 0 else self.converter.convert(original, curve[i:i+10])['action66'][:64]
            tokens.append(token)
        tokens = np.asarray(tokens, np.float32)
        if not np.isfinite(tokens).all() or np.abs(tokens).max()>1.25:
            raise ValueError('nonfinite/out-of-range encoded program')
        kin = self.converter.bridge.kin(tokens)
        q, dq = kin['q'], kin['dq']
        if (np.any(q < np.asarray(self.limits['body_min'])) or np.any(q > np.asarray(self.limits['body_max']))
                or np.any(np.abs(dq) > np.asarray(self.limits['body_speed_rad_s']))):
            raise ValueError('encoded body joint/rate bound')
        if np.array_equal(tokens, np.repeat(original[None, :64], n, axis=0)) and np.array_equal(handcurve[:n], np.repeat(current_hands[None], n, axis=0)):
            raise ValueError('program has no effect')
        return dict(tokens=tokens, hands=handcurve[:n].astype(np.float32),
                    metadata=dict(steps=n, token_sha256=hashlib.sha256(tokens.tobytes()).hexdigest(),
                                  hand_sha256=hashlib.sha256(handcurve[:n].astype(np.float32).tobytes()).hexdigest(),
                                  converter=self.converter.fingerprint, physics_validated=False))
