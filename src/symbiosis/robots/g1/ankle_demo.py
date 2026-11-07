"""Finite Pico-based token action with explicit audio prerequisite.

No hardware I/O. Source motor targets are never used as controller inputs.
"""
import json
from pathlib import Path

import numpy as np

from .state import NAMES, JOINTS
from .vendor import sha

SIDES={'left':13,'right':15}


def load_demos(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    result={}
    for side,episode in SIDES.items():
        path=directory/f'episode_{episode:06d}.npz'
        if sha(path)!=manifest[side]['sha256']:raise ValueError('Demonstration hash mismatch')
        with np.load(path,allow_pickle=False) as f:
            token=f['action.motion_token'].astype(np.float32)
            q=f['observation.state']
            times=f['timestamp'].reshape(-1)
            if not np.all(f['teleop.stream_mode']==1):raise ValueError('Expected SMPL demonstration')
        if token.shape!=(len(times),64) or not 2<=len(times)<=200 or not np.isfinite(token).all():
            raise ValueError('Invalid demonstration tokens')
        if q.shape!=(len(times),29) or not np.isfinite(q).all() or not np.allclose(np.diff(times),.02,atol=1e-5):
            raise ValueError('Invalid demonstration states or timing')
        result[side]=token
    return result


def validate_request(job,session_id,now):
    expected={'id','action','side','session_id','created_monotonic'}
    if set(job)!=expected or job['action']!='ankle_relax' or job['side'] not in SIDES:
        raise ValueError('Invalid ankle request')
    if job['session_id']!=session_id:raise ValueError('Changed ankle session')
    if not isinstance(job['id'],str) or len(job['id'])!=32 or any(c not in '0123456789abcdef' for c in job['id']):
        raise ValueError('Invalid request ID')
    if not isinstance(job['created_monotonic'],(float,int)) or not 0<=now-job['created_monotonic']<=2:
        raise ValueError('Expired ankle request')


def guard(measured,moving,*,check_speed=True):
    if measured['tilt_rad']>.25:raise ValueError('base_tilt_above_0.25_rad')
    q=np.asarray(measured['q']);dq=np.asarray(measured['dq'])
    if q.shape!=(29,) or dq.shape!=(29,) or not np.isfinite(q).all() or not np.isfinite(dq).all():
        raise ValueError('Invalid joint feedback')
    if any(q[i]<j['lower']-.02 or q[i]>j['upper']+.02 for i,j in enumerate(JOINTS)):
        raise ValueError('Measured joint outside model limit')
    if moving and check_speed:
        for i,name in enumerate(NAMES):
            bound=3. if any(x in name for x in ('hip','knee','ankle')) else 2.
            if abs(dq[i])>bound:raise ValueError('Ankle-demo active joint speed exceeded')


class DemoFlow:
    def __init__(self,tokens,anchor,now,*,audio_completed):
        if not audio_completed:raise ValueError('Successful audio required before movement')
        self.tokens=np.asarray(tokens,dtype=np.float32)
        self.anchor=np.asarray(anchor,dtype=np.float32).copy()
        if self.anchor.shape!=(64,) or not np.isfinite(self.anchor).all():raise ValueError('Invalid anchor token')
        self.stage='entry_blend';self.at=now;self.index=0;self.done=False
        self.settled_since=None;self.return_verified=False
        self.last=self.anchor.copy();self.source_frame=None

    def sample(self,now,settled):
        elapsed=now-self.at;self.source_frame=None
        if self.stage=='entry_blend':
            u=np.clip(elapsed/4.,0,1);b=u**3*(10-15*u+6*u*u)
            self.last=(1-b)*self.anchor+b*self.tokens[0]
            if elapsed>=4:self.stage='start_hold';self.at=now
        elif self.stage=='start_hold':
            self.last=self.tokens[0].copy()
            if elapsed>=2:self.stage='recorded_episode';self.at=now
        elif self.stage=='recorded_episode':
            # Submit every source frame once; never skip a pose to catch up.
            self.source_frame=self.index;self.last=self.tokens[self.index].copy();self.index+=1
            if self.index==len(self.tokens):self.stage='return_blend';self.at=now
        elif self.stage=='return_blend':
            u=np.clip(elapsed/3.,0,1);b=u**3*(10-15*u+6*u*u)
            self.last=(1-b)*self.tokens[-1]+b*self.tokens[0]
            if elapsed>=3:self.stage='return_hold';self.at=now
        elif self.stage=='return_hold':
            self.last=self.tokens[0].copy()
            self.settled_since=(self.settled_since if self.settled_since is not None else now) if settled else None
            if elapsed>=2 and self.settled_since is not None and now-self.settled_since>=1:
                self.done=True;self.return_verified=True
            elif elapsed>=8:self.done=True
        return self.last.astype(np.float32)


def announce(side,directory,job_id,thermal=False):
    import subprocess,time
    if side not in SIDES:raise ValueError('Invalid audio side')
    message=(f'Warning. High temperature at {side} ankle. ' if thermal else f'Ankle movement test. {side} side. ')
    message+=f'Lifting the {side} heel slightly, then returning.'
    wav=Path(directory)/(job_id+'.wav')
    if wav.exists():raise FileExistsError(wav)
    subprocess.run(['espeak','-s','155','-w',str(wav),message],check=True,timeout=10)
    start=time.monotonic()
    subprocess.run(['paplay',str(wav)],check=True,timeout=20)
    return dict(message=message,wav=str(wav),playback_started_monotonic=start,
                playback_completed_monotonic=time.monotonic(),playback_exit_code=0)
