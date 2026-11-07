"""Local control-source handoff: blend once per ownership edge, never per chunk."""
import numpy as np


class OwnershipHandoff:
    def __init__(self,initial,now,duration=.2):
        self.last=self._token(initial);self.baseline=self.last.copy()
        if not np.isfinite([now,duration]).all() or not 0<=duration<=1:
            raise ValueError('Invalid handoff time/duration')
        self.last_time=float(now);self.duration=float(duration)
        self.active=False;self.started=None;self.origin=self.last.copy()

    @staticmethod
    def _token(value):
        value=np.asarray(value,dtype=np.float64)
        if value.shape!=(64,) or not np.isfinite(value).all():raise ValueError('Invalid handoff token')
        return value.copy()

    def step(self,target,now,*,active=None):
        target=self._token(target)
        if not np.isfinite(now) or now<self.last_time:raise ValueError('Invalid handoff clock')
        if active is None:active=not np.array_equal(target,self.baseline)
        if type(active)is not bool:raise ValueError('Ownership must be boolean')
        if active!=self.active:
            self.origin=self.last.copy();self.started=float(now);self.active=active
        u=1. if self.started is None or self.duration==0 else np.clip((now-self.started)/self.duration,0,1)
        alpha=float(u*u*(3-2*u))
        self.last=target.copy() if alpha==1 else (1-alpha)*self.origin+alpha*target
        self.last_time=float(now)
        return self.last.copy(),alpha
