"""Operator session lifetime, independent of native transport watchdogs."""
import json
import math
import os
from pathlib import Path
import re
import time
import uuid

EXPIRED = 'Operator session lease expired'


def duration(seconds):
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or (seconds!=0 and seconds<30):
        raise ValueError('seconds must be 0 (unlimited) or a finite value >=30')
    return float(seconds)


def remaining(owner):
    if owner.get('lease_unlimited') is True:return math.inf
    value=owner.get('lease_remaining_seconds')
    return value if type(value) in (int,float) and math.isfinite(value) and value>=0 else 0.


def validate(action, seconds=None, job_id=None):
    if action=='status' and seconds is None and job_id is None:return
    if action=='set' and job_id is None:
        duration(seconds);return
    if action=='result' and seconds is None and isinstance(job_id,str) and re.fullmatch('[0-9a-f]{32}',job_id):return
    raise ValueError('Use status; set with seconds; or result with job_id')


def status(directory):
    if directory is None:raise RuntimeError('No body owner configured')
    state=json.loads((Path(directory)/'ankle_status.json').read_text())
    if not 0<=time.monotonic()-state['updated_monotonic']<=.25:raise RuntimeError('Body owner status stale')
    return dict(session_id=state['session_id'],phase=state['phase'],fault=state.get('fault'),
        lease_control_supported=state.get('lease_control_supported',False),
        lease_unlimited=state.get('lease_unlimited',False),
        lease_remaining_seconds=state.get('lease_remaining_seconds'),
        lease_duration_seconds=state.get('lease_duration_seconds'))


def submit(directory, seconds):
    seconds=duration(seconds);state=status(directory)
    if not state['lease_control_supported']:
        raise RuntimeError('Running owner predates lease control; install updated owner after operator-controlled native stop')
    request=dict(id=uuid.uuid4().hex,session_id=state['session_id'],seconds=seconds,created_monotonic=time.monotonic())
    folder=Path(directory)/'lease_requests';temp=folder/(request['id']+'.json')
    try:
        from .mailbox import publish
        temp.write_text(json.dumps(request,allow_nan=False));publish(temp,folder/'pending.json')
    finally:temp.unlink(missing_ok=True)
    return dict(job_id=request['id'],status='submitted',automatic_motion_resume=False)


def result(directory, job_id):
    validate('result',job_id=job_id)
    if directory is None:raise RuntimeError('No body owner configured')
    path=Path(directory)/'lease_results'/(job_id+'.json')
    return json.loads(path.read_text()) if path.exists() else dict(job_id=job_id,status='pending_or_unknown')


class SessionLease:
    def __init__(self, root, session, seconds=0):
        self.root=Path(root);self.session=session;self.seconds=duration(seconds);self.began=None
        for name in ('lease_requests','lease_results'):(self.root/name).mkdir(exist_ok=True)

    def start(self, now):self.began=now

    def remaining(self, now):
        if self.seconds==0:return math.inf
        return max(0.,self.seconds-(now-self.began)) if self.began is not None else self.seconds

    def expired(self, now):return self.began is not None and self.remaining(now)<=0

    def snapshot(self, now):
        return dict(lease_control_supported=True,lease_unlimited=self.seconds==0,
            lease_duration_seconds=self.seconds,
            lease_remaining_seconds=None if self.seconds==0 or self.began is None else self.remaining(now))

    def consume(self, now, phase, fault, settled, busy):
        """Called only AFTER fresh health/guard checks. Recover expiry alone, never other faults."""
        from .vla import atomic
        path=self.root/'lease_requests/pending.json'
        if not path.exists():return False
        claimed=path.with_name('claimed.json');path.replace(claimed);request=None;recover=False
        try:
            request=json.loads(claimed.read_text())
            if set(request)!={'id','session_id','seconds','created_monotonic'}:raise ValueError('Unexpected lease fields')
            if not re.fullmatch('[0-9a-f]{32}',str(request['id'])):raise ValueError('Invalid lease request ID')
            if request['session_id']!=self.session or not 0<=now-request['created_monotonic']<=2:
                raise ValueError('Lease request expired or owner session changed')
            seconds=duration(request['seconds'])
            recover=phase=='fault_idle' and fault==EXPIRED
            if phase=='fault_idle' and not recover:raise ValueError('Lease change cannot clear a non-lease fault')
            if recover and (not settled or busy):raise ValueError('Lease recovery requires settled idle feedback')
            self.seconds=seconds;self.began=now
            value=dict(completed=True,recovered_lease_fault=recover,automatic_motion_resume=False,
                       **self.snapshot(now))
        except (OSError,ValueError,KeyError,TypeError) as exc:
            recover=False;value=dict(completed=False,reason=str(exc))
        finally:claimed.unlink(missing_ok=True)
        if isinstance(request,dict) and re.fullmatch('[0-9a-f]{32}',str(request.get('id',''))):
            atomic(self.root/'lease_results'/(request['id']+'.json'),dict(job_id=request['id'],session_id=self.session,**value))
        return recover
