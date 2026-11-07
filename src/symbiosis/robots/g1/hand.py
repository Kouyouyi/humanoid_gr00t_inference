"""Finite right BrainCo commands with measured feedback; no body publisher.

Run under the existing teleop Python environment. Closing this process sends no
automatic release; a successful command is not evidence of a grasp.
"""
import argparse
import fcntl
import json
from pathlib import Path
import threading
import time

import numpy as np

POSES={'open':[0.,1.,0.,0.,0.,0.], 'close':[1.,1.,1.,1.,1.,1.]}


def command_values(start,target,fraction,speed):
    a,b=np.asarray(start,dtype=float),np.asarray(target,dtype=float)
    if (a.shape!=(6,) or b.shape!=(6,) or not np.isfinite([a,b]).all() or
        np.min([a,b])<0 or np.max([a,b])>1 or not 0<=fraction<=1 or not .05<=speed<=.3):
        raise ValueError('Invalid normalized BrainCo command')
    return (a+(b-a)*fraction).tolist(),[speed]*6


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--action',choices=['observe','open','close'],required=True)
    p.add_argument('--speed',type=float,default=.2)
    p.add_argument('--body-run',type=Path)
    a=p.parse_args()
    command_values([0]*6,[0]*6,0,a.speed)
    if a.action!='observe' and a.body_run is None:p.error('Body executor required for hand actions')
    a.run.mkdir(parents=True,exist_ok=False)
    lockfile=open('/tmp/rax_brainco_right.lock','a');fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize,ChannelSubscriber,ChannelPublisher
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_,MotorCmds_
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
    state={};mutex=threading.Lock()
    def receive(msg):
        row=dict(monotonic=time.monotonic(),q=[float(m.q) for m in msg.states],
                 dq=[float(m.dq) for m in msg.states],current_service_units=[float(m.tau_est) for m in msg.states])
        with mutex:state.update(row)
    def feedback():
        with mutex:r=dict(state)
        if not r or time.monotonic()-r['monotonic']>.2:return None
        q=np.asarray(r['q'])
        if q.shape!=(6,) or not np.isfinite(q).all() or min(q)<0 or max(q)>1:return None
        return r
    def body_ready():
        if a.body_run is None:return
        s=json.loads((a.body_run/'executor_status.json').read_text())
        if (time.monotonic()-s['updated_monotonic']>.25 or s['phase']!='reference_ready' or
            s.get('posture')!='clear_arms' or s['fault'] or not s.get('grasp_profile_enabled')):
            raise RuntimeError('Body must be holding a fresh cleared posture before finger movement')
    ChannelFactoryInitialize(0,'eno1')
    sub=ChannelSubscriber('rt/brainco/right/state',MotorStates_);sub.Init(receive,1);pub=None
    rows=[];sent=0
    try:
        deadline=time.monotonic()+5
        while feedback() is None:
            if time.monotonic()>deadline:raise RuntimeError('No fresh right hand feedback; no command sent')
            time.sleep(.02)
        before=feedback();start=time.monotonic()
        if a.action!='observe':
            body_ready();pub=ChannelPublisher('rt/brainco/right/cmd',MotorCmds_);pub.Init();time.sleep(.2)
        while time.monotonic()-start<3.:
            measured=feedback()
            if measured is None:raise RuntimeError('Hand feedback lost; no further commands')
            body_ready()
            elapsed=time.monotonic()-start
            record=dict(elapsed_s=elapsed,feedback=measured)
            if pub:
                q,dq=command_values(before['q'],POSES[a.action],min(elapsed/1.5,1.),a.speed)
                msg=MotorCmds_();msg.cmds=[unitree_go_msg_dds__MotorCmd_() for _ in range(6)]
                for motor,x,v in zip(msg.cmds,q,dq):motor.q=x;motor.dq=v
                if not pub.Write(msg):raise RuntimeError('Right hand DDS write failed')
                sent+=1;record.update(command_q=q,command_speed=dq)
            rows.append(record);time.sleep(.05)
        result=dict(status='command_cycle_finished' if pub else 'observed',action=a.action,commands_sent=sent,
                    before=before,after=feedback(),grasp_verified=False,body_commands_sent=0,
                    left_hand_commands_sent=0,feedback_is_normalized_actuator_position=True)
        (a.run/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    except BaseException as exc:
        (a.run/'failure.json').write_text(json.dumps(dict(error=repr(exc),commands_sent=sent,automatic_release_sent=False),indent=2)+'\n')
        raise
    finally:
        (a.run/'trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        if pub:pub.Close()
        sub.Close();lockfile.close()


if __name__=='__main__':main()
