"""Finite read-only live temperature observer or explicitly labelled replay.

Creates no DDS publisher and never writes to an executor request spool.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time

from .ankle_thermal import ANKLES, AlarmDetector, announce, BLOCK_REASON


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--seconds',type=float,default=20)
    p.add_argument('--replay',type=Path,help='Explicit synthetic/offline replay; never opens DDS')
    a=p.parse_args()
    if not 1<=a.seconds<=600:p.error('seconds must be 1..600')
    replay=json.loads(a.replay.read_text()) if a.replay else None
    if replay is not None:
        if not isinstance(replay,list) or not replay:p.error('Replay must be a nonempty sample list')
        offsets=[float(r['elapsed_s']) for r in replay]
        if any(not 0<=x<a.seconds for x in offsets) or any(y<=x for x,y in zip(offsets,offsets[1:])):
            p.error('Replay offsets must strictly increase within test duration')
    a.run.mkdir(parents=True,exist_ok=False)
    source='synthetic_or_offline_replay' if replay is not None else 'live_lowstate'
    provenance=dict(source=source,motion_enabled=False,dds_publishers=0,
        replay_file=str(a.replay) if a.replay else None,block_reason=BLOCK_REASON,
        source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ('ankle_monitor.py','ankle_thermal.py')})
    (a.run/'manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    for name in provenance['source_sha256']:
        (a.run/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    detector=AlarmDetector();latest={};lock=threading.Lock();sub=None
    futures=[];errors=[];samples=triggers=0;maxima={};invalid_samples=0
    pool=ThreadPoolExecutor(max_workers=1)
    events=(a.run/'events.jsonl').open('w');trace=(a.run/'temperature_trace.jsonl').open('w')
    def log(event):
        row=dict(monotonic=time.monotonic(),source=source,**event)
        events.write(json.dumps(row)+'\n');events.flush()
        print(json.dumps(row),flush=True)
    def receive(msg):
        row=dict(monotonic=time.monotonic(),motors=[dict(hardware_index=i,name=name,
            temperature_winding_driver_c=list(msg.motor_state[i].temperature),
            motorstate=int(msg.motor_state[i].motorstate),mode=int(msg.motor_state[i].mode))
            for i,name in ANKLES.items()])
        with lock:latest.update(row)
    try:
        if replay is None:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize,ChannelSubscriber
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
            ChannelFactoryInitialize(0,'eno1')
            sub=ChannelSubscriber('rt/lowstate',LowState_);sub.Init(receive,1)
        started=time.monotonic();index=0;last_seen=None;last_fresh=started
        while time.monotonic()-started<a.seconds:
            now=time.monotonic();row=None
            if replay is not None:
                if index<len(replay) and now-started>=replay[index]['elapsed_s']:
                    row=dict(replay[index]);row['monotonic']=now;index+=1
            else:
                with lock:row=dict(latest)
            if row and row.get('monotonic')!=last_seen:
                last_seen=row['monotonic']
                try:alarms=detector.update(row,time.monotonic())
                except (ValueError,KeyError,TypeError) as exc:
                    invalid_samples+=1;log(dict(event='sample_rejected',reason=str(exc)))
                else:
                    last_fresh=now;samples+=1
                    trace.write(json.dumps(dict(source=source,**row))+'\n');trace.flush()
                    for m in row['motors']:
                        maxima[m['name']]=max(maxima.get(m['name'],-float('inf')),max(m['temperature_winding_driver_c']))
                    for event in alarms:
                        log(event)
                        if event['event']=='ankle_temperature_trigger':
                            triggers+=1
                            if any(e['side']==event['side'] for e,_ in futures):
                                log(dict(event='audio_suppressed_pending',side=event['side']))
                            else:
                                futures.append((event,pool.submit(announce,event,a.run,replay is not None)))
            if replay is None and now-last_fresh>2:
                raise RuntimeError('No fresh valid ankle feedback for two seconds')
            for event,future in list(futures):
                if future.done():
                    try:log(future.result())
                    except Exception as exc:
                        errors.append(str(exc));log(dict(event='ankle_audio_failed',side=event['side'],reason=str(exc),motion_enabled=False))
                    futures.remove((event,future))
            time.sleep(.05)
        if not samples:raise RuntimeError('No valid samples collected')
    except BaseException as exc:
        errors.append(repr(exc));raise
    finally:
        if sub:sub.Close()
        # Audio is finite and independent of temperature sampling/body ownership.
        pool.shutdown(wait=True)
        for event,future in futures:
            try:log(future.result())
            except Exception as exc:
                errors.append(str(exc));log(dict(event='ankle_audio_failed',side=event['side'],reason=str(exc),motion_enabled=False))
        result=dict(source=source,valid_samples=samples,invalid_samples=invalid_samples,
            trigger_count=triggers,max_temperatures_c=maxima,errors=errors,
            body_commands_sent=0,hand_commands_sent=0,motion_enabled=False,
            thermal_motion_tested=False,dispatch_block_reason=BLOCK_REASON)
        (a.run/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        events.close();trace.close();print(json.dumps(result),flush=True)


if __name__=='__main__':main()
