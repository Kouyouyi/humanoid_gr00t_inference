"""Submit a demo or arm operator-configured / completed-test sides."""
import argparse
import json
from pathlib import Path
import time
import uuid

from .ankle_executor import atomic


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('operation',choices=['test','arm','disarm'])
    p.add_argument('--side',choices=['left','right'],default='right')
    p.add_argument('--sides',choices=['left','right'],nargs='+',default=['right'])
    a=p.parse_args();status=json.loads((a.run/'ankle_status.json').read_text());now=time.monotonic()
    if not 0<=now-status['updated_monotonic']<=.3 or status['phase']!='ready':
        raise ValueError('Ankle owner not fresh and ready')
    if a.operation=='test':
        path=a.run/'ankle_requests/pending.json'
        job=dict(id=uuid.uuid4().hex,action='ankle_relax',side=a.side,
                 session_id=status['session_id'],created_monotonic=now)
        # One caller per commissioning session; exclusive creation prevents overwriting.
        tmp=path.with_name('submit-'+job['id']+'.json');tmp.write_text(json.dumps(job)+'\n')
        try:
            import os
            from .mailbox import publish
            publish(tmp,path)
        finally:tmp.unlink(missing_ok=True)
        print(json.dumps(job))
    else:
        sides=a.sides if a.operation=='arm' else []
        allowed=set(status.get('configured_auto_sides',[]))|set(status['software_completed_sides'])
        if not set(sides)<=allowed:
            raise ValueError('Side must be configured at startup or complete a manual test before arming')
        value=dict(session_id=status['session_id'],sides=sides,monotonic=now)
        atomic(a.run/'thermal_arm.json',value);print(json.dumps(value))


if __name__=='__main__':main()
