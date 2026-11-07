"""Wall-clock evidence. Parallel lanes overlap; never label CLI gaps pure inference."""
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import re


def toml_value(value):
    if isinstance(value,dict):return '{ '+', '.join(json.dumps(k)+' = '+toml_value(v) for k,v in value.items())+' }'
    return json.dumps(value)


def scalar(value):
    if not isinstance(value,dict):return value
    return next((value[k] for k in ('stringValue','intValue','doubleValue','boolValue') if k in value),None)


def timing_attributes(values):
    # Export metadata only, never prompts, tool outputs, images, tokens or credentials.
    allowed=('event.name','event_name','model','attempt','success','status','status_code','kind','type','transport','operation','name')
    return {x['key']:scalar(x.get('value')) for x in values if x.get('key') in allowed
            or any(t in x.get('key','').lower() for t in ('duration','inference_time','ttft','tbt','token_count'))}


class TelemetryCollector:
    """Loopback OTLP/HTTP JSON timing sink, scoped to this experiment."""
    def __init__(self,path):
        journal=Journal(path)
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                size=int(self.headers.get('Content-Length',0))
                if not 0<size<=16*1024*1024:self.send_error(413);return
                try:payload=json.loads(self.rfile.read(size))
                except ValueError:self.send_error(400);return
                for resource in payload.get('resourceLogs',[]):
                    for scope in resource.get('scopeLogs',[]):
                        for row in scope.get('logRecords',[]):
                            body=scalar(row.get('body'));body=body if isinstance(body,str) and re.fullmatch(r'codex\.[a-zA-Z0-9_.]+',body) else None
                            attributes=timing_attributes(row.get('attributes',[]))
                            journal.write(event='otel_log',event_name=body or attributes.get('event.name'),time_unix_nano=row.get('timeUnixNano'),observed_time_unix_nano=row.get('observedTimeUnixNano'),attributes=attributes)
                for resource in payload.get('resourceSpans',[]):
                    for scope in resource.get('scopeSpans',[]):
                        for row in scope.get('spans',[]):
                            journal.write(event='otel_span',name=row.get('name'),start_unix_nano=row.get('startTimeUnixNano'),end_unix_nano=row.get('endTimeUnixNano'),attributes=timing_attributes(row.get('attributes',[])))
                for resource in payload.get('resourceMetrics',[]):
                    for scope in resource.get('scopeMetrics',[]):
                        for row in scope.get('metrics',[]):
                            name=row.get('name','')
                            if not any(k in name for k in ('duration','inference_time','ttft','tbt','api_request','websocket')):continue
                            for kind in ('histogram','sum','gauge','exponentialHistogram'):
                                for point in row.get(kind,{}).get('dataPoints',[]):
                                    journal.write(event='otel_metric',name=name,metric_kind=kind,point={k:v for k,v in point.items() if k in ('startTimeUnixNano','timeUnixNano','count','sum','min','max','asDouble','asInt')},attributes=timing_attributes(point.get('attributes',[])))
                data=b'{}';self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
    def overrides(self):
        endpoint=f'http://127.0.0.1:{self.server.server_port}'
        return [('otel.log_user_prompt',False),('otel.exporter',{'otlp-http':dict(endpoint=endpoint+'/v1/logs',protocol='json')}),
                ('otel.trace_exporter',{'otlp-http':dict(endpoint=endpoint+'/v1/traces',protocol='json')}),
                ('otel.metrics_exporter',{'otlp-http':dict(endpoint=endpoint+'/v1/metrics',protocol='json')})]
    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=5)


class Journal:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);self.lock=threading.Lock()
    def write(self,**row):
        row=dict(recorded_unix_s=time.time(),recorded_monotonic_s=time.monotonic(),**row)
        with self.lock,self.path.open('a') as stream:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
    @contextmanager
    def span(self,category,**details):
        wall=time.time();mono=time.monotonic();status='completed'
        self.write(event='start',category=category,start_unix_s=wall,**details)
        try:yield
        except BaseException:
            status='error';raise
        finally:self.write(event='end',category=category,start_unix_s=wall,end_unix_s=time.time(),
                           duration_s=time.monotonic()-mono,status=status,**details)


def capture_agent(stream,log,journal):
    """Receipt timestamps for public CLI events; no reasoning or image data in timing log."""
    for line in iter(stream.readline,b''):
        log.write(line);log.flush()
        try:row=json.loads(line)
        except ValueError:continue
        item=row.get('item',{})
        journal.write(event='cli_event',category='agent_event_receipt',cli_type=row.get('type'),
            item_id=item.get('id'),item_type=item.get('type'),tool=item.get('tool'),status=item.get('status'),
            usage=row.get('usage'),measurement='local stdout receipt; not server inference time')


def report(run):
    run=Path(run);requests={};tools=[];jobs={}
    path=run/'session/mcp.jsonl'
    for line in path.read_text().splitlines() if path.exists() else []:
        row=json.loads(line);ident=row.get('call_id')
        if row.get('event')=='mcp_request':requests[ident]=row
        elif row.get('event') in ('mcp_result','mcp_error') and ident in requests:
            req=requests[ident];name=req['tool'];start=req['time'];end=row['time']
            category='observation_rgbd' if name=='g1_observe' else 'depth_geometry' if name in ('g1_pixel_to_position','g1_hand_frames') else 'job_poll' if name.endswith('_result') else 'action_submit' if name in ('g1_move','g1_vla','g1_reach','g1_hand','g1_posture','g1_smpl_execute') else 'status_or_other'
            tools.append(dict(category=category,tool=name,start_unix_s=start,end_unix_s=end,duration_s=end-start,call_id=ident,status=row['event']))
            try:value=json.loads(row.get('result','{}'))
            except ValueError:value={}
            job=value.get('job_id')
            if job and category=='action_submit' and value.get('status')=='running':jobs[job]=dict(job_id=job,tool=name,arguments=req.get('arguments'),submitted_unix_s=start)
            if job and job in jobs and value.get('status') not in ('running',None):
                jobs[job].setdefault('first_completion_observed_unix_s',end)
                jobs[job]['completion_observation_upper_bound_s']=end-jobs[job]['submitted_unix_s']
    totals={}
    for row in tools:totals[row['category']]=totals.get(row['category'],0)+row['duration_s']
    result=dict(schema_version=1,clock='Unix wall clock + monotonic for new spans',tools=tools,jobs=list(jobs.values()),
        tool_duration_totals_s=totals,notes=[
        'Job submission/poll latency is not physical action duration. Legacy completion is an upper bound.',
        'Agent CLI gaps include model/network/local orchestration and cannot identify pure server inference.',
        'Physics, VLA inference and agent response may overlap. Do not add parallel lanes as exclusive totals.',
        'Third-person images are not read by this report.'],
        agent_request_server_inference_s=None)
    task=run/'task-result.json'
    if task.exists():result['task_wall_s']=json.loads(task.read_text()).get('wall_s')
    perf=run/'simulation/performance.json'
    if perf.exists():result['simulation_performance']=json.loads(perf.read_text())
    for job in jobs.values():
        evidence=run/'simulation'/(job['job_id']+'.json')
        if evidence.exists():
            raw=json.loads(evidence.read_text())
            job.update(actual_steps=raw.get('steps'),action=raw.get('action'),status=raw.get('status'),
                start_unix_s=raw.get('started_unix_s'),end_unix_s=raw.get('ended_unix_s'),duration_wall_s=raw.get('duration_wall_s'),
                preparation_wall_s=raw.get('preparation_wall_s'),actual_execution_wall_s=raw.get('actual_execution_wall_s'))
    gaps=[]
    for previous,next_row in zip(tools,tools[1:]):
        a=previous['end_unix_s'];b=next_row['start_unix_s']
        if b>a:gaps.append(dict(category='between_tools_unattributed',start_unix_s=a,end_unix_s=b,duration_s=b-a,
            note='May include model response, network, orchestration and overlapping physical action. Not pure inference.'))
    result['between_tool_intervals']=gaps
    result['between_tool_unattributed_s']=sum(x['duration_s'] for x in gaps)
    for name in ('vla-client','vla-server','codex-otel'):
        path=run/'timing'/(name+'.jsonl')
        rows=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
        result[name]=dict(collected=bool(rows),records=rows)
    # Independently measured concurrent lanes. Union partitions avoid double-counting.
    lanes=[];started={}
    retry=run/'retry/events.jsonl'
    for line in retry.read_text().splitlines() if retry.exists() else []:
        row=json.loads(line)
        if row['event']=='action_started':started[row['job_id']]=row
        elif row['event']=='action_finished' and row['job_id'] in started:
            origin=started.pop(row['job_id'])
            lanes.append(dict(category='action_active',tool=origin['action'],start_unix_s=origin['unix_s'],end_unix_s=row['unix_s'],duration_s=row['unix_s']-origin['unix_s']))
    monitor=run/'retry/monitor.jsonl';requests={};monitor_rows=[]
    for line in monitor.read_text().splitlines() if monitor.exists() else []:
        row=json.loads(line);monitor_rows.append(row)
        if row['event']=='monitor_request':requests[row['request_id']]=row
        elif row['event'] in ('monitor_result','monitor_unavailable') and row['request_id'] in requests:
            origin=requests.pop(row['request_id']);lanes.append(dict(category='qwen_request',start_unix_s=origin['unix_s'],end_unix_s=row['unix_s'],duration_s=row['unix_s']-origin['unix_s']))
    boundaries=sorted({r[k] for r in lanes for k in ('start_unix_s','end_unix_s')})
    result['concurrent_intervals']=[dict(start_unix_s=a,end_unix_s=b,duration_s=b-a,
        active=sorted({r['category'] for r in lanes if r['start_unix_s']<=a and r['end_unix_s']>=b})) for a,b in zip(boundaries,boundaries[1:])]
    result['retry_action_and_monitor_intervals']=lanes
    result['qwen_monitor']=dict(requests=sum(r['event']=='monitor_request' for r in monitor_rows),
        responses=sum(r['event']=='monitor_result' for r in monitor_rows),errors=sum(r['event']=='monitor_unavailable' for r in monitor_rows),
        unreturned_request_ids=list(requests),stop_signals=sum(r.get('verdict',{}).get('decision')=='stop' for r in monitor_rows if isinstance(r.get('verdict'),dict)))
    with (run/'timing-intervals.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['category','tool','start_unix_s','end_unix_s','duration_s','note'],extrasaction='ignore')
        writer.writeheader();writer.writerows(sorted([*tools,*gaps,*lanes],key=lambda r:r['start_unix_s']))
    (run/'timing-report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result
