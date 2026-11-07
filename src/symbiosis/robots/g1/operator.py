"""Authenticated operator requests. This interface is never exposed as an agent tool.

The web server only queues requests; a separate supervisor owns lifecycle changes.
Requests are bound to a supervisor epoch, owner session and (for results) task batch.
"""
from contextlib import contextmanager
import json
import sqlite3
import time
import uuid
from pathlib import Path

from symbiosis.experience.schema import read, write

BOOT = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
ACTIONS = ('start_control', 'prepare', 'start_harness', 'pause_harness', 'stop_control', 'end_episode',
           'confirm_success', 'confirm_failure')
LABELS = dict(start_control='启动身体控制', prepare='VLA 准备（打开双手）',
              start_harness='启动 harness 调度', pause_harness='暂停任务调度',
              stop_control='终止原生控制', end_episode='结束当前 episode', confirm_success='人工确认成功', confirm_failure='人工确认失败')


class Operator:
    def __init__(self, run):
        self.run = Path(run).resolve()
        self.path = self.run/'operator-controls.sqlite3'
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, payload TEXT, state TEXT, result TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS native_stops (id TEXT PRIMARY KEY, relay_pid INTEGER, session TEXT, epoch TEXT, expires REAL, consumed INTEGER)')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def begin(self):
        self.epoch = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("UPDATE requests SET state='failed', result=? WHERE state IN ('queued','executing')",
                       (json.dumps(dict(error='服务重启；旧请求未重放')),))
            db.execute('DELETE FROM native_stops')
        return self.epoch

    def heartbeat(self, phase, **extra):
        write(self.run/'operator-controls.json', dict(epoch=self.epoch, phase=phase,
              boot_id=BOOT, updated_monotonic=time.monotonic(), **extra))

    def snapshot(self):
        service = read(self.run/'operator-controls.json')
        online = (service.get('phase') != 'offline' and service.get('boot_id') == BOOT and
                  0 <= time.monotonic()-service.get('updated_monotonic', -1e9) < 3)
        owner = read(self.run/'ankle_status.json')
        sources = read(self.run/'sensor-sources.json')
        camera_directory=Path(sources.get('camera',self.run/'camera'))
        camera = read(camera_directory/'latest.json')
        camera_health=read(camera_directory/'health.json')
        telemetry = read(sources.get('telemetry',self.run/'telemetry.json'))
        from symbiosis.operations.g1_vla_trial_terminal import health_reason
        sensors_ready = (camera.get('host_boot_id') == BOOT and camera.get('source') == 'live_camera'
            and camera_health.get('running') is True and not camera_health.get('errors')
            and 0 <= time.monotonic()-camera_health.get('updated_monotonic',-1e9) <= 1
            and telemetry.get('host_boot_id') == BOOT and health_reason(telemetry,time.monotonic()) is None)
        fresh = 0 <= time.monotonic()-owner.get('updated_monotonic', -1e9) < .5
        simulated = service.get('transport') == 'simulation' and owner.get('transport') == 'simulation'
        if simulated:
            sensors_ready = fresh and camera.get('transport') == 'simulated_rgbd'
        session = owner.get('session_id')
        task = read(self.run/'operator-current-task.json')
        batch = task.get('batch') if task.get('session') == session else None
        outcome = verified_outcome(self.run, batch) if batch else {}
        gate = read(self.run/'harness-control.json')
        enabled = gate.get('enabled') is True and gate.get('session') == session
        ready = fresh and owner.get('phase') == 'ready' and not owner.get('fault') and not (self.run/'native-exit.json').exists()
        busy = service.get('phase') in ('starting_owner', 'starting_native', 'starting_harness','preparing', 'stopping','ending_episode')
        agent = read(self.run/'ui-agent-status.json')
        agent_online = (agent.get('phase') in ('idle','running') and
                        0 <= time.time()-agent.get('updated_at',0) < 3)
        agent_busy = agent_online and agent.get('phase') == 'running' and agent.get('session') == session
        relay = read(self.run/'native-relay.json')
        native_running = relay.get('boot_id') == BOOT and not (self.run/'native-exit.json').exists()
        result = read(self.run/'ui-agent-turns'/batch/'result.json') if batch else {}
        # A CLI startup failure with no robot-tool records is not a grasp trial.
        action_records = batch and (self.run/'ui-agent-turns'/batch/'compiled-task.json').exists()
        waiting_result = batch and not outcome and result and (result.get('exit_code') == 0 or action_records)
        waiting_memory = batch and outcome and not (self.run/'ui-agent-turns'/batch/'operator-memory.json').exists()
        ended = bool(batch and read(self.run/'ui-agent-turns'/batch/'episode-ended.json'))
        if ended:waiting_result=waiting_memory=False
        # A fault_idle snapshot can remain on disk after the owner process has
        # exited. It must not permanently disable K; the next K creates a
        # fresh owner session after the normal sensor checks.
        owner_recoverable = ((self.run/'native-exit.json').exists() or not owner or
            owner.get('phase') in ('fault_idle', 'owner_exited')) and not native_running
        # In simulation the physics worker owns and starts the simulated body
        # controller before the operator session. K therefore confirms that
        # transport is ready; it must never start a native/physical process.
        start_control_ready = (ready if simulated else sensors_ready and owner_recoverable)
        available = dict(start_control=start_control_ready and not busy,
            prepare=ready and not busy and not enabled and not agent_busy,
            start_harness=ready and not busy and (not enabled or not agent_online) and not agent_busy and not waiting_result and not waiting_memory,
            pause_harness=bool(enabled or agent_busy), stop_control=not (self.run/'native-exit.json').exists() and bool(owner or native_running or busy),
            end_episode=bool(batch and not ended and not busy),
            confirm_success=bool(batch and not outcome and fresh), confirm_failure=bool(batch and not outcome and fresh))
        with self.connect() as db:
            rows = db.execute('SELECT id,payload,state,result FROM requests ORDER BY rowid DESC LIMIT 8').fetchall()
        if not online:
            blocked = '操作服务离线，请恢复大屏服务后再提交。'
        elif not sensors_ready:
            blocked = '机器人相机或遥测未连接，暂不能提交任务。'
        elif owner.get('fault'):
            blocked = '身体控制故障：'+str(owner['fault'])+'；需恢复控制后再提交。'
        elif not ready and not agent_busy:
            blocked = '身体控制尚未就绪，请先启动控制。'
        elif not enabled:
            blocked = '任务调度暂停，请按 P 启动调度后再提交。'
        elif not agent_online:
            blocked = 'Harness 服务已停止，请按 P 重新启动调度。'
        else:
            blocked = ''
        effective_phase = service.get('phase','offline')
        if not busy and owner.get('fault'):
            effective_phase = 'error'
        message = blocked or ('Agent 正在执行任务，可发送补充指令。' if agent_busy else
                             '调度已就绪，请在输入框提交任务。')
        if busy:
            message = service.get('message') or message
        return dict(online=online,transport='simulation' if simulated else 'real', sensors_ready=bool(sensors_ready),epoch=service.get('epoch'), session=session, batch=batch,
                    control_active=bool(not (self.run/'native-exit.json').exists() and (native_running or fresh and owner or busy)),
                    phase=effective_phase, message=message,
                    harness_enabled=bool(enabled and agent_online), harness_online=agent_online,
                    task_submission_allowed=not bool(blocked), task_submission_error=blocked,
                    owner_fault=owner.get('fault'), task_outcome=outcome, task_instruction=task.get('instruction'),
                    available={k: bool(online and v) for k,v in available.items()},
                    requests=[dict(id=r['id'], action=json.loads(r['payload'])['action'], state=r['state'],
                                   result=json.loads(r['result'] or '{}')) for r in rows])

    def submit(self, body):
        if not isinstance(body, dict) or set(body) != {'request_id','action','epoch','session','batch'}:
            raise ValueError('Invalid operator request fields')
        ident, action = body['request_id'], body['action']
        if not isinstance(ident, str) or len(ident) != 32 or any(c not in '0123456789abcdef' for c in ident):
            raise ValueError('request_id must be a fresh UUID hex')
        if action not in ACTIONS:
            raise ValueError('Unknown operator action')
        payload = dict(body, created=time.time(), boot_id=BOOT)
        current = self.snapshot()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT payload,state FROM requests WHERE id=?', (ident,)).fetchone()
            if old:
                saved = json.loads(old['payload'])
                if any(saved.get(k) != v for k,v in body.items()):
                    raise ValueError('request_id cannot be reused for a different action')
                return dict(id=ident, state=old['state'], duplicate=True)
            if (body['epoch'] != current['epoch'] or body['session'] != current['session']
                    or body['batch'] != current['batch'] or not current['available'][action]):
                raise ValueError('状态已变化或操作不可用，请查看最新状态后再点击')
            pending = [json.loads(r[0]) for r in db.execute("SELECT payload FROM requests WHERE state IN ('queued','executing')")]
            if any(p['action']==action or action.startswith('confirm_') and p['action'].startswith('confirm_') for p in pending):
                raise ValueError('该操作正在处理，请等待反馈；不会重复下发')
            db.execute('INSERT INTO requests VALUES (?,?,?,NULL)', (ident,json.dumps(payload), 'queued'))
        return dict(id=ident, state='queued', action=action)

    def claim(self):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            rows = db.execute("SELECT * FROM requests WHERE state='queued' ORDER BY CASE WHEN payload LIKE '%stop_control%' THEN 0 ELSE 1 END,rowid").fetchall()
            for row in rows:
                p = json.loads(row['payload'])
                if p.get('epoch') != self.epoch or p.get('boot_id') != BOOT or not 0 <= time.time()-p['created'] <= 15:
                    db.execute("UPDATE requests SET state='failed',result=? WHERE id=?", (json.dumps(dict(error='请求过期；未执行')),row['id']))
                    continue
                db.execute("UPDATE requests SET state='executing' WHERE id=?", (row['id'],))
                return p

    def finish(self, request, result=None, error=None):
        with self.connect() as db:
            db.execute('UPDATE requests SET state=?,result=? WHERE id=?',
                ('failed' if error else 'done', json.dumps(dict(error=error) if error else result or {}, ensure_ascii=False),request['request_id']))

    def offer_native_stop(self, request, relay_pid):
        with self.connect() as db:
            db.execute('INSERT INTO native_stops VALUES (?,?,?,?,?,0)',
                (request['request_id'],relay_pid,request['session'],self.epoch,time.monotonic()+5))

    def take_native_stop(self, relay_pid):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            rows = db.execute("SELECT n.*,r.payload,r.state FROM native_stops n JOIN requests r ON r.id=n.id WHERE n.consumed=0 AND n.relay_pid=?", (relay_pid,)).fetchall()
            for row in rows:
                p = json.loads(row['payload'])
                if (p.get('action') == 'stop_control' and row['state'] == 'executing'
                        and row['epoch'] == read(self.run/'operator-controls.json').get('epoch')
                        and p.get('boot_id') == BOOT and time.monotonic() <= row['expires']
                        and row['session'] == read(self.run/'ankle_status.json').get('session_id')):
                    db.execute('UPDATE native_stops SET consumed=1 WHERE id=?', (row['id'],))
                    return row['id']


def verified_outcome(run, batch):
    """Only a completed operator request can certify this exact batch; model reports cannot."""
    if not batch:
        return {}
    run = Path(run)
    if not (run/'operator-controls.sqlite3').is_file():
        return {}
    folder = run/'ui-agent-turns'/batch
    result = read(folder/'operator-outcome.json')
    if not result or result.get('batch') != batch or result.get('verification_source') != 'operator_console':
        return {}
    with Operator(run).connect() as db:
        row = db.execute("SELECT payload,state,result FROM requests WHERE id=?", (result.get('request_id'),)).fetchone()
    if not row or row['state'] != 'done' or json.loads(row['result'] or '{}') != result:
        return {}
    p = json.loads(row['payload'])
    manifest = read(folder/'experience-turn.json')
    messages = manifest.get('messages', [])
    if (p.get('action') != 'confirm_'+result.get('outcome', '') or p.get('batch') != batch
            or p.get('session') != result.get('session') or not messages
            or messages[0].get('session') != result.get('session')):
        return {}
    return result
