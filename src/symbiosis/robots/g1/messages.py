"""Durable operator inbox shared by the UI and MCP; never executes commands."""
import json
from pathlib import Path
import sqlite3
import time
from contextlib import contextmanager


class Inbox:
    def __init__(self, directory, clock=time.time):
        self.root = Path(directory)
        self.clock = clock
        self.path = self.root / 'operator-messages.sqlite3'
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS messages (id TEXT PRIMARY KEY, text TEXT, time REAL, status TEXT, reply TEXT, session TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS reader (id INTEGER PRIMARY KEY, seen REAL)')
            db.execute('CREATE TABLE IF NOT EXISTS dispatches (message_id TEXT PRIMARY KEY, batch TEXT, state TEXT)')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    def session(self):
        try:
            return json.loads((self.root / 'ankle_status.json').read_text()).get('session_id')
        except (OSError, ValueError):
            return None

    def send(self, text, request_id):
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 1000:
            raise ValueError('请输入 1–1000 字的消息')
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 100:
            raise ValueError('request_id is required (1–100 characters)')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT text,status FROM messages WHERE id=?', (request_id,)).fetchone()
            if old and old[0] != text.strip():
                raise ValueError('同一 request_id 不能用于不同消息')
            if not old:
                if (self.root/'operator-controls.sqlite3').exists():
                    from .operator import Operator
                    state = Operator(self.root).snapshot()
                    if not state['task_submission_allowed']:
                        raise ValueError(state['task_submission_error'])
                if db.execute("SELECT count(*) FROM messages WHERE status IN ('queued','received')").fetchone()[0] >= 50:
                    raise RuntimeError('待处理消息过多，请等待 Agent 读取后重试')
                db.execute('INSERT INTO messages VALUES (?,?,?,?,?,?)',
                           (request_id, text.strip(), self.clock(), 'queued', '', self.session()))
        state = old[1] if old else 'queued'
        return dict(accepted=True, source='operator_inbox', action=state, message_id=request_id,
                    reply=('已提交，后台 Agent 将自动处理；尚未执行动作。' if self.dispatcher()['online'] else '已保存到 Agent 收件箱，等待读取；尚未执行动作。') if state=='queued'
                    else '此消息此前已提交，本次没有重复发送；请查看消息状态。')

    def snapshot(self, read=False):
        now = self.clock()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            # A stale instruction is visible in history but cannot silently apply to a later trial.
            db.execute("UPDATE messages SET status='expired', reply='消息已过期或试验已切换，请按当前现场重新发送。' WHERE status IN ('queued','received') AND (time < ? OR session IS NOT ?)", (now-120, self.session()))
            if read:
                db.execute('INSERT OR REPLACE INTO reader VALUES (1,?)', (now,))
                db.execute("UPDATE messages SET status='received' WHERE status='queued'")
            rows = db.execute('SELECT id,text,time,status,reply FROM messages ORDER BY time DESC,rowid DESC LIMIT 80').fetchall()
            seen = db.execute('SELECT seen FROM reader WHERE id=1').fetchone()
        messages = [dict(zip(('id','text','time','status','reply'), r)) for r in reversed(rows)]
        dispatcher = self.dispatcher()
        return dict(messages=messages, pending=[m for m in messages if m['status'] in ('queued','received')],
                    agent_connected=bool(seen and 0 <= now-seen[0] <= 5),
                    dispatcher=dispatcher,
                    note='接收/读取不等于执行。后台 Agent 在线时，新任务会自动唤醒执行。')

    def history(self):
        """Return the complete dialogue stored for this run/episode.

        Dispatch freshness remains session-scoped in ``snapshot`` and ``claim``;
        display history is run-scoped so a supervisor reconnect cannot hide
        earlier messages from the same episode. Each run/episode has its own
        control directory, so this does not mix prior rounds into the view.
        """
        with self.connect() as db:
            rows = db.execute(
                'SELECT id,text,time,status,reply FROM messages ORDER BY time,rowid'
            ).fetchall()
        return [dict(zip(('id','text','time','status','reply'), row)) for row in rows]

    def dispatcher(self):
        try:
            value=json.loads((self.root/'ui-agent-status.json').read_text())
            value['online']=0 <= self.clock()-value['updated_at'] <= 5 and value['phase'] in ('idle','running')
            return value
        except (OSError,ValueError,KeyError,TypeError):
            return dict(online=False,phase='offline')

    def claim(self, batch):
        """Durable at-most-once dispatch. A crash never silently replays an action."""
        self.snapshot()  # Expire old/session-mismatched instructions first.
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            rows=db.execute("SELECT m.id,m.text,m.session FROM messages m LEFT JOIN dispatches d ON d.message_id=m.id WHERE m.status IN ('queued','received') AND d.message_id IS NULL AND m.session IS ? AND m.time>=? ORDER BY m.time,m.rowid", (self.session(),self.clock()-120)).fetchall()
            for ident,_,_ in rows:db.execute('INSERT INTO dispatches VALUES (?,?,?)',(ident,batch,'running'))
        return [dict(id=i,text=t,session=s) for i,t,s in rows]

    def finish_dispatch(self, batch, reason):
        with self.connect() as db:
            db.execute("UPDATE messages SET status='failed',reply=? WHERE id IN (SELECT message_id FROM dispatches WHERE batch=?) AND status IN ('queued','received')", (reason,batch))
            db.execute("UPDATE dispatches SET state='finished' WHERE batch=?",(batch,))

    def acknowledge_dispatch(self,batch,reply,*,message_ids=None):
        """Host receipt for goals already included in this episode's model input.

        Never acknowledges later, unclaimed operator instructions. This is
        receipt of the goal, not confirmation of execution or task success.
        """
        if not isinstance(reply,str) or not reply.strip():raise ValueError('Public receipt required')
        allowed=None if message_ids is None else set(message_ids)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            rows=db.execute("SELECT m.id FROM messages m JOIN dispatches d ON d.message_id=m.id "
                "WHERE d.batch=? AND d.state='running' AND m.session IS ? AND m.status IN ('queued','received')",
                (batch,self.session())).fetchall()
            ids=[row[0] for row in rows if allowed is None or row[0] in allowed]
            for ident in ids:
                db.execute("UPDATE messages SET status='acknowledged',reply=? WHERE id=?",(reply.strip(),ident))
        return ids

    def recover_dispatches(self):
        with self.connect() as db:
            batches=[r[0] for r in db.execute("SELECT DISTINCT batch FROM dispatches WHERE state='running'")]
        for batch in batches:self.finish_dispatch(batch,'上次 Agent 会话中断，未自动重放。请观察当前状态后重新发送需要执行的指令。')

    def acknowledge(self, message_id, reply):
        if not isinstance(reply, str) or not 1 <= len(reply.strip()) <= 2000:
            raise ValueError('Provide a public reply (1–2000 characters)')
        self.snapshot()
        with self.connect() as db:
            row = db.execute('SELECT status,reply FROM messages WHERE id=?', (message_id,)).fetchone()
            if not row or row[0] not in ('received','acknowledged'):
                raise ValueError('Read a current message before acknowledging it')
            if row[0] == 'acknowledged' and row[1] != reply.strip():
                raise ValueError('Message already acknowledged with a different reply')
            db.execute("UPDATE messages SET status='acknowledged',reply=? WHERE id=?", (reply.strip(),message_id))
        return dict(message_id=message_id, status='acknowledged', reply=reply.strip(), action_executed=False)

    def events(self):
        events = []
        for m in self.history():
            events.append(dict(id=m['id'], time=m['time'], kind='message', role='user', text=m['text']))
            label = {'queued':'已保存，等待 Agent 读取；尚未执行动作。',
                     'received':'Agent 已读取，等待处理；读取不代表动作完成。'}
            events.append(dict(id=m['id']+':receipt',time=m['time']+.0001,kind='message',role='assistant',
                               text=m['reply'] or label.get(m['status'],m['status'])))
        return events
