import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('launcher', ROOT / 'run_real_vla.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
REPO = m.DEFAULT_REPO


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='gr00t-real-test-')
        self.run = Path(self.temp.name)
        self.base = self.run / 'runtime_source/g1_sonic'
        self.base.mkdir(parents=True)
        backbone=self.run/'backbone';backbone.mkdir();(backbone/'model.safetensors').write_bytes(b'test')
        self.backbone_patch=patch.object(m,'FIXED_BACKBONE',backbone);self.backbone_patch.start()
        for filename in ('vla.py', 'vla_runtime.py', 'ankle_demo.py'):
            shutil.copy2(REPO / 'src/symbiosis/robots/g1' / filename, self.base)
        m.adapt_snapshot(self.run, 16551)
        m.write(self.run / 'vla-instruction.json', dict(instruction='pick up mickey'))
        # Execute the actual adapted job/runtime methods with fake transports.
        tree = ast.parse((self.base / 'vla.py').read_text())
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 and n.name in ('validate', 'VLAJobs')]
        self.ns = dict(re=__import__('re'), time=time, Path=Path, json=json,
                       _capabilities=lambda: {'mickey': 'pick up mickey'},
                       atomic=m.write, validate_hand=Mock())
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'vla_snapshot', 'exec'), self.ns)
        self.runtime = Mock()
        self.runtime.steps = 0
        self.runtime.poll.return_value = (None, False)
        self.jobs = self.ns['VLAJobs'](self.run, 'session', self.runtime)

    def tearDown(self):
        self.backbone_patch.stop()
        self.temp.cleanup()

    def request(self, action, steps=None, now=None):
        req = dict(id=__import__('uuid').uuid4().hex, session_id='session',
                   created_monotonic=time.monotonic() if now is None else now, action=action)
        if action == 'run':
            req.update(task='mickey', steps=steps, evidence='operator i')
        name = 'pause.json' if action == 'pause' else 'pending.json'
        m.write(self.run / 'vla_requests' / name, req)
        return req['id']

    def tick(self, elapsed=0):
        return self.jobs.tick(time.monotonic() + elapsed, True, True, None, math.inf)

    def test_continuous_beyond_old_step_and_time_limits_then_pause(self):
        job = self.request('run', 0)
        self.tick()
        self.runtime.start.assert_called_once_with('pick up mickey', 0)
        self.runtime.steps = 1000000
        self.tick(3600 * 48)
        self.assertEqual(self.jobs.job['id'], job)
        self.assertFalse((self.run / 'vla_results' / (job + '.json')).exists())
        self.request('pause')
        self.tick()
        self.assertIsNone(self.jobs.job)
        self.assertEqual(m.read(self.run / 'vla_results' / (job + '.json'))['reason'], 'agent_pause')
        self.runtime.pause.assert_called()

    def test_bounded_deadline_unchanged(self):
        job = self.request('run', 10)
        self.tick()
        self.tick(100)
        self.assertIsNone(self.jobs.job)
        self.assertIn('deadline', m.read(self.run / 'vla_results' / (job + '.json'))['reason'])

    def test_fault_does_not_replay(self):
        job = self.request('run', 0)
        self.runtime.poll.side_effect = RuntimeError('camera stale')
        self.tick()
        self.assertIsNone(self.jobs.job)
        self.assertIn('camera stale', m.read(self.run / 'vla_results' / (job + '.json'))['reason'])
        self.tick()
        self.runtime.start.assert_called_once()

    def test_stale_and_wrong_session_rejected(self):
        job = self.request('run', 0, time.monotonic() - 10)
        self.tick()
        self.runtime.start.assert_not_called()
        self.assertIn('expired', m.read(self.run / 'vla_results' / (job + '.json'))['reason'])

    def test_runtime_unlimited_and_resume_are_real_not_repeated_chunks(self):
        tree = ast.parse((self.base / 'vla_runtime.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Runtime')
        start = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'start')
        ns = {}
        exec(compile(ast.Module(body=[start], type_ignores=[]), 'runtime', 'exec'), ns)
        fake = Mock()
        ns['start'](fake, 'pick up mickey', 0)
        self.assertEqual(fake.target_steps, math.inf)
        self.assertEqual(fake.steps, 0)
        fake.session.resume.assert_called_once()
        ns['start'](fake, 'pick up mickey', 50)
        self.assertEqual(fake.target_steps, 50)

    def test_pending_native_stop_keeps_owner_and_model(self):
        c = m.Console.__new__(m.Console)
        c.job = None
        c.native = Mock()
        c.native.master = 123
        c.native.alive.return_value = True
        c.native.wait.side_effect = __import__('subprocess').TimeoutExpired('native', 30)
        c.owner = Mock()
        with patch.object(m.os, 'write') as send:
            with self.assertRaisesRegex(RuntimeError, '保留 owner'):
                c.stop_body()
            send.assert_called_once_with(123, b'o')
        c.owner.stop.assert_not_called()

    def test_native_then_owner_then_model_shutdown(self):
        c = m.Console.__new__(m.Console)
        c.run = self.run
        c.job = None
        c.pending = None
        c.owner = Mock()
        c.native = Mock()
        c.native.master = 123
        c.native.wait.return_value = 0
        c.model = Mock()
        c.telemetry = Mock()
        c.pause_vla = Mock()
        order = []
        c.native.wait.side_effect = lambda timeout: order.append('native') or 0
        c.owner.stop.side_effect = lambda: order.append('owner')
        c.telemetry.stop.side_effect = lambda: order.append('telemetry')
        c.model.stop.side_effect = lambda: order.append('model')
        with patch.object(m.os, 'write'):
            c.close()
        self.assertEqual(order, ['native', 'owner', 'telemetry', 'model'])

    def test_repeated_body_cycle_after_native_exited(self):
        c = m.Console.__new__(m.Console)
        c.job = None
        c.native = Mock()
        c.native.alive.return_value = False
        c.owner = Mock()
        c.stop_body()
        self.assertIsNone(c.native)
        self.assertIsNone(c.owner)

    def test_only_new_snapshot_is_changed(self):
        original = (REPO / 'src/symbiosis/robots/g1/vla.py').read_text()
        self.assertIn('not 1 <= steps <= 1500', original)
        self.assertIn('not 0 <= steps <= 1500', (self.base / 'vla.py').read_text())
        with self.assertRaisesRegex(RuntimeError, '接口已变化'):
            m.adapt_snapshot(self.run, 16551)


    def test_tilt_guard_is_45_degrees_only_in_snapshot(self):
        import numpy as np
        tree = ast.parse((self.base / 'ankle_demo.py').read_text())
        guard = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'guard')
        ns = dict(np=np, JOINTS=[dict(lower=-5,upper=5) for _ in range(29)],NAMES=['joint']*29)
        exec(compile(ast.Module(body=[guard],type_ignores=[]),'guard', 'exec'), ns)
        measured=dict(tilt_rad=math.radians(44.99),q=[0]*29,dq=[0]*29)
        ns['guard'](measured, False)
        measured['tilt_rad']=math.pi/4
        ns['guard'](measured, False)
        measured['tilt_rad']=math.radians(45.01)
        with self.assertRaisesRegex(ValueError,'base_tilt_above_45_deg'):
            ns['guard'](measured, False)
        measured['tilt_rad']=0
        measured['q'][0]=6
        with self.assertRaisesRegex(ValueError,'outside model limit'):
            ns['guard'](measured, False)
        self.assertIn("measured['tilt_rad']>.25",(REPO / 'src/symbiosis/robots/g1/ankle_demo.py').read_text())
        self.assertEqual(m.read(self.run / 'standalone-adaptation.json')['base_tilt_limit_deg'],45)

    def test_no_rtc_override_and_staging_leave_checkpoint_unchanged(self):
        self.assertEqual(m.inference_mode(dict(action_horizon=40,rtc_max_delay_steps=10), no_rtc=True), 'chunk')
        self.assertEqual(m.inference_mode(dict(action_horizon=40,rtc_max_delay_steps=4), no_rtc=True), 'chunk')
        bundle = self.checkpoint_fixture()
        original = (bundle / 'config.json').read_bytes()
        stage = self.run / 'plain-staged'
        stage.mkdir()
        with patch.object(m.tempfile, 'mkdtemp', return_value=str(stage)):
            m.stage_model(bundle, self.run / 'plain-server', disable_rtc=True)
        self.assertEqual(m.read(stage / 'config.json')['rtc_max_delay_steps'], 0)
        self.assertEqual((bundle / 'config.json').read_bytes(), original)
        self.assertTrue(m.read(self.run / 'plain-server/staging.json')['rtc_disabled'])

    def test_mode_detection_does_not_fabricate_rtc_training(self):
        self.assertEqual(m.inference_mode(dict(action_horizon=40)), 'chunk')
        self.assertEqual(m.inference_mode(dict(action_horizon=40,rtc_max_delay_steps=10)), 'rtc')
        with self.assertRaises(ValueError):m.inference_mode(dict(action_horizon=40,rtc_max_delay_steps=4))

    def test_plain_runtime_snapshot_preserves_rtc_original(self):
        for filename in ('vla.py', 'vla_runtime.py', 'ankle_demo.py'):
            shutil.copy2(REPO / 'src/symbiosis/robots/g1' / filename, self.base)
        m.adapt_snapshot(self.run, 16551, 'chunk')
        runtime = (self.base / 'vla_runtime.py').read_text()
        self.assertIn('lambda:PlainChunkController(self.client_factory)', runtime)
        self.assertIn('validate_plain_server(self.client,adapter)', runtime)
        ast.parse(runtime)
        ast.parse((self.base / 'plain_chunk.py').read_text())

    def test_plain_chunk_holds_endpoint_and_uses_no_prefix(self):
        import numpy as np
        import threading
        ns = {}
        exec(m.PLAIN_RUNTIME_SOURCE, ns)
        release = threading.Event()
        calls = []
        def actions(value):
            return {k:np.full((1,40,d),value,dtype=np.float32) for k,d in
                [('motion_token',64),('left_hand_joints',6),('right_hand_joints',6)]}
        client = Mock()
        def infer(obs, options):
            calls.append((obs, options))
            if len(calls)>1:release.wait(2)
            return actions(len(calls)), {}
        client.get_action.side_effect = infer
        controller = ns['PlainChunkController'](lambda:client)
        try:
            controller.start('initial')
            for tick in range(40):controller.step('live',tick)
            held=controller.step('boundary',40)
            self.assertTrue(np.all(held['motion_token']==1))
            held=controller.step('still-live',41)
            self.assertTrue(np.all(held['motion_token']==1))
            release.set()
            controller.pending.result(timeout=2)
            next_action=controller.step('next-live',42)
            self.assertTrue(np.all(next_action['motion_token']==2))
            self.assertEqual(calls,[('initial',None),('boundary',None)])
            with self.assertRaises(RuntimeError):controller.step('missed',44)
        finally:
            release.set()
            controller.close()
        self.assertIsNone(controller.actions)
        client.close.assert_called_once()

    def checkpoint_fixture(self):
        bundle = self.run / 'checkpoint-without-shared'
        bundle.mkdir()
        values = {
            'config.json': dict(model_name='/training/machine/base', action_horizon=40, rtc_max_delay_steps=10),
            'processor_config.json': dict(processor_kwargs=dict(model_name='/training/machine/base')),
            'statistics.json': {}, 'embodiment_id.json': {},
            'model.safetensors.index.json': dict(weight_map={'weight': 'model-00001.safetensors'})}
        for name, value in values.items():m.write(bundle / name, value)
        (bundle / 'model-00001.safetensors').write_bytes(b'fake weight')
        return bundle

    def test_fixed_backbone_without_checkpoint_shared_and_no_source_changes(self):
        bundle = self.checkpoint_fixture()
        before = {p.name:p.read_bytes() for p in bundle.iterdir()}
        self.assertEqual(m.model_bundle(bundle), bundle)
        stage = self.run / 'staged'
        stage.mkdir()
        with patch.object(m.tempfile, 'mkdtemp', return_value=str(stage)):
            result = m.stage_model(bundle, self.run / 'server')
        self.assertEqual(result, stage)
        self.assertEqual(m.read(stage / 'config.json')['model_name'], str(m.FIXED_BACKBONE))
        self.assertEqual(m.read(stage / 'processor_config.json')['processor_kwargs']['model_name'], str(m.FIXED_BACKBONE))
        self.assertEqual((stage / 'shared/Cosmos-Reason2-2B').resolve(), m.FIXED_BACKBONE.resolve())
        self.assertTrue((stage / 'model-00001.safetensors').is_symlink())
        self.assertEqual({p.name:p.read_bytes() for p in bundle.iterdir()}, before)
        self.assertFalse((bundle / 'shared').exists())

    def test_corrupt_json_is_named_before_model_loading(self):
        bundle = self.checkpoint_fixture()
        (bundle / 'statistics.json').write_bytes(b'\xff\x00\xab')
        with self.assertRaisesRegex(ValueError, 'statistics.json'):
            m.model_bundle(bundle)

    def test_prepare_request_has_no_run_arguments_and_never_auto_runs(self):
        c = m.Console.__new__(m.Console)
        c.run = self.run
        c.pending = None
        c.native = Mock()
        c.native.alive.return_value = True
        c.job = None
        c.prepare_job = None
        c.state = lambda: dict(session_id='session',phase='ready',updated_monotonic=time.monotonic())
        def publish(source, dest):
            os.link(source, dest)
        c.publish = publish
        c.prepare_vla()
        request = m.read(self.run / 'vla_requests/pending.json')
        self.assertEqual(set(request), {'id','session_id','created_monotonic','action'})
        self.assertEqual(request['action'], 'prepare')
        self.runtime.poll.return_value = (None, True)
        self.tick()
        self.runtime.prepare.assert_called_once_with(None)
        self.runtime.start.assert_not_called()
        result = m.read(self.run / 'vla_results' / (c.prepare_job + '.json'))
        self.assertTrue(result['completed'])

    def test_p_blocked_during_preparation(self):
        c = m.Console.__new__(m.Console)
        c.pending = None
        c.native = Mock()
        c.native.alive.return_value = True
        c.prepare_job = 'preparing'
        c.job = None
        c.submit = Mock()
        with self.assertRaisesRegex(RuntimeError, '准备动作尚未完成'):
            c.toggle_vla()
        c.submit.assert_not_called()

    def test_prepare_blocked_while_vla_running(self):
        c = m.Console.__new__(m.Console)
        c.pending = None
        c.native = Mock()
        c.native.alive.return_value = True
        c.prepare_job = None
        c.job = 'running'
        c.submit = Mock()
        with self.assertRaisesRegex(RuntimeError, '推理中请按 p'):
            c.prepare_vla()
        c.submit.assert_not_called()

    def text_console(self, active=False):
        c = m.Console.__new__(m.Console)
        c.instruction = 'old'
        c.root = self.run
        c.run = self.run
        c.job = 'active-job' if active else None
        c.pause_vla = Mock()
        c.start_vla = Mock()
        return c

    def test_changed_text_pauses_then_restarts_once(self):
        c = self.text_console(active=True)
        calls = []
        c.pause_vla.side_effect = lambda: calls.append('pause')
        c.start_vla.side_effect = lambda: calls.append('resume:' + m.read(self.run / 'vla-instruction.json')['instruction'])
        c.set_instruction('Pick up 红色物体')
        self.assertEqual(calls, ['pause', 'resume:Pick up 红色物体'])
        self.assertEqual(c.instruction, 'Pick up 红色物体')

    def test_idle_text_does_not_start_motion(self):
        c = self.text_console()
        c.set_instruction('new')
        c.pause_vla.assert_not_called()
        c.start_vla.assert_not_called()
        self.assertEqual(m.read(self.run / 'vla-instruction.json')['instruction'], 'new')

    def test_pause_failure_preserves_old_instruction(self):
        c = self.text_console(active=True)
        c.pause_vla.side_effect = RuntimeError('no ack')
        with self.assertRaises(RuntimeError):
            c.set_instruction('new')
        self.assertEqual(c.instruction, 'old')
        c.start_vla.assert_not_called()

    def test_text_after_fault_does_not_resume(self):
        c = self.text_console(active=True)
        m.write(self.run / 'vla_results/active-job.json', dict(reason='camera stale'))
        c.set_instruction('new')
        c.start_vla.assert_not_called()

    def test_new_text_reaches_actual_owner_runtime_on_next_start(self):
        self.request('run', 0)
        self.tick()
        self.request('pause')
        self.tick()
        m.write(self.run / 'vla-instruction.json', dict(instruction='New instruction 中文'))
        self.request('run', 0)
        self.tick()
        self.runtime.start.assert_called_with('New instruction 中文', 0)
        self.assertEqual(self.runtime.start.call_count, 2)

    def test_actual_keyboard_loop_and_fault_remains_responsive(self):
        # Separate process with a real PTY; only action methods are mocked.
        child = self.run / 'keyboard_fixture.py'
        child.write_text("""import importlib.util,sys
spec=importlib.util.spec_from_file_location('launcher',sys.argv[1])
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
c=m.Console.__new__(m.Console)
c.owner=c.native=None;c.reported_tick_error=None;c.instruction='old';c.job=None
c.tick=lambda:(_ for _ in ()).throw(RuntimeError('fake fault'))
def event(value):
 with open(sys.argv[2],'a') as f:f.write(value+'\\n')
def start():
 c.owner=True;event('k-on')
def stop():
 c.owner=None;event('k-off')
c.start_body=start;c.stop_body=stop
def start_vla():
 c.job='running';event('p-on')
def pause_vla():
 c.job=None;event('pause')
c.start_vla=start_vla
c.pause_vla=pause_vla
c.prepare_vla=lambda:event('i-prepare')
c.close=lambda:event('q')
c.set_instruction=lambda text:event('text:'+text)
c.loop()
""")
        result = self.run / 'keys.txt'
        proc = m.Process([sys.executable, child, ROOT / 'run_real_vla.py', result],
                         os.environ.copy(), self.run, self.run / 'keys.log', terminal=True)
        try:
            until = time.monotonic() + 3
            while '按键无需回车'.encode() not in (self.run / 'keys.log').read_bytes() and time.monotonic() < until:
                time.sleep(.02)
            time.sleep(.05)
            os.write(proc.master, 'kipptPick up 红色 Q\nokq'.encode('utf-8'))
            self.assertEqual(proc.wait(5), 0)
            self.assertEqual(result.read_text().splitlines(), ['k-on', 'i-prepare', 'p-on', 'pause', 'text:Pick up 红色 Q', 'k-off', 'q'])
        finally:
            if proc.alive():proc.stop()

    def test_process_pty_receives_stop_key(self):
        p = m.Process([sys.executable, '-u', '-c',
            'import os,tty;tty.setraw(0);print("READY",flush=True);raise SystemExit(0 if os.read(0,1)==b"o" else 1)'],
            os.environ.copy(), self.run, self.run / 'fake-native.log', terminal=True)
        try:
            until = time.monotonic() + 3
            while b'READY' not in (self.run / 'fake-native.log').read_bytes() and time.monotonic() < until:
                time.sleep(.02)
            os.write(p.master, b'o')
            self.assertEqual(p.wait(3), 0)
        finally:
            if p.alive():
                p.stop()


if __name__ == '__main__':
    unittest.main()
