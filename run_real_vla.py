#!/usr/bin/env python3
"""Standalone G1/GR00T RTC console. Only stdlib required by this launcher.

python3 run_real_vla.py /absolute/path/to/model_bundle
Keys (no Enter): k body on/off; i prepare; p toggle continuous VLA; t edit instruction; q shut down.
Startup opens a dedicated first-person camera window. Each inference run records
a separate MP4; pause, fault and q finalize the video. Keep vla_camera_view.py
beside this launcher.
--no-rtc forces standard 40-step chunks with endpoint hold during inference.
--check validates files/software only: no model, sensors or robot processes.
"""
import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pty
import select
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import termios
import threading
import time
import tty
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_config import ROOT, SOURCE_PATHS, bootstrap
bootstrap()
DEFAULT_REPO = ROOT
CAMERA_HELPER = Path(__file__).resolve().with_name('vla_camera_view.py')
from symbiosis.common.deployment import external_path
FIXED_BACKBONE = external_path('backbone_path')


PLAIN_RUNTIME_SOURCE = (DEFAULT_REPO/'runtime/plain_chunk.py').read_text()


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def write(path, value):
    path = Path(path)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def replace_once(path, before, after):
    value = path.read_text()
    if value.count(before) != 1:
        raise RuntimeError(f'上游接口已变化，拒绝套用运行适配：{path.name}: {before!r}')
    path.write_text(value.replace(before, after))


def adapt_snapshot(run, port, mode="rtc"):
    """Zero steps is unlimited ONLY in this run's private owner snapshot."""
    base = run / 'runtime_source/g1_sonic'
    replace_once(base / 'ankle_demo.py',
        "if measured['tilt_rad']>.25:raise ValueError('base_tilt_above_0.25_rad')",
        "if measured['tilt_rad']>np.pi/4:raise ValueError('base_tilt_above_45_deg')")
    replace_once(base / 'vla.py', 'not 1 <= steps <= 1500', 'not 0 <= steps <= 1500')
    replace_once(base / 'vla.py', "now-self.at > self.job.get('steps',0)/50+15:",
                 "not (self.job['action'] == 'run' and self.job.get('steps') == 0) and now-self.at > self.job.get('steps',0)/50+15:")
    replace_once(base / 'vla.py', "self.runtime.start(_capabilities()[request['task']],request['steps'])",
                 "self.runtime.start(json.loads((self.root/'vla-instruction.json').read_text())['instruction'],request['steps'])")
    replace_once(base / 'vla_runtime.py', "host='127.0.0.1',port=5550,", f"host='127.0.0.1',port={port},")
    replace_once(base / 'vla_runtime.py', 'self.target_steps=steps;self.prompt=prompt',
                 "self.target_steps=float('inf') if steps == 0 else steps;self.prompt=prompt")
    if mode == 'chunk':
        (base / 'plain_chunk.py').write_text(PLAIN_RUNTIME_SOURCE)
        runtime_path = base / 'vla_runtime.py'
        replace_once(runtime_path, 'from .vla import DEFAULT_MODEL, DEFAULT_RTC',
                     'from .vla import DEFAULT_MODEL, DEFAULT_RTC\nfrom .plain_chunk import PlainChunkController, validate_plain_server')
        replace_once(runtime_path,
            "            interval=request_interval(self.client.get_rtc_config())\n            self.rtc_config=rtc.validate_server(self.client,adapter,rtc.Config(rtc_request_interval=interval))",
            "            self.rtc_config=validate_plain_server(self.client,adapter)")
        replace_once(runtime_path, 'try:self.request_interval=request_interval(client.rtc_config)',
                     'try:self.request_interval=40')
        replace_once(runtime_path,
            'lambda:rtc.AsyncRTCController(self.client_factory,request_interval=self.request_interval)',
            'lambda:PlainChunkController(self.client_factory)')
    write(run / 'standalone-adaptation.json', dict(
        unlimited_steps_sentinel=0, model_port=port, inference_mode=mode, base_tilt_limit_deg=45,
        description='Private snapshot: unlimited VLA until explicit pause or existing fault checks; mode selects RTC or standard chunks.',
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (base / 'vla.py', base / 'vla_runtime.py', base / 'ankle_demo.py')}))


def model_bundle(path, no_rtc=False):
    path = Path(path).expanduser().resolve()
    if not (path / 'config.json').is_file() and (path / path.name / 'config.json').is_file():
        path /= path.name
    names = ['config.json', 'processor_config.json', 'statistics.json', 'embodiment_id.json',
             'model.safetensors.index.json']
    missing = [name for name in names if not (path / name).is_file()]
    if missing:
        raise ValueError('checkpoint 缺少：' + ', '.join(missing))
    invalid = []
    values = {}
    for name in names:
        try:
            values[name] = json.loads((path / name).read_text(encoding='utf-8'))
        except (UnicodeError, ValueError) as exc:
            invalid.append(name)
    if invalid:
        raise ValueError('以下 checkpoint 文件不是可读取的 UTF-8 JSON，请核对导出或拷贝：' + ', '.join(invalid))
    if not (FIXED_BACKBONE / 'model.safetensors').is_file():
        raise ValueError('固定基础模型不存在：' + str(FIXED_BACKBONE))
    index = values['model.safetensors.index.json']
    names += sorted(set(index.get('weight_map', {}).values()))
    missing = [name for name in names if not (path / name).is_file()]
    if missing or not index.get('weight_map'):
        raise ValueError('需要完整 GR00T 模型目录，缺少：' + ', '.join(missing or ['weight_map']))
    config = read(path / 'config.json')
    inference_mode(config, no_rtc=no_rtc)
    return path


def inference_mode(config, no_rtc=False):
    if config.get('action_horizon') != 40:
        raise ValueError('当前控制接口要求 action_horizon=40')
    if no_rtc:
        return 'chunk'
    maximum = config.get('rtc_max_delay_steps', 0)
    if type(maximum) is not int or not 0 <= maximum < 40:
        raise ValueError('rtc_max_delay_steps 必须是 0～39 的整数；缺省按普通 GR00T')
    if maximum == 0:
        return 'chunk'
    if maximum < 8:
        raise ValueError('此 RTC 权重训练延迟不足 8 步，不能使用固定 delay=8 的 RTC 部署')
    return 'rtc'


def stage_model(bundle, output, disable_rtc=False):
    """Stage small configs on ext4; preserve the delivered checkpoint unchanged."""
    bundle = Path(bundle).resolve()
    output = Path(output)
    local = Path(tempfile.mkdtemp(prefix='gr00t-fixed-backbone-', dir='/tmp'))
    for source in bundle.iterdir():
        if source.name in ('config.json', 'processor_config.json'):
            config = json.loads(source.read_text(encoding='utf-8'))
            if source.name == 'config.json':
                config['model_name'] = str(FIXED_BACKBONE)
                if disable_rtc:
                    config['rtc_max_delay_steps'] = 0
            else:
                config['processor_kwargs']['model_name'] = str(FIXED_BACKBONE)
            write(local / source.name, config)
        elif source.name != 'shared':
            (local / source.name).symlink_to(source, target_is_directory=source.is_dir())
    # Also provide the conventional relative layout, pointing to the SAME fixed base.
    (local / 'shared').mkdir()
    (local / 'shared/Cosmos-Reason2-2B').symlink_to(FIXED_BACKBONE, target_is_directory=True)
    output.mkdir(parents=True, exist_ok=True)
    write(output / 'staging.json', dict(model_bundle=str(bundle), staged_model=str(local),
                                      fixed_backbone=str(FIXED_BACKBONE), rtc_disabled=disable_rtc))
    return local


def serve_model():
    """Reuse the repository's service, overriding only its staging function."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--serve-model', action='store_true')
    parser.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    parser.add_argument('--mode', choices=('rtc', 'chunk'), default='rtc')
    args, service_args = parser.parse_known_args()
    sys.path[:0] = [str(args.repo / 'src'), str(args.repo)]
    if args.mode == 'chunk':
        serve_plain_model(args.repo, service_args)
        return
    from symbiosis.operations import g1_vla_server as service
    service.prepare = lambda bundle, rtc_repo, output: stage_model(bundle, output)
    sys.argv = [str(args.repo / 'scripts/g1_vla_server.py'), *service_args]
    service.main()


def serve_plain_model(repo, arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--rtc-delay-steps')  # Shared launch argument, unused in plain mode.
    parser.add_argument('--cpu-threads', type=int, default=1)
    args = parser.parse_args(arguments)
    local = stage_model(args.model_path, args.output, disable_rtc=True)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', NO_ALBUMENTATIONS_UPDATE='1')
    from symbiosis.common.deployment import external_path
    sys.path.insert(0, str(external_path('rtc_root')))
    import torch
    torch.set_num_threads(args.cpu_threads)
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.server_client import PolicyServer
    from symbiosis.robots.g1.vla_runtime import identity
    from symbiosis.operations.run_timing import Journal
    selected = identity(args.model_path)
    write(args.output / 'identity.json', selected)
    policy = Gr00tPolicy('UNITREE_G1_SONIC', str(local), device='cuda',
                         sonic_token_mode='continuous', rtc_delay_steps=None)
    journal = Journal(args.output / 'timing/vla-server.jsonl')
    def action(*a, **kw):
        with journal.span('plain_gr00t_get_action', measurement='server wall: preprocessing + inference + postprocessing'):
            return policy.get_action(*a, **kw)
    with PolicyServer(policy, host='127.0.0.1', port=args.port) as server:
        server.register_endpoint('get_action', action)
        server.register_endpoint('get_rax_identity', lambda:selected, requires_input=False)
        print(f'Standard GR00T ready: 40-step chunks, no RTC prefix, localhost:{args.port}', flush=True)
        server.run()


def load_config(repo):
    if Path(repo).resolve() != ROOT:
        raise ValueError('This repository is standalone; external --repo is not supported')
    from symbiosis.common.deployment import load
    return load()


def occupied(port):
    with socket.socket() as probe:
        return probe.connect_ex(('127.0.0.1', port)) == 0


def control_conflicts():
    conflicts = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            args = [s.decode(errors='replace') for s in (p / 'cmdline').read_bytes().split(b'\0') if s]
        except OSError:
            continue
        if args and (Path(args[0]).name == 'g1_field' or
                     any(s in args for s in ('g1_sonic.ankle_executor', 'g1_sonic.executor', 'g1_sonic.commission'))):
            conflicts.append(int(p.name))
    if conflicts or occupied(17586) or occupied(17587):
        raise RuntimeError(f'已有机器人控制或端口占用（PID {conflicts}）；不会关闭已有控制')


class Process:
    def __init__(self, command, env, cwd, log, terminal=False):
        self.master = None
        self.output = Path(log).open('ab', buffering=0)
        if terminal:
            master, slave = pty.openpty()
            self.master = master
            # Native script creates its own terminal; only this private PTY receives O.
            try:
                self.proc = subprocess.Popen([str(x) for x in command], env=env, cwd=cwd,
                    stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
            finally:
                os.close(slave)
            self.thread = threading.Thread(target=self.drain, daemon=True)
            self.thread.start()
        else:
            self.proc = subprocess.Popen([str(x) for x in command], env=env, cwd=cwd,
                stdin=subprocess.DEVNULL, stdout=self.output, stderr=subprocess.STDOUT, start_new_session=True)

    def drain(self):
        try:
            while True:
                data = os.read(self.master, 65536)
                if not data:
                    break
                self.output.write(data)
        except OSError:
            pass

    def alive(self):
        return self.proc.poll() is None

    def wait(self, seconds):
        result = self.proc.wait(timeout=seconds)
        if self.master is not None:
            self.thread.join(timeout=2)
            os.close(self.master)
            self.master = None
        self.output.close()
        return result

    def stop(self):
        if self.alive():
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                return self.wait(15)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        return self.wait(5)


class Console:
    def __init__(self, args, cfg):
        self.args, self.cfg = args, cfg
        self.repo = args.repo
        self.paths = cfg['external_paths']
        self.mode = inference_mode(read(args.model / 'config.json'), no_rtc=getattr(args, 'no_rtc', False))
        self.root = Path(self.paths['artifact_root']) / 'runs' / (
            'standalone-vla-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
        self.root.mkdir(parents=True)
        cfg['external_paths']['vla_model'] = str(args.model)
        self.deployment = self.root / 'deployment.json'
        write(self.deployment, cfg)
        self.env = os.environ | dict(GROOT_REAL_DEPLOYMENT_FILE=str(self.deployment),
            PYTHONPATH=os.pathsep.join(str(p) for p in [*SOURCE_PATHS,self.repo]), PYTHONDONTWRITEBYTECODE='1')
        # Staging must be on a filesystem supporting symlinks; isolate same-named bundles.
        self.env['G1_MODEL_CACHE'] = str(Path('/tmp') / ('g1-vla-staging-' + self.root.name))
        self.model = self.owner = self.native = self.telemetry = None
        self.camera = self.camera_window = None
        self.recording = None
        self.camera_warning = None
        self.run = None
        self.cycle = 0
        self.job = None
        self.prepare_job = None
        self.instruction = args.instruction
        self.pending = None
        self.body_start_at = None
        self.reported_fault = None
        self.reported_tick_error = None
        spec = importlib.util.spec_from_file_location('standalone_mailbox', self.repo / 'src/symbiosis/robots/g1/mailbox.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.publish = module.publish
        spec = importlib.util.spec_from_file_location('standalone_native_checks', self.repo / 'src/symbiosis/operations/g1_vla_trial_terminal.py')
        checks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checks)
        self.health_reason = checks.health_reason

    def python(self, key, code, *arguments):
        result = subprocess.run([self.paths[key], '-c', code, *map(str, arguments)],
            env=self.env, cwd=self.repo, text=True, capture_output=True, timeout=180)
        if result.returncode:
            raise RuntimeError((result.stdout + result.stderr)[-5000:])
        return result.stdout

    def start_camera(self):
        self.camera = Process([self.paths['inference_python'], CAMERA_HELPER,
            '--root', self.root, '--repo', self.repo], self.env, self.repo, self.root / 'camera.log')
        deadline = time.monotonic() + 15
        while not (self.root / 'camera-viewer.json').is_file():
            if not self.camera.alive():
                raise RuntimeError('相机服务已退出，见 camera.log')
            if time.monotonic() > deadline:
                raise TimeoutError('相机视口启动超时，见 camera.log')
            time.sleep(.1)
        url = read(self.root / 'camera-viewer.json')['url']
        profile = tempfile.mkdtemp(prefix='g1-vla-camera-window-', dir='/tmp')
        chrome = shutil.which('google-chrome') or shutil.which('chromium') or shutil.which('chromium-browser')
        if chrome:
            command = [chrome, '--user-data-dir=' + profile, '--no-first-run',
                '--no-default-browser-check', '--disable-background-mode', '--disable-gpu',
                '--disable-dev-shm-usage', '--window-size=1000,700', '--app=' + url]
        else:
            command = [shutil.which('firefox'), '--no-remote', '--profile', profile, '--new-window', url]
        self.camera_window = Process(command, self.env, self.repo, self.root / 'camera-window.log')
        time.sleep(.3)
        if not self.camera_window.alive():
            raise RuntimeError('相机窗口未能打开，见 camera-window.log')
        print('已打开机器人相机第一视角窗口；请在本终端输入 k/i/p/t/q。', flush=True)
        print('视频将自动保存在：' + str(self.root / 'recordings'), flush=True)

    def camera_request(self, action, **values):
        if not getattr(self, 'camera', None) or not self.camera.alive():
            raise RuntimeError('相机录像服务未运行，见 camera.log')
        request = dict(id=uuid.uuid4().hex, action=action, **values)
        write(self.root / 'camera-command.json', request)
        deadline = time.monotonic() + (30 if action != 'start' else 5)
        while time.monotonic() < deadline:
            state = read(self.root / 'camera-state.json')
            if state.get('ack') == request['id']:
                if state.get('command_error'):
                    raise RuntimeError(state['command_error'])
                return state
            if not self.camera.alive():
                raise RuntimeError('相机录像服务已退出，见 camera.log')
            time.sleep(.03)
        raise TimeoutError('相机录像请求未确认：' + action)

    def start_recording(self):
        if getattr(self, 'recording', None):
            raise RuntimeError('上一段录像尚未结束')
        folder = self.root / 'recordings' / ('vla-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
        self.recording = folder
        try:
            self.camera_request('start', folder=str(folder))
        except Exception:
            self.stop_recording('recording_start_failed')
            raise
        print('开始录像：' + str(folder / 'camera.mp4'), flush=True)

    def stop_recording(self, reason):
        if not getattr(self, 'recording', None):
            return
        folder = self.recording
        try:
            self.camera_request('stop', reason=reason)
            result = read(folder / 'recording.json')
            if result.get('status') != 'saved':
                raise RuntimeError(result.get('error') or '录像未确认保存，见 recording.json')
            print('视频已保存：' + str(folder / 'camera.mp4'), flush=True)
        except Exception as exc:
            print('录像保存异常：' + str(exc) + '；保留已有视频及记录：' + str(folder), flush=True)
        finally:
            self.recording = None

    def close_camera(self):
        if getattr(self, 'camera', None):
            try:
                if self.camera.alive():
                    self.camera_request('shutdown', reason='operator_q')
                    self.camera.wait(10)
                else:
                    self.camera.wait(2)
            except Exception as exc:
                print('相机服务关闭：' + str(exc), flush=True)
                self.camera.stop()
            self.camera = None
        if getattr(self, 'camera_window', None):
            self.camera_window.stop()
            self.camera_window = None

    def start_model(self):
        if occupied(self.args.port):
            raise RuntimeError(f'端口 {self.args.port} 已占用；请使用 --port 选择空闲端口')
        self.model = Process([self.paths['rtc_python'], Path(__file__).resolve(), '--serve-model', '--repo', self.repo, '--mode', self.mode,
            '--model-path', self.args.model, '--port', self.args.port, '--rtc-delay-steps', '8',
            '--cpu-threads', '1', '--output', self.root / 'vla-model'], self.env, self.repo, self.root / 'model.log')
        print(f'推理模式：{self.mode}；固定基础模型：{FIXED_BACKBONE}', flush=True)
        print(f'正在加载权重；日志：{self.root / "model.log"}', flush=True)
        deadline = time.monotonic() + 600
        while not occupied(self.args.port):
            if not self.model.alive():
                raise RuntimeError('VLA 服务已退出：' + str(self.root / 'model.log'))
            if time.monotonic() > deadline:
                raise TimeoutError('模型加载超时；见 model.log')
            time.sleep(.2)
        self.python('inference_python', '''import sys
from symbiosis.robots.g1.vla_runtime import dependencies, identity, request_interval
from symbiosis.robots.g1.vla import DEFAULT_MODEL
rtc=dependencies(); c=rtc.PolicyClient('127.0.0.1',int(sys.argv[1]),timeout_ms=5000)
try:
 assert c.call_endpoint('get_rax_identity',requires_input=False)==identity(DEFAULT_MODEL), '模型身份不匹配'
 cfg=c.get_rtc_config()
 adapter=rtc.legacy.BraincoModelAdapter(DEFAULT_MODEL,None)
 if sys.argv[2]=='rtc':
  assert cfg['rtc_delay_steps']==8 and request_interval(cfg)==30, 'RTC 配置不匹配'
  rtc.validate_server(c,adapter,rtc.Config(rtc_request_interval=30))
 else:
  assert cfg['rtc_delay_steps'] is None and cfg['action_horizon']==40, '普通 GR00T 配置不匹配'
  remote=c.get_modality_config()
  for mode,expected in adapter.modalities.items():
   for field in ('modality_keys','delta_indices'):
    actual=remote[mode]
    value=actual[field] if isinstance(actual,dict) else getattr(actual,field)
    assert list(value)==list(expected[field]), '输入输出布局不匹配'
finally: c.close()
''', self.args.port, self.mode)
        print(f'VLA 服务就绪：127.0.0.1:{self.args.port}\n记录目录：{self.root}', flush=True)

    def start_telemetry(self):
        self.telemetry = Process([self.paths['telemetry_python'], '-m', 'symbiosis.web.console.telemetry',
            '--interface', 'eno1', '--output', self.root / 'telemetry.json'],
            self.env, self.repo, self.root / 'telemetry.log')

    def start_body(self):
        control_conflicts()
        self.cycle += 1
        self.run = self.root / f'control-{self.cycle:03d}'
        self.python('data_python', '''import sys
from pathlib import Path
from symbiosis.operations.g1_prepare_native import prepare
from symbiosis.task_runtime import default_bundle
prepare(Path(sys.argv[1]),vla=True,task_bundle=default_bundle())
''', self.run)
        adapt_snapshot(self.run, self.args.port, self.mode)
        snapshot = self.run / 'runtime_source/g1_sonic/deployment.json'
        deployment = read(snapshot)
        if self.instruction:
            deployment['task_capabilities']['standalone'] = self.instruction
            self.task = 'standalone'
        else:
            self.task = self.args.task
            if self.task not in deployment['task_capabilities']:
                raise ValueError('任务不存在，可用任务：' + ', '.join(deployment['task_capabilities']))
        self.instruction = self.instruction or deployment['task_capabilities'][self.task]
        write(self.run / 'vla-instruction.json', dict(instruction=self.instruction))
        write(snapshot, deployment)
        manifest_path = self.run / 'manifest.json'
        manifest = read(manifest_path)
        manifest['upstream_owner_source_sha256'] = manifest['owner_source_sha256']
        manifest['owner_source_sha256'] = {str(p.relative_to(self.run / 'runtime_source')):
            hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((self.run / 'runtime_source').rglob('*')) if p.is_file()}
        manifest['standalone_continuous_vla'] = dict(port=self.args.port, task=self.task, steps=0)
        write(manifest_path, manifest)
        # Use the prepared exact owner command, without its unrelated terminal cleanup wrapper.
        command = read(self.run / 'manifest.json')['owner']
        self.owner = Process(command, self.env, self.repo, self.run / 'owner.log')
        self.pending = 'owner'
        self.body_start_at = time.monotonic()
        self.reported_fault = None
        print('当前 VLA 文本：' + self.instruction, flush=True)
        print('身体倾角保护阈值：45°。', flush=True)
        print('正在启动身体 owner 和原生控制；等待 ready。', flush=True)

    def state(self):
        return read(self.run / 'ankle_status.json') if self.run else {}

    def submit(self, action):
        from_path = self.run / 'vla_requests'
        state = self.state()
        if not 0 <= time.monotonic() - state.get('updated_monotonic', -1e9) <= .25:
            raise RuntimeError('身体控制反馈过期')
        if action in ('run', 'prepare') and (state.get('phase') != 'ready' or state.get('fault')):
            raise RuntimeError('身体控制未 ready 或存在故障')
        request = dict(id=uuid.uuid4().hex, session_id=state['session_id'],
                       created_monotonic=time.monotonic(), action=action)
        if action in ('run', 'prepare'):
            if state.get('active_job') or (state.get('vla') or {}).get('active_job'):
                raise RuntimeError('已有动作；推理中请按 p 停止，准备动作请等待完成')
        if action == 'run':
            request.update(task=self.task, steps=0, evidence='Operator pressed p in the standalone continuous VLA console.')
        temporary = from_path / (request['id'] + '.json')
        write(temporary, request)
        # Reuse the exact exFAT-compatible no-overwrite mailbox implementation.
        try:
            self.publish(temporary, from_path / ('pause.json' if action == 'pause' else 'pending.json'))
        finally:
            temporary.unlink(missing_ok=True)
        return request['id']

    def start_vla(self):
        if self.pending or not self.native or not self.native.alive():
            raise RuntimeError('先按 k 并等待身体控制 ready')
        if getattr(self, 'prepare_job', None):
            raise RuntimeError('准备动作尚未完成；等待完成后再按 p')
        if self.job:
            print('VLA 已在运行。', flush=True)
            return
        self.start_recording()
        try:
            self.job = self.submit('run')
        except Exception:
            self.stop_recording('vla_submit_failed')
            raise
        try:
            write(self.recording / 'vla-job.json', dict(job_id=self.job, instruction=self.instruction,
                control_directory=str(self.run), inference_mode=self.mode))
        except OSError as exc:
            print('录像任务索引保存异常：' + str(exc), flush=True)
        print('已提交持续 VLA：无步数或总时长上限；再按 p 暂停。', flush=True)

    def prepare_vla(self):
        if self.pending or not self.native or not self.native.alive():
            raise RuntimeError('先按 k 并等待身体控制 ready')
        if self.job or getattr(self, 'prepare_job', None):
            raise RuntimeError('已有 VLA 或准备动作；推理中请按 p 停止，准备动作请等待完成')
        self.prepare_job = self.submit('prepare')
        print('已提交准备动作：约 6 秒过渡至初始姿态并张开双手；完成后按 p 开始 VLA。', flush=True)

    def toggle_vla(self):
        if self.job:
            self.pause_vla()
        else:
            self.start_vla()

    def set_instruction(self, instruction):
        instruction = instruction.strip()
        if not 1 <= len(instruction) <= 2000:
            raise ValueError('VLA 文本必须为 1～2000 个字符')
        if instruction == self.instruction:
            print('文本未变化。', flush=True)
            return
        # Only a still-active job is resumed. Editing after a fault never restarts it.
        running = bool(self.job and self.run and
            not (self.run / 'vla_results' / (self.job + '.json')).exists())
        if running:
            self.pause_vla()  # Require owner acknowledgement before changing the prompt.
        self.instruction = instruction
        value = dict(instruction=instruction, changed_unix=time.time(), restart_requested=running)
        write(self.root / 'current-instruction.json', value)
        if self.run:
            write(self.run / 'vla-instruction.json', value)
        with (self.root / 'instruction-history.jsonl').open('a') as log:
            log.write(json.dumps(value, ensure_ascii=False) + '\n')
        print('VLA 文本已更新：' + instruction, flush=True)
        if running:
            self.start_vla()  # New observation / new RTC queue; keep the model loaded.

    def pause_vla(self):
        if not self.owner or not self.owner.alive():
            self.stop_recording('owner_unavailable')
            return
        job = self.submit('pause')
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            result = read(self.run / 'vla_results' / (job + '.json'))
            if result:
                if not result.get('completed'):
                    raise RuntimeError('VLA 暂停被拒绝：' + str(result))
                self.job = None
                self.prepare_job = None
                self.stop_recording('operator_pause')
                print('VLA 已退出，RTC 队列已清空；身体控制保持。', flush=True)
                return
            time.sleep(.05)
        raise TimeoutError('VLA 暂停尚未确认；保留身体控制，查看 owner.log')

    def stop_body(self):
        if self.job or getattr(self, 'prepare_job', None):
            raise RuntimeError('先结束当前动作（推理按 p，准备动作等待完成），再按 k 关闭身体控制')
        if self.native and self.native.alive():
            os.write(self.native.master, b'o')
            print('已按本次 k/q 请求向本轮原生终端发送 O，等待退出。', flush=True)
            try:
                code = self.native.wait(30)
            except subprocess.TimeoutExpired:
                raise RuntimeError('原生 O 尚未确认退出；保留 owner/模型，请检查 native.log 后重试 k/q')
            write(self.run / 'native-exit.json', dict(exit_code=code, reason='standalone_operator_stop'))
            self.native = None
        elif self.native:
            self.native.wait(2)
            self.native = None
        if self.owner:
            self.owner.stop()
            self.owner = None
        self.pending = None
        print('身体控制程序已关闭；k 可新建控制会话。', flush=True)

    def tick(self):
        if getattr(self, 'camera', None) and not self.camera.alive() and not self.camera_warning:
            self.camera_warning = '相机／录像服务已退出；控制保持原状，见 camera.log'
            print(self.camera_warning, flush=True)
        if self.model and not self.model.alive():
            if self.job:
                with contextlib.suppress(Exception):
                    self.pause_vla()
            self.stop_recording('model_service_exit')
            raise RuntimeError('模型服务已退出；请按 q 结束，见 model.log')
        state = self.state()
        if self.owner and not self.owner.alive():
            self.stop_recording('owner_exit')
            self.job = None
            self.prepare_job = None
            if self.pending:
                self.pending = None
            raise RuntimeError('身体 owner 已退出，请按 k 清理本轮后再重新启动；见 owner.log')
        if self.pending and time.monotonic() - self.body_start_at > 120:
            self.pending = None
            print('身体启动尚未 ready；不会自动重试。按 k 关闭本轮，查看 owner.log/native.log。', flush=True)
        if self.pending == 'owner' and state.get('session_id') and not self.health_reason(
                read(self.root / 'telemetry.json'), time.monotonic()):
            self.native = Process(['bash', self.run / 'controller.sh'], self.env, self.repo,
                                  self.run / 'native.log', terminal=True)
            self.pending = 'native'
        if self.pending == 'native' and state.get('phase') == 'ready' and not state.get('fault'):
            self.pending = None
            print('身体控制 ready。按 i 准备，完成后按 p 开始持续 VLA。', flush=True)
        if state.get('fault') and state['fault'] != self.reported_fault:
            self.reported_fault = state['fault']
            print('控制器反馈：' + str(state['fault']) + '；不会自动恢复 VLA。', flush=True)
        if getattr(self, 'prepare_job', None):
            result = read(self.run / 'vla_results' / (self.prepare_job + '.json'))
            if result:
                self.prepare_job = None
                message = '准备动作完成；按 p 开始持续 VLA。' if result.get('completed') else '准备动作未完成：' + str(result.get('reason'))
                print(message, flush=True)
        if self.job:
            result = read(self.run / 'vla_results' / (self.job + '.json'))
            if result:
                self.job = None
                self.stop_recording('vla_result: ' + str(result.get('reason')))
                print('VLA 已停止：' + str(result.get('reason')) + '；需要再次按 p 才会重启。', flush=True)

    def close(self):
        # q, Ctrl-C and termination signals explicitly request the same orderly shutdown.
        if self.owner and self.owner.alive():
            try:
                self.pause_vla()
            except Exception as exc:
                print('暂停未确认：' + str(exc) + '；继续执行本次明确请求的原生 O。', flush=True)
                self.job = None
        self.stop_recording('operator_q')
        self.job = None  # q explicitly authorizes native O even if the owner has died.
        self.prepare_job = None
        self.stop_body()  # Never SIGKILL native / remove its owner before confirmed native exit.
        self.close_camera()
        if self.telemetry:
            self.telemetry.stop()
            self.telemetry = None
        if self.model:
            self.model.stop()
            self.model = None
        print('本脚本启动的 VLA 服务与机器人控制程序已结束。', flush=True)

    def loop(self):
        print('\n按键无需回车：k 身体控制开/关 | i 准备动作 | p 启动/停止 VLA | t 更换文本 | q 全部退出', flush=True)
        old = termios.tcgetattr(sys.stdin.fileno())
        text_buffer = None
        try:
            tty.setcbreak(sys.stdin.fileno())
            editing = termios.tcgetattr(sys.stdin.fileno())
            editing[3] &= ~termios.ECHO
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, editing)
            while True:
                try:
                    try:
                        self.tick()
                    except Exception as exc:
                        if str(exc) != self.reported_tick_error:
                            print('状态异常：' + str(exc) + '；仍可按 p/k/q。', flush=True)
                            self.reported_tick_error = str(exc)
                    ready, _, _ = select.select([sys.stdin], [], [], .1)
                    if not ready:
                        continue
                    raw = os.read(sys.stdin.fileno(), 1)
                    key = raw.decode(errors='ignore').lower()
                    if key in ('\x03', '\x04') or (text_buffer is None and key == 'q'):
                        self.close()
                        return
                    if text_buffer is not None:
                        if raw in (b'\r', b'\n'):
                            instruction = text_buffer.decode('utf-8')
                            text_buffer = None
                            print(flush=True)
                            self.set_instruction(instruction)
                        elif raw == b'\x1b':
                            text_buffer = None
                            print('\n已取消文本编辑。', flush=True)
                        elif raw in (b'\x7f', b'\x08'):
                            text_buffer = bytearray(text_buffer.decode('utf-8', errors='ignore')[:-1].encode('utf-8'))
                            print('\r\033[2K文本> ' + text_buffer.decode('utf-8', errors='ignore'), end='', flush=True)
                        elif raw and raw[0] >= 32 and len(text_buffer) < 8000:
                            text_buffer.extend(raw)
                            print('\r\033[2K文本> ' + text_buffer.decode('utf-8', errors='ignore'), end='', flush=True)
                        continue
                    if key == 't':
                        text_buffer = bytearray()
                        print('\n当前文本：' + str(self.instruction or '(默认训练任务)') +
                              '\n输入新文本后回车生效；Esc 取消。编辑期间继续使用原文本；Esc 后可按 p 切换推理。\n文本> ',
                              end='', flush=True)
                    elif key == 'k':
                        if self.owner or self.native:
                            self.stop_body()
                        else:
                            self.start_body()
                    elif key == 'i':
                        self.prepare_vla()
                    elif key == 'p':
                        self.toggle_vla()
                except Exception as exc:
                    print('\n操作未完成：' + str(exc), flush=True)
                    # Keep keyboard available so O / shutdown can be retried.
                    time.sleep(.1)
        finally:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)


def main():
    if '--serve-model' in sys.argv[1:]:
        serve_model()
        return 0
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', type=Path, help='完整 GR00T 模型目录')
    parser.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    parser.add_argument('--port', type=int, default=16551, help='本脚本独占的 VLA 端口，默认 16551')
    parser.add_argument('--task', default='mickey', help='现有训练任务名，默认 mickey')
    parser.add_argument('--instruction', help='覆盖任务语言指令；应与该权重训练任务一致')
    parser.add_argument('--no-rtc', action='store_true', help='强制普通 GR00T 推理：不使用 RTC 前缀/延迟补偿，分段执行并在推理等待时保持末端姿态')
    parser.add_argument('--check', action='store_true', help='仅检查软件/权重，不启动服务或机器人')
    args = parser.parse_args()
    args.repo = args.repo.expanduser().resolve()
    if args.repo != ROOT:
        parser.error('--repo must refer to this standalone checkout')
    args.model = model_bundle(args.model, no_rtc=args.no_rtc)
    if not 1024 <= args.port <= 65535 or args.port in (5550, 5555, 17586, 17587):
        parser.error('请选择独立模型端口，例如 16551')
    cfg = load_config(args.repo)
    for key in ('data_python', 'inference_python', 'rtc_python', 'telemetry_python'):
        if not os.access(cfg['external_paths'][key], os.X_OK):
            raise ValueError('解释器不可用：' + key)
    if not (args.repo / 'scripts/g1_vla_server.py').is_file():
        raise ValueError('VLA 服务入口不存在')
    if not CAMERA_HELPER.is_file():
        raise ValueError('缺少相机组件：' + str(CAMERA_HELPER))
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise ValueError('相机录像需要本机 ffmpeg 和 ffprobe')
    if not any(shutil.which(name) for name in ('google-chrome', 'chromium', 'chromium-browser', 'firefox')):
        raise ValueError('相机窗口需要本机 Chrome/Chromium 或 Firefox')
    if args.check:
        native = Path(cfg['external_paths']['native_build'])
        if not (native / 'g1_field').is_file() or not (native / 'manifest.json').is_file():
            raise ValueError('原生控制部署不完整')
        sys.path[:0] = [str(args.repo / 'src'), str(args.repo)]
        from symbiosis.robots.g1.native_deployment import validate
        validate(native, read(native / 'manifest.json'))
        # Validate adaptation against source in memory, no snapshots or service startup.
        checks = [('ankle_demo.py', "if measured['tilt_rad']>.25:raise ValueError('base_tilt_above_0.25_rad')"),
                  ('vla.py', 'not 1 <= steps <= 1500'),
                  ('vla.py', "now-self.at > self.job.get('steps',0)/50+15:"),
                  ('vla.py', "self.runtime.start(_capabilities()[request['task']],request['steps'])"),
                  ('vla_runtime.py', "host='127.0.0.1',port=5550,"),
                  ('vla_runtime.py', 'self.target_steps=steps;self.prompt=prompt')]
        for filename, source in checks:
            if (args.repo / 'src/symbiosis/robots/g1' / filename).read_text().count(source) != 1:
                raise ValueError('上游适配接口变化：' + filename)
        print(f'文件与原生部署检查通过。推理模式：{inference_mode(read(args.model / "config.json"), no_rtc=args.no_rtc)}。权重：{args.model}\n模型端口：{args.port}；未启动服务、未连接机器人。')
        return 0
    if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        parser.error('自动相机窗口需要桌面 DISPLAY/WAYLAND_DISPLAY')
    if not sys.stdin.isatty():
        parser.error('请在交互终端运行，以接收 k/i/p/t/q')
    control_conflicts()
    with open('/tmp/symbiosis-standalone-vla.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        console = Console(args, cfg)
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGHUP, interrupted)
        try:
            console.start_camera()
            console.start_model()
            console.start_telemetry()
            console.loop()
        except (KeyboardInterrupt, Exception) as exc:
            print('\n退出请求/异常：' + str(exc), flush=True)
            console.close()
            return 1
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print('未完成：' + str(exc), file=sys.stderr)
        raise SystemExit(1)
