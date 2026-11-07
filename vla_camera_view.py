#!/usr/bin/env python3
"""First-person preview and per-inference wall-clock video; never robot controls."""
import argparse
from runtime_config import bootstrap
bootstrap()
import json
import math
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name('.' + path.name + '.' + secrets.token_hex(4))
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


class Recording:
    """Resample received frames to 30 fps; duplicate gaps explicitly, never speed up time."""
    def __init__(self, folder, frame, sequence, received_at, fps=30, clock=time.monotonic):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.final = self.folder / 'camera.mp4'
        self.partial = self.folder / 'camera.partial.mp4'
        self.clock, self.fps = clock, fps
        self.height, self.width = frame.shape[:2]
        self.started = clock()
        self.started_unix = time.time()
        self.frames = self.duplicates = self.stale_frames = 0
        self.last_sequence = None
        self.frame = frame.copy()
        self.sequence, self.received_at = sequence, received_at
        self.error = None
        self.closed = False
        self.result = None
        self.log = (self.folder / 'ffmpeg.log').open('ab')
        self.timing = (self.folder / 'frame-timing.jsonl').open('a')
        self.encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', f'{self.width}x{self.height}',
            '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'ultrafast',
            '-crf', '20', '-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
            '-movflags', '+faststart', str(self.partial)], stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=self.log, start_new_session=True)
        try:
            self.advance(self.started)
            self.publish('recording')
        except Exception:
            self.finish('start_failed')
            raise

    def advance(self, now, frame=None, sequence=None, received_at=None):
        if self.closed:
            return
        # Fill intervals before a new frame arrives with the PREVIOUS frame.
        target = max(1, int(math.floor((now - self.started) * self.fps)) + 1)
        while self.frames < target:
            if self.encoder.poll() is not None:
                raise RuntimeError('录像编码进程已退出，见 ffmpeg.log')
            self.encoder.stdin.write(self.frame.tobytes())
            repeated = self.last_sequence == self.sequence
            self.duplicates += int(repeated)
            frame_at = self.started + self.frames / self.fps
            age = max(0., frame_at - self.received_at)
            self.stale_frames += int(age > .5)
            self.timing.write(json.dumps(dict(video_frame=self.frames, frame_monotonic=frame_at,
                source_sequence=self.sequence, source_received_monotonic=self.received_at,
                source_age_s=age, duplicated=repeated)) + '\n')
            self.last_sequence = self.sequence
            self.frames += 1
        if frame is not None:
            if frame.shape[:2] != (self.height, self.width):
                raise RuntimeError('相机分辨率变化，当前视频已结束')
            self.frame = frame.copy()
            self.sequence, self.received_at = sequence, received_at

    def publish(self, status, **extra):
        value = dict(status=status, video=str(self.final if status == 'saved' else self.partial),
            width=self.width, height=self.height, fps=self.fps, frames=self.frames,
            started_unix=self.started_unix, elapsed_s=max(0., self.clock()-self.started),
            duplicate_frames=self.duplicates, stale_frames=self.stale_frames,
            timing_source='workstation receive monotonic time; not camera exposure time',
            frame_timing=str(self.folder / 'frame-timing.jsonl'), error=self.error, **extra)
        atomic(self.folder / 'recording.json', value)
        return value

    def finish(self, reason):
        if self.closed:
            return self.result
        ended = self.clock()
        try:
            self.advance(ended)
        except Exception as exc:
            self.error = str(exc)
        self.closed = True
        code = None
        try:
            self.encoder.stdin.close()
            code = self.encoder.wait(timeout=15)
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.error = self.error or str(exc)
            self.encoder.kill()
            self.encoder.wait(timeout=5)
        finally:
            self.log.close()
            self.timing.close()
        probe = {}
        if code == 0 and not self.error and self.frames:
            try:
                checked = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                    '-show_entries', 'stream=codec_name,width,height,nb_frames,duration',
                    '-of', 'json', str(self.partial)], capture_output=True, text=True, timeout=10, check=True)
                probe = json.loads(checked.stdout)['streams'][0]
                if int(probe.get('nb_frames', 0)) != self.frames:
                    raise ValueError('封装帧数与编码帧数不一致')
                self.partial.replace(self.final)
            except Exception as exc:
                self.error = str(exc)
        else:
            self.error = self.error or f'ffmpeg exit={code}'
        self.result = self.publish('error' if self.error else 'saved', reason=reason,
                                  ended_unix=time.time(), wall_duration_s=ended-self.started, probe=probe)
        return self.result


PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>G1 相机第一视角</title>
<style>body{margin:0;background:#10151d;color:#ecf1f7;font:16px system-ui}main{max-width:1100px;margin:30px auto;padding:0 24px}header{display:flex;justify-content:space-between;align-items:center}h1{font-size:22px}#badge{color:#9ab1c8}section{background:#080b10;min-height:320px;display:flex;align-items:center;justify-content:center;border-radius:12px;margin:20px 0}img{max-width:100%;max-height:75vh;object-fit:contain}p{color:#a9b7c7;overflow-wrap:anywhere}.rec{color:#ff7676!important}</style>
<main><header><h1>G1 · 相机第一视角</h1><span id="badge">正在连接</span></header><section><img id="view" hidden><span id="waiting">等待机器人相机画面…</span></section><p id="record">未录像</p><p>终端按 p 开始／结束推理并自动录像；q 关闭全部程序。</p></main>
<script>const token=new URLSearchParams(location.search).get('token');let busy=false;
async function refresh(){if(busy)return;busy=true;try{const s=await(await fetch('/state?token='+token,{cache:'no-store'})).json();const b=document.getElementById('badge');b.textContent=s.frame_age_s===null?'等待画面':s.frame_age_s>.5?'画面更新中断 · '+s.frame_age_s.toFixed(1)+' 秒':s.width+' × '+s.height+' · 实时';if(s.has_frame){document.getElementById('view').hidden=false;document.getElementById('waiting').hidden=true;document.getElementById('view').src='/frame.jpg?token='+token+'&v='+s.sequence;}const r=document.getElementById('record');r.textContent=s.recording?'● 录像中 · '+s.recording.frames+' 帧':s.last_recording?(s.last_recording.status==='saved'?'已保存：':'录像异常：')+s.last_recording.video:s.error||'未录像';r.className=s.recording?'rec':'';}catch(e){document.getElementById('badge').textContent='相机服务已结束';}finally{busy=false;}}setInterval(refresh,150);refresh();</script></html>'''


def run_worker(root, repo, fixture=False):
    import cv2
    root = Path(root).resolve()
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    recording = None
    last_recording = None
    frame = None
    jpeg = None
    sequence = -1
    received_at = None
    last_ack = None
    command_error = None
    token = secrets.token_urlsafe(24)
    lock = threading.Lock()
    shared = dict(has_frame=False, frame_age_s=None, width=None, height=None, sequence=None,
                  recording=None, last_recording=None, error=None)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            parsed = urlparse(self.path)
            if parse_qs(parsed.query).get('token') != [token]:
                self.send_error(403)
                return
            with lock:
                if parsed.path == '/':
                    payload, kind = PAGE.encode(), 'text/html; charset=utf-8'
                elif parsed.path == '/state':
                    payload, kind = json.dumps(shared).encode(), 'application/json'
                elif parsed.path == '/frame.jpg' and jpeg is not None:
                    payload, kind = jpeg, 'image/jpeg'
                else:
                    self.send_error(404)
                    return
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    atomic(root / 'camera-viewer.json', dict(url=f'http://127.0.0.1:{server.server_port}/?token={token}',
        pid=os.getpid(), fixture=fixture))
    client = None
    try:
        if not fixture:
            sys.path[:0] = [str(repo / 'src'), str(repo)]
            from symbiosis.robots.g1.vla_runtime import dependencies
            client = dependencies().legacy.runner.ComposedCameraClientSensor(server_ip='192.168.123.164', port=5555)
        last_publish = 0.
        fixture_at = 0.
        while not stop.is_set():
            now = time.monotonic()
            image = None
            next_sequence = sequence
            if fixture and now-fixture_at >= 1/20:
                import numpy as np
                image = np.zeros((180, 320, 3), dtype=np.uint8)
                cv2.putText(image, 'OFFLINE TEST', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
                next_sequence += 1
                fixture_at = now
            elif client is not None:
                data = client.read(blocking=False)
                if data and client.idx != sequence and 'ego_view' in data.get('images', {}):
                    image = cv2.cvtColor(data['images']['ego_view'], cv2.COLOR_RGB2BGR)
                    next_sequence = client.idx
            if recording:
                try:
                    recording.advance(now)
                except Exception as exc:
                    recording.error = str(exc)
                    last_recording = recording.finish('recording_error')
                    recording = None
            if image is not None:
                frame, sequence, received_at = image, next_sequence, now
                ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    with lock:
                        jpeg = encoded.tobytes()
                if recording:
                    recording.advance(now, frame, sequence, received_at)
            command = read(root / 'camera-command.json')
            if command.get('id') and command['id'] != last_ack:
                command_error = None
                try:
                    action = command['action']
                    if action == 'start':
                        if recording:
                            raise RuntimeError('已有录像正在进行')
                        if frame is None or now-received_at > .5:
                            raise RuntimeError('相机尚无新鲜画面，未开始录像／推理')
                        folder = Path(command['folder']).resolve()
                        if folder.parent != root / 'recordings':
                            raise ValueError('录像目录不属于本轮')
                        folder.parent.mkdir(exist_ok=True)
                        recording = Recording(folder, frame, sequence, received_at)
                    elif action in ('stop', 'shutdown'):
                        if recording:
                            last_recording = recording.finish(command.get('reason', action))
                            recording = None
                        if action == 'shutdown':
                            stop.set()
                    else:
                        raise ValueError('未知相机命令')
                except Exception as exc:
                    command_error = str(exc)
                last_ack = command['id']
            if now-last_publish >= .1 or last_ack != read(root / 'camera-state.json').get('ack') or stop.is_set():
                value = dict(online=True, updated_monotonic=time.monotonic(), ack=last_ack,
                    command_error=command_error, has_frame=frame is not None,
                    frame_age_s=now-received_at if received_at is not None else None,
                    width=frame.shape[1] if frame is not None else None,
                    height=frame.shape[0] if frame is not None else None, sequence=sequence,
                    recording=recording.publish('recording') if recording else None,
                    last_recording=last_recording, error=command_error, fixture=fixture)
                with lock:
                    shared = value
                atomic(root / 'camera-state.json', value)
                last_publish = now
            stop.wait(.005)
    except Exception as exc:
        command_error = str(exc)
        raise
    finally:
        if recording:
            last_recording = recording.finish('camera_service_exit')
        atomic(root / 'camera-state.json', dict(online=False, updated_monotonic=time.monotonic(),
            ack=last_ack, command_error=command_error, last_recording=last_recording, recording=None))
        if client:
            client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--fixture', action='store_true', help='Offline generated test frames; no robot connection')
    args = parser.parse_args()
    run_worker(args.root, args.repo, args.fixture)


if __name__ == '__main__':
    main()
