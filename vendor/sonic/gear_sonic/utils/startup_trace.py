"""Opt-in passive recorder. No sockets, robot imports or control decisions.

STARTUP_TRACE_DIR enables logging; STARTUP_SESSION_ID groups processes.
All array files are lossless NPZ. Host clocks must not be subtracted across hosts.

Self-contained copy of the deployment tracing helper originally maintained in
world_action_model.utils.startup_trace; no external checkout is required.
"""
import atexit
from contextvars import ContextVar
from collections.abc import Mapping

request_context = ContextVar("startup_request_id", default=None)
import copy
import json
import os
from pathlib import Path
import queue
import socket
import threading
import time
import uuid

import numpy as np


class TensorSnapshot:
    def __init__(self, value):
        import torch
        self.value = value.detach().clone()
        self.event = None
        if value.is_cuda:
            self.event = torch.cuda.Event()
            self.event.record(torch.cuda.current_stream(value.device))

    def numpy(self):
        if self.event is not None:
            self.event.synchronize()
        # NumPy has no native BF16; FP32 represents every BF16 value exactly.
        # Preserve float64 and integer tensors instead of silently rounding them.
        import torch
        value = self.value.float() if self.value.dtype == torch.bfloat16 else self.value
        return value.cpu().numpy()


def snapshot(value):
    if type(value).__module__.startswith('torch') and hasattr(value, 'detach'):
        return TensorSnapshot(value)
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {str(k): snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [snapshot(v) for v in value]
    return copy.deepcopy(value)


class NullRecorder:
    enabled = False
    def record(self, event, **fields):
        return None
    def close(self, timeout=5):
        pass


class Recorder:
    enabled = True

    def __init__(self, root, role, capacity=256, metadata=None):
        self.session_id = os.environ.get('STARTUP_SESSION_ID') or uuid.uuid4().hex
        self.process_id = uuid.uuid4().hex
        self.directory = Path(root) / self.session_id / (role + '-' + self.process_id[:8])
        self.directory.mkdir(parents=True, exist_ok=False)
        self.queue = queue.Queue(maxsize=capacity)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.sequence = self.written = self.dropped = self.errors = 0
        self.last_error = None
        self.closed = False
        self.inflight = 0
        self.host = socket.gethostname()
        before = time.monotonic_ns()
        utc = time.time_ns()
        after = time.monotonic_ns()
        self._json('manifest.json', dict(
            session_id=self.session_id, process_id=self.process_id, role=role,
            host=self.host, pid=os.getpid(), clock_domain=f'{self.host}:CLOCK_MONOTONIC',
            clock_mapping=dict(monotonic_before_ns=before, utc_ns=utc,
                               monotonic_after_ns=after), metadata=metadata or {},
            scope='Passive process trace; does not establish robot execution by itself.',
        ))
        self.thread = threading.Thread(target=self._writer, name='startup-trace-writer', daemon=True)
        self.thread.start()
        atexit.register(self.close)

    def _json(self, name, value):
        path = self.directory / name
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(json.dumps(value, default=str, allow_nan=False, indent=2) + '\n')
        tmp.replace(path)

    def record(self, event, **fields):
        # Never wait for another producer or for disk. Sequence gaps represent drops.
        captured = time.monotonic_ns()
        if not self.lock.acquire(blocking=False):
            self.dropped += 1
            return None
        try:
            if self.closed:
                self.dropped += 1
                return None
            self.sequence += 1
            seq = self.sequence
            if self.queue.full():
                self.dropped += 1
                return None
            item = dict(event=event, event_sequence=seq, monotonic_ns=captured,
                        utc_ns=time.time_ns(), session_id=self.session_id,
                        process_id=self.process_id, host=self.host, fields=snapshot(fields))
            self.queue.put_nowait(item)
            return seq
        except Exception as exc:
            self.errors += 1
            self.last_error = repr(exc)
            return None
        finally:
            self.lock.release()

    def _extract(self, value, arrays):
        if isinstance(value, TensorSnapshot):
            return dict(original_dtype=str(value.value.dtype), values=self._extract(value.numpy(), arrays))
        if isinstance(value, np.ndarray):
            key = 'array_' + str(len(arrays))
            arrays[key] = value
            return dict(array=key, shape=list(value.shape), dtype=str(value.dtype))
        if isinstance(value, dict):
            return {k: self._extract(v, arrays) for k, v in value.items()}
        if isinstance(value, list):
            return [self._extract(v, arrays) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _write_item(self, stream, item):
        arrays = {}
        item = self._extract(item, arrays)
        if arrays:
            filename = f"event_{item['event_sequence']:09d}.npz"
            np.savez(self.directory / filename, **arrays)
            item['arrays_file'] = filename
        stream.write(json.dumps(item, default=str, allow_nan=False) + '\n')
        stream.flush()
        self.written += 1

    def _status(self):
        self._json('recorder_status.json', dict(
            attempted=self.sequence, written=self.written, dropped=self.dropped,
            errors=self.errors, unfinished=self.queue.qsize()+self.inflight, closed=self.closed,
            last_error=self.last_error, writer_alive=self.thread.is_alive(),
        ))

    def _writer(self):
        try:
            with (self.directory / 'events.jsonl').open('a') as stream:
                while not self.stop.is_set() or not self.queue.empty():
                    try:
                        item = self.queue.get(timeout=.1)
                    except queue.Empty:
                        self._status()
                        continue
                    self.inflight = 1
                    try:
                        self._write_item(stream, item)
                    except Exception as exc:
                        self.errors += 1
                        self.last_error = repr(exc)
                    finally:
                        self.inflight = 0
                        self.queue.task_done()
                    self._status()
        except Exception as exc:
            self.errors += 1
            self.last_error = repr(exc)

    def close(self, timeout=5):
        self.closed = True
        self.stop.set()
        self.thread.join(timeout=timeout)
        # Shutdown only; never called from control tick.
        if not self.thread.is_alive():
            try:
                self._status()
            except Exception:
                pass


def from_env(role, metadata=None):
    root = os.environ.get('STARTUP_TRACE_DIR')
    return Recorder(root, role, metadata=metadata) if root else NullRecorder()
