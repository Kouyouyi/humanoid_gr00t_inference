"""Six-channel hand I/O over a private socket; no body publisher or auto pose.

Preserves the deployed BrainCo protocol, freshness and paired-target checks.
Read-only mode creates subscribers only, for deployment preflight.
"""
import argparse
import json
import socket
import subprocess
import sys
import threading
import time

import numpy as np

from .deployment import external_path


def hand_vector(value):
    q = np.asarray(value, dtype=np.float32)
    if q.shape != (6,) or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError('BrainCo requires six finite normalized positions in [0,1]')
    return q


def feedback_hands(payload):
    if not payload['ready']:
        raise RuntimeError('Waiting for fresh feedback from BOTH hands')
    result = {}
    for side in ('left', 'right'):
        state = payload['feedback'][side]
        if not state['valid'] or not 0 <= state['age_s'] <= .2:
            raise RuntimeError('Invalid/stale BrainCo '+side+' feedback')
        result[side+'_hand'] = hand_vector(state['q'])
    return result


def handle_request(hands, request, *, read_only=False):
    ready, feedback = hands.feedback()
    result = dict(ready=ready, feedback=feedback)
    if request['op'] == 'feedback':return result
    if request['op'] != 'command' or read_only:
        raise ValueError('Hand commands unavailable')
    if not 0 <= time.monotonic()-request['monotonic_s'] <= .2:
        raise RuntimeError('Expired BrainCo command')
    feedback_hands(result)
    targets = {s:hand_vector(request[s]) for s in ('left', 'right')}
    for side, q in targets.items():
        message = hands._command_type()
        message.cmds = [hands._motor_type() for _ in range(6)]
        for motor, position in zip(message.cmds, q, strict=True):
            motor.q, motor.dq = float(position), 1.
        if not hands.publishers[side].Write(message):
            raise RuntimeError('BrainCo '+side+' DDS write failed')
    return result


class BraincoPolicyBridge:
    def __init__(self, interface, *, read_only=False):
        self.lock = threading.Lock()
        self.failed = False
        self.socket, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.socket.settimeout(15)
        self.process = None
        try:
            command = [str(external_path('telemetry_python')), '-m', __package__+'.brainco_bridge',
                       '--fd', str(child.fileno()), '--interface', interface]
            if read_only:command.append('--read-only')
            self.process = subprocess.Popen(command, pass_fds=(child.fileno(),))
            child.close()
            self.feedback()  # Initializes the SDK; does not move hands.
            self.socket.settimeout(2)
        except BaseException:
            child.close();self.close();raise

    def request(self, data):
        with self.lock:
            if self.failed:raise RuntimeError('BrainCo bridge disconnected; restart the client')
            try:
                self.socket.sendall(json.dumps(data, allow_nan=False).encode())
                raw = self.socket.recv(65536)
                if not raw:raise RuntimeError('BrainCo bridge exited')
                response = json.loads(raw)
            except (OSError, ValueError, RuntimeError):
                self.failed = True;self.socket.close();raise
            if 'error' in response:raise RuntimeError(response['error'])
            return response

    def feedback(self):return self.request(dict(op='feedback'))

    def command(self, left, right):
        return self.request(dict(op='command', monotonic_s=time.monotonic(),
                                 left=hand_vector(left).tolist(), right=hand_vector(right).tolist()))

    def close(self):
        self.socket.close()
        if self.process is not None:
            try:self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill();self.process.wait(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fd', type=int, required=True)
    parser.add_argument('--interface', required=True)
    parser.add_argument('--read-only', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(external_path('sonic_root')))
    from gear_sonic.utils.teleop.brainco_hands import BraincoHands
    with socket.socket(fileno=args.fd) as channel:
        hands = BraincoHands(args.interface, dry_run=args.read_only)
        try:
            while raw := channel.recv(65536):
                try:response = handle_request(hands, json.loads(raw), read_only=args.read_only)
                except Exception as error:response = dict(error=str(error))
                channel.sendall(json.dumps(response, allow_nan=False).encode())
        finally:hands.close()


if __name__ == '__main__':main()
