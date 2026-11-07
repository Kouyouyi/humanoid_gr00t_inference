"""Read-only LowState bridge across Python environments, over a private socketpair."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import threading
import time

from .ankle_thermal import ANKLES
from .deployment import external_path


class HealthBridge:
    def __init__(self):
        self.socket,child=socket.socketpair(socket.AF_UNIX,socket.SOCK_SEQPACKET)
        self.socket.setblocking(False)
        self.process=None;self.latest={}
        try:
            self.process=subprocess.Popen([
                str(external_path('telemetry_python')),
                '-m',__package__+'.health_bridge','--fd',str(child.fileno())],pass_fds=(child.fileno(),))
        except BaseException:
            self.socket.close();raise
        finally:child.close()

    def poll(self,now):
        if self.process.poll() is not None:
            raise RuntimeError('Read-only ankle health bridge exited')
        for _ in range(64):
            try:raw=self.socket.recv(65536)
            except BlockingIOError:break
            if not raw:raise RuntimeError('Read-only ankle health bridge disconnected')
            self.latest=json.loads(raw)
        return self.latest.copy()

    def close(self):
        self.socket.close()
        if self.process:
            try:self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill();self.process.wait(timeout=2)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fd',type=int,required=True)
    a=p.parse_args()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize,ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    latest={};lock=threading.Lock()
    def receive(msg):
        row=dict(monotonic=time.monotonic(),motors=[dict(hardware_index=i,name=name,
            temperature_winding_driver_c=list(msg.motor_state[i].temperature),
            motorstate=int(msg.motor_state[i].motorstate),tau_est=float(msg.motor_state[i].tau_est))
            for i,name in ANKLES.items()])
        with lock:latest.clear();latest.update(row)
    ChannelFactoryInitialize(0,'eno1')
    sub=ChannelSubscriber('rt/lowstate',LowState_);sub.Init(receive,1)
    try:
        with socket.socket(fileno=a.fd) as channel:
            channel.settimeout(.1)
            while True:
                with lock:row=dict(latest)
                try:channel.send(json.dumps(row,allow_nan=False).encode())
                except socket.timeout:pass
                except (BrokenPipeError,ConnectionResetError):break
                time.sleep(.02)
    finally:sub.Close()


if __name__=='__main__':main()
