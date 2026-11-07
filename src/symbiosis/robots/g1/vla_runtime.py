"""RTC adapter used inside the sole SONIC owner. Never creates a body publisher."""
import fcntl
from pathlib import Path
import sys
import time

import numpy as np

from .vla import DEFAULT_MODEL, DEFAULT_RTC
import hashlib


def identity(path):
    path = Path(path).resolve()
    return dict(model_path=str(path), files={name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in
        ('config.json','processor_config.json','statistics.json','model.safetensors.index.json')})


def dependencies(rtc_repo=DEFAULT_RTC):
    from .rtc_adapter import dependencies as load_dependencies
    return load_dependencies(rtc_repo)


def request_interval(config):
    """Leave the queue room for the server's trained latency prefix."""
    delay=config.get('rtc_delay_steps')
    maximum=config.get('rtc_max_delay_steps')
    horizon=config.get('action_horizon')
    if (type(delay) is not int or type(maximum) is not int or
        not 1 <= delay <= min(10, maximum) or horizon != 40):
        raise ValueError('Invalid trained RTC delay or action horizon')
    interval=min(30,horizon-delay-1)
    if interval < delay:
        raise ValueError('Insufficient RTC horizon for trained delay')
    return interval


class Client:
    def __init__(self, rtc, adapter, host, port, expected):
        self.client = rtc.PolicyClient(host,port,timeout_ms=2000)
        try:
            if self.client.call_endpoint('get_rax_identity', requires_input=False) != expected:
                raise ValueError('VLA server model identity differs from selected bundle')
            interval=request_interval(self.client.get_rtc_config())
            self.rtc_config=rtc.validate_server(self.client,adapter,rtc.Config(rtc_request_interval=interval))
        except BaseException:
            self.client.close()
            raise
        self.adapter = adapter

    def get_rtc_config(self):
        return self.client.get_rtc_config()

    def get_action(self, observation, options=None):
        started=time.monotonic()
        actions, info = self.client.get_action(observation,options)
        elapsed=time.monotonic()-started
        if options is not None and elapsed > self.rtc_config['rtc_delay_steps']/50:
            print(f'RTC model round trip exceeded prefix: {elapsed*1000:.1f} ms; '
                  f'budget={self.rtc_config["rtc_delay_steps"]*20} ms',flush=True)
        actions.pop('task_progress',None);actions.pop('action.task_progress',None)
        result = self.adapter.action(None,actions)
        if np.max(np.abs(result['motion_token'])) > 1.25:
            raise ValueError('VLA token exceeds SONIC inference bound 1.25')
        return result, info

    def close(self):
        self.client.close()


class FreshBodyState:
    """Accept both native timestamp schemas without refreshing cached samples.

    The pinned field controller exports StateLogger.timestamp_monotonic as a
    one-element trace_monotonic_s array. Newer SONIC exports integer nanoseconds.
    Both clocks originate on this workstation; arrival time is never substituted.
    """
    def __init__(self, subscriber):
        self.subscriber = subscriber

    def get_msg(self):
        state = self.subscriber.get_msg(clear=False)
        if state is None:
            return None
        stamp = state.get('state_monotonic_ns')
        if 'state_monotonic_ns' not in state:
            seconds = np.asarray(state.get('trace_monotonic_s'))
            if (seconds.shape != (1,) or seconds.dtype.kind not in 'fi' or
                    not np.isfinite(seconds[0]) or seconds[0] <= 0):
                raise RuntimeError('Body telemetry lacks a valid native source timestamp')
            stamp = int(float(seconds[0]) * 1_000_000_000)
        if type(stamp) is not int:
            raise RuntimeError('Invalid native state_monotonic_ns')
        if not 0 <= time.monotonic_ns() - stamp <= 200_000_000:
            raise RuntimeError('Stale body state (>0.2 s) or future source timestamp; RTC paused')
        # The observation helper edits placeholder hand arrays in place.
        result = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in state.items()}
        result['state_monotonic_ns'] = stamp
        return result


class Runtime:
    def __init__(self, model_path=DEFAULT_MODEL, rtc_repo=DEFAULT_RTC,
                 host='127.0.0.1',port=5550, camera_host='192.168.123.164',camera_port=5555,
                 state_port=17587):
        self.model_path=Path(model_path)
        self.rtc=rtc=dependencies(rtc_repo)
        self.expected=identity(self.model_path)
        self.adapter=rtc.legacy.BraincoModelAdapter(self.model_path,None)
        self.client_factory=lambda: Client(rtc,self.adapter,host,port,self.expected)
        self.bridge=None;self.camera=None;self.raw=None;self.session=None;self.locks=[]
        self.steps=0;self.pending=None;self.preparing=False;self.target_steps=0
        self.hand_motion=None;self.last_hand_targets=None
        self.smpl_future=None;self.smpl_program=None;self.smpl_pool=None
        self.prompt='';self.camera_index=None;self.camera_at=None
        self.pose=rtc.legacy.runner.InitialPoseTransition(rtc.legacy.runner.LATENT_INITIAL_MOTION_TOKEN,6.,50)
        try:
            for side in ('left','right'):
                lock=open('/tmp/rax_brainco_'+side+'.lock','a')
                self.locks.append(lock)
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            client=self.client_factory()
            try:self.request_interval=request_interval(client.rtc_config)
            finally:client.close()  # No motion; fail early on wrong model.
            self.bridge=rtc.legacy.BraincoPolicyBridge('eno1')
            self.adapter.bridge=self.bridge
            self.camera=rtc.legacy.runner.ComposedCameraClientSensor(server_ip=camera_host,port=camera_port)
            self.raw=rtc.legacy.runner.ZMQStateSubscriber(host='127.0.0.1',port=state_port)
            self.state=FreshBodyState(self.raw)
            self.robot=rtc.robot_model(waist_location='lower_and_upper_body')
            self.session=rtc.RobotRTCSession(
                lambda:rtc.AsyncRTCController(self.client_factory,request_interval=self.request_interval),
                self.observation,self._stage,50)
        except BaseException:
            self.close();raise

    def observation(self):
        obs=self.rtc.legacy.runner.prepare_observation_from_sensors(self.camera,self.state,self.robot,self.prompt)
        if obs is None:
            raise RuntimeError('Missing live VLA observation')
        index=self.camera.idx
        now=time.monotonic()
        if index is not None and index!=self.camera_index:
            self.camera_index=index;self.camera_at=now
        if self.camera_at is None or now-self.camera_at>.2:
            raise RuntimeError('VLA camera not advancing')
        return self.adapter.observation(obs)

    def _stage(self, action):
        # Hand writes happen only when the owner consumes this staged body action.
        self.pending=action
        self.steps+=1

    def prepare(self, anchor):
        self.pause();self.steps=0
        self.pose.start(dict(token_state=anchor,state_monotonic_ns=time.monotonic_ns()))
        self.hand_origin=self.rtc.legacy.feedback_hands(self.bridge.feedback())
        self.open_poses=self.rtc.legacy.load_poses()
        self.preparing=True

    def release_hand(self, side):
        self.control_hands(side, 'open')

    def control_hands(self, side, action):
        if side not in ('left','right','both') or action not in ('open','close'):
            raise ValueError('Choose left/right/both and open/close')
        self.pause()
        self.steps=0
        if self.last_hand_targets is None:
            raise RuntimeError('No previous paired hand targets; prepare first')
        if self.state.get_msg() is None:
            raise RuntimeError('No fresh body telemetry')
        measured=self.rtc.legacy.feedback_hands(self.bridge.feedback())
        poses=self.rtc.legacy.load_poses()
        sides=('left','right') if side=='both' else (side,)
        targets={s:np.asarray(poses[s]['open' if action=='open' else 'closed'],np.float32).copy() for s in sides}
        # Validate every selected target before publishing either hand.
        for target in targets.values():
            if target.shape != (6,) or not np.isfinite(target).all() or np.any((target<0)|(target>1)):
                raise ValueError('Invalid configured hand pose')
        self.hand_origin={s:np.asarray(measured[s+'_hand'],np.float32).copy() for s in sides}
        self.hand_targets=targets
        self.hand_hold={key:np.asarray(value,np.float32).copy() for key,value in self.last_hand_targets.items()}
        self.hand_motion=sides;self.steps=0

    def start(self,prompt,steps):
        self.pause();self.steps=0;self.target_steps=steps;self.prompt=prompt
        self.session.resume()

    def start_smpl(self,program,anchor):
        from concurrent.futures import ThreadPoolExecutor
        from .smpl import Executor, validate
        validate(program)
        self.pause();self.steps=0
        hands=self.rtc.legacy.feedback_hands(self.bridge.feedback())
        captured=dict(token=np.asarray(anchor,float).copy(),
            hands=np.stack([hands[s+'_hand'] for s in ('left','right')]))
        if self.smpl_pool is None:self.smpl_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='comotion')
        def compile_program():return Executor().compile(captured,program)
        self.smpl_future=self.smpl_pool.submit(compile_program)

    def poll(self):
        self.pending=None
        if self.smpl_future is not None:
            if not self.smpl_future.done():return None,False
            self.smpl_program=self.smpl_future.result();self.smpl_future=None
        if self.smpl_program is not None:
            i=self.steps;compiled=self.smpl_program
            action=dict(motion_token=compiled['tokens'][i],
                **{side+'_hand_joints':compiled['hands'][i,j] for j,side in enumerate(('left','right'))})
            self.steps+=1
            return action,self.steps>=len(compiled['tokens'])
        if self.hand_motion:
            if self.state.get_msg() is None:raise RuntimeError('No fresh body telemetry')
            self.rtc.legacy.feedback_hands(self.bridge.feedback())
            u=min(self.steps/100,1.);b=u*u*(3-2*u)
            action={key:value.copy() for key,value in self.hand_hold.items()}
            for side in self.hand_motion:
                action[side+'_hand_joints']=(1-b)*self.hand_origin[side]+b*self.hand_targets[side]
            self.steps+=1
            return action,self.steps>=101
        if self.preparing:
            if self.state.get_msg() is None:raise RuntimeError('No fresh body telemetry')
            self.rtc.legacy.feedback_hands(self.bridge.feedback())
            step=self.pose.step;token,done=self.pose.next_token()
            u=min(step/self.pose.steps,1.);b=u*u*(3-2*u)
            hands={side+'_hand_joints':(1-b)*self.hand_origin[side+'_hand']+
                b*np.asarray(self.open_poses[side]['open'],np.float32) for side in ('left','right')}
            self.steps+=1
            return dict(motion_token=token,**hands),done
        self.session.poll()
        return self.pending,self.steps>=self.target_steps

    def apply_hands(self,action):
        self.bridge.command(action['left_hand_joints'],action['right_hand_joints'])
        self.last_hand_targets={side+'_hand_joints':np.asarray(action[side+'_hand_joints'],np.float32).copy() for side in ('left','right')}

    def pause(self):
        if self.session:self.session.pause()
        self.pose.cancel();self.preparing=False;self.pending=None;self.hand_motion=None
        if getattr(self,'smpl_future',None) is not None:self.smpl_future.cancel()
        self.smpl_future=None;self.smpl_program=None

    def close(self):
        if getattr(self,'smpl_pool',None):self.smpl_pool.shutdown(wait=False,cancel_futures=True)
        if self.session:self.session.close()
        if self.raw:self.raw.close()
        if self.camera:self.camera.close()
        if self.bridge:self.bridge.close()
        for lock in self.locks:lock.close()
