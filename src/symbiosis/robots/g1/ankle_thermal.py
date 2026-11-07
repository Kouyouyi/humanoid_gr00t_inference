"""Ankle alarm detection with legacy strict91 and native90 hysteresis modes.

No motion command API. The standalone observer retains historical blocked
metadata; the separate ankle executor owns configured action dispatch/repeats.
"""
import math
from pathlib import Path
import subprocess
import time

ANKLES = {4: 'left_ankle_pitch_joint', 5: 'left_ankle_roll_joint',
          10: 'right_ankle_pitch_joint', 11: 'right_ankle_roll_joint'}
TRIGGER_C = 91.
REARM_C = 85.
REARM_SECONDS = 5.
BLOCK_REASON = 'heel_motion_not_validated_for_thermal_relief; E064/E065 ankle_roll_torque_increased'


class AlarmDetector:
    def __init__(self, rule='strict91'):
        if rule not in ('strict91','native90'):raise ValueError('Unknown ankle alarm rule')
        self.rule=rule
        self.latched_motors=set()
        self.active = {s: False for s in ('left', 'right')}
        self.cool_since = {s: None for s in self.active}
        self.episodes = {s: 0 for s in self.active}
        self.last_sample = None

    def update(self, sample, now):
        """Fresh complete samples; emit side alarm edges for the selected rule."""
        stamp = sample['monotonic']
        if not math.isfinite(stamp) or not 0 <= now-stamp <= .2:
            self.cool_since = dict.fromkeys(self.active)
            raise ValueError('stale_or_future_lowstate')
        if self.last_sample is not None and stamp <= self.last_sample:
            self.cool_since = dict.fromkeys(self.active)
            raise ValueError('duplicate_or_out_of_order_lowstate')
        motors = {}
        for motor in sample['motors']:
            index = motor['hardware_index']
            if index not in ANKLES:
                continue
            temps = motor['temperature_winding_driver_c']
            if (index in motors or motor['name'] != ANKLES[index] or len(temps) != 2 or
                    any(not isinstance(t, (int, float)) or not math.isfinite(t) for t in temps)):
                self.cool_since = dict.fromkeys(self.active)
                raise ValueError('invalid_ankle_temperature_sample')
            motors[index] = motor
        if set(motors) != set(ANKLES):
            self.cool_since = dict.fromkeys(self.active)
            raise ValueError('missing_ankle_temperature')
        if self.last_sample is not None and stamp-self.last_sample > .25:
            self.cool_since = dict.fromkeys(self.active)
        self.last_sample = stamp
        events = []
        if self.rule=='native90':
            # Mirror native per-motor hysteresis: enter >=90, exit <85.
            # A latched 87C ankle is still named by the native voice warning.
            for i,motor in motors.items():
                temperature=max(motor['temperature_winding_driver_c'])
                if temperature>=90:self.latched_motors.add(i)
                elif temperature<85:self.latched_motors.discard(i)
            for side in self.active:
                latched=[m for i,m in motors.items() if i in self.latched_motors and m['name'].startswith(side)]
                if latched and not self.active[side]:
                    self.active[side]=True;self.episodes[side]+=1
                    events.append(dict(event='ankle_temperature_trigger',side=side,episode=self.episodes[side],
                        trigger_rule='native alarm: enter >=90 C, clear each motor <85 C',joints=latched,
                        motion_enabled=False,requested_skill='ankle_heel_lift',dispatch_block_reason=BLOCK_REASON))
                elif not latched and self.active[side]:
                    self.active[side]=False
                    events.append(dict(event='ankle_alarm_rearmed',side=side))
            return events
        for side in self.active:
            readings = [m for m in motors.values() if m['name'].startswith(side)]
            hot = [m for m in readings if max(m['temperature_winding_driver_c']) > TRIGGER_C]
            if hot:
                self.cool_since[side] = None
                if not self.active[side]:
                    self.active[side] = True
                    self.episodes[side] += 1
                    events.append(dict(event='ankle_temperature_trigger',side=side,
                        episode=self.episodes[side],trigger_rule='max(winding,driver) > 91 C',
                        joints=hot,motion_enabled=False,requested_skill='ankle_heel_lift',
                        dispatch_block_reason=BLOCK_REASON))
            elif all(max(m['temperature_winding_driver_c']) < REARM_C for m in readings):
                if self.cool_since[side] is None:
                    self.cool_since[side] = stamp
                if self.active[side] and stamp-self.cool_since[side] >= REARM_SECONDS:
                    self.active[side] = False
                    events.append(dict(event='ankle_alarm_rearmed',side=side))
            else:
                self.cool_since[side] = None
        return events


def announce(event, directory, replay=False):
    """Actual local audio playback; never says the disabled motion will execute."""
    side = event['side']
    if side not in ('left', 'right'):
        raise ValueError('Unknown ankle side')
    prefix = 'Replay test. ' if replay else ''
    message = prefix + f'Warning! High temperature at {side} ankle. Automatic heel lift is not enabled.'
    wav = Path(directory)/f"{side}-{event['episode']}.wav"
    if wav.exists():
        raise FileExistsError(wav)
    subprocess.run(['espeak','-s','165','-w',str(wav),message],check=True,timeout=10)
    playback_started = time.monotonic()
    subprocess.run(['paplay',str(wav)],check=True,timeout=20)
    return dict(event='ankle_audio_completed',side=side,episode=event['episode'],message=message,
                playback_started_monotonic=playback_started,playback_completed_monotonic=time.monotonic(),
                wav=str(wav),playback_exit_code=0,motion_enabled=False,
                dispatch_status='blocked',dispatch_block_reason=BLOCK_REASON)
