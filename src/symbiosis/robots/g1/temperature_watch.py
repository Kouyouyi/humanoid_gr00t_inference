"""One-second G1 body motor temperature display. DDS subscriber only."""
import argparse
import json
import math
from pathlib import Path
import sys
import threading
import time

from .state import NAMES, HARDWARE_TO_ISAAC


def owner_lines(run, now):
    if run is None:
        return ['Owner: not selected']
    try:
        status = json.loads((run / 'ankle_status.json').read_text())
    except (OSError, ValueError):
        return ['Owner: waiting for status (temperature display is independent)']
    age = now - status['updated_monotonic']
    phase = status['phase']
    if not 0 <= age <= .5:
        return [f'Owner: {phase}, status STALE ({age:.1f}s); auto state unknown']
    sides = ','.join(status.get('thermal_auto_sides', [])) or 'OFF'
    remaining = status.get('lease_remaining_seconds')
    lease = ' | lease unlimited' if status.get('lease_unlimited') else f' | remaining {remaining:.0f}s' if remaining is not None else ''
    job = status.get('active_job') or {}
    detail = status.get('fault') or status.get('stage') or 'none'
    rule=status.get('thermal_rule','strict91')
    repeat=status.get('thermal_repeat_seconds',0)
    pending=', '.join(f'{s}:{max(0.,t-now):.0f}s' for s,t in status.get('repeat_ready_at',{}).items()) or '-'
    return [f'Owner: {phase} | Auto: {sides}{lease}',
            f'Action: {job.get("side", "-")} / {detail}',
            f'Rule: {rule} | repeat gap {repeat:g}s | next repeat {pending}']


def render(sample, now, run=None, color=False):
    native_rule=False
    if run is not None:
        try:native_rule=json.loads((run/'ankle_status.json').read_text()).get('thermal_rule')=='native90'
        except (OSError,ValueError):
            try:native_rule=json.loads((run/'manifest.json').read_text()).get('thermal_rule')=='native90'
            except (OSError,ValueError):pass
    lines = ['G1 body temperatures | ' + time.strftime('%H:%M:%S') + ' | refresh 1s | Ctrl+C: close display']
    lines.extend(owner_lines(run, now))
    if not sample:
        return '\n'.join(lines + ['Waiting for rt/lowstate on DDS 0 / eno1...'])
    age = now - sample['monotonic']
    fresh = 0 <= age <= .2
    lines.append(f'DDS: {"LIVE" if fresh else "STALE - readings below are OLD"} | age {age * 1000:.0f} ms | 29 body joints')
    lines.append('W=windings D=driver M=max (C); *=ankle '+('>=90' if native_rule else '>91')+'; !=motor error')
    header = f'{"ID":2} {"Joint":15} {"W":>3} {"D":>3} {"M":>3} {"Err":>4} '
    lines.append(header + ' | ' + header)
    rows = []
    for motor in sample['motors']:
        name = motor['name'].removesuffix('_joint').replace('left_', 'L_').replace('right_', 'R_').replace('shoulder', 'shld')
        winding, driver = motor['temperature_winding_driver_c']
        maximum = max(winding, driver)
        error = motor['motorstate']
        trigger=maximum>=90 if native_rule else maximum>91
        mark = '!' if error else '*' if 'ankle' in motor['name'] and trigger else ' '
        row = f'{motor["hardware_index"]:02} {name:15} {winding:3d} {driver:3d} {maximum:3d} {error:4d} {mark}'
        if color:
            shade = '90' if not fresh else '31' if error or trigger else '33' if maximum >= 85 else ''
            if shade:
                row = f'\033[{shade}m{row}\033[0m'
        rows.append(row)
    for i in range(15):
        lines.append(rows[i] + (' | ' + rows[i + 15] if i + 15 < len(rows) else ''))
    lines.append('Ankle alarm: enter >=90, remains active until <85; repeat after settling.' if native_rule else
                 'Only ankle >91 triggers an ENABLED owner; native speech starts at >=90.')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, help='Show this ankle owner status alongside temperatures')
    parser.add_argument('--output', type=Path, help='Append one displayed sample per second to JSONL')
    parser.add_argument('--seconds', type=float, default=0, help='0: continuous; positive: bounded read-only check')
    parser.add_argument('--plain', action='store_true', help='Print frames without terminal escape codes')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error('--seconds must be finite and nonnegative')
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    names = dict(zip(HARDWARE_TO_ISAAC, NAMES))
    latest = {}
    lock = threading.Lock()

    def receive(msg):
        if len(msg.motor_state) < 29:
            return
        sample = dict(monotonic=time.monotonic(), motors=[dict(
            hardware_index=i, name=names[i],
            temperature_winding_driver_c=[int(v) for v in motor.temperature],
            motorstate=int(motor.motorstate)) for i, motor in enumerate(msg.motor_state[:29])])
        with lock:
            latest.clear()
            latest.update(sample)

    ChannelFactoryInitialize(0, 'eno1')
    subscriber = ChannelSubscriber('rt/lowstate', LowState_)
    subscriber.Init(receive, 1)
    output = None
    terminal = sys.stdout.isatty() and not args.plain
    try:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output = args.output.open('a')
        if terminal:
            print('\033[?1049h\033[?25l', end='', flush=True)
        start = time.monotonic()
        deadline = start
        while not args.seconds or time.monotonic() - start < args.seconds:
            now = time.monotonic()
            with lock:
                sample = dict(latest)
            frame = render(sample, now, args.run, terminal)
            print(('\033[H\033[J' if terminal else '') + frame, flush=True)
            if output:
                output.write(json.dumps(dict(display_monotonic=now, sample=sample,
                    source='live_lowstate', commands_sent=0)) + '\n')
                output.flush()
            deadline += 1.
            time.sleep(max(0., deadline - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        if terminal:
            print('\033[?25h\033[?1049l', end='', flush=True)
        if output:
            output.close()
        subscriber.Close()


if __name__ == '__main__':
    main()
