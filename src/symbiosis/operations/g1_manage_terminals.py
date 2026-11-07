"""Close only retired G1 desktop wrappers and their known read-only monitors."""
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal


@dataclass(frozen=True)
class Process:
    pid: int
    parent: int
    start: str
    argv: tuple


def snapshot():
    result = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
            argv = tuple(x.decode(errors='replace') for x in (path/'cmdline').read_bytes().split(b'\0') if x)
            if argv:
                result[int(path.name)] = Process(int(path.name), int(fields[1]), fields[19], argv)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return result


def descendants(processes, pid):
    found = []
    for child in processes.values():
        if child.parent == pid:
            found.extend(descendants(processes, child.pid))
            found.append(child)
    return found


def active_control(processes):
    modules = {'g1_sonic.ankle_executor', 'g1_sonic.executor', 'g1_sonic.commission'}
    return any(Path(p.argv[0]).name == 'g1_field' or
               any(p.argv[i:i+2] == ('-m', module) for module in modules for i in range(len(p.argv)-1))
               for p in processes.values())


def readonly_child(p, run):
    executable = Path(p.argv[0]).name
    if executable in {'bash', 'sh', 'script'}:
        return True  # All descendants must also pass; a live service prevents closing.
    if not executable.startswith('python') or str(run) not in p.argv:
        return False
    return (any(Path(arg).name == 'g1_watch_agent.py' for arg in p.argv[1:2]) or
            p.argv[1:3] == ('-m', 'g1_sonic.temperature_watch'))


def candidates(processes, runs_root, keep_run=None):
    result = []
    busy = active_control(processes)
    for p in processes.values():
        if Path(p.argv[0]).name != 'bash' or len(p.argv) != 2:
            continue
        wrapper = Path(p.argv[1])
        if not wrapper.is_absolute() or wrapper.parent.parent != runs_root:
            continue
        if wrapper.name not in {'window-owner.sh', 'window-native.sh', 'window-codex.sh',
                                'window-agent-watch.sh', 'window-temperature.sh', 'window-camera.sh', 'window-model.sh'}:
            continue
        parent = processes.get(p.parent)
        if parent is None or Path(parent.argv[0]).name != 'gnome-terminal-server':
            continue
        children = descendants(processes, p.pid)
        reason = ('current run' if wrapper.parent == keep_run else
                  'model/camera service window' if children and wrapper.name in {'window-model.sh', 'window-camera.sh'} else
                  'controller/owner still running' if busy else
                  'live agent/service or unrecognized child' if any(not readonly_child(c, wrapper.parent) for c in children) else '')
        result.append((p, children, wrapper, reason))
    return result


def clean(runs_root, keep_run=None, apply=False):
    report = []
    for p, children, wrapper, reason in candidates(snapshot(), runs_root, keep_run):
        entry = dict(pid=p.pid, window=str(wrapper), action='keep' if reason else 'would_close', reason=reason)
        if apply and not reason:
            # Recompute immediately before signalling: never close a wrapper that acquired a service.
            current = next((row for row in candidates(snapshot(), runs_root, keep_run) if row[0] == p), None)
            if current is None or current[3] or current[1] != children:
                entry.update(action='keep', reason='process tree changed')
            else:
                for target in [*children, p]:
                    try:
                        fd = os.pidfd_open(target.pid)
                        try:
                            if snapshot().get(target.pid) == target:
                                signal.pidfd_send_signal(fd, signal.SIGTERM)
                        finally:
                            os.close(fd)
                    except ProcessLookupError:
                        pass
                entry['action'] = 'close_requested'
        report.append(entry)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--keep-run', type=Path)
    parser.add_argument('--apply', action='store_true', help='Default is a read-only preview')
    args = parser.parse_args()
    print(json.dumps(clean(args.runs_root.resolve(), args.keep_run.resolve() if args.keep_run else None,
                           args.apply), ensure_ascii=False, indent=2))
