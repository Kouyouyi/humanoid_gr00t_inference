"""Repository-local deployment; explicit frozen snapshots retain absolute paths."""
import json
import os
from pathlib import Path
import sys

def load():
    snapshot = Path(__file__).with_name('deployment.json')
    if snapshot.is_file():
        return json.loads(snapshot.read_text())
    explicit = os.environ.get('GROOT_REAL_DEPLOYMENT_FILE')
    if explicit:
        return json.loads(Path(explicit).read_text())
    root = Path(os.environ.get('SYMBIOSIS_PROJECT_ROOT',Path(__file__).resolve().parents[3]))
    values = json.loads((root/'configs/defaults.json').read_text())
    local = root/'configs/local/g1.json'
    if local.is_file():
        update = json.loads(local.read_text())
        values['external_paths'].update(update.get('external_paths',{}))
    for name,value in values['external_paths'].items():
        if value == '@python':
            value = sys.executable
        value = Path(os.path.expandvars(value)).expanduser()
        values['external_paths'][name] = str(value if value.is_absolute() else (root/value).resolve())
    return values

def external_path(name):
    value = load().get('external_paths',{}).get(name)
    if not isinstance(value,str) or not Path(value).is_absolute():
        raise ValueError('Configure external_paths.'+name+' in configs/local/g1.json')
    return Path(value)
