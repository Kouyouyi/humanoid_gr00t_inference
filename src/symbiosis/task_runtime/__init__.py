"""Standalone task-name lookup; no external Harness/task package dependency."""
import json
from symbiosis.common.project import project_root
class CapabilityBundle:
    package = 'standalone_vla'
    profile = 'manual'
    def policy(self, name):
        if name != 'capabilities':
            raise ValueError('Standalone runtime only provides task capabilities')
        return json.loads((project_root()/'configs/tasks.json').read_text())
def default_bundle(*args, **kwargs):
    return CapabilityBundle()
def for_run(*args, **kwargs):
    return CapabilityBundle()
