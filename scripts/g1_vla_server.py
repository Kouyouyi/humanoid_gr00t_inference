"""Compatibility command entry; implementation lives in src/symbiosis/operations."""
from pathlib import Path
import runpy
import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]
if __name__ == "__main__":
    runpy.run_module("symbiosis.operations.g1_vla_server", run_name="__main__")
else:
    from importlib import import_module
    globals().update({k: v for k, v in vars(import_module("symbiosis.operations.g1_vla_server")).items() if not k.startswith("__")})
