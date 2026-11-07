"""Only the immutable joint ordering needed by read-only DDS telemetry."""
import json
from pathlib import Path
JOINTS = sorted(json.loads((Path(__file__).parents[2]/'robots/g1/joints.json').read_text())['joints'],key=lambda j:j['hardware_index'])
BOOT_ID = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
