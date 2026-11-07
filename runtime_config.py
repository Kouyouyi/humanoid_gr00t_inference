"""Put this checkout ahead of installed/editable copies of application code."""
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
SOURCE_PATHS = [ROOT/'src', ROOT/'vendor/rtc', ROOT/'vendor/rtc/scripts/local',ROOT/'vendor/sonic',ROOT/'vendor/unitree']
def bootstrap():
    os.environ['SYMBIOSIS_PROJECT_ROOT'] = str(ROOT)
    sys.path[:0] = [str(p) for p in SOURCE_PATHS]
    return ROOT
bootstrap()
