"""Paths for a source checkout or editable installation; never import the old repo."""
import os
from pathlib import Path


def project_root():
    return Path(os.environ.get("SYMBIOSIS_PROJECT_ROOT", Path(__file__).resolve().parents[3])).resolve()
