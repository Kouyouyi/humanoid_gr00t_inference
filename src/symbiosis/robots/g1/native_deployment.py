"""Resolve field assets without depending on the original workstation layout."""
from pathlib import Path

from .deployment import external_path, load
from .vendor import sha


def optional_path(name, fallback):
    if load().get('external_paths', {}).get(name):
        return external_path(name)
    return fallback()


def paths(native_build=None):
    def experiment():
        return external_path('comotion_root')/'experiments/shared_control'
    assets = optional_path('controller_assets', lambda: experiment()/'assets/local_release_20261004')
    return dict(
        build=Path(native_build).resolve() if native_build else optional_path(
            'native_build', lambda: experiment()/'build/controller_field'),
        assets=assets,
        planner=optional_path('controller_planner', lambda: assets/'planner/target_vel/V2/planner_sonic.onnx'),
        reference=optional_path('controller_reference', lambda: external_path('sonic_root')/'gear_sonic_deploy/reference/example'),
        environment=optional_path('native_env', lambda: external_path('comotion_root')/'env.local.sh'),
        workdir=optional_path('native_workdir', lambda: external_path('comotion_root')))


def validate(build, manifest):
    if (manifest.get('target'), manifest.get('dds_domain'), manifest.get('interface'),
            manifest.get('transport_lease_ms')) != ('field', 0, 'eno1', 350):
        raise ValueError('Unexpected native controller deployment')
    if sha(Path(build)/'g1_field') != manifest['binary_sha256']:
        raise ValueError('Native controller binary hash mismatch')

