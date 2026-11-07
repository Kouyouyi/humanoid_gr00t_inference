"""Same configuration contract in the source package and frozen native package."""
if __package__ == "g1_sonic":
    from ._deployment import external_path, load
else:
    from symbiosis.common.deployment import external_path, load
