"""Checkpoint/package compatibility, checked before allocating model weights."""

from packaging.version import Version


def check_checkpoint_compatibility(config):
    from . import __version__

    raw = config.to_dict() if hasattr(config, "to_dict") else config
    required = raw.get("sensenova_loop_min_version")
    if required is not None and Version(__version__) < Version(required):
        raise RuntimeError(
            f"Checkpoint requires sensenova-loop >= {required}; installed {__version__}. "
            "Upgrade the package before loading this checkpoint."
        )
