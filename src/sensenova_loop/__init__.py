"""SenseNova Loop inference and local Transformers model registration."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sensenova-loop")
except PackageNotFoundError:
    __version__ = "0.1.0"


def register():
    """Register the repository's model classes with Transformers Auto classes."""
    from .models import register as register_models

    register_models()
