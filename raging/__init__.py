"""Top-level package for the Raging RAG toolkit."""

from importlib import metadata

try:
    __version__ = metadata.version("raging")
except metadata.PackageNotFoundError:  # pragma: no cover - during local dev
    __version__ = "0.1.0"

__all__ = ["__version__"]
