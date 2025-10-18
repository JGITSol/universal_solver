"""Library metadata for the universal_solver package."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path


def _read_local_version() -> str:
	version_path = Path(__file__).resolve().parent.parent / "VERSION"
	return version_path.read_text(encoding="utf-8").strip()


try:
	__version__ = metadata.version("universal_solver")
except metadata.PackageNotFoundError:
	__version__ = _read_local_version()


__all__ = ["__version__"]
