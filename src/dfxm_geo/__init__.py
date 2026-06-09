"""Geometrical-optics forward model for dark-field X-ray microscopy."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dfxm-geo")
except PackageNotFoundError:  # source tree without an install
    __version__ = "0.0.0+unknown"
