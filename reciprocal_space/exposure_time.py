"""DEPRECATED — see `dfxm_geo.reciprocal_space.exposure`.

This shim runs the new exposure-time script. New invocations should be:

    python -m dfxm_geo.reciprocal_space.exposure
"""

from dfxm_geo.reciprocal_space import exposure  # noqa: F401
