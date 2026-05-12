"""DEPRECATED — see `dfxm_geo.reciprocal_space.kernel`.

This shim runs the new kernel-generation script. New invocations should be:

    python -m dfxm_geo.reciprocal_space.kernel
"""

from dfxm_geo.reciprocal_space import kernel  # noqa: F401
