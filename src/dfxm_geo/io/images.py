"""Worker-count helper for identification HDF5 writer (io/hdf5.py).

The legacy `.npy` write path (``save_images_parallel`` / ``save_image``) was
removed in S4 of the ForwardContext refactor (#16) — identification migrated
to HDF5 output in v1.2.0 and the dead code had no remaining caller in src/.
All read helpers (``load_image``, ``load_images``, ``load_images_parallel``)
and the legacy EDF writer (``save_edfs``) were removed in v1.1.0; see
``io/migrate.py`` for the migration helper and ``io/hdf5.py`` for the HDF5
equivalents.

The only public surface is ``_auto_max_workers``, re-exported by ``io/hdf5.py``.
"""

import os

from dfxm_geo.direct_space import forward_model as _fm

# Measured per-worker transient peak of `forward_from_static`, in bytes per
# ray (NN1*NN2*NN3). The live intermediates per call are `qi` (3*8 B/ray),
# the three int16 index arrays (3*2 B/ray), the bool mask, and `prob`
# (~8 B/ray plus a float32 copy) — measured at ~48 B/ray on the px510/Nsub=1
# detector. We budget 2x that as a safety envelope (allocator overhead +
# transient double-buffering like `prob.astype(np.float32)`). This SCALES
# with ray count, so Nsub=2 publication runs (8x rays) reserve ~8x per worker
# automatically. `base_qc` is shared read-only across workers (24 B/ray,
# counted once below), not per-worker. See tests/_perf_static_hoist.py and
# the static-hoist handoff for the measurement.
_PER_WORKER_BYTES_PER_RAY = 96
_BASE_QC_BYTES_PER_RAY = 24


def _auto_max_workers() -> int:
    """Cap thread-pool workers by both CPU count and free memory.

    Per-worker memory is estimated from the *actual* detector ray count
    (``NN1*NN2*NN3``) at ~96 B/ray (see ``_PER_WORKER_BYTES_PER_RAY``), not a
    fixed slab — the static hoist made ``base_qc`` a shared read-only array, so
    the per-worker footprint is just the dynamic intermediates and is small at
    Nsub=1 (~67 MiB/worker) while scaling up for Nsub=2 publication runs.

    Resolution order (highest precedence wins):
      1. ``DFXM_MAX_WORKERS`` env var, if set to a positive int.
      2. ``min(cpu_count, usable_gb / per_worker_gb)`` if psutil is installed,
         where ``usable_gb`` reserves ~2 GiB for persistent forward_model state
         (the Resq_i LUT, Hg) + OS, minus the once-shared ``base_qc``.
      3. ``min(cpu_count, 4)`` fixed conservative cap when psutil is missing.
    """
    env = os.environ.get("DFXM_MAX_WORKERS")
    if env is not None:
        try:
            n = int(env)
            if n >= 1:
                return n
        except ValueError:
            pass  # fall through to auto-detect

    cpu = os.cpu_count() or 1
    try:
        import psutil
    except ImportError:
        return min(cpu, 4)

    ray_count = (_fm.NN1 * _fm.NN2 * _fm.NN3) or 1
    base_qc_gb = ray_count * _BASE_QC_BYTES_PER_RAY / (1024**3)  # shared once
    per_worker_gb = ray_count * _PER_WORKER_BYTES_PER_RAY / (1024**3)

    avail_gb = psutil.virtual_memory().available / (1024**3)
    # Reserve ~2 GiB for persistent state + OS and the once-shared base_qc.
    usable_gb = max(0.5, avail_gb - 2.0 - base_qc_gb)
    mem_cap = max(1, int(usable_gb / per_worker_gb))
    return min(cpu, mem_cap)
