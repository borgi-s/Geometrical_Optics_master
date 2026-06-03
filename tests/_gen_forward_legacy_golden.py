"""RETIRED — golden was captured before v1.2.0; do not re-run.

Historical note: this script originally generated
``tests/data/golden/forward_legacy_writer_4frames_8x8.npy`` via the legacy
``.npy`` writer (``save_images_parallel``).  ``save_images_parallel`` was
removed in S4 of the ForwardContext refactor (#16) — identification migrated
to HDF5 output in v1.2.0 and the dead code had no remaining caller in src/.

The golden file is still valid and still consumed by test_hdf5_bit_equiv.py
(via the HDF5 writer path).  To regenerate it, use
``tests/_gen_forward_legacy_golden_hdf5.py`` (future replacement) or simply
delete and let the allclose tolerance in test_hdf5_bit_equiv.py catch any
real regression.
"""
