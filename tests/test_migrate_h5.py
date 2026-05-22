"""Round-trip test for dfxm-migrate-h5 (v1.1.0 single-file -> v1.2.0 master+per-scan)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from dfxm_geo.io.hdf5 import DETECTOR_INTERNAL_PATH
from dfxm_geo.io.migrate import migrate_h5_master_to_master


def _build_v110_fixture(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Build a minimal v1.1.0 single-file dfxm_geo.h5 fixture in-place.

    Returns (dislocs_stack, perfect_stack) for later equality checks.
    """
    dislocs = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    perfect = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4) * -1.0
    with h5py.File(path, "w") as f:
        # /dfxm_geo/ provenance
        g = f.create_group("dfxm_geo")
        g.create_dataset("cli", data="dfxm-forward (fixture)")
        g.create_dataset("version", data="1.1.0")
        g.create_dataset("config_toml", data="")
        # /1.1
        s1 = f.create_group("1.1")
        s1.attrs["NX_class"] = "NXentry"
        s1.create_dataset("title", data="fscan2d phi -0.01 0.01 2 chi 0 0 1 1.0")
        s1.create_dataset("start_time", data="2026-05-20T00:00:00")
        s1.create_dataset("end_time", data="2026-05-20T00:00:01")
        instr = s1.create_group("instrument")
        instr.attrs["NX_class"] = "NXinstrument"
        det = instr.create_group("dfxm_sim_detector")
        det.attrs["NX_class"] = "NXdetector"
        det.create_dataset("data", data=dislocs)
        pos = instr.create_group("positioners")
        pos.attrs["NX_class"] = "NXcollection"
        pos.create_dataset("phi", data=np.array([-0.01, 0.01]))
        pos["phi"].attrs["units"] = "degree"
        pos.create_dataset("chi", data=np.zeros(2))
        pos["chi"].attrs["units"] = "degree"
        samp = s1.create_group("sample")
        samp.attrs["NX_class"] = "NXsample"
        samp.create_dataset("name", data="simulated, dislocations")
        samp.create_dataset("dis", data=4.0)
        samp.create_dataset("ndis", data=151)
        samp.create_dataset("sample_remount", data="S1")
        d = s1.create_group("dfxm_geo")
        d.create_dataset("Hg", data=np.eye(3).reshape(1, 3, 3))
        d.create_dataset("q_hkl", data=np.array([0.0, 0.0, 1.0]))
        # /2.1 perfect crystal
        s2 = f.create_group("2.1")
        s2.attrs["NX_class"] = "NXentry"
        det2 = s2.create_group("instrument/dfxm_sim_detector")
        det2.attrs["NX_class"] = "NXdetector"
        det2.create_dataset("data", data=perfect)
    return dislocs, perfect


def test_migrate_h5_master_to_master_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "v110.h5"
    dst_dir = tmp_path / "v120"
    dislocs, perfect = _build_v110_fixture(src)
    migrate_h5_master_to_master(src, dst_dir)

    new_master = dst_dir / "dfxm_geo.h5"
    assert new_master.is_file()
    assert (dst_dir / "scan0001" / "dfxm_sim_detector_0000.h5").is_file()
    assert (dst_dir / "scan0002" / "dfxm_sim_detector_0000.h5").is_file()
    with h5py.File(new_master, "r") as f:
        # Pixels follow the ExternalLink to the new layout
        np.testing.assert_array_equal(f["/1.1/instrument/dfxm_sim_detector/data"][...], dislocs)
        np.testing.assert_array_equal(f["/2.1/instrument/dfxm_sim_detector/data"][...], perfect)
        # Provenance copied
        assert f["/dfxm_geo/version"][()].decode() == "1.1.0"
        # Sample, positioners, dfxm_geo nodes preserved
        assert "/1.1/sample/name" in f
        assert "/1.1/instrument/positioners/phi" in f
        assert "/1.1/dfxm_geo/Hg" in f
        # DETECTOR_INTERNAL_PATH constant is the link target inside per-scan file
        assert DETECTOR_INTERNAL_PATH == "/entry_0000/dfxm_sim_detector/image"
