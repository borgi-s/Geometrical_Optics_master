# tests/test_scoring_library.py
import h5py
import numpy as np
import pytest

from dfxm_geo.scoring.library import load_library
from dfxm_geo.scoring.types import GridSpec


def _write_master(tmp_path, name, candidates, psize_m=1.0e-7):
    """Build an identify master + per-scan detector files mirroring the real
    BLISS layout: frames live in a per-scan file linked via ExternalLink."""
    master = tmp_path / name
    with h5py.File(master, "w") as h:
        for i, c in enumerate(candidates, start=1):
            entry = h.create_group(f"{i}.1")
            s = entry.create_group("sample")
            s["slip_plane_normal"] = np.asarray(c["plane"], dtype=np.int32)
            s["burgers"] = np.asarray(c["b"], dtype=np.int32)
            s["rotation_deg"] = float(c["alpha"])
            g = entry.create_group("dfxm_geo")
            g["gb_cos"] = float(c.get("gb_cos", 0.8))
            g["gb_visible"] = np.int8(c.get("gb_visible", 1))
            g["q_hkl"] = np.asarray(c.get("q_hkl", [0.0, 2.0, 0.0]), dtype=np.float64)
            g["psize"] = float(psize_m)
            # per-scan detector file + external link
            scan = tmp_path / f"{name}_scan{i:04d}.h5"
            with h5py.File(scan, "w") as d:
                d.create_dataset(
                    "/entry_0000/dfxm_sim_detector/image", data=c["frame"][None].astype(np.float32)
                )
            inst = entry.create_group("instrument").create_group("dfxm_sim_detector")
            inst["data"] = h5py.ExternalLink(scan.name, "/entry_0000/dfxm_sim_detector/image")
    return master


def _blob(shape, yx):
    a = np.zeros(shape, dtype=np.float32)
    y, x = yx
    a[y : y + 3, x : x + 3] = 100.0
    return a


def test_load_single_master(tmp_path):
    cands = [
        {"plane": (1, 1, 1), "b": (1, 0, 1), "alpha": 0.0, "frame": _blob((12, 12), (2, 2))},
        {"plane": (1, 1, 1), "b": (0, 1, 1), "alpha": 30.0, "frame": _blob((12, 12), (7, 7))},
    ]
    master = _write_master(tmp_path, "dfxm_identify.h5", cands)
    lib = load_library(master)
    assert len(lib) == 2
    assert lib.frames.shape == (2, 12, 12)
    assert lib.frames.dtype == np.float32
    assert lib.labels[0].slip_plane_normal == (1, 1, 1)
    assert lib.labels[0].burgers == (1, 0, 1)
    assert lib.labels[1].rotation_deg == 30.0
    assert lib.labels[0].scan_index == 1
    assert lib.grid == GridSpec(pitch_um=(0.1, 0.1), shape=(12, 12))  # psize 1e-7 m -> 0.1 um


def test_load_missing_psize_raises(tmp_path):
    cands = [{"plane": (1, 1, 1), "b": (1, 0, 1), "alpha": 0.0, "frame": _blob((8, 8), (1, 1))}]
    master = _write_master(tmp_path, "dfxm_identify.h5", cands)
    with h5py.File(master, "r+") as h:
        del h["1.1/dfxm_geo/psize"]
    with pytest.raises(ValueError, match="psize"):
        load_library(master)


def test_invisible_filter(tmp_path):
    cands = [
        {
            "plane": (1, 1, 1),
            "b": (1, 0, 1),
            "alpha": 0.0,
            "gb_visible": 1,
            "frame": _blob((8, 8), (1, 1)),
        },
        {
            "plane": (1, 1, 1),
            "b": (1, -1, 0),
            "alpha": 0.0,
            "gb_visible": 0,
            "frame": _blob((8, 8), (4, 4)),
        },
    ]
    master = _write_master(tmp_path, "dfxm_identify.h5", cands)
    assert len(load_library(master)) == 1  # invisible dropped
    assert len(load_library(master, include_invisible=True)) == 2


def test_directory_concatenation(tmp_path):
    d1 = tmp_path / "seed00001"
    d1.mkdir()
    d2 = tmp_path / "seed00002"
    d2.mkdir()
    _write_master(
        d1,
        "dfxm_identify.h5",
        [{"plane": (1, 1, 1), "b": (1, 0, 1), "alpha": 0.0, "frame": _blob((8, 8), (1, 1))}],
    )
    _write_master(
        d2,
        "dfxm_identify.h5",
        [{"plane": (1, 1, 1), "b": (0, 1, 1), "alpha": 0.0, "frame": _blob((8, 8), (4, 4))}],
    )
    lib = load_library(tmp_path)
    assert len(lib) == 2


def test_multiframe_reduction(tmp_path):
    master = tmp_path / "dfxm_identify.h5"
    with h5py.File(master, "w") as h:
        e = h.create_group("1.1")
        s = e.create_group("sample")
        s["slip_plane_normal"] = np.asarray((1, 1, 1), np.int32)
        s["burgers"] = np.asarray((1, 0, 1), np.int32)
        s["rotation_deg"] = 0.0
        g = e.create_group("dfxm_geo")
        g["gb_cos"] = 0.8
        g["gb_visible"] = np.int8(1)
        g["q_hkl"] = np.asarray([0.0, 2.0, 0.0])
        g["psize"] = 1.0e-7
        scan = tmp_path / "scan1.h5"
        stack = np.zeros((3, 8, 8), np.float32)
        stack[1, 2, 2] = 50.0
        with h5py.File(scan, "w") as d:
            d["/entry_0000/dfxm_sim_detector/image"] = stack
        e.create_group("instrument").create_group("dfxm_sim_detector")["data"] = h5py.ExternalLink(
            "scan1.h5", "/entry_0000/dfxm_sim_detector/image"
        )
    lib = load_library(master, frame_reduction="max")
    assert lib.frames.shape == (1, 8, 8)
    assert lib.frames[0, 2, 2] == 50.0  # max-projection kept the peak


def test_non_uniform_grid_raises(tmp_path):
    cands = [
        {"plane": (1, 1, 1), "b": (1, 0, 1), "alpha": 0.0, "frame": _blob((8, 8), (1, 1))},
        {"plane": (1, 1, 1), "b": (0, 1, 1), "alpha": 0.0, "frame": _blob((10, 10), (4, 4))},
    ]
    master = _write_master(tmp_path, "dfxm_identify.h5", cands)
    with pytest.raises(ValueError, match="non-uniform grid"):
        load_library(master)
