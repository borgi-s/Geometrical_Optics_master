"""Derive the absolute calibration constant ``counts_scale`` (ADU/s per
normalized-intensity unit) by matching a simulated weak-beam dislocation
feature to a measured ID03 frame.

This is a LOCAL scientific calibration, NOT a CI test. It is *provenance*:
it records how the ``DetectorConfig.counts_scale`` default was anchored to
real data. The experimental frames are Sina's ID03 beamtime data and live
ONLY on the local machine (not in the repo). Reading the LIMA frames
REQUIRES ``hdf5plugin`` (bitshuffle) imported before ``h5py.File``.

The conversion the pipeline applies (orchestrator ``_apply_detector_model``)::

    adu = ideal_intensity * counts_scale * exposure_time          # then noise

so, matching the *integrated* feature signal between data and simulation::

    counts_scale = ADU_integral / (sim_integral * exposure_time)

Both integrals are taken over the largest connected weak-beam feature, after
``subtract_background(img, k=2.0)`` (mean + 2*std clamp at zero) — the same
reduction the beamline viewer uses and that ``viz/detector.subtract_background``
implements.

Physics-first cross-check (order of magnitude):

    expected_ADU_per_s ~= flux_ID03 * FOV_fraction
                          * run_exposure_simulation()[1] * gain(2.14)

``flux_ID03`` is the photon flux on the sample. We do NOT have web access here;
the value below is a DOCUMENTED ASSUMPTION to be confirmed against the beamline
record. ID03 is the ESRF Surface Diffraction / microdiffraction beamline; a
focused monochromatic beam at a 3rd-generation/EBS undulator source typically
delivers ~1e11-1e13 ph/s into a focal spot. We adopt FLUX_ID03 = 1e12 ph/s as
a mid-range order-of-magnitude assumption.
  ASSUMPTION — confirm with the ID03 beamline page / logbook for this beamtime
  (record number + URL to be filled in by the domain expert):
    ESRF ID03: https://www.esrf.fr/  (beamline ID03; flux figure unverified here)
The cross-check only needs to agree within ~1 order of magnitude (a factor of
~10); a larger disagreement points to a units bug, not a calibration tweak.

Run standalone with the venv python::

    C:\\Users\\borgi\\Documents\\GM-reworked\\.venv\\Scripts\\python.exe \\
        docs/calibration/derive_counts_scale.py

It PRINTS every key number (ADU_integral, sim_integral, counts_scale,
core-peak*counts_scale, and the physics cross-check).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  # REQUIRED before h5py.File for LIMA bitshuffle frames
import numpy as np
from scipy.ndimage import label

from dfxm_geo.detector import PCO_EDGE_4P2_ID03
from dfxm_geo.pipeline import (
    AxisScanConfig,
    CenteredCrystalConfig,
    CrystalConfig,
    DetectorConfig,
    IOConfig,
    ReciprocalConfig,
    ScanConfig,
    SimulationConfig,
    run_simulation,
)
from dfxm_geo.reciprocal_space.exposure import run_exposure_simulation
from dfxm_geo.viz.detector import subtract_background

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
DATA_ROOT = Path(r"C:\Users\borgi\Documents\GM-reworked\experimental_data")

# 61x1s focus sweep over a field of individual dislocations; frame 30 is the
# in-focus middle frame.
DISLOCATION_H5 = (
    DATA_ROOT / "111_individual_dislocations_10x_focusing_2" / "scan0001" / "pco_ff_0000.h5"
)
LIMA_DATA_PATH = "entry_0000/ESRF-ID03/pco_ff/data"
FRAME_INDEX = 30

EXPOSURE_TIME = 1.0  # s, both data (LIMA acquisition/exposure_time) and sim

# Measured detector offset at 1 s (the dark floor the model reproduces):
# DetectorModel.offset(1.0) = offset_base + dark_rate*1 = 102.5 + 7.5 = 110 ADU.
MEASURED_OFFSET_ADU = PCO_EDGE_4P2_ID03.offset(EXPOSURE_TIME)  # 110.0 ADU

# Weak-beam forward scene (reuse M5 notebook-03 weak-beam settings):
#   - Al (1,-1,1)-family reflection at 17 keV  -> kernel hkl=(-1,1,-1)
#   - single dislocation (centered)            -> CrystalConfig centered default
#   - weak-beam phi offset a few times the rocking width: phi = 1.25e-4 rad
#     (the notebook-03 rocking *half-range* is 1.25e-4 rad; the weak-beam frame
#     sits at +1.25e-4 rad, i.e. ~3x a ~40 urad rocking width)
#   - model = "ideal" so the detector image stays normalized-intensity float
REFLECTION_HKL = (-1, 1, -1)
ENERGY_KEV = 17.0
WEAK_BEAM_PHI = 1.25e-4  # rad

# Physics cross-check: documented assumption (see module docstring).
FLUX_ID03_PH_PER_S = 1.0e12  # ph/s on sample — ASSUMPTION, confirm at beamline

# Guard thresholds (the GATE).
GUARD_A_LO, GUARD_A_HI = 2_000.0, 6_000.0  # core-peak * counts_scale, ADU
GUARD_B_FACTOR = 10.0  # data anchor vs physics estimate must agree within ~10x


def _largest_feature_integral(img: np.ndarray, k: float = 2.0) -> tuple[float, float, int]:
    """Background-subtract, select the brightest connected feature, integrate.

    Returns ``(integral, peak, n_pixels)`` where ``integral`` is the sum of the
    background-subtracted values over the largest (by pixel count) connected
    above-background feature, ``peak`` is that feature's brightest pixel, and
    ``n_pixels`` is its footprint. All in the *input* image's units (ADU for
    data, normalized intensity for the sim).
    """
    above = subtract_background(img, k=k)
    mask = above > 0.0
    if not mask.any():
        raise RuntimeError("no above-background pixels — degenerate feature")
    labels, n = label(mask)
    if n == 0:
        raise RuntimeError("scipy.ndimage.label found no features")
    # Largest feature by pixel count (the dislocation core+arms; isolated hot
    # pixels survive as tiny 1-px labels and are not the feature of interest).
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # background label
    feat = int(sizes.argmax())
    feat_mask = labels == feat
    integral = float(above[feat_mask].sum())
    peak = float(above[feat_mask].max())
    return integral, peak, int(feat_mask.sum())


def measure_experimental() -> tuple[float, float, int]:
    """Per-feature integrated ADU above background, from ID03 frame 30."""
    with h5py.File(DISLOCATION_H5, "r") as f:
        frame = f[LIMA_DATA_PATH][FRAME_INDEX].astype(np.float64)
    # Subtract the measured dark offset, then background-select + integrate.
    frame_minus_offset = np.clip(frame - MEASURED_OFFSET_ADU, 0.0, None)
    integral, peak, npix = _largest_feature_integral(frame_minus_offset, k=2.0)
    return integral, peak, npix


def simulate_weak_beam() -> tuple[np.ndarray, float, float, float, int]:
    """Run the weak-beam single-dislocation forward scene (model='ideal').

    Returns ``(image, sim_integral, sim_core_peak, fov_fraction, n_pixels)``.
    The image is the noiseless normalized-intensity float frame.
    """
    import tempfile

    out = Path(tempfile.mkdtemp(prefix="counts_scale_")) / "run"
    cfg = SimulationConfig(
        crystal=CrystalConfig(mode="centered", centered=CenteredCrystalConfig()),
        scan=ScanConfig(phi=AxisScanConfig(value=WEAK_BEAM_PHI)),
        io=IOConfig(include_perfect_crystal=False, write_strain_provenance=False),
        detector=DetectorConfig(model="ideal"),  # keep float normalized-intensity
        reciprocal=ReciprocalConfig(hkl=REFLECTION_HKL, keV=ENERGY_KEV, backend="mc"),
    )
    run_simulation(cfg, out)
    from dfxm_geo.io.hdf5 import DETECTOR_INTERNAL_PATH

    det = out / "scan0001" / "dfxm_sim_detector_0000.h5"
    with h5py.File(det, "r") as f:
        img = f[DETECTOR_INTERNAL_PATH][0].astype(np.float64)  # (ny, nx)
    integral, peak, npix = _largest_feature_integral(img, k=2.0)
    fov_fraction = npix / img.size
    return img, integral, peak, fov_fraction, npix


def main() -> None:
    print("=" * 70)
    print("counts_scale derivation — ID03 weak-beam dislocation anchor")
    print("=" * 70)
    print(f"data frame : {DISLOCATION_H5}")
    print(f"             frame {FRAME_INDEX}, exposure {EXPOSURE_TIME} s")
    print(f"offset     : {MEASURED_OFFSET_ADU:.1f} ADU (subtracted)")
    print(
        f"sim scene  : Al hkl={REFLECTION_HKL} @ {ENERGY_KEV} keV, single dislocation,"
        f" weak-beam phi={WEAK_BEAM_PHI:.2e} rad, model='ideal', MC backend, px510"
    )
    print("-" * 70)

    # --- experimental integral ------------------------------------------------
    adu_integral, adu_peak, adu_npix = measure_experimental()
    print(f"ADU_integral (data, largest feature) : {adu_integral:.6g} ADU")
    print(f"  feature footprint                  : {adu_npix} px, peak {adu_peak:.1f} ADU")

    # --- simulated integral ---------------------------------------------------
    _img, sim_integral, sim_peak, fov_fraction, sim_npix = simulate_weak_beam()
    print(f"sim_integral (sim, largest feature)  : {sim_integral:.6g}  (norm. intensity)")
    print(f"  feature footprint                  : {sim_npix} px, core peak {sim_peak:.6g}")
    print(f"  FOV fraction (feature/frame)       : {fov_fraction:.4g}")

    # --- the calibration constant --------------------------------------------
    counts_scale = adu_integral / (sim_integral * EXPOSURE_TIME)
    print("-" * 70)
    print("==> counts_scale = ADU_integral / (sim_integral * t)")
    print(f"==> counts_scale = {counts_scale:.6g}  ADU/s per normalized-intensity unit")
    print("-" * 70)

    # --- Guard A: core-peak sanity -------------------------------------------
    core_peak_adu = sim_peak * counts_scale * EXPOSURE_TIME
    guard_a = GUARD_A_LO <= core_peak_adu <= GUARD_A_HI
    print(
        f"Guard A (core-peak): sim_core_peak * counts_scale * t = {core_peak_adu:.1f} ADU "
        f"(target {GUARD_A_LO:.0f}-{GUARD_A_HI:.0f}) -> {'PASS' if guard_a else 'FAIL'}"
    )

    # --- Guard B: physics-first cross-check ----------------------------------
    _total, fraction_transmitted = run_exposure_simulation()
    expected_adu_per_s = (
        FLUX_ID03_PH_PER_S * fov_fraction * fraction_transmitted * PCO_EDGE_4P2_ID03.gain
    )
    # Compare against the data-anchored ADU/s the feature actually carries:
    derived_adu_per_s = adu_integral / EXPOSURE_TIME
    ratio = expected_adu_per_s / derived_adu_per_s if derived_adu_per_s else float("inf")
    guard_b = (1.0 / GUARD_B_FACTOR) <= ratio <= GUARD_B_FACTOR
    print(
        f"Guard B (physics):  flux({FLUX_ID03_PH_PER_S:.1e}) * FOV_frac({fov_fraction:.3g})"
        f" * transmission({fraction_transmitted:.4g}) * gain({PCO_EDGE_4P2_ID03.gain})"
    )
    print(f"  expected_ADU_per_s (physics) : {expected_adu_per_s:.6g} ADU/s")
    print(f"  derived_ADU_per_s  (data)    : {derived_adu_per_s:.6g} ADU/s")
    print(
        f"  ratio expected/derived       : {ratio:.4g} "
        f"(within {GUARD_B_FACTOR:.0f}x?) -> {'PASS' if guard_b else 'FAIL'}"
    )
    print("    [flux_ID03 is a documented ASSUMPTION — confirm at beamline]")

    print("=" * 70)
    print(
        f"GATE: Guard A {'PASS' if guard_a else 'FAIL'} / "
        f"Guard B {'PASS' if guard_b else 'FAIL'} -> "
        f"{'PIN' if (guard_a and guard_b) else 'DO NOT PIN'}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
