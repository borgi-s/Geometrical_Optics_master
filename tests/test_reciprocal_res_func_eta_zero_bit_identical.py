"""Eta=0 in the new code path must produce a bit-identical LUT to the legacy path.

This is the gate for adding `eta` without breaking v2.2.0.

Kwarg adaptations from the plan:
- The plan's test called ``reciprocal_res_func(**common, output_path=None, rng=rng_a)``
  and expected a LUT array back.  In reality ``reciprocal_res_func`` returns ``None``
  (not a LUT) unless ``return_qs=True``.  This test uses ``return_qs=True``, which
  returns ``(qrock, qroll, qpar, qrock_prime, q2th, delta_2theta)`` — the full point
  cloud after all transforms.  Comparing these arrays with
  ``np.testing.assert_array_equal`` is strictly stronger than comparing the binned
  histogram, so the test is TIGHTER than what the plan described.

- The plan's ``phys_aper=4.0e-4`` is too narrow for ``Nrays=1e5``: roughly half
  the Gaussian NA draws fall outside the aperture, leaving < Nrays survivors.
  Replaced with the canonical test value from ``test_reciprocal_resolution.py``:
  ``phys_aper = 2*sqrt(50e-6 * 1.6e-3) / 0.274`` (~6.6e-3), which is the
  physical aperture from the published CDD_inc setup.  ``Nrays`` is reduced to
  10_000 (fast; the bit-identity check doesn't benefit from large Nrays).

- ``output_path=None`` is kept (no file written); ``save_resqi=False`` likewise.
- All other kwargs match the plan exactly.
"""

import numpy as np

from dfxm_geo.reciprocal_space.resolution import reciprocal_res_func

# Physical aperture from the published CDD_inc / test_reciprocal_resolution canon.
_PHYS_APER = float(2 * np.sqrt(50e-6 * 1.6e-3) / 0.274)


def test_eta_zero_bit_identical_to_omitted_eta(tmp_path) -> None:
    common = dict(
        Nrays=10_000,
        npoints1=50,
        npoints2=40,
        npoints3=40,
        qi1_range=5e-4,
        qi2_range=7.5e-3,
        qi3_range=7.5e-3,
        plot_figs=False,
        save_resqi=False,
        zeta_v_fwhm=5.3e-4,
        zeta_h_fwhm=0.0,
        NA_rms=7.31e-4 / 2.35,
        eps_rms=1.41e-4 / 2.35,
        theta=0.2691,  # paper Figure 3B Bragg angle
        phys_aper=_PHYS_APER,
        date="20260528_test",
        beamstop=False,
        bs_height=0.025,
        aperture=False,
        knife_edge=False,
        dphi_range=0.0,
        output_path=None,
        return_qs=True,
    )
    rng_a = np.random.default_rng(seed=42)
    rng_b = np.random.default_rng(seed=42)
    # result_a: no eta keyword (legacy path)
    result_a = reciprocal_res_func(**common, rng=rng_a)
    # result_b: explicit eta=0.0 (must be bit-identical to legacy path)
    result_b = reciprocal_res_func(**common, eta=0.0, rng=rng_b)
    assert result_a is not None
    assert result_b is not None
    # Compare every returned array element-wise (bit-exact)
    for i, (arr_a, arr_b) in enumerate(zip(result_a, result_b, strict=True)):
        np.testing.assert_array_equal(
            arr_a,
            arr_b,
            err_msg=f"return_qs tuple element {i} differs between eta=omitted and eta=0.0",
        )
