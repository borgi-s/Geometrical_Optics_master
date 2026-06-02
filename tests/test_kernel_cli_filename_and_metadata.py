"""New oblique LUT filename pattern + extended metadata; legacy pattern preserved."""

from dfxm_geo.reciprocal_space.kernel import _build_kernel_filename


def test_simplified_filename_unchanged_from_v220() -> None:
    name = _build_kernel_filename(
        mode="simplified",
        hkl=(-1, 1, -1),
        keV=17.0,
        theta=0.0,
        eta=0.0,
        date="20260528_1430",
    )
    assert name == "Resq_i_h-1_k1_l-1_17keV_20260528_1430.npz"


def test_oblique_filename_uses_theta_eta_pattern() -> None:
    name = _build_kernel_filename(
        mode="oblique",
        hkl=(-1, -1, 3),
        keV=19.1,
        theta=0.2691,
        eta=0.3531,
        date="20260528_1430",
    )
    assert name == "Resq_i_theta0.2691rad_eta0.3531rad_19.1keV_20260528_1430.npz"
