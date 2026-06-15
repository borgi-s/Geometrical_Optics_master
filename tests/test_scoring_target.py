import numpy as np

from dfxm_geo.scoring.target import _crop_or_pad, resample_to_grid
from dfxm_geo.scoring.types import GridSpec


def test_crop_or_pad_crops_centered():
    a = np.arange(36).reshape(6, 6).astype(float)
    out = _crop_or_pad(a, (4, 4))
    assert out.shape == (4, 4)
    assert out[0, 0] == a[1, 1]  # centered crop


def test_crop_or_pad_pads_centered():
    a = np.ones((2, 2))
    out = _crop_or_pad(a, (4, 4))
    assert out.shape == (4, 4)
    assert out.sum() == 4.0  # only original ones survive
    assert out[0, 0] == 0.0 and out[1, 1] == 1.0


def test_resample_matches_native_scale():
    # a blob rendered at 2x coarser pitch, resampled to the fine grid, should
    # closely match the natively-fine blob (same physical extent).
    grid = GridSpec(pitch_um=(0.5, 0.5), shape=(40, 40))
    yy, xx = np.mgrid[0:40, 0:40]
    fine = np.exp(-(((yy - 20) ** 2 + (xx - 20) ** 2) / (2 * 4.0**2)))
    yy2, xx2 = np.mgrid[0:20, 0:20]
    coarse = np.exp(-(((yy2 - 10) ** 2 + (xx2 - 10) ** 2) / (2 * 2.0**2)))
    out = resample_to_grid(coarse, src_pitch_um=(1.0, 1.0), grid=grid)
    assert out.shape == (40, 40)
    # peak near centre, correlation with the native fine blob is high
    num = float((out * fine).sum())
    den = float(np.sqrt((out**2).sum() * (fine**2).sum()))
    assert num / den > 0.95
