import inspect

from dfxm_geo import pipeline


def test_multi_identify_generator_does_not_assign_fm_q_hkl():
    """_iter_identification_multi must not assign the fm.q_hkl global.

    Regression for #10: a generator writing a process global is a data race
    on the planned persistent-worker pool. The original write was a no-op
    (assigned back the value just read), so this guards the source directly.
    """
    src = inspect.getsource(pipeline._iter_identification_multi)
    # No assignment to fm.q_hkl in any spacing (reads `... = np.asarray(fm.q_hkl...)`
    # are fine; only `fm.q_hkl = ...` is the forbidden write).
    compact = src.replace(" ", "")
    assert "fm.q_hkl=" not in compact, "generator still assigns fm.q_hkl"
