"""One-shot wall-clock check that the static hoist landed. Not a pytest test.

Run: python tests/_perf_static_hoist.py
Reports per-frame cost of the OLD path (forward(Hg) per frame, recomputing qs)
vs the NEW path (precompute_forward_static once + forward_from_static per frame).
"""

import time

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.pipeline import _lookup_and_load_kernel

_lookup_and_load_kernel((-1, 1, -1), 17.0)
Hg, q_hkl = fm.Find_Hg(4.0, 1, fm.psize, fm.zl_rms)
fm.q_hkl = q_hkl

N = 25  # one 5x5 scan's worth of frames
angles = [(1e-4 * (k % 5), 2e-4 * (k // 5), 0.0) for k in range(N)]

# warm
fm.forward(Hg, phi=0.0)

# OLD: recompute qs every frame
t = time.perf_counter()
for phi, chi, dt in angles:
    fm.forward(Hg, phi=phi, chi=chi, TwoDeltaTheta=dt)
old = time.perf_counter() - t

# NEW: precompute once, dynamic per frame
t = time.perf_counter()
base_qc = fm.precompute_forward_static(Hg)
for phi, chi, dt in angles:
    fm.forward_from_static(base_qc, phi=phi, chi=chi, TwoDeltaTheta=dt)
new = time.perf_counter() - t

print(f"OLD  {N} frames (forward per frame):        {old:7.2f}s  ({old / N * 1000:6.1f} ms/frame)")
print(f"NEW  {N} frames (precompute + from_static): {new:7.2f}s  ({new / N * 1000:6.1f} ms/frame)")
print(f"per-scan speedup: {old / new:4.2f}x")
