"""Forward + identification run orchestration — extracted from pipeline.py
(refactor gate, 2026-06-11).

``run_simulation`` / ``run_identification`` / ``run_postprocess`` and their
helpers live here. Imports config names DIRECTLY from ``dfxm_geo.config``
(never from ``dfxm_geo.pipeline`` — the facade re-exports orchestrator, so a
back-import would cycle). Tests that monkeypatch orchestration internals
(``_load_resolution``, ``find_hg_scene``, the ``_run_identification_*``
runners, ``write_simulation_h5``, ``_KERNEL_CTX_CACHE``) patch THIS module —
the facade binding does not reach the bare-name call sites here.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import dfxm_geo.direct_space.forward_model as fm
from dfxm_geo.analysis.mosaicity import compute_com_maps
from dfxm_geo.config import (
    _CANONICAL_AXES,
    GeometryConfig,
    IdentificationConfig,
    ReciprocalConfig,
    ScanConfig,
    ScanFrames,
    SimulationConfig,
    _dataclass_to_toml_str,
    _identification_config_to_toml_str,
    run_theta,
)
from dfxm_geo.crystal.burgers import (
    burgers_vectors as _burgers_vectors,
)
from dfxm_geo.crystal.burgers import (
    gb_cos as _gb_cos,
)
from dfxm_geo.crystal.burgers import (
    gb_visible as _gb_visible,
)
from dfxm_geo.crystal.burgers import (
    rotated_t_vectors as _rotated_t_vectors,
)
from dfxm_geo.crystal.burgers import (
    ud_matrices as _ud_matrices,
)
from dfxm_geo.crystal.dislocations import (
    MixedDislocSpec,
    find_hg_scene,
)
from dfxm_geo.crystal.reflections import (
    ReflectionRun as _ReflectionRun,
)
from dfxm_geo.crystal.remount import SAMPLE_REMOUNT_OPTIONS
from dfxm_geo.io.hdf5 import (
    DETECTOR_FILE_FMT,
    DETECTOR_INTERNAL_PATH,
    SCAN_DIR_FMT,
    ScanSpec,
    geometry_provenance_attrs,
    load_h5_scan,
    write_identification_h5,
    write_multi_reflection_master,
    write_simulation_h5,
)
from dfxm_geo.viz.mosaicity import plot_mosaicity_maps, plot_qi_cross_section

# #16: module-level cache so repeated _lookup_and_load_kernel calls for the same
# kernel path return the already-loaded ResolutionContext without hitting disk
# again. The sole idempotency authority (Slice 5 removed the per-process
# kernel-path global the check used to cross-reference).
_KERNEL_CTX_CACHE: dict[Path, fm.ResolutionContext] = {}

# === FORWARD ORCHESTRATION (kernel/resolution, run_simulation, postprocess) ===


def _lookup_and_load_kernel(
    hkl: tuple[int, int, int],
    keV: float,
    *,
    geometry: GeometryConfig | None = None,
    skip_hkl_check: bool = False,
) -> fm.ResolutionContext:
    """Pre-flight: look up the kernel npz for this geometry and load it.

    Sub-project D replacement for `_ensure_kernel_loaded()`. Composes a lookup
    and a load:

    - simplified (default / `geometry=None`): glob the legacy
      ``Resq_i_h{h}_k{k}_l{l}_{keV}keV_*.npz`` pattern by (hkl, keV).
    - oblique (v2.3.0): glob the ``Resq_i_theta{θ}rad_eta{η}rad_{keV}keV_*.npz``
      pattern by the validated (theta, eta, keV). ``geometry.theta_validated``
      is the solver value baked into the bootstrap filename, so the match is
      exact.

    Both modes still verify the npz's bundled (hkl, keV) metadata on load — the
    oblique bootstrap records them too.

    Idempotent: returns the cached ResolutionContext if the resolved target was
    already loaded this process. The module-level _KERNEL_CTX_CACHE is the sole
    authority (#16 Slice 5 removed the per-process kernel-path global it used to
    cross-check against).

    Raises KeyError on lookup miss, ValueError on metadata mismatch,
    KeyError on pre-sub-project-D legacy npz lacking metadata.

    Args:
        skip_hkl_check: Pass True for multi-reflection group members whose
            kernel was bootstrapped for the group REPRESENTATIVE's hkl (the
            LUT covers the group, not one hkl — spec §6). Single-reflection
            callers always leave this False so the sub-project-D metadata guard
            remains active.

    Returns:
        ResolutionContext snapshotting the newly (or previously) loaded kernel.
    """
    if geometry is not None and geometry.mode == "oblique":
        if geometry.theta_validated is None:
            raise ValueError(
                "oblique kernel lookup requires geometry.theta_validated "
                "(set by SimulationConfig.from_toml from the [geometry] block)."
            )
        target = fm._lookup_kernel_path(
            directory=fm.pkl_fpath,
            mode="oblique",
            theta=geometry.theta_validated,
            eta=geometry.eta,
            keV=keV,
        )
    else:
        target = fm._lookup_kernel_path(directory=fm.pkl_fpath, mode="simplified", hkl=hkl, keV=keV)
    # The module-level cache is authoritative now that no per-run globals exist
    # to fall out of sync with it (#16 Slice 5): a cache hit on the resolved path
    # returns the previously-built context without a disk reload.
    cached = _KERNEL_CTX_CACHE.get(target)
    if cached is not None:
        return cached
    res = fm._load_default_kernel(
        str(target),
        expected_hkl=None if skip_hkl_check else hkl,
        expected_keV=keV,
    )
    _KERNEL_CTX_CACHE[target] = res
    return res


def _load_resolution(
    config: ReciprocalConfig,
    geometry: GeometryConfig | None = None,
    *,
    skip_hkl_check: bool = False,
) -> fm.ResolutionContext:
    """Select and load the resolution backend per config (spec sec. 5.4).

    auto     -> analytic if beamstop off, else MC
    analytic -> analytic; ValueError if beamstop on (cannot represent it)
    mc       -> MC

    ``geometry`` routes the MC LUT lookup to the oblique pattern when
    ``geometry.mode == 'oblique'``; the analytic backend reads ``eta`` straight
    off ``config`` (already propagated from [geometry] by
    SimulationConfig.from_toml). Defaults to simplified when omitted.

    ``skip_hkl_check`` is forwarded to ``_lookup_and_load_kernel`` for
    multi-reflection group members whose kernel was bootstrapped for the group
    representative's hkl (spec §6). Single-reflection callers leave this False.

    Returns:
        ResolutionContext snapshotting the loaded/cached backend. Threaded into
        build_forward_context by the run functions (#16 Slice 5).
    """
    use_analytic = config.backend == "analytic" or (
        config.backend == "auto" and not config.beamstop
    )
    if config.backend == "analytic" and config.beamstop:
        raise ValueError(
            "backend='analytic' is incompatible with beamstop=True (the wire/"
            "knife-edge/aperture stop cannot be represented in closed form). "
            "Use backend='mc', or disable the beamstop."
        )
    # #16 Slice 5: the chosen backend is carried entirely on the returned
    # ResolutionContext (analytic_eval set for analytic, None for the LUT path),
    # so forward() reads ctx.resolution — no module-global toggling needed.
    if use_analytic:
        return fm._load_analytic_resolution(config)
    return _lookup_and_load_kernel(
        config.hkl, config.keV, geometry=geometry, skip_hkl_check=skip_hkl_check
    )


def _resolution_for_run(
    reciprocal: ReciprocalConfig,
    geometry: GeometryConfig,
    run: _ReflectionRun,
) -> fm.ResolutionContext:
    """Per-reflection resolution: build a run-specific (ReciprocalConfig,
    GeometryConfig) pair and delegate to ``_load_resolution``.

    Constructs ``recip_run`` (hkl and eta overridden to the run's values) and
    ``geom_run`` (oblique mode pinned to run.theta/run.eta/run.omega), then
    calls ``_load_resolution(recip_run, geom_run, skip_hkl_check=True)``.

    Full backend dispatch — including the ``backend='analytic' + beamstop=True``
    ValueError guard — is inherited from ``_load_resolution``.  No dispatch logic
    is duplicated here.

    For the analytic backend the per-run hkl drives the Bragg-angle derivation
    (via ``_load_analytic_resolution``), and the per-run eta feeds the analytic
    resolution function.  For the MC backend the group kernel is looked up by
    (run.theta, run.eta, keV) — the kernel's bundled hkl metadata is the
    bootstrap group REPRESENTATIVE's hkl, which may differ from run.hkl for
    group members; the hkl metadata check is therefore relaxed with
    ``skip_hkl_check=True`` (the (theta, eta, keV) match is the contract, spec §6).
    Single-reflection callers go through ``_load_resolution`` directly and keep
    the sub-project-D hkl guard active.
    """
    recip_run = replace(reciprocal, hkl=run.hkl, eta=run.eta)
    geom_run = GeometryConfig(
        mode="oblique",
        eta=run.eta,
        theta_validated=run.theta,
        omega=run.omega,
        mount=geometry.mount,
    )
    # For the MC path the group kernel's bundled hkl is the representative's,
    # not necessarily run.hkl — bypass the per-hkl metadata check (spec §6).
    # All other dispatch (including the analytic+beamstop guard) is handled by
    # _load_resolution — no duplicate logic here.
    return _load_resolution(recip_run, geom_run, skip_hkl_check=True)


def _context_for_run(
    res: fm.ResolutionContext,
    run: _ReflectionRun,
) -> fm.ForwardContext:
    """Build a ``ForwardContext`` for a single ``ReflectionRun``.

    Wraps ``fm.build_forward_context`` with the run's Bragg angle, hkl, and
    omega so the orchestrator loop has a single call-site.
    """
    return fm.build_forward_context(run.theta, res, run.hkl, omega=run.omega)


def run_simulation(config: SimulationConfig, output_dir: Path) -> dict[str, Any]:
    """Execute a DFXM forward-simulation run from a config object.

    Single-reflection (no ``[[reflections]]``): writes one
    ``<output_dir>/dfxm_geo.h5`` containing BLISS scan ``/1.1``
    (dislocations) and, if ``io.include_perfect_crystal=True``, ``/2.1``
    (Hg=0 reference). Returns ``{"h5_path", "Hg", "q_hkl",
    "include_perfect_crystal"}``.

    Multi-reflection (``[[reflections]]`` present): loops over the resolved
    ``ReflectionRun`` list, writes one standard ``dfxm_geo.h5`` per
    reflection into ``output_dir/reflection_NNN/``, then writes a thin
    super-master ``dfxm_geo_multi.h5`` in ``output_dir`` with ExternalLinks.
    Returns ``{"n_reflections": int, "reflections": list[dict]}``.

    For ``crystal.mode='random_dislocations'``, also writes a
    ``<output_dir>/dfxm_geo_random_dislocations.json`` sidecar.
    """
    if config.reflections:
        # M3 plan 2 (B'): loop over [[reflections]], one standard forward master
        # per reflection in reflection_NNN/ subdirs, plus a thin super-master.
        runs = config.reflections
        results: list[dict[str, Any]] = []
        for idx, run in enumerate(runs, start=1):
            res_run = _resolution_for_run(config.reciprocal, config.geometry, run)
            sub_dir = output_dir / f"reflection_{idx:03d}"
            results.append(
                _run_simulation_inner(
                    config,
                    sub_dir,
                    res_run,
                    reflection=run,
                    reflection_index=idx,
                    n_reflections=len(runs),
                )
            )
        write_multi_reflection_master(
            output_dir,
            runs,
            master_name="dfxm_geo.h5",
            mount=config.geometry.mount,
            keV=config.reciprocal.keV,
        )
        return {"n_reflections": len(runs), "reflections": results}
    res = _load_resolution(config.reciprocal, config.geometry)
    # #16: geometry is built from run_theta(config) inside build_forward_context
    # (called in _run_simulation_inner); Z_shift callers pass
    # xl_range=ctx.geometry.xl_range so oblique z-scans stay correct.
    return _run_simulation_inner(config, output_dir, res)


def _run_simulation_inner(
    config: SimulationConfig,
    output_dir: Path,
    res: fm.ResolutionContext,
    *,
    reflection: _ReflectionRun | None = None,
    reflection_index: int = 0,
    n_reflections: int = 0,
) -> dict[str, Any]:
    """Body of ``run_simulation``.

    Builds the dislocation population, snapshots a ForwardContext via
    ``build_forward_context(run_theta(config), ...)`` (oblique-safe: geometry
    is derived from the correct Bragg angle rather than module globals), then
    runs the scan frames and writes the HDF5 output.

    For multi-reflection runs (M3 plan 2, B'), ``reflection`` carries the
    per-run hkl/omega/eta/theta; ``reflection_index`` (1-based) and
    ``n_reflections`` are written as per-scan attrs in the master.  All three
    default to the single-reflection no-op values so the existing call path
    is byte-identical when ``reflection is None``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build dislocation population (dispatches on crystal.mode).
    # M4 Stage 4.3a: the slip systems + |b| come from the SAME [crystal] mount
    # the kernel was built from. In simplified geometry (the default FCC path)
    # config.geometry.mount is None -> build_dislocation_population resolves to
    # "fcc" + BURGERS_VECTOR (byte-identical to v2.x).
    fov_lateral_um = fm.Npixels * fm.psize * 1e6  # m -> um
    population = fm.build_dislocation_population(
        config.crystal, fov_lateral_um=fov_lateral_um, rng=None, mount=config.geometry.mount
    )

    # Write sidecar BEFORE forward kernel so a forward crash still leaves
    # the realized draw recoverable.
    if population.sidecar is not None:
        from dfxm_geo.io.sidecar import write_random_dislocations_sidecar

        sidecar_path = write_random_dislocations_sidecar(
            output_dir / "dfxm_geo", population.sidecar
        )
        print(f"[dfxm-forward] sidecar: {sidecar_path}", flush=True)

    # Effective-config print.
    print(
        f"[dfxm-forward] effective config:\n"
        f"  Nsub={fm.Nsub}  Npixels={fm.Npixels}  NN1={fm.NN1}  NN2={fm.NN2}\n"
        f"  kernel={res.loaded_kernel_path}\n"
        f"  crystal.mode={config.crystal.mode}  ndis={len(population.positions_um)}\n"
        f"  scan.mode={config.scan.derived_mode_name()}  "
        f"axes_scanned={config.scan.scanned_axes()}",
        flush=True,
    )

    # Snapshot the run's ForwardContext.  build_forward_context calls
    # build_geometry_context(run_theta(config), ...) which computes Theta/rl/prob_z
    # from the correct Bragg angle (oblique or simplified) — no module globals needed.
    # S2a+S3 (#16): CM call site removed; ctx is the sole geometry source.
    # M3 plan 2 (B'): per-reflection runs use _context_for_run which threads the
    # run's hkl and omega into the ForwardContext; single-reflection uses the
    # config-level hkl with omega=0 (the original code path).
    if reflection is not None:
        ctx = _context_for_run(res, reflection)
    else:
        ctx = fm.build_forward_context(
            run_theta(config),
            res,
            config.reciprocal.hkl,
        )

    # Wall mode preserves legacy Find_Hg path (Fg cache + sidecar _vars.txt).
    # Centered + random_dislocations use Find_Hg_from_population.
    # Hg_provider(z) returns (Hg, q_hkl) for a given z offset in micrometers.
    # For non-z scans z is always 0.0 — this preserves v1.2.0 behavior exactly.
    if config.crystal.mode == "wall":
        w = config.crystal.wall
        assert w is not None
        S = SAMPLE_REMOUNT_OPTIONS[w.sample_remount]

        def Hg_provider(z: float) -> tuple[np.ndarray, np.ndarray]:
            # population.b_um is BURGERS_VECTOR for FCC (byte-identical to v2.x)
            # and the cell-derived |b| for non-FCC walls (M4 Stage 4.3a).
            return fm.Find_Hg(
                w.dis,
                w.ndis,
                fm.psize,
                fm.zl_rms,
                h=config.reciprocal.hkl[0],
                k=config.reciprocal.hkl[1],
                l=config.reciprocal.hkl[2],
                S=S,
                remount_name=w.sample_remount,
                z_offset_um=z,
                b=population.b_um,
                ctx=ctx,
            )

        sample_dis = w.dis
        sample_ndis = w.ndis
        sample_remount = w.sample_remount
    else:

        def Hg_provider(z: float) -> tuple[np.ndarray, np.ndarray]:
            # z != 0: use Z_shift for the per-z-layer rl (takes precedence over ctx).
            # z == 0: no explicit rl → Find_Hg_from_population uses ctx.geometry.rl
            # (the oblique-aware grid built above).
            # Pass xl_range so oblique runs use the correct x-lateral extent
            # (S3 of #16: CM removed, ctx carries the oblique geometry).
            rl_eff = fm.Z_shift(z, xl_range=ctx.geometry.xl_range) if z != 0.0 else None
            return fm.Find_Hg_from_population(
                population,
                h=config.reciprocal.hkl[0],
                k=config.reciprocal.hkl[1],
                l=config.reciprocal.hkl[2],
                rl=rl_eff,
                ctx=ctx,
            )

        sample_dis = None  # not applicable for centered/random_dislocations
        sample_ndis = len(population.positions_um)
        sample_remount = "N/A"

    # Use Hg at z=0 as the "base" provenance Hg stored in /dfxm_geo/Hg.
    # Per-frame variation is captured in the detector stack + positioners[z].
    Hg_base, q_hkl = Hg_provider(0.0)

    config_toml = _dataclass_to_toml_str(config)

    h5_path = output_dir / "dfxm_geo.h5"
    frames = _build_scan_frames(config.scan)
    positioners = _positioners_for_scan_frames(frames, config.scan)
    # Per-run eta: use the reflection's eta for multi-reflection runs; fall back
    # to config.geometry.eta for single-reflection (preserves existing behaviour).
    _eta = reflection.eta if reflection is not None else config.geometry.eta
    # Multi-reflection per-scan attrs (M3 plan 2). None for single-reflection →
    # attrs dict is unchanged (byte-identical existing path).
    _reflection_attrs: dict[str, object] | None = (
        {
            "hkl_reflection": np.asarray(reflection.hkl, dtype=np.int64),
            "omega": float(reflection.omega),
            "reflection_index": reflection_index,
            "n_reflections": n_reflections,
        }
        if reflection is not None
        else None
    )
    write_simulation_h5(
        h5_path,
        Hg=Hg_base,
        q_hkl=q_hkl,
        frames=frames,
        include_perfect_crystal=config.io.include_perfect_crystal,
        sample_dis=sample_dis,
        sample_ndis=sample_ndis,
        sample_remount=sample_remount,
        config_toml=config_toml,
        cli=" ".join(sys.argv),
        max_workers=config.io.max_workers,
        crystal_mode=config.crystal.mode,
        scan_mode=config.scan.derived_mode_name(),
        scanned_axes=list(config.scan.scanned_axes()),
        positioners=positioners,
        Hg_provider=Hg_provider,
        write_strain_provenance=config.io.write_strain_provenance,
        geometry_mode=config.geometry.mode,
        eta=_eta,
        mount=config.geometry.mount,
        ctx=ctx,
        reflection_attrs=_reflection_attrs,
    )
    return {
        "h5_path": h5_path,
        "Hg": Hg_base,
        "q_hkl": q_hkl,
        "include_perfect_crystal": config.io.include_perfect_crystal,
    }


def _resolve_postprocess_Hg(h5_path: Path, Hg: np.ndarray | None) -> np.ndarray:
    """Strain source for postprocess, in priority order.

    1. explicit ``Hg`` argument;
    2. persisted ``/1.1/dfxm_geo/Hg`` from the run's HDF5 (written when
       ``io.write_strain_provenance`` is True).

    #16 Slice 5 removed the legacy module-global ``Hg`` fallback — provenance is
    now always recovered from the HDF5 (or passed explicitly).
    """
    if Hg is not None:
        return np.asarray(Hg, dtype=float)
    with h5py.File(h5_path, "r") as f:
        if "/1.1/dfxm_geo/Hg" in f:
            return np.asarray(f["/1.1/dfxm_geo/Hg"][()], dtype=float)
    raise RuntimeError(
        "Cannot postprocess: no Hg passed and none persisted at "
        "/1.1/dfxm_geo/Hg (run with io.write_strain_provenance=true)."
    )


def run_postprocess(
    output_dir: Path,
    config: SimulationConfig,
    *,
    Hg: np.ndarray | None = None,
    q_hkl: np.ndarray | None = None,
) -> dict[str, Any]:
    """Read /1.1 from dfxm_geo.h5; compute COM maps and the qi field.

    Analysis outputs are written into /1.1/dfxm_geo/analysis/ inside the same
    HDF5 file. SVG figures land on disk under <output_dir>/figures/ (F1).

    The strain field ``Hg`` used for qi computation is resolved in priority order:

    1. the explicit ``Hg`` keyword argument (if supplied);
    2. ``/1.1/dfxm_geo/Hg`` persisted in the HDF5 (written when
       ``io.write_strain_provenance`` is True, the default).

    This means ``--postprocess-only`` on a fresh process works without any manual
    setup, as long as the HDF5 was produced with strain provenance enabled
    (the default).

    Args:
        output_dir: Directory containing ``dfxm_geo.h5`` from a prior run.
        config: ``SimulationConfig`` matching the original run.
        Hg: Optional explicit strain-gradient tensor to use for qi computation.
            Overrides the persisted HDF5 value.
        q_hkl: Accepted for API symmetry; currently unused by the postprocess body.

    Raises:
        FileNotFoundError: if the expected dfxm_geo.h5 file is absent.
        FileNotFoundError: if /2.1 (perfect crystal scan) is missing from the .h5.
        RuntimeError: if no Hg source can be found (explicit or HDF5).
    """
    h5_path = output_dir / "dfxm_geo.h5"
    if not h5_path.is_file():
        raise FileNotFoundError(
            f"Expected {h5_path}; run dfxm-forward without --postprocess-only first."
        )

    # Sanity-check that /2.1 exists (the perfect-crystal scan is still part of
    # the forward-output contract, even though postprocessing no longer reads
    # it). Data validation comes BEFORE the kernel load so a malformed run
    # fails with the documented FileNotFoundError, not a kernel-lookup error.
    with h5py.File(h5_path, "r") as f:
        if "/2.1" not in f:
            raise FileNotFoundError(
                f"{h5_path} has no /2.1 scan (perfect crystal). Re-run with "
                "include_perfect_crystal=True, or skip postprocess."
            )

    res = _load_resolution(config.reciprocal, config.geometry)
    # Build run ctx after resolution is loaded (analytic_eval / Resq_i are now live).
    # run_postprocess is NOT inside an oblique CM — its qi_field uses the default
    # theta, unchanged from today; that pre-existing detail is intentional.
    # S2a (#16): explicit theta + resolution + hkl.
    ctx = fm.build_forward_context(
        run_theta(config),
        res,
        config.reciprocal.hkl,
    )

    phi_steps = config.scan.phi.steps or 1
    chi_steps = config.scan.chi.steps or 1
    phi_range = config.scan.phi.range or 0.0
    chi_range = config.scan.chi.range or 0.0

    _, dis_reshape, _, _ = load_h5_scan(
        h5_path,
        scan_id="1.1",
        phi_steps=phi_steps,
        chi_steps=chi_steps,
    )

    # COM maps are an exact intensity-weighted mean (v2.0.2) read directly off the
    # nominal χ grid. The earlier runtime χ-offset calibration (compute_chi_shift,
    # from the perfect-crystal corner pixel) was dropped: it carried an abs()
    # sign-loss that shifted the χ-COM map off the article axis, and the
    # de-quantized weighted mean needs no such correction. The phi/chi oversample
    # knobs on PostprocessConfig are now inert (kept for TOML back-compat only).
    phi_list, chi_list = compute_com_maps(
        dis_reshape,
        phi_range,
        phi_steps,
        chi_range,
        chi_steps,
        chi_shift=0.0,
    )
    Hg_pp = _resolve_postprocess_Hg(h5_path, Hg)
    _, qi_field = fm.forward(Hg_pp, ctx=ctx, phi=0, qi_return=True)

    # Append analysis to /1.1/dfxm_geo/analysis/ inside the existing .h5
    with h5py.File(h5_path, "a") as f:
        analysis = f.require_group("/1.1/dfxm_geo/analysis")
        for name, val in [
            ("phi_list", phi_list),
            ("chi_list", chi_list),
            ("qi_field", qi_field),
        ]:
            if name in analysis:
                del analysis[name]
            analysis.create_dataset(name, data=val)

    # F1: render SVG figures on disk alongside the .h5
    fig_dir = output_dir / config.postprocess.figures_dirname
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_mosaicity_maps(
        phi_list,
        chi_list,
        ctx.geometry.xl_start,  # S4 (#16): xl_start is geometry (theta-dependent; deleted in S5)
        ctx.instrument.yl_start,
        fig_dir / "mosaicity_maps.svg",
    )
    plot_qi_cross_section(
        qi_field,
        ctx.geometry.xl_start,  # S4 (#16): xl_start is geometry (theta-dependent; deleted in S5)
        ctx.instrument.yl_start,
        ctx.instrument.xl_steps,
        ctx.instrument.yl_steps,
        ctx.instrument.zl_steps,
        fig_dir / "qi_cross_section.svg",
    )
    return {
        "phi_list": phi_list,
        "chi_list": chi_list,
        "qi_field": qi_field,
        "h5_path": h5_path,
        "figures_dir": fig_dir,
    }


def _build_scan_frames(scan: ScanConfig) -> ScanFrames:
    """Flatten the 4-axis Cartesian product of `build_scan_grid` into per-frame arrays.

    Order: phi-innermost (stride 1), then chi, then two_dtheta,
    then z-outermost (largest stride). Fixed axes contribute a
    singleton sample, so they degenerate to a constant column.
    """
    grid = fm.build_scan_grid(scan)
    phi, chi, two_dtheta, z = grid.samples
    # `np.meshgrid(..., indexing="ij")` returns arrays ordered (phi, chi, two_dtheta, z).
    # Ravel in Fortran order so the FIRST index (phi) varies fastest -- giving
    # phi-innermost, z-outermost flat layout.
    phi_g, chi_g, twodt_g, z_g = np.meshgrid(phi, chi, two_dtheta, z, indexing="ij")
    phi_pf = phi_g.ravel(order="F")
    chi_pf = chi_g.ravel(order="F")
    two_dtheta_pf = twodt_g.ravel(order="F")
    z_pf = z_g.ravel(order="F")
    return ScanFrames(
        phi_pf=phi_pf,
        chi_pf=chi_pf,
        two_dtheta_pf=two_dtheta_pf,
        z_pf=z_pf,
        n_frames=int(phi_pf.size),
    )


def _build_scan_frames_at_z(scan: ScanConfig, z_value: float) -> ScanFrames:
    """Inner (phi × chi × two_dtheta) trajectory with z_pf fixed to z_value.

    Used by identification iterators that loop outer over z themselves.
    The scan's `[scan.z]` configuration is ignored; only z_value is used.
    """
    grid = fm.build_scan_grid(scan)
    phi, chi, two_dtheta, _z_ignored = grid.samples
    phi_g, chi_g, twodt_g = np.meshgrid(phi, chi, two_dtheta, indexing="ij")
    phi_pf = phi_g.ravel(order="F")
    chi_pf = chi_g.ravel(order="F")
    two_dtheta_pf = twodt_g.ravel(order="F")
    n = int(phi_pf.size)
    return ScanFrames(
        phi_pf=phi_pf,
        chi_pf=chi_pf,
        two_dtheta_pf=two_dtheta_pf,
        z_pf=np.full(n, float(z_value), dtype=np.float64),
        n_frames=n,
    )


def _iterate_simulation_frames(
    frames: ScanFrames,
    Hg_provider: Callable[[float], np.ndarray],
) -> Iterator[tuple[int, np.ndarray, float, float, float]]:
    """Yield (idx, Hg, phi, chi, two_dtheta) per frame; Hg_provider called once per unique z.

    Memory mitigation: because z is the outermost loop in `_build_scan_frames`
    (z-outermost frame order), all frames sharing a z value are contiguous in
    `frames.z_pf`. After the last frame at a given z is yielded, the cached
    Hg is dropped — only one Hg lives in memory at any time during the walk.
    """
    z_to_Hg: dict[float, np.ndarray] = {}
    z_pf = frames.z_pf
    n = frames.n_frames
    for k in range(n):
        z = float(z_pf[k])
        if z not in z_to_Hg:
            z_to_Hg[z] = Hg_provider(z)
        yield (
            k,
            z_to_Hg[z],
            float(frames.phi_pf[k]),
            float(frames.chi_pf[k]),
            float(frames.two_dtheta_pf[k]),
        )
        # If this is the last frame at this z (either we're at the end,
        # or the next frame has a different z), free the Hg array.
        if k == n - 1 or float(z_pf[k + 1]) != z:
            del z_to_Hg[z]


def _scan_frames_args(
    Hg: np.ndarray,
    frames: ScanFrames,
    scan: ScanConfig,
    ctx: fm.ForwardContext,
) -> tuple[
    list[tuple[int, np.ndarray, float, float, float, fm.ForwardContext]],
    dict[str, np.ndarray | float],
]:
    """Build (args_list, positioners) for one ScanSpec.

    args_list elements: (frame_idx, base_qc, phi_rad, chi_rad, two_dtheta_rad, ctx),
    where base_qc = precompute_forward_static(Hg) is computed ONCE here and
    shared (read-only) across every frame -- the per-frame worker runs the
    cheap forward_from_static(base_qc, ctx, ...) instead of recomputing the
    Hg-only qs matmul each frame.
    positioners: dict keyed by canonical axis; scanned axes -> per-frame array,
    fixed axes -> scalar.

    ctx is the run's ForwardContext, built once inside the oblique CM (if any)
    and passed in by the caller — not rebuilt from globals here.
    """
    base_qc = fm.precompute_forward_static(Hg, ctx)
    args_list: list[tuple[int, np.ndarray, float, float, float, fm.ForwardContext]] = []
    for k in range(frames.n_frames):
        args_list.append(
            (
                k,
                base_qc,
                float(frames.phi_pf[k]),
                float(frames.chi_pf[k]),
                float(frames.two_dtheta_pf[k]),
                ctx,
            )
        )
    positioners = _positioners_for_scan_frames(frames, scan)
    return args_list, positioners


def _positioners_for_scan_frames(
    frames: ScanFrames, scan: ScanConfig
) -> dict[str, np.ndarray | float]:
    """Build the positioners dict for a ScanSpec.

    Scanned axes -> the full per-frame array (np.ndarray of length n_frames).
    Fixed axes -> the scalar axis value (float). Matches v1.2.0 convention
    for phi/chi; now extended to two_dtheta + z.

    Special case: when z is configured as scanned but all per-frame z values
    are identical (i.e. z was looped externally as a between-scan axis in
    identification mode), z is stored as a scalar.  This preserves the
    HDF5 contract that z is a 0-D positioner within a single scan.
    """
    pf_arrays: dict[str, np.ndarray] = {
        "phi": frames.phi_pf,
        "chi": frames.chi_pf,
        "two_dtheta": frames.two_dtheta_pf,
        "z": frames.z_pf,
    }
    out: dict[str, np.ndarray | float] = {}
    for axis_name in _CANONICAL_AXES:
        axis = getattr(scan, axis_name)
        arr = pf_arrays[axis_name]
        if axis.is_scanned and not (axis_name == "z" and np.all(arr == arr[0])):
            out[axis_name] = arr
        else:
            out[axis_name] = float(arr[0])
    return out


# === IDENTIFICATION ORCHESTRATION ===


def _identify_title(scan_mode: str, n_frames: int, scan: ScanConfig) -> str:
    """Compact human title for /N.1/title in identification masters."""
    return f"identify-{scan_mode} N_frames={n_frames}"


def _q_unit(hkl: tuple[int, int, int] | list[int]) -> np.ndarray:
    """Return the unit diffraction vector for hkl (float64, length 1)."""
    q = np.asarray(hkl, dtype=float)
    return q / np.linalg.norm(q)


def _passes_invisibility(
    q_hkl: np.ndarray,
    b_vec: np.ndarray,
    threshold_deg: float,
) -> bool:
    """True if |G·b| / (|G| |b|) >= cos(90° - threshold) — NOT near-orthogonal.

    Paper convention (Borgi 2025): a configuration is excluded if the Burgers
    vector is within `threshold_deg` degrees of perpendicular to G.
    cos(90° - 10°) = cos(80°) ≈ 0.174.
    """
    return _gb_visible(q_hkl, b_vec, threshold_deg)


def _iter_identification_single(
    config: IdentificationConfig,
    ctx: fm.ForwardContext,
    *,
    visibility_qs: list[np.ndarray] | None = None,
) -> Iterator[ScanSpec]:
    """Yield one ScanSpec per (z, plane, b_idx, alpha) configuration.

    Supports `[scan.phi]` / `[scan.chi]` from the shared ScanConfig: when
    either axis is scanned, each scan dir contains a (N_frames, H, W)
    stack with frame ordering phi-inner, chi-outer.

    When `[scan.z]` is configured, the iterator loops z outermost: each
    unique z value produces its own set of (plane × b × alpha) ScanSpecs,
    with `rl_eff = fm.Z_shift(z)` substituted into `find_hg_scene`.
    When z is fixed, z_samples has length 1 (no extra scans emitted).

    ctx is the run's ForwardContext, built by run_identification via
    build_forward_context(run_theta(config), ...) and passed in. Geometry globals
    (Theta/rl/Us/psize/zl_rms/theta_0) are NOT read here — all geometry is
    sourced from ``ctx`` (S2a+S3 of #16).

    visibility_qs: when None, the exclude_invisibility gate uses ctx.q_hkl
    (single-reflection, today's behaviour). When provided (multi-reflection
    orchestrator), a config is kept if it is visible to AT LEAST ONE reflection
    q — so the sweep grid is IDENTICAL across all reflection masters, and ML
    labels align by scan index (spec §8, M3 plan 2).
    """
    crystal_cfg = config.crystal
    # Noiseless frames are emitted here; intensity scaling and optional
    # Poisson noise are applied to the combined detector file post-write
    # by `_maybe_apply_poisson_noise` (called from the dispatcher).

    planes = (
        _ALL_111_PLANES if crystal_cfg.sweep_all_slip_planes else [crystal_cfg.slip_plane_normal]
    )

    angles_deg = np.arange(
        crystal_cfg.angle_start_deg,
        crystal_cfg.angle_stop_deg + crystal_cfg.angle_step_deg * 0.5,
        crystal_cfg.angle_step_deg,
    )

    q_hkl = np.asarray(ctx.q_hkl, dtype=float)
    scan_mode = config.scan.derived_mode_name()
    scanned_axes = list(config.scan.scanned_axes())

    # Source geometry + instrument from ctx (oblique-safe).
    Us_ = ctx.instrument.Us
    Theta_ = ctx.geometry.Theta
    rl_ = ctx.geometry.rl
    xl_range_ = ctx.geometry.xl_range
    psize_ = ctx.instrument.psize
    zl_rms_ = ctx.instrument.zl_rms
    theta_0_ = ctx.geometry.theta_0

    # Outer z loop. When z is fixed, z_samples is a length-1 array so the
    # loop body executes once — identical to the pre-z-aware behaviour.
    z_samples = fm.build_scan_grid(config.scan).samples[3]

    for z in z_samples:
        z_float = float(z)
        # find_hg_scene expects the ray grid in MICROMETRES (b is in µm); rl_
        # and fm.Z_shift(...) are both in metres, so scale by 1e6 — matching the
        # forward path (strain_cache.py:61, forward_model.py:1139).
        # xl_range keeps the oblique x-lateral extent correct (S3 #16).
        rl_eff = (fm.Z_shift(z_float, xl_range=xl_range_) if z_float != 0.0 else rl_) * 1e6
        frames_at_z = _build_scan_frames_at_z(config.scan, z_float)

        for plane in planes:
            b_table = _burgers_vectors(plane)
            b_indices = (
                crystal_cfg.b_vector_indices
                if crystal_cfg.b_vector_indices is not None
                else list(range(len(b_table)))
            )
            b_subset = b_table[b_indices]
            n_arr_unnorm = np.asarray(plane, dtype=float)
            n_arr = n_arr_unnorm / np.linalg.norm(n_arr_unnorm)
            rotated = _rotated_t_vectors(n_arr, b_subset, angles_deg)
            Ud_all = _ud_matrices(n_arr, rotated)

            # Resolve visibility query vectors: None -> [ctx.q_hkl] (single-reflection);
            # provided list -> all-reflections gate (keep if visible to any).
            _vis_qs: list[np.ndarray] = visibility_qs if visibility_qs is not None else [q_hkl]
            for j, b_idx in enumerate(b_indices):
                if crystal_cfg.exclude_invisibility and not any(
                    _passes_invisibility(q, b_table[b_idx], crystal_cfg.invisibility_threshold_deg)
                    for q in _vis_qs
                ):
                    continue
                for i, alpha in enumerate(angles_deg):
                    Ud_mix = Ud_all[i, j]
                    Hg, _ = find_hg_scene(
                        rl_eff,
                        Us_,
                        [
                            MixedDislocSpec(
                                Ud_mix=Ud_mix,
                                rotation_deg=float(alpha),
                            )
                        ],
                        Theta_,
                    )

                    args_list, positioners = _scan_frames_args(Hg, frames_at_z, config.scan, ctx)

                    burgers_int = (
                        int(round(b_table[b_idx, 0] * np.sqrt(2))),
                        int(round(b_table[b_idx, 1] * np.sqrt(2))),
                        int(round(b_table[b_idx, 2] * np.sqrt(2))),
                    )
                    # g·b labels (Task 5, M3 plan 2): per-scan visibility scalars
                    # computed from this run's q_hkl and the scan's Burgers vector.
                    # gb_cos normalises both inputs, so any non-zero multiple of b works.
                    _b_vec_single = b_table[b_idx]
                    _gb_cos_val = _gb_cos(q_hkl, _b_vec_single)
                    _gb_vis_val = np.int8(
                        _gb_visible(q_hkl, _b_vec_single, crystal_cfg.invisibility_threshold_deg)
                    )
                    yield ScanSpec(
                        title=_identify_title(scan_mode, frames_at_z.n_frames, config.scan),
                        sample={
                            "name": "simulated, dislocation identification (single)",
                            "slip_plane_normal": np.asarray(plane, dtype=np.int32),
                            "burgers": np.asarray(burgers_int, dtype=np.int32),
                            "rotation_deg": float(alpha),
                        },
                        positioners=positioners,
                        dfxm_geo={
                            "Hg": Hg,
                            "q_hkl": q_hkl,
                            "theta": float(theta_0_),
                            "psize": float(psize_),
                            "zl_rms": float(zl_rms_),
                            "gb_cos": _gb_cos_val,
                            "gb_visible": _gb_vis_val,
                        },
                        detectors={"dfxm_sim_detector": args_list},
                        attrs={
                            "scan_mode": scan_mode,
                            "scanned_axes": scanned_axes,
                            "identify_mode": "single",
                        },
                    )


def _run_identification_single(
    config: IdentificationConfig,
    output_dir: Path,
    ctx: fm.ForwardContext,
    *,
    reflection_index: int = 0,
    visibility_qs: list[np.ndarray] | None = None,
    reflection_attrs: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Dispatcher: feed `_iter_identification_single` into write_identification_h5.

    Empty sweep (e.g. exclude_invisibility filters out everything) is
    allowed: the orchestrator writes an empty master with `n_images=0`.
    Mirrors the old behavior which also emitted an empty manifest.

    reflection_index, visibility_qs, reflection_attrs:
    forwarded from _dispatch_identification; defaults preserve the
    single-reflection byte-identical path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    config_toml = _identification_config_to_toml_str(config)
    n_scans = write_identification_h5(
        output_dir,
        scan_iter=_iter_identification_single(config, ctx, visibility_qs=visibility_qs),
        cli=" ".join(sys.argv),
        config_toml=config_toml,
        max_workers=config.io.max_workers,
        write_strain_provenance=config.io.write_strain_provenance,
        geometry_attrs=_identify_geometry_attrs(config, ctx),
        ctx=ctx,
        reflection_attrs=reflection_attrs,
    )
    _maybe_apply_poisson_noise(config, output_dir, n_scans, reflection_index=reflection_index)
    return {
        "n_images": n_scans,
        "output_dir": output_dir,
        "master_path": output_dir / "dfxm_identify.h5",
    }


_ALL_111_PLANES: list[tuple[int, int, int]] = [
    (1, 1, 1),
    (1, -1, 1),
    (1, 1, -1),
    (-1, 1, 1),
]


def _draw_dislocation(rng: np.random.Generator, pos_std_um: float) -> dict[str, Any]:
    """Draw a single random dislocation (slip plane, Burgers idx, angle, position)."""
    plane_idx = int(rng.integers(0, len(_ALL_111_PLANES)))
    plane = _ALL_111_PLANES[plane_idx]
    b_table = _burgers_vectors(plane)
    b_idx = int(rng.integers(0, len(b_table)))
    alpha = float(rng.uniform(0.0, 360.0))
    pos = (float(rng.normal(0.0, pos_std_um)), float(rng.normal(0.0, pos_std_um)), 0.0)

    n = np.asarray(plane, dtype=float)
    n_unit = n / np.linalg.norm(n)
    rotated = _rotated_t_vectors(n_unit, b_table[b_idx : b_idx + 1], np.array([alpha]))
    Ud = _ud_matrices(n_unit, rotated)[0, 0]

    return {
        "plane": plane,
        "b_idx": b_idx,
        "b_vec": b_table[b_idx] * np.sqrt(2),
        "alpha_deg": alpha,
        "pos_um": pos,
        "Ud": Ud,
    }


def _build_dislocation_sample_entry(d: dict[str, Any]) -> dict[str, Any]:
    """Convert a `_draw_dislocation` output to an NXsample-shaped dict.

    Used to populate `/N.1/sample/dislocations/<idx>` inside the master
    identification HDF5. The Burgers vector is rounded to integer
    components (matches the `[h, k, l]` convention used for single mode).
    """
    return {
        "slip_plane_normal": np.asarray(d["plane"], dtype=np.int32),
        "burgers": np.asarray([int(round(c)) for c in d["b_vec"]], dtype=np.int32),
        "rotation_deg": float(d["alpha_deg"]),
        "position_um": np.asarray(d["pos_um"], dtype=float),
    }


def _iter_identification_multi(
    config: IdentificationConfig,
    ctx: fm.ForwardContext,
) -> Iterator[ScanSpec]:
    """Yield one ScanSpec per (z, Monte Carlo sample) pair.

    `render_per_dislocation=False` (default): a single detector
    (`dfxm_sim_detector`) holds the combined-scene frames. `=True`: two
    additional detectors (`dfxm_sim_detector_dis0`, `_dis1`) hold each
    dislocation rendered in isolation; these per-dislocation files are
    NOISELESS by construction (they bypass the post-write Poisson pass).

    All frames yielded here are NOISELESS; intensity scaling and
    optional Poisson noise are applied to the combined detector file
    post-write by `_maybe_apply_poisson_noise`.

    When `[scan.z]` is configured, the iterator loops z outermost: each
    unique z value produces its own set of n_samples ScanSpecs, with
    `rl_eff = fm.Z_shift(z)` substituted into all Fd_find calls.
    When z is fixed, z_samples has length 1 — identical to pre-z-aware
    behaviour.

    ctx is the run's ForwardContext, built by run_identification via
    build_forward_context(run_theta(config), ...) and passed in. Geometry globals
    (Theta/rl/Us/psize/zl_rms/theta_0) are NOT read here — all geometry is
    sourced from ``ctx`` (S2a+S3 of #16).
    """
    assert config.multi is not None  # validated in __post_init__
    mc = config.multi
    noise_cfg = config.noise
    q_hkl = np.asarray(ctx.q_hkl, dtype=float)

    # Source geometry + instrument from ctx (oblique-safe).
    Us_ = ctx.instrument.Us
    Theta_ = ctx.geometry.Theta
    rl_ = ctx.geometry.rl
    xl_range_ = ctx.geometry.xl_range
    psize_ = ctx.instrument.psize
    zl_rms_ = ctx.instrument.zl_rms
    theta_0_ = ctx.geometry.theta_0

    # Split master rng → child streams. [0] = param draws (consumed here);
    # [1] = Poisson noise (consumed by _maybe_apply_poisson_noise, which
    # re-spawns with the same seed to get the same noise stream).
    param_rng, _noise_rng = np.random.default_rng(noise_cfg.rng_seed).spawn(2)

    scan_mode = config.scan.derived_mode_name()
    scanned_axes = list(config.scan.scanned_axes())

    # Outer z loop. When z is fixed, z_samples is a length-1 array so the
    # loop body executes once — identical to the pre-z-aware behaviour.
    z_samples = fm.build_scan_grid(config.scan).samples[3]

    # param_rng walks continuously across z; dislocation draws are NOT z-invariant
    # by design (sample i at z=-2 and sample i at z=+2 use different dislocations).
    for z in z_samples:
        z_float = float(z)
        # find_hg_scene expects the ray grid in MICROMETRES (b is in µm); rl_
        # and fm.Z_shift(...) are both in metres, so scale by 1e6 — matching the
        # forward path (strain_cache.py:61, forward_model.py:1139).
        # xl_range keeps the oblique x-lateral extent correct (S3 #16).
        rl_eff = (fm.Z_shift(z_float, xl_range=xl_range_) if z_float != 0.0 else rl_) * 1e6
        frames_at_z = _build_scan_frames_at_z(config.scan, z_float)

        for _ in range(mc.n_samples):
            d1 = _draw_dislocation(param_rng, mc.pos_std_um)
            d2 = _draw_dislocation(param_rng, mc.pos_std_um)

            # Combined-scene Hg (sum of both dislocations)
            specs = [
                MixedDislocSpec(
                    Ud_mix=d1["Ud"],
                    rotation_deg=d1["alpha_deg"],
                    position_lab_um=d1["pos_um"],
                ),
                MixedDislocSpec(
                    Ud_mix=d2["Ud"],
                    rotation_deg=d2["alpha_deg"],
                    position_lab_um=d2["pos_um"],
                ),
            ]
            # W2 dedup (v2.6.0): each dislocation's field is computed ONCE;
            # the combined scene is Σ(Fg_i − I) + I — bit-identical to the
            # old Fd_find_multi_dislocs_mixed + per-solo recompute path.
            Hg_combined, solo_hgs = find_hg_scene(
                rl_eff,
                Us_,
                specs,
                Theta_,
                per_dislocation=mc.render_per_dislocation,
            )

            combined_args, positioners = _scan_frames_args(
                Hg_combined, frames_at_z, config.scan, ctx
            )
            detectors: dict[
                str, list[tuple[int, np.ndarray, float, float, float, fm.ForwardContext]]
            ] = {
                "dfxm_sim_detector": combined_args,
            }

            if mc.render_per_dislocation:
                # Per-dislocation Hg: each rendered alone (other one absent), at
                # its own scene position so the renders overlay the combined
                # image as ground-truth instance labels. Noiseless by design.
                assert solo_hgs is not None
                dis0_args, _ = _scan_frames_args(solo_hgs[0], frames_at_z, config.scan, ctx)
                dis1_args, _ = _scan_frames_args(solo_hgs[1], frames_at_z, config.scan, ctx)
                detectors["dfxm_sim_detector_dis0"] = dis0_args
                detectors["dfxm_sim_detector_dis1"] = dis1_args

            sample: dict[str, Any] = {
                "name": "simulated, dislocation identification (multi)",
                "dislocations": {
                    "0": _build_dislocation_sample_entry(d1),
                    "1": _build_dislocation_sample_entry(d2),
                },
            }

            # g·b labels (Task 5, M3 plan 2): per-dislocation arrays (length = 2).
            # Order matches the specs list (d1 at index 0, d2 at index 1),
            # which matches per-dislocation file naming (_dis0, _dis1).
            _thr_multi = config.crystal.invisibility_threshold_deg
            _gb_cos_multi = np.array([_gb_cos(q_hkl, d["b_vec"]) for d in (d1, d2)])
            _gb_vis_multi = np.array(
                [int(_gb_visible(q_hkl, d["b_vec"], _thr_multi)) for d in (d1, d2)],
                dtype=np.int8,
            )
            yield ScanSpec(
                title=_identify_title(scan_mode, frames_at_z.n_frames, config.scan),
                sample=sample,
                positioners=positioners,
                dfxm_geo={
                    "Hg": Hg_combined,
                    "q_hkl": q_hkl,
                    "theta": float(theta_0_),
                    "psize": float(psize_),
                    "zl_rms": float(zl_rms_),
                    "gb_cos": _gb_cos_multi,
                    "gb_visible": _gb_vis_multi,
                },
                detectors=detectors,
                attrs={
                    "scan_mode": scan_mode,
                    "scanned_axes": scanned_axes,
                    "identify_mode": "multi",
                },
            )


def _run_identification_multi(
    config: IdentificationConfig,
    output_dir: Path,
    ctx: fm.ForwardContext,
    *,
    reflection_index: int = 0,
    reflection_attrs: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Dispatcher: feed `_iter_identification_multi` into write_identification_h5.

    After the master + per-scan detector files are written, applies a
    post-write Poisson noise pass (and intensity scaling) to combined
    detector files only — per-dislocation files stay noiseless.

    reflection_index, reflection_attrs: forwarded from _dispatch_identification;
    defaults preserve the single-reflection byte-identical path.
    Multi mode has no deterministic visibility gate — visibility_qs unused.
    """
    assert config.multi is not None  # validated in __post_init__
    output_dir.mkdir(parents=True, exist_ok=True)
    n_scans = write_identification_h5(
        output_dir,
        scan_iter=_iter_identification_multi(config, ctx),
        cli=" ".join(sys.argv),
        config_toml=_identification_config_to_toml_str(config),
        max_workers=config.io.max_workers,
        write_strain_provenance=config.io.write_strain_provenance,
        geometry_attrs=_identify_geometry_attrs(config, ctx),
        ctx=ctx,
        reflection_attrs=reflection_attrs,
    )
    _maybe_apply_poisson_noise(config, output_dir, n_scans, reflection_index=reflection_index)
    return {
        "n_samples": config.multi.n_samples,
        "output_dir": output_dir,
        "master_path": output_dir / "dfxm_identify.h5",
    }


def _maybe_apply_poisson_noise(
    config: IdentificationConfig, output_dir: Path, n_scans: int, *, reflection_index: int = 0
) -> None:
    """Apply intensity scaling (always) and Poisson noise (if enabled) to
    combined-detector files post-write.

    Only modifies `dfxm_sim_detector_0000.h5` files; per-dislocation
    detector files (`*_dis0_0000.h5`, `*_dis1_0000.h5`) are intentionally
    left untouched so they remain deterministic ground-truth labels.

    Determinism contract (RNG policy, M3 plan 2):
    - reflection_index == 0 (single-reflection / default): noise stream is
      the second spawn of `np.random.default_rng(rng_seed).spawn(2)` — the
      same split used by `_iter_identification_*` for parameter draws.
      Bit-compatible with all v2.5.x releases.
    - reflection_index > 0 (multi-reflection loop): noise seeded from
      `np.random.default_rng([rng_seed, reflection_index])` so every
      reflection gets an INDEPENDENT noise draw (multi-reflection data must
      have independent detector noise even though all reflections share the
      same crystal/dislocation realization).
    """
    noise_cfg = config.noise
    scale = noise_cfg.intensity_scale
    if scale == 1.0 and not noise_cfg.poisson_noise:
        return  # nothing to do; skip the per-scan file open
    if noise_cfg.poisson_noise:
        if reflection_index > 0:
            # Multi-reflection: per-reflection independent noise stream
            # (same crystal / dislocation params, independent detector noise).
            rng: np.random.Generator | None = np.random.default_rng(
                [noise_cfg.rng_seed, reflection_index]
            )
        else:
            # Single-reflection: bit-compatible stream (v2.5.x behaviour).
            rng = np.random.default_rng(noise_cfg.rng_seed).spawn(2)[1]
    else:
        rng = None
    for k in range(1, n_scans + 1):
        det_file = (
            output_dir / SCAN_DIR_FMT.format(k) / DETECTOR_FILE_FMT.format(name="dfxm_sim_detector")
        )
        if not det_file.is_file():
            continue
        with h5py.File(det_file, "a") as f:
            img = f[DETECTOR_INTERNAL_PATH]
            arr = img[...] * scale
            if rng is not None:
                arr = rng.poisson(np.clip(arr, a_min=0.0, a_max=None)).astype(float)
            img[...] = arr


def _iter_identification_zscan(
    config: IdentificationConfig,
    ctx: fm.ForwardContext,
    *,
    visibility_qs: list[np.ndarray] | None = None,
) -> Iterator[ScanSpec]:
    """Yield one ScanSpec per (z_offset, plane, b_idx, alpha) configuration.

    Each ScanSpec carries the primary (deterministic, on-axis) dislocation
    in ``sample["primary"]`` and, when ``zscan.include_secondary`` is True,
    a randomly-drawn ``sample["secondary"]`` (one draw per configuration,
    shared across the rocking grid). The (phi, chi) rocking grid comes
    from ``config.scan.phi`` / ``config.scan.chi`` (shared B+C schema).

    All frames are noiseless; intensity scaling / Poisson noise are
    applied post-write by ``_maybe_apply_poisson_noise``.

    ctx is the run's ForwardContext, built by run_identification via
    build_forward_context(run_theta(config), ...) and passed in. Geometry globals
    (Theta/rl/Us/psize/zl_rms/theta_0) are NOT read here — all geometry is
    sourced from ``ctx`` (S2a+S3 of #16).

    visibility_qs: when None, the exclude_invisibility gate uses ctx.q_hkl
    (single-reflection, today's behaviour). When provided (multi-reflection
    orchestrator), a config is kept if it is visible to AT LEAST ONE reflection
    q — identical grid semantics to _iter_identification_single (spec §8).
    """
    assert config.zscan is not None  # validated in __post_init__
    zscan = config.zscan
    crystal_cfg = config.crystal
    noise_cfg = config.noise

    planes = (
        _ALL_111_PLANES if crystal_cfg.sweep_all_slip_planes else [crystal_cfg.slip_plane_normal]
    )
    angles_deg = np.arange(
        crystal_cfg.angle_start_deg,
        crystal_cfg.angle_stop_deg + crystal_cfg.angle_step_deg * 0.5,
        crystal_cfg.angle_step_deg,
    )

    # Secondary stream uses SeedSequence child [secondary_rng_offset]. Default
    # is 0; _maybe_apply_poisson_noise uses child [1] from a spawn(2), so the
    # two streams are independent SeedSequence siblings.
    spawned = np.random.default_rng(noise_cfg.rng_seed).spawn(zscan.secondary_rng_offset + 1)
    secondary_rng = spawned[zscan.secondary_rng_offset]

    q_hkl = np.asarray(ctx.q_hkl, dtype=float)
    scan_mode = config.scan.derived_mode_name()
    scanned_axes = list(config.scan.scanned_axes())

    # Source geometry + instrument from ctx (oblique-safe).
    Us_ = ctx.instrument.Us
    Theta_ = ctx.geometry.Theta
    xl_range_ = ctx.geometry.xl_range
    psize_ = ctx.instrument.psize
    zl_rms_ = ctx.instrument.zl_rms
    theta_0_ = ctx.geometry.theta_0

    for z_off in zscan.z_offsets_um:
        frames_at_z = _build_scan_frames_at_z(config.scan, z_value=float(z_off))
        n_frames = frames_at_z.n_frames
        # Fd_find_* expect the ray grid in MICROMETRES (b is in µm); Z_shift
        # returns metres, so scale by 1e6 — matching the forward path.
        # xl_range keeps the oblique x-lateral extent correct (S3 #16).
        rl_shifted = fm.Z_shift(z_off, xl_range=xl_range_) * 1e6
        for plane in planes:
            b_table = _burgers_vectors(plane)
            b_indices = (
                crystal_cfg.b_vector_indices
                if crystal_cfg.b_vector_indices is not None
                else list(range(len(b_table)))
            )
            b_subset = b_table[b_indices]
            n_arr_unnorm = np.asarray(plane, dtype=float)
            n_arr = n_arr_unnorm / np.linalg.norm(n_arr_unnorm)
            rotated = _rotated_t_vectors(n_arr, b_subset, angles_deg)
            Ud_all = _ud_matrices(n_arr, rotated)

            # Resolve visibility query vectors: None -> [ctx.q_hkl] (single-reflection);
            # provided list -> all-reflections gate (keep if visible to any).
            _vis_qs: list[np.ndarray] = visibility_qs if visibility_qs is not None else [q_hkl]
            for j, b_idx in enumerate(b_indices):
                if crystal_cfg.exclude_invisibility and not any(
                    _passes_invisibility(q, b_table[b_idx], crystal_cfg.invisibility_threshold_deg)
                    for q in _vis_qs
                ):
                    continue
                for i, alpha in enumerate(angles_deg):
                    Ud_primary = Ud_all[i, j]
                    primary_spec = MixedDislocSpec(
                        Ud_mix=Ud_primary,
                        rotation_deg=float(alpha),
                        position_lab_um=(0.0, 0.0, 0.0),
                    )

                    burgers_int = np.asarray(
                        [
                            int(round(b_table[b_idx, 0] * np.sqrt(2))),
                            int(round(b_table[b_idx, 1] * np.sqrt(2))),
                            int(round(b_table[b_idx, 2] * np.sqrt(2))),
                        ],
                        dtype=np.int32,
                    )
                    sample: dict[str, Any] = {
                        "name": "simulated, dislocation identification (z-scan)",
                        "z_offset_um": float(z_off),
                        "primary": {
                            "slip_plane_normal": np.asarray(plane, dtype=np.int32),
                            "burgers": burgers_int,
                            "rotation_deg": float(alpha),
                            "position_um": np.asarray([0.0, 0.0, 0.0]),
                        },
                    }

                    detectors: dict[
                        str, list[tuple[int, np.ndarray, float, float, float, fm.ForwardContext]]
                    ] = {}
                    if zscan.include_secondary:
                        sec = _draw_dislocation(secondary_rng, pos_std_um=0.0)
                        secondary_spec = MixedDislocSpec(
                            Ud_mix=sec["Ud"],
                            rotation_deg=sec["alpha_deg"],
                            position_lab_um=sec["pos_um"],
                        )
                        sample["secondary"] = _build_dislocation_sample_entry(sec)
                        # W2 dedup (v2.6.0): primary+secondary fields computed once;
                        # combined = Σ(Fg_i − I) + I. The solo secondary used
                        # to be rendered without its position offset — drawn
                        # at pos_std_um=0.0 the offset is (0,0,0), so passing
                        # it through the spec is bit-identical.
                        Hg, solo_hgs = find_hg_scene(
                            rl_shifted,
                            Us_,
                            [primary_spec, secondary_spec],
                            Theta_,
                            per_dislocation=zscan.render_per_dislocation,
                        )
                        if zscan.render_per_dislocation:
                            # Primary + secondary each rendered alone (noiseless
                            # ground-truth instance labels). Bypass the Poisson
                            # pass, which only touches `dfxm_sim_detector`.
                            assert solo_hgs is not None
                            prim_args, _ = _scan_frames_args(
                                solo_hgs[0], frames_at_z, config.scan, ctx
                            )
                            sec_args, _ = _scan_frames_args(
                                solo_hgs[1], frames_at_z, config.scan, ctx
                            )
                            detectors["dfxm_sim_detector_primary"] = prim_args
                            detectors["dfxm_sim_detector_secondary"] = sec_args
                    else:
                        Hg, _ = find_hg_scene(
                            rl_shifted,
                            Us_,
                            [primary_spec],
                            Theta_,
                        )

                    args_list, positioners = _scan_frames_args(Hg, frames_at_z, config.scan, ctx)
                    detectors["dfxm_sim_detector"] = args_list

                    # g·b labels (Task 5, M3 plan 2): primary dislocation scalars.
                    # Label the PRIMARY (deterministic sweep b) with gb_cos/gb_visible.
                    # If a secondary is present and its b is distinct, also write
                    # gb_cos_secondary / gb_visible_secondary for completeness.
                    _thr_z = crystal_cfg.invisibility_threshold_deg
                    _b_primary = b_table[b_idx]
                    _zscan_dfxm_geo: dict[str, Any] = {
                        "Hg": Hg,
                        "q_hkl": q_hkl,
                        "theta": float(theta_0_),
                        "psize": float(psize_),
                        "zl_rms": float(zl_rms_),
                        "gb_cos": _gb_cos(q_hkl, _b_primary),
                        "gb_visible": np.int8(_gb_visible(q_hkl, _b_primary, _thr_z)),
                    }
                    if zscan.include_secondary and "secondary" in sample:
                        # `sec` is the dict from _draw_dislocation; b_vec is the
                        # physical Burgers direction (before integer rounding).
                        _b_sec = sec["b_vec"]
                        _zscan_dfxm_geo["gb_cos_secondary"] = _gb_cos(q_hkl, _b_sec)
                        _zscan_dfxm_geo["gb_visible_secondary"] = np.int8(
                            _gb_visible(q_hkl, _b_sec, _thr_z)
                        )
                    yield ScanSpec(
                        title=_identify_title(scan_mode, n_frames, config.scan),
                        sample=sample,
                        positioners=positioners,
                        dfxm_geo=_zscan_dfxm_geo,
                        detectors=detectors,
                        attrs={
                            "scan_mode": scan_mode,
                            "scanned_axes": scanned_axes,
                            "identify_mode": "z-scan",
                        },
                    )


def _run_identification_zscan(
    config: IdentificationConfig,
    output_dir: Path,
    ctx: fm.ForwardContext,
    *,
    reflection_index: int = 0,
    visibility_qs: list[np.ndarray] | None = None,
    reflection_attrs: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Dispatcher: feed ``_iter_identification_zscan`` into write_identification_h5.

    After the master + per-scan detector files are written, applies the
    post-write Poisson / intensity-scale pass to the combined detector only;
    any per-dislocation files (`render_per_dislocation=True`) stay noiseless.

    reflection_index, visibility_qs, reflection_attrs: forwarded from
    _dispatch_identification; defaults preserve the single-reflection path.
    """
    assert config.zscan is not None  # validated in __post_init__
    output_dir.mkdir(parents=True, exist_ok=True)
    n_scans = write_identification_h5(
        output_dir,
        scan_iter=_iter_identification_zscan(config, ctx, visibility_qs=visibility_qs),
        cli=" ".join(sys.argv),
        config_toml=_identification_config_to_toml_str(config),
        max_workers=config.io.max_workers,
        write_strain_provenance=config.io.write_strain_provenance,
        geometry_attrs=_identify_geometry_attrs(config, ctx),
        ctx=ctx,
        reflection_attrs=reflection_attrs,
    )
    _maybe_apply_poisson_noise(config, output_dir, n_scans, reflection_index=reflection_index)
    return {
        "n_configurations": n_scans,
        "output_dir": output_dir,
        "master_path": output_dir / "dfxm_identify.h5",
    }


def _identify_geometry_attrs(
    config: IdentificationConfig, ctx: fm.ForwardContext
) -> dict[str, Any]:
    """Geometry provenance attrs for every /N.1 of an identify run (M2 parity)."""
    return geometry_provenance_attrs(
        geometry_mode=config.geometry.mode,
        eta=config.geometry.eta,
        theta_0=float(ctx.geometry.theta_0),
        mount=config.geometry.mount,
    )


def _dispatch_identification(
    config: IdentificationConfig,
    output_dir: Path,
    ctx: fm.ForwardContext,
    *,
    reflection: _ReflectionRun | None = None,
    reflection_index: int = 0,
    n_reflections: int = 0,
    visibility_qs: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Route to the appropriate mode runner, threading multi-reflection params.

    This factors the 3-way mode dispatch so it is shared between:
    - single-reflection run_identification (reflection=None, all other params 0/None
      → identical byte output to pre-M3 code)
    - the multi-reflection loop (reflection given, reflection_index > 0, etc.)

    reflection_attrs is assembled locally from `reflection` when given, else None.
    Note: multi mode drops visibility_qs — no deterministic sweep grid to gate.
    """
    reflection_attrs: dict[str, object] | None = None
    if reflection is not None:
        reflection_attrs = {
            "hkl_reflection": np.asarray(reflection.hkl, dtype=np.int64),
            "omega": float(reflection.omega),
            "reflection_index": reflection_index,
            "n_reflections": n_reflections,
        }
    if config.mode == "single":
        return _run_identification_single(
            config,
            output_dir,
            ctx,
            reflection_index=reflection_index,
            visibility_qs=visibility_qs,
            reflection_attrs=reflection_attrs,
        )
    if config.mode == "multi":
        return _run_identification_multi(
            config,
            output_dir,
            ctx,
            reflection_index=reflection_index,
            reflection_attrs=reflection_attrs,
        )
    return _run_identification_zscan(
        config,
        output_dir,
        ctx,
        reflection_index=reflection_index,
        visibility_qs=visibility_qs,
        reflection_attrs=reflection_attrs,
    )


def run_identification(
    config: IdentificationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Dispatch to single / multi / z-scan runner based on config.mode.

    Single-reflection (no ``[[reflections]]``): writes one standard
    ``dfxm_identify.h5`` master in ``output_dir``.  Return value depends
    on mode: ``{"n_images", "output_dir", "master_path"}`` for single,
    ``{"n_samples", ...}`` for multi, ``{"n_configurations", ...}`` for
    z-scan.

    Multi-reflection (``[[reflections]]`` present): loops over the resolved
    ``ReflectionRun`` list, writes one standard ``dfxm_identify.h5`` per
    reflection into ``output_dir/reflection_NNN/``, then writes a thin
    super-master ``dfxm_identify_multi.h5`` in ``output_dir`` with
    ExternalLinks.  Returns ``{"n_reflections": int, "reflections":
    list[dict]}``.

    RNG policy (M3 plan 2): the dislocation parameter stream uses
    ``config.noise.rng_seed`` UNCHANGED for every reflection — all
    reflections image the SAME crystal realization (the scientific
    requirement for multi-reflection data). Poisson noise is seeded
    independently per reflection via ``[rng_seed, reflection_index]``
    so detector noise is not correlated across reflections.
    """
    if config.reflections:
        # M3 plan 2 (B'): loop over [[reflections]], one standard identify
        # master per reflection in reflection_NNN/ subdirs.
        runs = config.reflections
        results: list[dict[str, Any]] = []
        # All-reflections visibility: collect unit q for every reflection so
        # the sweep grid is IDENTICAL across all masters (spec §8).
        all_vis_qs = [_q_unit(r.hkl) for r in runs]
        for idx, run in enumerate(runs, start=1):
            res_run = _resolution_for_run(config.reciprocal, config.geometry, run)
            ctx_run = _context_for_run(res_run, run)
            sub_dir = output_dir / f"reflection_{idx:03d}"
            results.append(
                _dispatch_identification(
                    config,
                    sub_dir,
                    ctx_run,
                    reflection=run,
                    reflection_index=idx,
                    n_reflections=len(runs),
                    visibility_qs=all_vis_qs,
                )
            )
        write_multi_reflection_master(
            output_dir,
            runs,
            master_name="dfxm_identify.h5",
            mount=config.geometry.mount,
            keV=config.reciprocal.keV,
        )
        return {"n_reflections": len(runs), "reflections": results}

    res = _load_resolution(config.reciprocal, config.geometry)

    # S3 (#16): CM call site retired. build_forward_context uses run_theta(config)
    # (the correct Bragg angle) to rebuild Theta/rl/prob_z via build_geometry_context
    # — same expressions the CM used. Z_shift callers pass xl_range so
    # oblique z-scans remain correct without touching module globals.
    ctx = fm.build_forward_context(
        run_theta(config),
        res,
        config.reciprocal.hkl,
    )
    return _dispatch_identification(config, output_dir, ctx)
