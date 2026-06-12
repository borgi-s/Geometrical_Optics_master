"""M4 Stage 4.2: crystal/cif.py loader unit tests (spec 2026-06-12)."""

from pathlib import Path

import pytest

pytest.importorskip("gemmi")

from dfxm_geo.crystal.cif import load_cif

DATA = Path(__file__).parent / "data" / "cif"


class TestLoadCif:
    def test_al_cubic(self) -> None:
        cc = load_cif(DATA / "al_fm3m.cif")
        assert cc.lattice == "cubic"
        assert cc.a == pytest.approx(4.0495e-10)  # Angstrom -> metres
        assert cc.b == pytest.approx(4.0495e-10)
        assert cc.alpha_deg == 90.0 and cc.gamma_deg == 90.0
        assert cc.space_group == "F m -3 m"  # canonical H-M

    def test_mg_hexagonal(self) -> None:
        cc = load_cif(DATA / "mg_p63mmc.cif")
        assert cc.lattice == "hexagonal"
        assert cc.a == pytest.approx(3.2094e-10)
        assert cc.c == pytest.approx(5.2108e-10)
        assert cc.gamma_deg == 120.0
        assert cc.space_group == "P 63/m m c"

    def test_corundum_trigonal_sg_maps_to_hexagonal_lattice(self) -> None:
        # R-3c is crystal-system 'trigonal' but the CIF is in hexagonal axes
        # (gamma=120) -> lattice='hexagonal' per the Stage 4.1 convention.
        cc = load_cif(DATA / "al2o3_r3c.cif")
        assert cc.lattice == "hexagonal"
        assert cc.space_group == "R -3 c"
        assert cc.c == pytest.approx(12.9933e-10)

    def test_no_spacegroup_is_none_not_error(self) -> None:
        cc = load_cif(DATA / "nospacegroup.cif")
        assert cc.space_group is None
        assert cc.lattice == "cubic"  # inferred from cell parameters

    def test_no_cell_raises(self) -> None:
        with pytest.raises(ValueError, match="no cell parameters"):
            load_cif(DATA / "nocell.cif")

    def test_missing_file_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            load_cif(DATA / "does_not_exist.cif")


class TestGemmiMissing:
    def test_import_error_mentions_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import dfxm_geo.crystal.cif as cif_mod

        def _boom() -> None:
            raise ImportError(
                "CIF/space-group support requires gemmi — install with "
                "`pip install dfxm-geo[cif]` (or `conda install -c conda-forge gemmi`)."
            )

        monkeypatch.setattr(cif_mod, "_import_gemmi", _boom)
        with pytest.raises(ImportError, match=r"dfxm-geo\[cif\]"):
            cif_mod.load_cif(DATA / "al_fm3m.cif")
