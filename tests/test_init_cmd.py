"""Tests for the bundled-config accessor and the dfxm-init command."""

from __future__ import annotations

from pathlib import Path


class TestBundledConfigAccessor:
    def test_configs_root_contains_default(self) -> None:
        from dfxm_geo.data import configs_root

        root = configs_root()
        assert (root / "default.toml").is_file()

    def test_iter_config_files_lists_full_tree(self) -> None:
        from dfxm_geo.data import iter_config_files

        rels = {rel for rel, _ in iter_config_files()}
        assert "default.toml" in rels
        assert "identification_single.toml" in rels
        assert "variants/dis_1.toml" in rels
        assert "al_oblique_figure3.toml" in rels
        # 14 shipped templates: default + 3 identification_* + 7 variants
        # + al_oblique_figure3 (v2.3.0 oblique-angle paper Fig 3B reproduction)
        # + gnb_leds_eq11 + gnb_frankus (gnb wall mode examples, feature/gnb-walls).
        assert len(rels) == 14


class TestDfxmInit:
    def test_writes_full_tree(self, tmp_path: Path) -> None:
        from dfxm_geo.data import iter_config_files
        from dfxm_geo.init_cmd import cli_main

        dest = tmp_path / "configs"
        rc = cli_main(["--dest", str(dest)])
        assert rc == 0
        for rel, src in iter_config_files():
            written = dest / rel
            assert written.is_file(), f"missing {rel}"
            assert written.read_bytes() == src.read_bytes(), f"content mismatch {rel}"

    def test_skips_existing_without_force(self, tmp_path: Path) -> None:
        from dfxm_geo.init_cmd import cli_main

        dest = tmp_path / "configs"
        (dest).mkdir()
        sentinel = dest / "default.toml"
        sentinel.write_text("DO NOT OVERWRITE", encoding="utf-8")

        rc = cli_main(["--dest", str(dest)])
        assert rc == 0
        assert sentinel.read_text(encoding="utf-8") == "DO NOT OVERWRITE"

    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        from dfxm_geo.data import configs_root
        from dfxm_geo.init_cmd import cli_main

        dest = tmp_path / "configs"
        dest.mkdir()
        sentinel = dest / "default.toml"
        sentinel.write_text("DO NOT OVERWRITE", encoding="utf-8")

        rc = cli_main(["--dest", str(dest), "--force"])
        assert rc == 0
        expected = (configs_root() / "default.toml").read_bytes()
        assert sentinel.read_bytes() == expected
