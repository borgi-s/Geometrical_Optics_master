"""Tests for the bundled-config accessor and the dfxm-init command."""

from __future__ import annotations


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
        # 11 shipped templates: default + 3 identification_* + 7 variants
        assert len(rels) == 11
