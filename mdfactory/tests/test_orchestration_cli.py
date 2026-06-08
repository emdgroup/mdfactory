# ABOUTME: Integration tests for the CLI build command orchestration paths
# ABOUTME: Tests CSV/YAML/summary-YAML dispatch, dry-run, and error handling
"""Integration tests for CLI build command with orchestration."""

from unittest.mock import patch

import pytest
import yaml

parsl = pytest.importorskip("parsl", reason="parsl not installed")


@pytest.fixture()
def sample_csv(tmp_path):
    """Create a minimal CSV input file for testing."""
    csv_content = (
        "simulation_type,system.species.SOL.smiles,system.species.SOL.count,parametrization\n"
        "mixedbox,O,100,cgenff\n"
    )
    csv_path = tmp_path / "input.csv"
    csv_path.write_text(csv_content)
    return csv_path


@pytest.fixture()
def sample_yaml(tmp_path):
    """Create a minimal single-build YAML input file."""
    data = {
        "simulation_type": "mixedbox",
        "system": {"species": [{"smiles": "O", "count": 100, "resname": "SOL"}]},
        "parametrization": "cgenff",
        "engine": "gromacs",
    }
    yaml_path = tmp_path / "build.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f)
    return yaml_path


@pytest.fixture()
def sample_config(tmp_path):
    """Create a minimal executor config YAML."""
    data = {"provider": "local", "max_workers_per_node": 1}
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(data, f)
    return cfg_path


class TestBuildCommandCSV:
    """Tests for CSV input dispatch path."""

    def test_csv_dry_run(self, sample_csv, tmp_path, monkeypatch):
        """CSV input with --dry-run calls build_systems with dry_run=True."""
        from mdfactory.cli import _build_from_csv

        with patch("mdfactory.orchestration.build_systems") as mock_build:
            mock_build.return_value = [{"hash": "X", "simulation_type": "mixedbox"}]
            _build_from_csv(sample_csv, tmp_path, dry_run=True)

        mock_build.assert_called_once()
        assert mock_build.call_args[1]["dry_run"] is True

    def test_csv_dry_run_no_filesystem_side_effects(self, sample_csv, tmp_path, monkeypatch):
        """CSV dry-run does not create directories or summary YAML."""
        from mdfactory.cli import _build_from_csv

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch("mdfactory.orchestration.build_systems") as mock_build:
            mock_build.return_value = []
            _build_from_csv(sample_csv, output_dir, dry_run=True)

        # No summary YAML or hash directories should be created in output
        assert not (output_dir / "input.yaml").exists()
        assert len(list(output_dir.iterdir())) == 0

    def test_csv_sequential_builds_locally(self, sample_csv, tmp_path, monkeypatch):
        """CSV input without --config builds sequentially."""
        from mdfactory.cli import _build_from_csv

        with patch("mdfactory.cli.run_build_from_dict") as mock_build:
            _build_from_csv(sample_csv, tmp_path, config=None, dry_run=False)

        mock_build.assert_called_once()

    def test_csv_generates_summary_yaml(self, sample_csv, tmp_path, monkeypatch):
        """CSV build generates a summary YAML matching prepare-build format."""
        from mdfactory.cli import _build_from_csv

        with patch("mdfactory.cli.run_build_from_dict"):
            _build_from_csv(sample_csv, tmp_path, config=None, dry_run=False)

        summary_path = tmp_path / "input.yaml"
        assert summary_path.exists()
        with open(summary_path) as f:
            summary = yaml.safe_load(f)
        assert summary["n_systems"] == 1
        assert str(sample_csv) == summary["input"]
        assert len(summary["hash"]) == 1
        assert len(summary["system_directory"]) == 1
        assert "date" in summary
        assert "simulation_type" in summary

    def test_csv_with_config_uses_parsl(self, sample_csv, sample_config, tmp_path, monkeypatch):
        """CSV input with --config dispatches via build_systems."""
        from mdfactory.cli import _build_from_csv

        with patch("mdfactory.orchestration.build_systems") as mock_build:
            mock_build.return_value = [{"hash": "X", "status": "success"}]
            _build_from_csv(sample_csv, tmp_path, config=sample_config, dry_run=False)

        mock_build.assert_called_once()


class TestBuildCommandYAML:
    """Tests for single-YAML input dispatch path."""

    def test_yaml_sequential_builds_into_hash_dir(self, sample_yaml, tmp_path, monkeypatch):
        """Single YAML without --config builds into output/{hash}/."""
        from mdfactory.cli import _build_from_yaml

        with patch("mdfactory.cli.run_build_from_file") as mock_build:
            _build_from_yaml(sample_yaml, tmp_path)

        mock_build.assert_called_once()
        # Verify the working directory was a hash-based subdirectory
        call_args = mock_build.call_args
        # run_build_from_file is called with the input path
        assert call_args[0][0] == sample_yaml

    def test_yaml_dry_run(self, sample_yaml, tmp_path, monkeypatch):
        """Single YAML with --dry-run calls build_systems with dry_run=True."""
        from mdfactory.cli import _build_from_yaml

        with patch("mdfactory.orchestration.build_systems") as mock_build:
            mock_build.return_value = [{"hash": "X", "simulation_type": "mixedbox"}]
            _build_from_yaml(sample_yaml, tmp_path, dry_run=True)

        mock_build.assert_called_once()
        assert mock_build.call_args[1]["dry_run"] is True

    def test_yaml_with_config_uses_parsl(self, sample_yaml, sample_config, tmp_path, monkeypatch):
        """Single YAML with --config dispatches via build_systems."""
        from mdfactory.cli import _build_from_yaml

        with patch("mdfactory.orchestration.build_systems") as mock_build:
            mock_build.return_value = [{"hash": "X", "status": "success"}]
            _build_from_yaml(sample_yaml, tmp_path, config=sample_config, dry_run=False)

        mock_build.assert_called_once()


class TestBuildCommandSummaryYAML:
    """Tests for summary YAML (prepare-build output) dispatch path."""

    def test_summary_yaml_empty_dirs_exits(self, tmp_path):
        """Summary YAML with no directories causes exit."""
        from mdfactory.cli import _build_from_summary_yaml

        data = {"system_directory": [], "hash": []}
        with pytest.raises(SystemExit):
            _build_from_summary_yaml(data, tmp_path)

    def test_summary_yaml_missing_build_file_exits(self, tmp_path):
        """Summary YAML referencing non-existent build file exits."""
        from mdfactory.cli import _build_from_summary_yaml

        data = {"system_directory": [str(tmp_path / "nonexistent")], "hash": ["ABC123"]}
        with pytest.raises(SystemExit):
            _build_from_summary_yaml(data, tmp_path)

    def test_summary_yaml_mismatched_lengths_exits(self, tmp_path):
        """Summary YAML with mismatched list lengths causes exit."""
        from mdfactory.cli import _build_from_summary_yaml

        data = {"system_directory": [str(tmp_path), str(tmp_path)], "hash": ["A"]}
        with pytest.raises(SystemExit):
            _build_from_summary_yaml(data, tmp_path)

    def test_summary_yaml_dry_run(self, tmp_path):
        """Summary YAML with --dry-run calls build_systems with dry_run=True."""
        from mdfactory.cli import _build_from_summary_yaml

        hash_val = "TESTHASH123"
        build_dir = tmp_path / hash_val
        build_dir.mkdir()
        yaml_data = {
            "simulation_type": "mixedbox",
            "parametrization": "cgenff",
            "engine": "gromacs",
            "system": {"species": [{"smiles": "O", "count": 100, "resname": "SOL"}]},
        }
        (build_dir / f"{hash_val}.yaml").write_text(yaml.safe_dump(yaml_data))
        data = {"system_directory": [str(build_dir)], "hash": [hash_val]}

        with patch("mdfactory.orchestration.build_systems") as mock_build:
            mock_build.return_value = [{"hash": hash_val}]
            _build_from_summary_yaml(data, tmp_path, dry_run=True)

        mock_build.assert_called_once()
        assert mock_build.call_args[1]["dry_run"] is True

    def test_summary_yaml_sequential(self, tmp_path):
        """Summary YAML without --config builds sequentially."""
        from mdfactory.cli import _build_from_summary_yaml

        hash_val = "SEQHASH456"
        build_dir = tmp_path / hash_val
        build_dir.mkdir()
        yaml_data = {
            "simulation_type": "mixedbox",
            "parametrization": "cgenff",
            "engine": "gromacs",
            "system": {"species": [{"smiles": "O", "count": 100, "resname": "SOL"}]},
        }
        (build_dir / f"{hash_val}.yaml").write_text(yaml.safe_dump(yaml_data))
        data = {"system_directory": [str(build_dir)], "hash": [hash_val]}

        with patch("mdfactory.cli.run_build_from_dict") as mock_build:
            _build_from_summary_yaml(data, tmp_path, config=None, dry_run=False)

        mock_build.assert_called_once()

    def test_summary_yaml_parallel(self, tmp_path, sample_config):
        """Summary YAML with --config dispatches via build_systems."""
        from mdfactory.cli import _build_from_summary_yaml

        hash_val = "PARHASH789"
        build_dir = tmp_path / hash_val
        build_dir.mkdir()
        yaml_data = {
            "simulation_type": "mixedbox",
            "parametrization": "cgenff",
            "engine": "gromacs",
            "system": {"species": [{"smiles": "O", "count": 100, "resname": "SOL"}]},
        }
        (build_dir / f"{hash_val}.yaml").write_text(yaml.safe_dump(yaml_data))
        data = {"system_directory": [str(build_dir)], "hash": [hash_val]}

        with patch("mdfactory.orchestration.build_systems") as mock_build:
            mock_build.return_value = [{"hash": hash_val, "status": "success"}]
            _build_from_summary_yaml(data, tmp_path, config=sample_config, dry_run=False)

        mock_build.assert_called_once()


class TestBuildCommandErrors:
    """Tests for error handling in build command."""

    def test_unsupported_extension_exits(self, tmp_path):
        """Unsupported file extension causes sys.exit."""
        from mdfactory.cli import build_system

        txt_file = tmp_path / "input.txt"
        txt_file.write_text("hello")
        with pytest.raises(SystemExit):
            build_system(txt_file, output=tmp_path)

    def test_malformed_csv_exits(self, tmp_path):
        """CSV with missing required columns causes sys.exit."""
        from mdfactory.cli import _build_from_csv

        # Missing simulation_type column
        csv_content = "system.species.SOL.smiles,system.species.SOL.count\nO,100\n"
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text(csv_content)
        with pytest.raises(SystemExit):
            _build_from_csv(csv_path, tmp_path, config=None, dry_run=False)

    def test_empty_yaml_exits(self, tmp_path):
        """Empty YAML file causes sys.exit with clear message."""
        from mdfactory.cli import _build_from_yaml

        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("")
        with pytest.raises(SystemExit):
            _build_from_yaml(empty_yaml, tmp_path)


class TestBuildCommandRouting:
    """Tests for build_system() top-level routing logic."""

    def test_csv_routes_to_csv_handler(self, sample_csv, tmp_path):
        """build_system() with .csv file routes to CSV handler."""
        from mdfactory.cli import build_system

        with patch("mdfactory.cli.run_build_from_dict") as mock_build:
            build_system(sample_csv, output=tmp_path)

        mock_build.assert_called_once()

    def test_yaml_routes_to_yaml_handler(self, sample_yaml, tmp_path):
        """build_system() with .yaml file routes to YAML handler."""
        from mdfactory.cli import build_system

        with patch("mdfactory.cli.run_build_from_file") as mock_build:
            build_system(sample_yaml, output=tmp_path)

        mock_build.assert_called_once()
