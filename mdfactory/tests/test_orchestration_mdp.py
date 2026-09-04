# ABOUTME: Tests for MDP parser/modifier used in rescue retry
# ABOUTME: Verifies parsing, modification, and binary-division rescue tier logic
"""Tests for mdfactory.orchestration.mdp."""

import pytest

from mdfactory.orchestration.mdp import (
    RESCUE_ELIGIBLE_STAGES,
    RESCUE_PARAMS,
    apply_rescue_tier,
    get_mdp_value,
    modify_mdp_value,
    parse_mdp,
    write_mdp,
)


@pytest.fixture
def em_mdp(tmp_path):
    """Create a minimal EM MDP file."""
    content = """; minim.mdp
integrator  = steep
emtol       = 1000.0        ; Stop at max force < 1000
emstep      = 0.01          ; Step size
nsteps      = 50000         ; Max steps
nstlist     = 1
"""
    p = tmp_path / "em.mdp"
    p.write_text(content)
    return p


@pytest.fixture
def nvt_mdp(tmp_path):
    """Create a minimal NVT MDP file."""
    content = """; NVT equilibration
integrator  = md
dt          = 0.002         ; 2 fs
nsteps      = 50000         ; 100 ps
constraints = h-bonds
"""
    p = tmp_path / "nvt.mdp"
    p.write_text(content)
    return p


class TestParseMdp:
    def test_parses_key_value_pairs(self, em_mdp):
        parsed = parse_mdp(em_mdp)
        assert get_mdp_value(parsed, "integrator") == "steep"
        assert get_mdp_value(parsed, "emtol") == "1000.0"
        assert get_mdp_value(parsed, "emstep") == "0.01"
        assert get_mdp_value(parsed, "nsteps") == "50000"

    def test_preserves_comment_lines(self, em_mdp):
        parsed = parse_mdp(em_mdp)
        assert parsed[0][1] is None
        assert parsed[0][2] is None

    def test_handles_semicolon_inline_comments(self, em_mdp):
        parsed = parse_mdp(em_mdp)
        assert get_mdp_value(parsed, "emtol") == "1000.0"

    def test_normalizes_dashes_to_underscores(self, tmp_path):
        p = tmp_path / "test.mdp"
        p.write_text("lincs-iter = 2\nnstxout-compressed = 500\n")
        parsed = parse_mdp(p)
        assert get_mdp_value(parsed, "lincs_iter") == "2"
        assert get_mdp_value(parsed, "nstxout_compressed") == "500"

    def test_missing_key_returns_none(self, em_mdp):
        parsed = parse_mdp(em_mdp)
        assert get_mdp_value(parsed, "nonexistent") is None


class TestModifyMdpValue:
    def test_replaces_value(self, em_mdp):
        parsed = parse_mdp(em_mdp)
        modified = modify_mdp_value(parsed, "emstep", "0.005")
        assert get_mdp_value(modified, "emstep") == "0.005"

    def test_preserves_other_values(self, em_mdp):
        parsed = parse_mdp(em_mdp)
        modified = modify_mdp_value(parsed, "emstep", "0.005")
        assert get_mdp_value(modified, "nsteps") == "50000"
        assert get_mdp_value(modified, "emtol") == "1000.0"


class TestWriteMdp:
    def test_roundtrip_preserves_structure(self, em_mdp, tmp_path):
        parsed = parse_mdp(em_mdp)
        out = tmp_path / "out.mdp"
        write_mdp(parsed, out)
        reparsed = parse_mdp(out)
        assert get_mdp_value(reparsed, "emstep") == "0.01"
        assert get_mdp_value(reparsed, "nsteps") == "50000"

    def test_modified_roundtrip(self, em_mdp, tmp_path):
        parsed = parse_mdp(em_mdp)
        modified = modify_mdp_value(parsed, "emstep", "0.005")
        out = tmp_path / "modified.mdp"
        write_mdp(modified, out)
        reparsed = parse_mdp(out)
        assert get_mdp_value(reparsed, "emstep") == "0.005"


class TestApplyRescueTier:
    def test_em_tier1_halves_emstep_doubles_nsteps(self, tmp_path, em_mdp, monkeypatch):
        from mdfactory.orchestration.stages import StageSpec

        mock_spec = StageSpec(
            name="EM",
            deffnm="min",
            mdp_file="em.mdp",
            gro_in="system.pdb",
            gro_out="min.gro",
            tpr_file="min.tpr",
            cpt_file="min.cpt",
            prereq_cpt=None,
            traj_files=(),
            ref_file=None,
            maxwarn=1,
            supports_pme_gpu=False,
        )
        monkeypatch.setattr("mdfactory.orchestration.stages.STAGE_BY_NAME", {"EM": mock_spec})
        rescue_path = apply_rescue_tier(tmp_path, "EM", 1)
        assert rescue_path.name == "em_rescue_t1.mdp"
        assert rescue_path.exists()
        parsed = parse_mdp(rescue_path)
        assert get_mdp_value(parsed, "emstep") == "0.005"
        assert get_mdp_value(parsed, "nsteps") == "100000"

    def test_em_tier2_compounds(self, tmp_path, em_mdp, monkeypatch):
        from mdfactory.orchestration.stages import StageSpec

        mock_spec = StageSpec(
            name="EM",
            deffnm="min",
            mdp_file="em.mdp",
            gro_in="system.pdb",
            gro_out="min.gro",
            tpr_file="min.tpr",
            cpt_file="min.cpt",
            prereq_cpt=None,
            traj_files=(),
            ref_file=None,
            maxwarn=1,
            supports_pme_gpu=False,
        )
        monkeypatch.setattr("mdfactory.orchestration.stages.STAGE_BY_NAME", {"EM": mock_spec})
        rescue_path = apply_rescue_tier(tmp_path, "EM", 2)
        parsed = parse_mdp(rescue_path)
        assert get_mdp_value(parsed, "emstep") == "0.0025"
        assert get_mdp_value(parsed, "nsteps") == "200000"

    def test_nvt_tier1_halves_dt_doubles_nsteps(self, tmp_path, nvt_mdp, monkeypatch):
        from mdfactory.orchestration.stages import StageSpec

        mock_spec = StageSpec(
            name="NVT",
            deffnm="nvt",
            mdp_file="nvt.mdp",
            gro_in="min.gro",
            gro_out="nvt.gro",
            tpr_file="nvt.tpr",
            cpt_file="nvt.cpt",
            prereq_cpt=None,
            traj_files=(),
            ref_file="min.gro",
            maxwarn=1,
            supports_pme_gpu=True,
        )
        monkeypatch.setattr("mdfactory.orchestration.stages.STAGE_BY_NAME", {"NVT": mock_spec})
        rescue_path = apply_rescue_tier(tmp_path, "NVT", 1)
        assert rescue_path.name == "nvt_rescue_t1.mdp"
        parsed = parse_mdp(rescue_path)
        assert get_mdp_value(parsed, "dt") == "0.001"
        assert get_mdp_value(parsed, "nsteps") == "100000"

    def test_invalid_stage_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not rescue-eligible"):
            apply_rescue_tier(tmp_path, "Production", 1)

    def test_tier_zero_raises(self, tmp_path):
        with pytest.raises(ValueError, match="tier must be >= 1"):
            apply_rescue_tier(tmp_path, "EM", 0)

    def test_missing_mdp_raises(self, tmp_path, monkeypatch):
        from mdfactory.orchestration.stages import StageSpec

        mock_spec = StageSpec(
            name="EM",
            deffnm="min",
            mdp_file="em.mdp",
            gro_in="system.pdb",
            gro_out="min.gro",
            tpr_file="min.tpr",
            cpt_file="min.cpt",
            prereq_cpt=None,
            traj_files=(),
            ref_file=None,
            maxwarn=1,
            supports_pme_gpu=False,
        )
        monkeypatch.setattr("mdfactory.orchestration.stages.STAGE_BY_NAME", {"EM": mock_spec})
        with pytest.raises(FileNotFoundError):
            apply_rescue_tier(tmp_path, "EM", 1)


class TestRescueConstants:
    def test_eligible_stages(self):
        assert RESCUE_ELIGIBLE_STAGES == {"EM", "NVT", "NPT"}

    def test_production_not_eligible(self):
        assert "Production" not in RESCUE_ELIGIBLE_STAGES

    def test_rescue_params_keys_match_eligible(self):
        assert set(RESCUE_PARAMS.keys()) == RESCUE_ELIGIBLE_STAGES
