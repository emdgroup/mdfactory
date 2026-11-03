# ABOUTME: Tests for proteinbox models, topology parsing, and YAML validation
# ABOUTME: Covers ProteinSpecies, ProteinBoxComposition, Pdb2gmxConfig, and BuildInput
"""Tests for proteinbox models, topology parsing, and YAML validation."""

import hashlib
import io
import shutil
import tarfile
import textwrap
from pathlib import Path
from types import SimpleNamespace

import MDAnalysis as mda
import numpy as np
import pytest
import yaml

from mdfactory import workflows
from mdfactory.build import _center_in_cubic_box, _ion_counts
from mdfactory.models.composition import ProteinBoxComposition, ProteinMixedBoxComposition
from mdfactory.models.input import BuildInput
from mdfactory.models.parametrization import CharmmConfig, Pdb2gmxConfig
from mdfactory.models.species import ProteinSpecies, SingleMoleculeSpecies
from mdfactory.prepare import resolve_protein_pdb_path
from mdfactory.setup import protein
from mdfactory.setup.protein import (
    _apply_protonation_states,
    _build_disulfide_prompt_input,
    _sum_atoms_column,
    bundle_forcefield_into_topology,
    check_forcefield_available,
    clean_pdb,
    download_forcefield,
    run_pdb2gmx,
    update_topology_molecules,
)

from .conftest import protein_input_data

PROTEIN_SIM_TYPES = ["proteinbox", "protein_mixedbox"]
CONFIG_CLASSES = {"proteinbox": Pdb2gmxConfig, "protein_mixedbox": CharmmConfig}
COMPOSITION_CLASSES = {
    "proteinbox": ProteinBoxComposition,
    "protein_mixedbox": ProteinMixedBoxComposition,
}


def _pdb_atom(serial, atom_name, resname, resid, chain="A", x=0.0, insertion_code=""):
    return (
        f"ATOM  {serial:5d} {atom_name:<4s} {resname:>3s} "
        f"{chain:1s}{resid:4d}{insertion_code:1s}   "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00\n"
    )


def _write_pdb(tmp_path, *atoms, atom_name="N"):
    """Write input.pdb from ``(resname, resid[, chain, x, insertion_code])`` atom tuples."""
    pdb = tmp_path / "input.pdb"
    pdb.write_text("".join(_pdb_atom(i, atom_name, *a) for i, a in enumerate(atoms, 1)))
    return pdb


def _patch_pdb2gmx(monkeypatch, molecule_names, extra_files=(), calls=None):
    """Stub gmx, force-field and environment lookups and fake the pdb2gmx subprocess.

    The fake writes a topology listing *molecule_names* (plus any *extra_files*) and,
    when *calls* is a dict, records the invocation in it.
    """

    def fake_run(cmd, cwd, text, input, capture_output, timeout, env=None, check=None):
        if calls is not None:
            calls.update(cmd=cmd, cwd=cwd, input=input, env=env, check=check)
        output_dir = Path(cwd)
        (output_dir / "processed.gro").write_text("mock gro\n")
        molecules = "".join(f"{name} 1\n" for name in molecule_names)
        (output_dir / "topol.top").write_text(f"[ molecules ]\n{molecules}")
        for name in extra_files:
            (output_dir / name).write_text(f"; {name}\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("mdfactory.setup.protein.subprocess.run", fake_run)
    monkeypatch.setattr("mdfactory.setup.protein.check_gmx_available", lambda: Path("/mock/gmx"))
    monkeypatch.setattr("mdfactory.setup.protein.check_forcefield_available", lambda ff: None)
    monkeypatch.setattr(
        protein.settings, "gromacs_env", lambda base_env=None: {"GMXLIB": "/mock/forcefields"}
    )


def _run(tmp_path, pdb, **kwargs):
    """Call run_pdb2gmx with default (empty) disulfide/protonation arguments."""
    kwargs.setdefault("config", Pdb2gmxConfig(forcefield="charmm36m", water_model="tip3p"))
    kwargs.setdefault("disulfide_bonds", [])
    kwargs.setdefault("protonation_states", {})
    kwargs.setdefault("output_dir", tmp_path / "out")
    return run_pdb2gmx(pdb_path=pdb, **kwargs)


@pytest.fixture
def gmx_search_paths(monkeypatch):
    """Patch the GROMACS force-field search path; tests append directories to the list."""
    paths = []
    monkeypatch.setattr("mdfactory.setup.protein._get_gmx_search_paths", lambda: paths)
    return paths


def _make_ff_tarball(dirname):
    """Build an in-memory gzip tarball containing a minimal .ff directory."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        content = b"; forcefield\n"
        info = tarfile.TarInfo(name=f"{dirname}/forcefield.itp")
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload


class TestProteinSpecies:
    def test_creation_with_optional_fields(self, protein_pdb):
        spec = ProteinSpecies(
            resname="LYZ",
            pdb_path=protein_pdb,
            disulfide_bonds=[("CYS6", "CYS127"), ("CYS30", "CYS115")],
            protonation_states={"HIS15": "HIE", "GLU35": "GLH"},
            chains=["H", "L", "Y"],
        )
        assert spec.disulfide_bonds == [("CYS6", "CYS127"), ("CYS30", "CYS115")]
        assert spec.protonation_states == {"HIS15": "HIE", "GLU35": "GLH"}
        assert spec.chains == ["H", "L", "Y"]
        assert spec.charge is None

    def test_resname_validation(self, protein_pdb):
        with pytest.raises(ValueError, match="Residue name must be less"):
            ProteinSpecies(resname="TOOLONG", pdb_path=protein_pdb)

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"count": 7}, "Input should be 1"),
            ({"count": None}, "Input should be 1"),
            ({"fraction": 0.25}, "Input should be 1.0"),
            ({"fraction": None}, "Input should be 1.0"),
        ],
    )
    def test_rejects_multiple_copies_or_null(self, protein_pdb, overrides, match):
        with pytest.raises(ValueError, match=match):
            ProteinSpecies(resname="LYZ", pdb_path=protein_pdb, **overrides)


class TestProteinBoxComposition:
    def test_creation_with_custom_ionization(self, protein_pdb):
        comp = ProteinBoxComposition(
            protein=ProteinSpecies(resname="LYZ", pdb_path=protein_pdb),
            padding=12.0,
            ionization={"neutralize": True, "concentration": 0.10},
        )
        assert comp.padding == 12.0
        assert comp.ionization.concentration == 0.10


@pytest.mark.parametrize("config_cls", [Pdb2gmxConfig, CharmmConfig])
class TestForcefieldConfigs:
    """Constraints shared by the pdb2gmx and CHARMM parametrization configs."""

    def test_frozen(self, config_cls):
        with pytest.raises(Exception):
            config_cls().forcefield = "other"

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"forcefield": "amber99sb-ildn"}, "CHARMM"),
            ({"water_model": "tip4p"}, "3-site"),
            ({"forcefield": "charmm36m-ljpme"}, "LJ-PME"),
        ],
    )
    def test_rejects_unsupported_choices(self, config_cls, overrides, match):
        with pytest.raises(ValueError, match=match):
            config_cls(**overrides)


class TestForceFieldCheck:
    def test_available_forcefield_passes(self, tmp_path, gmx_search_paths):
        (tmp_path / "charmm27.ff").mkdir()
        gmx_search_paths.append(tmp_path)
        check_forcefield_available("charmm27")

    def test_unavailable_forcefield_raises_and_lists_available(self, tmp_path, gmx_search_paths):
        (tmp_path / "charmm27.ff").mkdir()
        gmx_search_paths.append(tmp_path)
        with pytest.raises(ValueError, match="not found") as exc_info:
            check_forcefield_available("nonexistent_ff_xyz")
        assert "charmm27" in str(exc_info.value)
        assert "Available force fields" in str(exc_info.value)

    def test_registered_forcefield_check_does_not_download(
        self, tmp_path, monkeypatch, gmx_search_paths
    ):
        gmx_search_paths.append(tmp_path)
        monkeypatch.setattr(
            "mdfactory.setup.protein.download_forcefield",
            lambda name: pytest.fail("check_forcefield_available should not download"),
        )
        with pytest.raises(ValueError, match="Downloadable"):
            check_forcefield_available("charmm36m")


class TestForceFieldDownload:
    def _install(self, monkeypatch, tmp_path, payload, sha256):
        """Register a stub force field served from an in-memory tarball."""
        monkeypatch.setattr(
            "mdfactory.setup.protein.urlopen",
            lambda url, timeout=0: _FakeResponse(payload),
        )
        monkeypatch.setattr(
            "mdfactory.setup.protein.settings", SimpleNamespace(gromacs_forcefield_dir=tmp_path)
        )
        monkeypatch.setitem(
            protein.FORCEFIELD_REGISTRY,
            "test-ff",
            {
                "url": "https://example.invalid/test.ff.tgz",
                "dirname": "test.ff",
                "sha256": sha256,
            },
        )

    def test_verifies_matching_sha256_and_extracts(self, tmp_path, monkeypatch):
        payload = _make_ff_tarball("test.ff")
        self._install(monkeypatch, tmp_path, payload, hashlib.sha256(payload).hexdigest())
        result = download_forcefield("test-ff")
        assert result == tmp_path / "test.ff"
        assert (result / "forcefield.itp").is_file()

    def test_rejects_mismatched_sha256(self, tmp_path, monkeypatch):
        payload = _make_ff_tarball("test.ff")
        self._install(monkeypatch, tmp_path, payload, "0" * 64)
        with pytest.raises(RuntimeError, match="failed integrity check"):
            download_forcefield("test-ff")
        assert not (tmp_path / "test.ff").exists()


@pytest.mark.parametrize("sim_type", PROTEIN_SIM_TYPES)
class TestProteinInput:
    """BuildInput validation shared by proteinbox and protein_mixedbox."""

    def test_default_parametrization_config(self, sim_type, protein_pdb):
        inp = BuildInput(**protein_input_data(sim_type, protein_pdb))
        assert isinstance(inp.system, COMPOSITION_CLASSES[sim_type])
        assert isinstance(inp.parametrization_config, CONFIG_CLASSES[sim_type])

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"parametrization": "cgenff"}, "not valid for simulation type"),
            ({"config": {"type": "cgenff"}}, "does not match"),
            (
                {
                    "config": {"merge_all": True},
                    "protein": {"resname": "INS", "chains": ["A", "B"]},
                },
                "merge_all",
            ),
        ],
    )
    def test_rejects_invalid_combinations(self, sim_type, protein_pdb, overrides, match):
        with pytest.raises(ValueError, match=match):
            BuildInput(**protein_input_data(sim_type, protein_pdb, **overrides))

    def test_merge_all_without_declared_chains_allowed(self, sim_type, protein_pdb):
        data = protein_input_data(
            sim_type, protein_pdb, config={"merge_all": True}, protein={"resname": "INS"}
        )
        inp = BuildInput(**data)
        assert inp.parametrization_config.merge_all is True
        assert inp.system.protein.chains == []


class TestProteinboxInput:
    def test_yaml_with_explicit_config_and_protein_extras(self, protein_pdb):
        yaml_str = f"""
simulation_type: proteinbox
engine: gromacs
parametrization: pdb2gmx
parametrization_config:
  type: pdb2gmx
  forcefield: charmm36m
  water_model: tip3p
  ignore_hydrogens: true
system:
  protein:
    resname: LYZ
    pdb_path: {protein_pdb}
    disulfide_bonds:
      - [CYS6, CYS127]
      - [CYS30, CYS115]
    protonation_states:
      HIS15: HIE
  padding: 12.0
  ionization:
    neutralize: true
    concentration: 0.15
"""
        inp = BuildInput(**yaml.safe_load(yaml_str))
        assert isinstance(inp.system, ProteinBoxComposition)
        assert isinstance(inp.parametrization_config, Pdb2gmxConfig)
        assert inp.system.protein.disulfide_bonds == [("CYS6", "CYS127"), ("CYS30", "CYS115")]
        assert inp.system.protein.protonation_states == {"HIS15": "HIE"}

    def test_metadata(self, protein_pdb):
        meta = BuildInput(**protein_input_data("proteinbox", protein_pdb)).metadata
        assert meta["simulation_type"] == "proteinbox"
        assert meta["total_count"] == 1
        assert meta["system_specific"]["padding"] == 12.0


class TestTopologyParsing:
    def test_sum_charges_from_itp(self, tmp_path):
        itp = tmp_path / "protein.itp"
        itp.write_text(
            textwrap.dedent("""\
            [ moleculetype ]
            Protein   3

            [ atoms ]
            ;   nr  type  resnr residue  atom   cgnr   charge     mass
                 1    NH3    1    LYS      N      1    -0.300   14.007
                 2     CT    1    LYS     CA      2     0.210   12.011
                 3      C    1    LYS      C      3     0.510   12.011
                 4      O    1    LYS      O      4    -0.510   15.999

            [ bonds ]
            1 2
        """)
        )
        charge = _sum_atoms_column(itp, 6)
        assert abs(charge - (-0.090)) < 1e-6

    def test_update_topology_molecules(self, tmp_path):
        top = tmp_path / "topol.top"
        top.write_text(
            textwrap.dedent("""\
            #include "charmm36m.ff/forcefield.itp"
            #include "protein.itp"

            [ system ]
            Protein in water

            [ molecules ]
            Protein_chain_A  1
        """)
        )
        update_topology_molecules(top, n_water=5000, num_na=10, num_cl=18)
        content = top.read_text()
        assert "SOL" in content
        assert "5000" in content
        assert "NA" in content
        assert "10" in content
        assert "CL" in content
        assert "18" in content

    @pytest.mark.parametrize(
        ("atoms", "states", "forcefield", "expected"),
        [
            # Plain renames; untouched residues keep their name.
            (
                [("HIS", 15), ("HIS", 15), ("ALA", 16), ("GLU", 35)],
                {"HIS15": "HIE", "GLU35": "GLH"},
                None,
                ["HIE", "HIE", "ALA", "GLH"],
            ),
            # CHARMM histidine aliases.
            ([("HIS", 15)], {"HIS15": "HIE"}, "charmm36m", ["HSE"]),
            # An existing tautomer (HID) still matches an unqualified HIS key.
            ([("HID", 15)], {"HIS15": "HIE"}, "charmm36m", ["HSE"]),
            # A chain-qualified key renames only the residue in that chain.
            ([("HIS", 15, "A"), ("HIS", 15, "B")], {"A:HIS15": "HIE"}, None, ["HIE", "HIS"]),
            # Insertion codes distinguish otherwise identical residue numbers.
            (
                [("HIS", 100, "H"), ("HIS", 100, "H", 0.0, "A")],
                {"H:HIS100A": "HIE"},
                None,
                ["HIS", "HIE"],
            ),
            (
                [("HIS", 100, "H"), ("HIS", 100, "H", 0.0, "A")],
                {"H:HIS100": "HID"},
                None,
                ["HID", "HIS"],
            ),
        ],
    )
    def test_apply_protonation_states(self, tmp_path, atoms, states, forcefield, expected):
        pdb = _write_pdb(tmp_path, *atoms)
        kwargs = {"forcefield": forcefield} if forcefield else {}
        result = _apply_protonation_states(pdb, states, tmp_path, **kwargs)
        lines = [ln for ln in result.read_text().splitlines() if ln.startswith("ATOM")]
        assert [ln[17:20].strip() for ln in lines] == expected

    @pytest.mark.parametrize(
        ("atoms", "states", "forcefield", "match"),
        [
            ([("GLU", 35)], {"GLU35": "GLUP"}, None, "cannot be written to a PDB"),
            # GLH aliases to CHARMM GLUP (4 chars), which cannot fit a PDB column.
            ([("GLU", 35)], {"GLU35": "GLH"}, "charmm36m", "cannot be written to a PDB"),
            ([("HIS", 15)], {"HIS99": "HIE"}, None, "matched no residue"),
            # HIS15 exists in both chains; an unqualified key cannot say which.
            ([("HIS", 15, "A"), ("HIS", 15, "B")], {"HIS15": "HIE"}, None, "ambiguous across"),
        ],
    )
    def test_apply_protonation_states_rejects(self, tmp_path, atoms, states, forcefield, match):
        pdb = _write_pdb(tmp_path, *atoms)
        kwargs = {"forcefield": forcefield} if forcefield else {}
        with pytest.raises(ValueError, match=match):
            _apply_protonation_states(pdb, states, tmp_path, **kwargs)

    @pytest.mark.parametrize(
        ("atoms", "bonds", "expected"),
        [
            # Answers follow the order pdb2gmx lists close SG pairs, not residue numbers.
            (
                [
                    ("CYS", 1, "A", 0.0),
                    ("CYS", 4, "A", 2.0),
                    ("CYS", 2, "A", 10.0),
                    ("CYS", 3, "A", 12.0),
                ],
                [("CYS2", "CYS3")],
                "n\ny\n",
            ),
            # Inter-chain disulfide (e.g. insulin A7-B7) via chain-qualified references.
            (
                [("CYS", 7, "A", 0.0), ("CYS", 7, "B", 2.0), ("CYS", 20, "A", 20.0)],
                [("A:CYS7", "B:CYS7")],
                "y\n",
            ),
            # PDB chain order, not alphabetical, determines the prompt order.
            (
                [
                    ("CYS", 6, "B", 0.0),
                    ("CYS", 127, "B", 2.0),
                    ("CYS", 6, "A", 10.0),
                    ("CYS", 127, "A", 12.0),
                ],
                [("B:CYS6", "B:CYS127")],
                "y\nn\n",
            ),
            (
                [("CYS", 100, "H", 0.0), ("CYS", 100, "H", 2.0, "A")],
                [("H:CYS100A", "H:CYS100")],
                "y\n",
            ),
        ],
    )
    def test_disulfide_prompt_input(self, tmp_path, atoms, bonds, expected):
        pdb = _write_pdb(tmp_path, *atoms, atom_name="SG")
        assert _build_disulfide_prompt_input(pdb, bonds) == expected

    @pytest.mark.parametrize(
        ("atoms", "bonds", "match"),
        [
            (
                [("CYS", 6, "A", 0.0), ("CYS", 127, "A", 8.0)],
                [("CYS6", "CYS127")],
                "not detected as close CYS SG pairs",
            ),
            # CYS7 exists in both chains; an unqualified reference is ambiguous.
            (
                [("CYS", 7, "A", 0.0), ("CYS", 7, "B", 2.0)],
                [("CYS7", "CYS7")],
                "ambiguous across chains",
            ),
        ],
    )
    def test_disulfide_prompt_input_rejects(self, tmp_path, atoms, bonds, match):
        pdb = _write_pdb(tmp_path, *atoms, atom_name="SG")
        with pytest.raises(ValueError, match=match):
            _build_disulfide_prompt_input(pdb, bonds)

    def test_run_pdb2gmx_passes_disulfide_answers_to_subprocess(self, tmp_path, monkeypatch):
        pdb = _write_pdb(
            tmp_path,
            ("CYS", 1, "A", 0.0),
            ("CYS", 4, "A", 2.0),
            ("CYS", 2, "A", 10.0),
            ("CYS", 3, "A", 12.0),
            atom_name="SG",
        )
        calls = {}
        _patch_pdb2gmx(monkeypatch, ["Protein_chain_A"], ["posre.itp"], calls)

        _run(tmp_path, pdb, disulfide_bonds=[("CYS2", "CYS3")])

        assert "-ss" in calls["cmd"]
        assert calls["cmd"][:2] == ["/mock/gmx", "pdb2gmx"]
        assert calls["input"] == "n\ny\n"
        assert calls["cwd"] == str((tmp_path / "out").resolve())
        assert calls["env"] == {"GMXLIB": "/mock/forcefields"}
        assert calls["check"] is False

    def test_run_pdb2gmx_multichain_collects_chain_includes(self, tmp_path, monkeypatch):
        pdb = _write_pdb(tmp_path, ("ALA", 1))
        chains = ["H", "L", "Y"]
        extra = [f"topol_Protein_chain_{c}.itp" for c in chains] + [
            f"posre_Protein_chain_{c}.itp" for c in chains
        ]
        _patch_pdb2gmx(monkeypatch, [f"Protein_chain_{c}" for c in chains], extra)

        params = _run(tmp_path, pdb, chains=chains)

        assert params.chains == chains
        assert sorted(p.name for p in params.topology_include_files) == sorted(extra)

    @pytest.mark.parametrize(
        ("produced", "declared", "match"),
        [
            (["H", "L"], None, "none were declared"),
            (["H", "L", "Y"], ["H", "L"], "do not match"),
        ],
    )
    def test_run_pdb2gmx_chain_declaration_mismatch_raises(
        self, tmp_path, monkeypatch, produced, declared, match
    ):
        pdb = _write_pdb(tmp_path, ("ALA", 1))
        _patch_pdb2gmx(monkeypatch, [f"Protein_chain_{c}" for c in produced])
        with pytest.raises(ValueError, match=match):
            _run(tmp_path, pdb, chains=declared)


class TestBundleForcefield:
    def test_copies_ff_dir_once_and_rewrites_absolute_includes(self, tmp_path, monkeypatch):
        ff_source = tmp_path / "ffsource" / "charmm36-feb2026_cgenff-5.0.ff"
        ff_source.mkdir(parents=True)
        (ff_source / "forcefield.itp").write_text("; forcefield\n")
        (ff_source / "tip3p.itp").write_text("; water\n")

        build_dir = tmp_path / "build"
        build_dir.mkdir()
        top = build_dir / "topology.top"
        top.write_text(
            textwrap.dedent(f"""\
            #include "{ff_source / "forcefield.itp"}"
            #include "protein.itp"
            #include "{ff_source / "tip3p.itp"}"
            #include "posre.itp"
            """)
        )
        copies = []
        real_copytree = protein.shutil.copytree

        def counting_copytree(src, dst, *args, **kwargs):
            copies.append(src)
            return real_copytree(src, dst, *args, **kwargs)

        monkeypatch.setattr(protein.shutil, "copytree", counting_copytree)

        ff_name = bundle_forcefield_into_topology(top)

        assert ff_name == "charmm36-feb2026_cgenff-5.0.ff"
        assert (build_dir / ff_name / "forcefield.itp").is_file()
        assert (build_dir / ff_name / "tip3p.itp").is_file()
        # Two includes into the same .ff directory, copied exactly once.
        assert len(copies) == 1

        content = top.read_text()
        assert '#include "charmm36-feb2026_cgenff-5.0.ff/forcefield.itp"' in content
        assert '#include "charmm36-feb2026_cgenff-5.0.ff/tip3p.itp"' in content
        # Relative includes are left untouched.
        assert '#include "protein.itp"' in content
        assert '#include "posre.itp"' in content
        # No absolute paths remain.
        assert str(ff_source) not in content

    def test_no_absolute_ff_include_returns_none(self, tmp_path):
        top = tmp_path / "topology.top"
        top.write_text('#include "protein.itp"\n#include "posre.itp"\n')
        original = top.read_text()

        assert bundle_forcefield_into_topology(top) is None
        assert top.read_text() == original

    def test_bundles_relative_ff_include(self, tmp_path, gmx_search_paths):
        # Built-in force fields emit relative includes like charmm27.ff/forcefield.itp;
        # the directory must be resolved from a search path and copied next to the topology.
        ff_source = tmp_path / "gmxtop" / "charmm27.ff"
        ff_source.mkdir(parents=True)
        (ff_source / "forcefield.itp").write_text("; forcefield\n")
        gmx_search_paths.append(tmp_path / "gmxtop")

        build_dir = tmp_path / "build"
        build_dir.mkdir()
        top = build_dir / "topology.top"
        top.write_text('#include "charmm27.ff/forcefield.itp"\n#include "posre.itp"\n')

        ff_name = bundle_forcefield_into_topology(top)

        assert ff_name == "charmm27.ff"
        assert (build_dir / "charmm27.ff" / "forcefield.itp").is_file()
        # A relative include already resolves from the topology directory once the
        # force field sits alongside it, so the line is left untouched.
        assert '#include "charmm27.ff/forcefield.itp"' in top.read_text()

    def test_relative_ff_include_not_found_raises(self, tmp_path, gmx_search_paths):
        gmx_search_paths.append(tmp_path / "empty")
        top = tmp_path / "topology.top"
        top.write_text('#include "charmm27.ff/forcefield.itp"\n')
        with pytest.raises(RuntimeError, match="not found in any GROMACS search path"):
            bundle_forcefield_into_topology(top)


class TestCleanPdb:
    def test_preserves_chain_and_residue_numbers(self, tmp_path):
        # A residue in chain H numbered 15: cleaning must not renumber it to A/1,
        # or later chain/disulfide/protonation references stop resolving.
        pdb = tmp_path / "in.pdb"
        pdb.write_text(
            "ATOM      1  N   ALA H  15A      0.000   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA H  15A      1.458   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  C   ALA H  15A      2.009   1.420   0.000  1.00  0.00           C\n"
            "ATOM      4  O   ALA H  15A      1.251   2.390   0.000  1.00  0.00           O\n"
            "ATOM      5  CB  ALA H  15A      1.988  -0.773   1.199  1.00  0.00           C\n"
            "TER\n"
            "END\n"
        )
        out = clean_pdb(pdb, tmp_path / "clean.pdb")
        atom_lines = [ln for ln in out.read_text().splitlines() if ln.startswith("ATOM")]
        assert atom_lines
        assert all(ln[21] == "H" for ln in atom_lines)
        assert all(ln[22:26].strip() == "15" for ln in atom_lines)
        assert all(ln[26] == "A" for ln in atom_lines)

    def test_models_internal_gaps_but_not_terminal_tails(self, tmp_path):
        # SEQRES lists 9 residues; only ALA 3-5 and ALA 7 have coordinates.
        # The internal gap (GLY 6) is modeled, the unresolved tails
        # (GLY 1-2, SER 8-9) are not, and every segment is reported.
        from loguru import logger

        seqres = "GLY GLY ALA ALA ALA GLY ALA SER SER".split()
        lines = [f"SEQRES   1 A    9  {' '.join(seqres)}"]
        serial = 1
        for resid in (3, 4, 5, 7):
            x0 = 3.8 * (resid - 1)
            for name, dx, dy, element in (
                ("N", -1.2, 0.3, "N"),
                ("CA", 0.0, 0.0, "C"),
                ("C", 1.2, 0.3, "C"),
                ("O", 1.4, 1.5, "O"),
            ):
                lines.append(
                    f"ATOM  {serial:5d}  {name:<3s} ALA A{resid:4d}    "
                    f"{x0 + dx:8.3f}{dy:8.3f}{0.0:8.3f}  1.00  0.00          {element:>2s}"
                )
                serial += 1
        lines += ["TER", "END"]
        pdb = tmp_path / "gapped.pdb"
        pdb.write_text("\n".join(lines) + "\n")

        messages = []
        handler_id = logger.add(
            lambda m: messages.append(str(m)), level="WARNING", format="{message}"
        )
        try:
            out = clean_pdb(pdb, tmp_path / "clean.pdb")
        finally:
            logger.remove(handler_id)

        u = mda.Universe(str(out))
        assert list(u.residues.resids) == [3, 4, 5, 6, 7]
        assert list(u.residues.resnames) == ["ALA", "ALA", "ALA", "GLY", "ALA"]
        text = "\n".join(messages)
        assert "N-terminal" in text and "GLY-GLY" in text
        assert "C-terminal" in text and "SER-SER" in text
        assert "internal" in text


@pytest.mark.parametrize("sim_type", PROTEIN_SIM_TYPES)
class TestPdbPathResolution:
    def test_relative_pdb_path_resolved_against_yaml_dir(self, tmp_path, sim_type):
        base = tmp_path / "specs"
        base.mkdir()
        dct = {"simulation_type": sim_type, "system": {"protein": {"pdb_path": "1aki.pdb"}}}
        resolve_protein_pdb_path(dct, base)
        assert dct["system"]["protein"]["pdb_path"] == str((base / "1aki.pdb").resolve())

    def test_absolute_pdb_path_unchanged(self, tmp_path, sim_type):
        abs_path = str((tmp_path / "abs.pdb").resolve())
        dct = {"simulation_type": sim_type, "system": {"protein": {"pdb_path": abs_path}}}
        resolve_protein_pdb_path(dct, tmp_path / "other")
        assert dct["system"]["protein"]["pdb_path"] == abs_path


def test_non_protein_simulation_type_untouched(tmp_path):
    dct = {"simulation_type": "mixedbox", "system": {"foo": "bar"}}
    resolve_protein_pdb_path(dct, tmp_path)
    assert dct == {"simulation_type": "mixedbox", "system": {"foo": "bar"}}


def _write_spec_dir(tmp_path, protein_pdb, sim_type="proteinbox"):
    """Create a spec dir with a PDB and YAML using a relative pdb_path, plus an unrelated cwd."""
    spec_dir = tmp_path / "specs"
    spec_dir.mkdir()
    pdb = spec_dir / "prot.pdb"
    shutil.copy(protein_pdb, pdb)
    yaml_file = spec_dir / "spec.yaml"
    yaml_file.write_text(yaml.safe_dump(protein_input_data(sim_type, "prot.pdb")))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    return spec_dir, pdb, yaml_file, elsewhere


class TestRelativePdbPathAcrossEntryPoints:
    """Every entry point must resolve a relative pdb_path the same way (same hash)."""

    def test_csv_push_and_prepare_compute_identical_hashes(
        self, tmp_path, protein_pdb, monkeypatch
    ):
        from mdfactory.prepare import load_build_input_csv
        from mdfactory.utils.push import load_models_from_csv

        spec_dir, pdb, _, elsewhere = _write_spec_dir(tmp_path, protein_pdb)
        csv_path = spec_dir / "proteins.csv"
        csv_path.write_text(
            "simulation_type,parametrization,system.protein.resname,"
            "system.protein.pdb_path,system.padding\n"
            "proteinbox,pdb2gmx,LYZ,prot.pdb,12.0\n"
        )
        # Run from an unrelated directory so cwd-relative resolution would differ.
        monkeypatch.chdir(elsewhere)

        _, prepared, prep_errors = load_build_input_csv(csv_path)
        pushed, push_errors = load_models_from_csv(csv_path)

        assert prep_errors == {} and push_errors == {}
        assert prepared[0].system.protein.pdb_path == pdb.resolve()
        assert [m.hash for m in pushed] == [m.hash for m in prepared]

    @pytest.mark.parametrize("sim_type", PROTEIN_SIM_TYPES)
    def test_run_from_file_resolves_relative_path(
        self, tmp_path, protein_pdb, monkeypatch, sim_type
    ):
        _, pdb, yaml_file, _ = _write_spec_dir(tmp_path, protein_pdb, sim_type)
        captured = {}
        monkeypatch.setattr(workflows, "run_build_from_dict", captured.update)
        workflows.run_build_from_file(yaml_file)
        assert captured["system"]["protein"]["pdb_path"] == str(pdb.resolve())

    def test_yaml_dry_run_resolves_before_dispatch(self, tmp_path, protein_pdb, monkeypatch):
        import mdfactory.orchestration
        from mdfactory.cli import _build_from_yaml

        _, pdb, yaml_file, elsewhere = _write_spec_dir(tmp_path, protein_pdb)
        monkeypatch.chdir(elsewhere)

        captured = []
        monkeypatch.setattr(
            mdfactory.orchestration,
            "build_systems",
            lambda models, *args, **kwargs: captured.extend(models) or [],
        )
        _build_from_yaml(yaml_file, tmp_path / "out", dry_run=True)

        assert len(captured) == 1
        assert captured[0].system.protein.pdb_path == pdb.resolve()

    def test_yaml_sequential_run_dir_matches_model_hash(self, tmp_path, protein_pdb, monkeypatch):
        from mdfactory import cli

        _, _, yaml_file, elsewhere = _write_spec_dir(tmp_path, protein_pdb)
        monkeypatch.chdir(elsewhere)

        built = {}

        def fake_run_build_from_dict(dct):
            built["hash"] = BuildInput(**dct).hash
            built["cwd"] = Path.cwd()

        monkeypatch.setattr(workflows, "run_build_from_dict", fake_run_build_from_dict)
        out = tmp_path / "out"
        cli._build_from_yaml(yaml_file, out)

        assert built["cwd"] == (out / built["hash"]).resolve()


class TestCenterInCubicBox:
    def test_padding_guaranteed_on_every_face(self):
        # Asymmetric molecule: long in x, short in y/z, offset from the origin.
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [40.0, 5.0, 2.0],
                [10.0, -3.0, 8.0],
            ],
            dtype=float,
        )
        u = mda.Universe.empty(len(positions), trajectory=True)
        u.atoms.positions = positions
        padding = 12.0
        box_size = _center_in_cubic_box(u, padding)

        pos = u.atoms.positions
        bbox_min = pos.min(axis=0)
        bbox_max = pos.max(axis=0)
        # Every face is at least `padding` from the protein.
        assert bbox_min.min() >= padding - 1e-3
        assert (box_size - bbox_max.max()) >= padding - 1e-3
        # The longest-extent axis sits exactly `padding` from both faces.
        long_axis = int(np.argmax(bbox_max - bbox_min))
        assert abs(bbox_min[long_axis] - padding) < 1e-3
        assert abs((box_size - bbox_max[long_axis]) - padding) < 1e-3


class TestIonCounts:
    """ionize_solvated_system returns only the ions it added; counts map by resname."""

    @pytest.mark.parametrize(
        ("ions", "expected"),
        [
            ([("[Na+]", "NA", 7), ("[Cl-]", "CL", 9)], (7, 9)),
            # Positive protein with no salt: only Cl- is added, not reported as sodium.
            ([("[Cl-]", "CL", 3)], (0, 3)),
            ([("[Na+]", "NA", 4)], (4, 0)),
            ([], (0, 0)),
        ],
    )
    def test_ion_counts(self, ions, expected):
        species = [
            SingleMoleculeSpecies(smiles=smiles, resname=resname, count=count)
            for smiles, resname, count in ions
        ]
        assert _ion_counts(species) == expected
