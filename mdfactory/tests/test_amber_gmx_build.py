# ABOUTME: Opt-in real-GROMACS tests: Amber ff14SB + Sage protein topology and generated mdps
# ABOUTME: Runs pdb2gmx amber14sb and grompp on every generated stage within its maxwarn
"""Opt-in real-GROMACS tests of the Amber ff14SB + OpenFF Sage route and the run schedules.

Skipped unless a GROMACS >= 2026 binary is available (configured or on PATH).
The module name contains ``build``, so ``pytest -k "not build"`` (CI) skips it.
"""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import MDAnalysis as mda
import numpy as np
import pytest

from mdfactory.build import _stage_protein_topology
from mdfactory.models.parametrization import (
    AmberParametrization,
    CharmmParametrization,
    OpenFFParametrization,
)
from mdfactory.models.simulation_settings import SimulationSettings
from mdfactory.models.species import (
    PositionRestraint,
    ProteinSpecies,
    SingleMoleculeSpecies,
    SolutionSpecies,
)
from mdfactory.orchestration.stages import STAGE_REGISTRY
from mdfactory.parametrize import NativeForceField, resolve_parametrizer
from mdfactory.run_schedules import write_mdp_files
from mdfactory.settings import settings
from mdfactory.setup import protein
from mdfactory.setup.restraints import apply_restraints
from mdfactory.utils.setup_utilities import topology_atom_names

# Lys1-Val2-Phe3 heavy atoms from PDB 1AKI (hen egg-white lysozyme).
PEPTIDE_PDB = """\
ATOM      1  N   LYS A   1      35.365  22.342 -11.980  1.00 22.28           N
ATOM      2  CA  LYS A   1      35.892  21.073 -11.427  1.00 21.12           C
ATOM      3  C   LYS A   1      34.741  20.264 -10.844  1.00 16.85           C
ATOM      4  O   LYS A   1      33.945  20.813 -10.081  1.00 18.94           O
ATOM      5  CB  LYS A   1      36.872  21.435 -10.306  1.00 20.78           C
ATOM      6  CG  LYS A   1      37.453  20.248  -9.565  1.00 18.47           C
ATOM      7  CD  LYS A   1      38.688  20.649  -8.775  1.00 20.32           C
ATOM      8  CE  LYS A   1      39.057  19.508  -7.837  1.00 24.76           C
ATOM      9  NZ  LYS A   1      40.423  19.771  -7.299  1.00 28.27           N
ATOM     10  N   VAL A   2      34.739  18.961 -11.042  1.00 19.96           N
ATOM     11  CA  VAL A   2      33.903  17.998 -10.333  1.00 18.10           C
ATOM     12  C   VAL A   2      34.800  17.312  -9.294  1.00 19.39           C
ATOM     13  O   VAL A   2      35.759  16.605  -9.665  1.00 22.14           O
ATOM     14  CB  VAL A   2      33.140  17.034 -11.232  1.00 16.81           C
ATOM     15  CG1 VAL A   2      32.251  16.084 -10.434  1.00 21.92           C
ATOM     16  CG2 VAL A   2      32.294  17.714 -12.290  1.00 19.46           C
ATOM     17  N   PHE A   3      34.491  17.546  -8.038  1.00 19.89           N
ATOM     18  CA  PHE A   3      35.185  16.903  -6.918  1.00 17.43           C
ATOM     19  C   PHE A   3      34.742  15.441  -6.771  1.00 15.70           C
ATOM     20  O   PHE A   3      33.525  15.162  -6.862  1.00 18.52           O
ATOM     21  CB  PHE A   3      34.967  17.632  -5.594  1.00 17.94           C
ATOM     22  CG  PHE A   3      35.944  18.737  -5.375  1.00 16.78           C
ATOM     23  CD1 PHE A   3      35.666  20.050  -5.798  1.00 15.97           C
ATOM     24  CD2 PHE A   3      37.000  18.557  -4.473  1.00 19.95           C
ATOM     25  CE1 PHE A   3      36.577  21.076  -5.568  1.00 17.32           C
ATOM     26  CE2 PHE A   3      37.869  19.589  -4.157  1.00 17.65           C
ATOM     27  CZ  PHE A   3      37.636  20.873  -4.666  1.00 17.91           C
END
"""

BOX_A = 40.0


def _new_enough_gmx():
    try:
        gmx = protein.check_gmx_available()
        version, _ = protein.get_gmx_version(gmx)
    except RuntimeError:
        return None
    return gmx if version >= (2026, 0) else None


GMX = _new_enough_gmx()
pytestmark = pytest.mark.skipif(GMX is None, reason="needs GROMACS >= 2026")


def _grompp_all_stages(build_dir: Path, structure: Path, topology: Path) -> None:
    """Run grompp on every generated stage within the stage's ``maxwarn``.

    As in the orchestration, every stage gets ``-r`` equal to ``-c`` and, when
    the build wrote one, ``-n index.ndx``; the stages after EM start from the
    build structure, which is enough for grompp's checks.
    """
    index = ["-n", "index.ndx"] if (build_dir / "index.ndx").is_file() else []
    for spec in STAGE_REGISTRY:
        cmd = [str(GMX), "grompp", "-f", spec.mdp_file, "-c", str(structure), "-r", str(structure)]
        cmd += ["-p", str(topology), "-o", f"{spec.deffnm}.tpr", "-po", f"{spec.deffnm}_out.mdp"]
        cmd += index
        cmd += ["-maxwarn", str(spec.maxwarn)]
        result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True, check=False)
        assert result.returncode == 0, f"{spec.name}:\n{result.stdout}\n{result.stderr}"


def _solution_universe(species, names, offsets):
    """Place each species' RDKit conformer at an offset; use the topology atom names."""
    positions, atom_names, resindex, resnames = [], [], [], []
    for i, (spec, spec_names, offset) in enumerate(zip(species, names, offsets)):
        xyz = spec.rdkit_molecule.GetConformer().GetPositions()
        positions.append(xyz - xyz.mean(axis=0) + offset)
        atom_names += spec_names
        resindex += [i] * len(spec_names)
        resnames.append(spec.resname)
    u = mda.Universe.empty(
        len(atom_names), n_residues=len(species), atom_resindex=resindex, trajectory=True
    )
    u.add_TopologyAttr("names", atom_names)
    u.add_TopologyAttr("resnames", resnames)
    u.add_TopologyAttr("resids", np.arange(1, len(species) + 1))
    u.atoms.positions = np.vstack(positions)
    return u


def test_amber_sage_composite_topology_passes_grompp_build(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "parameter_store", tmp_path / "parameters")
    pdb = tmp_path / "peptide.pdb"
    pdb.write_text(PEPTIDE_PDB)
    spec = AmberParametrization().resolve("protein_mixedbox")

    params = protein.run_pdb2gmx(
        pdb_path=pdb,
        forcefield=spec.protein,
        water_model=spec.water_model,
        ignore_hydrogens=True,
        merge_all=False,
        disulfide_bonds=[],
        protonation_states={},
        output_dir=tmp_path / "pdb2gmx",
        chains=[],
    )
    restraints = [
        PositionRestraint(name="backbone", selection="name CA", k=500.0),
        PositionRestraint(
            name="tail",
            selection="resid 3 and not name H*",
            type="flat_bottom",
            radius=0.2,
            k=800.0,
        ),
    ]
    params, _ = apply_restraints(params, restraints, pdb)
    ff_name = protein.bundle_forcefield_into_topology(params.topology_file)
    assert ff_name == "amber14sb.ff"
    ff_dir = params.topology_file.parent / ff_name

    native = NativeForceField(name=spec.protein, directory=ff_dir, water_model=spec.water_model)
    parametrize = resolve_parametrizer(spec, "gromacs", native)
    species = [
        SolutionSpecies(smiles="CCO", resname="ETH", count=1),
        SolutionSpecies(smiles="O", resname="SOL", count=1),
        SingleMoleculeSpecies(smiles="[Cl-]", resname="CL", count=1),
    ]
    parameters = [parametrize(spec) for spec in species]
    assert parameters[0].parameter_itp is not None
    assert parameters[1].itp == ff_dir / "tip3p.itp"

    build_dir = tmp_path / "system"
    build_dir.mkdir()
    topology = _stage_protein_topology(params, ff_dir, build_dir, species, parameters, [1, 1, 1])

    u_protein = mda.Universe(str(params.structure_file))
    u_protein.atoms.translate(np.full(3, BOX_A / 2) - u_protein.atoms.center_of_geometry())
    solution = _solution_universe(
        species,
        [topology_atom_names(spec, par) for spec, par in zip(species, parameters)],
        [np.array([5.0, 5.0, 5.0]), np.array([35.0, 35.0, 35.0]), np.array([5.0, 35.0, 5.0])],
    )
    u = mda.Merge(u_protein.atoms, solution.atoms)
    u.dimensions = [BOX_A, BOX_A, BOX_A, 90, 90, 90]
    structure = build_dir / "system.gro"
    u.atoms.write(str(structure))
    # OpenMM's PDB reader drops atoms whose name repeats within a residue, so the
    # Sage solute must carry unique names for the OpenMM compression/relaxation.
    from openmm import app

    pdb_out = build_dir / "system.pdb"
    u.atoms.write(str(pdb_out))
    assert app.PDBFile(str(pdb_out)).topology.getNumAtoms() == u.atoms.n_atoms

    inp = SimpleNamespace(
        simulation_type="protein_mixedbox",
        parametrization=spec,
        simulation_settings=SimulationSettings().resolve("protein_mixedbox"),
        system=SimpleNamespace(
            protein=ProteinSpecies(resname="PEP", pdb_path=pdb, restraints=restraints)
        ),
    )
    write_mdp_files(inp, build_dir)
    for itp in ("posre_backbone_Protein_chain_A.itp", "posre_tail_Protein_chain_A.itp"):
        assert (build_dir / itp).is_file()

    text = topology.read_text()
    for include in ("forcefield.itp", "tip3p.itp", "ions_tip3p.itp"):
        assert f'#include "amber14sb.ff/{include}"' in text
    assert f'#include "{parameters[0].itp.name}"' in text
    extra = (build_dir / "extra_params.itp").read_text()
    assert "[ defaults ]" not in extra
    # OpenFF atom types carry the parameter set's prefix, not the residue name.
    provenance = json.loads((parameters[0].itp.parent / "provenance.json").read_text())
    assert "ETH_" not in extra and f"{provenance['atomtype_prefix']}_0 " in extra

    # One Cl- neutralizes the peptide (+1), so the build-time check runs at -maxwarn 0.
    protein.validate_with_grompp(topology, structure, build_dir / "em.mdp", build_dir)
    # md.mdp switches the selection restraints on and needs the -r reference.
    protein.validate_with_grompp(
        topology, structure, build_dir / "md.mdp", build_dir, reference=structure
    )
    assert Path(build_dir / "topology.top").is_file()
    _grompp_all_stages(build_dir, structure, topology)


WATER_TOPOLOGY = """\
#include "amber14sb.ff/forcefield.itp"
#include "amber14sb.ff/tip3p.itp"

[ system ]
water

[ molecules ]
SOL {n_water}
"""

SCHEDULE_CASES = [
    ("mixedbox", CharmmParametrization(), {}),
    ("mixedbox", OpenFFParametrization(), {}),
    ("bilayer", CharmmParametrization(), {}),
    ("bilayer", OpenFFParametrization(), {}),
    ("lnp", CharmmParametrization(), {}),
    (
        "mixedbox",
        CharmmParametrization(),
        {
            "temperature": 310.0,
            "pressure": 1.5,
            "timestep": 1.0,
            "nvt": {"length": 20.0, "output_interval": 0.5},
            "production": {"length": 1.0, "output_interval": 5.0},
            "gromacs_overrides": {"npt": {"pcoupl": "C-rescale"}, "production": {"tau_p": 4.0}},
        },
    ),
]


@pytest.mark.parametrize(("simulation_type", "family", "settings_data"), SCHEDULE_CASES)
def test_generated_schedules_pass_grompp_build(tmp_path, simulation_type, family, settings_data):
    """Every stage of the solvent schedules passes grompp on a 3.2 nm water box."""
    structure = tmp_path / "water.gro"
    result = subprocess.run(
        [str(GMX), "solvate", "-cs", "spc216.gro", "-box", "3.2", "3.2", "3.2", "-o", "water.gro"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    n_water = mda.Universe(str(structure)).residues.n_residues
    # amber14sb.ff ships with GROMACS 2026; its TIP3P is enough to check the mdp parameters.
    topology = tmp_path / "topol.top"
    topology.write_text(WATER_TOPOLOGY.format(n_water=n_water))
    inp = SimpleNamespace(
        simulation_type=simulation_type,
        parametrization=family.resolve(simulation_type),
        simulation_settings=SimulationSettings(**settings_data).resolve(simulation_type),
    )
    write_mdp_files(inp, tmp_path)
    _grompp_all_stages(tmp_path, structure, topology)


EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "proteinbox"
needs_example_pdbs = pytest.mark.skipif(
    not (EXAMPLES / "1zni.pdb").is_file() or not (EXAMPLES / "1aki.pdb").is_file(),
    reason="needs examples/proteinbox/1aki.pdb and 1zni.pdb (examples/fetch_pdbs.sh)",
)


def _restrained_labels(itp: Path, moleculetype_file: Path) -> list[tuple[str, str]]:
    """Return the (residue, atom name) of each atom restrained by *itp*."""
    from mdfactory.utils.topology_utilities import iter_topology_entries

    def entries(path, section):
        return [
            entry.code.split()
            for entry in iter_topology_entries(path.read_text().splitlines())
            if entry.section == section
        ]

    atoms = entries(moleculetype_file, "atoms")
    return [tuple(atoms[int(row[0]) - 1][3:5]) for row in entries(itp, "position_restraints")]


@needs_example_pdbs
@pytest.mark.parametrize("merge_all", [False, True])
def test_chain_qualified_restraints_on_real_insulin_build(tmp_path, merge_all):
    """Insulin chains share resids; chain IDs come from the input via residue order."""
    from mdfactory.models.species import PositionRestraint
    from mdfactory.run_schedules.gromacs import format_mdp, render_stage

    cleaned = protein.clean_pdb(EXAMPLES / "1zni.pdb", tmp_path / "cleaned.pdb")
    params = protein.run_pdb2gmx(
        pdb_path=cleaned,
        forcefield="amber14sb",
        water_model="tip3p",
        ignore_hydrogens=True,
        merge_all=merge_all,
        disulfide_bonds=[],
        protonation_states={},
        output_dir=tmp_path / "pdb2gmx",
        chains=[] if merge_all else ["A", "B", "C", "D"],
    )
    restraints = [
        PositionRestraint(name="anchor", selection="chainID B and resid 1:3 and name CA", k=500.0),
        PositionRestraint(
            name="tail",
            selection="chainID D and resid 30 and not name H*",
            type="flat_bottom",
            radius=0.5,
            k=1000.0,
        ),
    ]
    params, _ = apply_restraints(params, restraints, cleaned)

    out = params.topology_file.parent
    chain_b, chain_d = (
        ("Protein_chain_A",) * 2 if merge_all else ("Protein_chain_B", "Protein_chain_D")
    )
    defined_in = {
        name: params.topology_file if merge_all else out / f"topol_{name}.itp"
        for name in (chain_b, chain_d)
    }
    # Chain B of 1ZNI starts Phe-Val-Asn; chain D ends with Ala30.
    assert _restrained_labels(out / f"posre_anchor_{chain_b}.itp", defined_in[chain_b]) == [
        ("PHE", "CA"),
        ("VAL", "CA"),
        ("ASN", "CA"),
    ]
    tail = _restrained_labels(out / f"posre_tail_{chain_d}.itp", defined_in[chain_d])
    assert {residue for residue, _ in tail} == {"ALA"}
    assert {"CA", "CB", "OXT"} <= {name for _, name in tail}

    settings = SimulationSettings(gromacs_overrides={"production": {"pcoupl": "C-rescale"}})
    md = render_stage(
        "production",
        simulation_type="mixedbox",
        nonbonded_scheme="sage",
        settings=settings.resolve("mixedbox"),
        restraint_macros=tuple(r.macro for r in restraints),
    )
    mdp = out / "md.mdp"
    mdp.write_text(format_mdp(md, ["restraint check"]))
    # Vacuum insulin has a net charge; grompp accepts only that warning.
    protein.validate_with_grompp(
        params.topology_file,
        params.structure_file,
        mdp,
        out,
        allow_net_charge=True,
        reference=params.structure_file,
    )


@needs_example_pdbs
def test_restrained_proteinbox_example_build(tmp_path, monkeypatch):
    """The restraint example builds (incl. the md.mdp grompp check) and every stage preprocesses."""
    import yaml

    from mdfactory.build import build_proteinbox
    from mdfactory.models.input import BuildInput
    from mdfactory.prepare import resolve_protein_pdb_path

    data = yaml.safe_load((EXAMPLES / "lysozyme_restraints.yaml").read_text())
    resolve_protein_pdb_path(data, EXAMPLES)
    data["system"]["relax_steps"] = 200
    inp = BuildInput(**data)
    monkeypatch.chdir(tmp_path)

    build_proteinbox(inp)

    for name in ("helices", "loop"):
        assert (tmp_path / f"posre_{name}_Protein_chain_A.itp").is_file()
    md = (tmp_path / "md.mdp").read_text()
    assert "-DPOSRES_HELICES -DPOSRES_LOOP" in md
    _grompp_all_stages(tmp_path, tmp_path / "system.pdb", tmp_path / "topology.top")


@needs_example_pdbs
def test_com_restrained_insulin_example_build(tmp_path, monkeypatch):
    """Position and COM-distance restraints build; every stage preprocesses with -n."""
    import yaml

    from mdfactory.build import build_proteinbox
    from mdfactory.models.input import BuildInput
    from mdfactory.orchestration.mdp import get_mdp_value, parse_mdp
    from mdfactory.prepare import resolve_protein_pdb_path

    data = yaml.safe_load((EXAMPLES / "insulin_restraints.yaml").read_text())
    resolve_protein_pdb_path(data, EXAMPLES)
    data["system"]["relax_steps"] = 200
    inp = BuildInput(**data)
    monkeypatch.chdir(tmp_path)

    build_proteinbox(inp)

    index = (tmp_path / "index.ndx").read_text()
    for group in ("Protein", "non-Protein", "COM_DIMER_1", "COM_DIMER_2", "COM_MONOMER_1"):
        assert f"[ {group} ]" in index
    measured = json.loads((tmp_path / "com_restraints.json").read_text())
    md = parse_mdp(tmp_path / "md.mdp")
    assert get_mdp_value(md, "define") == "-DPOSRES_HELICES"
    assert get_mdp_value(md, "pull_coord2_type") == "flat-bottom"
    init = float(get_mdp_value(md, "pull_coord2_init"))
    assert init == pytest.approx(measured["monomer"]["measured_distance"] + 0.2, abs=1e-6)
    _grompp_all_stages(tmp_path, tmp_path / "system.pdb", tmp_path / "topology.top")
