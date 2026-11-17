# ABOUTME: Tests for selection-based position restraints on synthetic pdb2gmx outputs
# ABOUTME: Covers index mapping per layout, itp text, include insertion and the error paths
"""Tests for :mod:`mdfactory.setup.restraints` (no GROMACS needed)."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdfactory.models.species import ComDistanceRestraint, PositionRestraint
from mdfactory.setup.restraints import (
    apply_restraints,
    format_index_groups,
    format_position_restraints,
    insert_restraint_includes,
    measure_com_distance,
    read_com_restraints,
    read_protein_moleculetypes,
    reference_distance,
    write_com_restraint_files,
)
from mdfactory.utils.topology_utilities import iter_topology_entries

from .conftest import PDB2GMX_RESIDUES, write_pdb2gmx_output

CHAIN_A = PDB2GMX_RESIDUES[:2]


def _apply(params, restraints, cleaned):
    """Apply *restraints* and return the updated parameter set."""
    return apply_restraints(params, restraints, cleaned)[0]


def _harmonic(name="core", selection="name CA", k=500.0):
    return PositionRestraint(name=name, selection=selection, k=k)


def _flat_bottom(name="loop", selection="name CA", radius=0.3, k=1000.0):
    return PositionRestraint(name=name, selection=selection, type="flat_bottom", radius=radius, k=k)


def _atom_labels(topology_or_itp: Path) -> list[tuple[int, str, str]]:
    """Return ``(resnr, residue, atom name)`` of each ``[ atoms ]`` line, in order."""
    labels = []
    for entry in iter_topology_entries(topology_or_itp.read_text().splitlines()):
        if entry.section == "atoms":
            parts = entry.code.split()
            labels.append((int(parts[2]), parts[3], parts[4]))
    return labels


def _restrained(itp: Path) -> list[list[str]]:
    """Return the fields of each ``[ position_restraints ]`` line of *itp*."""
    return [
        entry.code.split()
        for entry in iter_topology_entries(itp.read_text().splitlines())
        if entry.section == "position_restraints"
    ]


def _restrained_labels(itp: Path, moleculetype_file: Path) -> list[tuple[int, str, str]]:
    labels = _atom_labels(moleculetype_file)
    return [labels[int(fields[0]) - 1] for fields in _restrained(itp)]


# --- index mapping per topology layout ---------------------------------------------------


def test_single_chain_maps_selection_to_moleculetype_atoms(tmp_path):
    params, cleaned = write_pdb2gmx_output(tmp_path, "single", CHAIN_A)
    result = _apply(params, [_harmonic(selection="resid 2 and name CA CB")], cleaned)

    itp = params.topology_file.parent / "posre_core_Protein_chain_A.itp"
    assert result.topology_include_files == [*params.topology_include_files, itp]
    assert _restrained_labels(itp, params.topology_file) == [(2, "ALA", "CA"), (2, "ALA", "CB")]


def test_unmerged_chains_restrain_only_the_selected_chain(tmp_path):
    """Chains share resids; a chain-qualified selection picks one chain's moleculetype."""
    params, cleaned = write_pdb2gmx_output(tmp_path, "chains")
    out = params.topology_file.parent
    chain_a_before = (out / "topol_Protein_chain_A.itp").read_text()

    restraint = _harmonic(selection="chainID B and resid 1 and not name H*")
    result = _apply(params, [restraint], cleaned)

    itp = out / "posre_core_Protein_chain_B.itp"
    assert [p.name for p in result.topology_include_files][-1] == itp.name
    assert not (out / "posre_core_Protein_chain_A.itp").exists()
    assert _restrained_labels(itp, out / "topol_Protein_chain_B.itp") == [
        (1, "SER", name) for name in ("N", "CA", "CB", "OG", "C", "O")
    ]
    assert (out / "topol_Protein_chain_A.itp").read_text() == chain_a_before


def test_unmerged_chains_split_a_selection_across_moleculetypes(tmp_path):
    params, cleaned = write_pdb2gmx_output(tmp_path, "chains")
    out = params.topology_file.parent
    _apply(params, [_harmonic(selection="resid 1 and name CA")], cleaned)
    for chain, resname in (("A", "GLY"), ("B", "SER")):
        itp = out / f"posre_core_Protein_chain_{chain}.itp"
        assert _restrained_labels(itp, out / f"topol_Protein_chain_{chain}.itp") == [
            (1, resname, "CA")
        ]


def test_merged_chains_use_input_chain_ids_and_resids(tmp_path):
    """merge_all: one moleculetype; resids come from the input even if the gro renumbers."""
    params, cleaned = write_pdb2gmx_output(tmp_path, "single", gro_resids=[1, 2, 3, 4])
    _apply(params, [_harmonic(selection="chainID B and resid 2")], cleaned)

    itp = params.topology_file.parent / "posre_core_Protein_chain_A.itp"
    labels = _restrained_labels(itp, params.topology_file)
    # Residue 4 of the merged moleculetype is chain B's GLY 2.
    assert labels == [(4, "GLY", name) for name in ("N", "H", "CA", "C", "O", "OXT")]


def test_read_protein_moleculetypes_in_molecules_order(tmp_path):
    params, _ = write_pdb2gmx_output(tmp_path, "chains")
    moltypes = read_protein_moleculetypes(params.topology_file)
    assert [(m.name, m.path.name) for m in moltypes] == [
        ("Protein_chain_A", "topol_Protein_chain_A.itp"),
        ("Protein_chain_B", "topol_Protein_chain_B.itp"),
    ]
    assert moltypes[0].atom_names[:3] == ("N", "H1", "CA")


def test_no_restraints_leaves_output_untouched(tmp_path):
    params, cleaned = write_pdb2gmx_output(tmp_path, "single")
    before = params.topology_file.read_text()
    result, com_groups = apply_restraints(params, [], cleaned)
    assert result is params and com_groups == {}
    assert params.topology_file.read_text() == before


# --- itp text ----------------------------------------------------------------------------


def test_harmonic_itp_is_funct_1_isotropic():
    text = format_position_restraints(_harmonic(k=750.0), [0, 4])
    rows = [line.split() for line in text.splitlines() if line and line[0] not in ";["]
    assert "[ position_restraints ]" in text and "-DPOSRES_CORE" in text
    assert rows == [["1", "1", "750.0", "750.0", "750.0"], ["5", "1", "750.0", "750.0", "750.0"]]


def test_flat_bottom_itp_is_funct_2_sphere():
    text = format_position_restraints(_flat_bottom(radius=0.25, k=1200.0), [2])
    rows = [line.split() for line in text.splitlines() if line and line[0] not in ";["]
    # ai funct g r k, with g = 1 (sphere).
    assert rows == [["3", "2", "1", "0.25", "1200.0"]]


# --- include insertion -------------------------------------------------------------------


def _ifdef_blocks(path: Path) -> list[tuple[str, str]]:
    """Return ``(macro, included file)`` of each ``#ifdef`` + ``#include`` pair in *path*."""
    lines = [line.strip() for line in path.read_text().splitlines()]
    return [
        (line.split()[1], lines[i + 1].split('"')[1])
        for i, line in enumerate(lines)
        if line.startswith("#ifdef") and lines[i + 1].startswith("#include")
    ]


def test_includes_follow_pdb2gmx_block_in_topology(tmp_path):
    params, cleaned = write_pdb2gmx_output(tmp_path, "single")
    _apply(params, [_harmonic(), _flat_bottom(selection="chainID A and name C")], cleaned)
    assert _ifdef_blocks(params.topology_file) == [
        ("POSRES", "posre.itp"),
        ("POSRES_CORE", "posre_core_Protein_chain_A.itp"),
        ("POSRES_LOOP", "posre_loop_Protein_chain_A.itp"),
    ]
    text = params.topology_file.read_text()
    # Still inside the protein moleculetype: before the water include and [ molecules ].
    assert text.index("POSRES_LOOP") < text.index("tip3p.itp") < text.index("[ molecules ]")


def test_includes_go_into_each_chain_itp(tmp_path):
    params, cleaned = write_pdb2gmx_output(tmp_path, "chains")
    out = params.topology_file.parent
    top_before = params.topology_file.read_text()
    _apply(params, [_harmonic(selection="name CA")], cleaned)
    for chain in "AB":
        assert _ifdef_blocks(out / f"topol_Protein_chain_{chain}.itp") == [
            ("POSRES", f"posre_Protein_chain_{chain}.itp"),
            ("POSRES_CORE", f"posre_core_Protein_chain_{chain}.itp"),
        ]
    assert params.topology_file.read_text() == top_before


def test_missing_posres_anchor_raises(tmp_path):
    params, _ = write_pdb2gmx_output(tmp_path, "single")
    text = params.topology_file.read_text().replace("#ifdef POSRES\n", "")
    params.topology_file.write_text(text)
    with pytest.raises(RuntimeError, match="No pdb2gmx '#ifdef POSRES' block"):
        insert_restraint_includes(
            params.topology_file, "Protein_chain_A", [("POSRES_CORE", "posre_core.itp")]
        )


# --- errors ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("selection", "match"),
    [("chainID C", "matches no protein atom"), ("resid 1 and", "invalid selection")],
)
def test_bad_selection_raises(tmp_path, selection, match):
    params, cleaned = write_pdb2gmx_output(tmp_path, "chains")
    with pytest.raises(ValueError, match=match):
        _apply(params, [_harmonic(selection=selection)], cleaned)


def test_atom_name_mismatch_raises(tmp_path):
    params, cleaned = write_pdb2gmx_output(tmp_path, "single")
    gro = params.structure_file.read_text().replace("  OXT", "  OT2", 1)
    params.structure_file.write_text(gro)
    with pytest.raises(ValueError, match="'OT2' in the pdb2gmx structure but 'OXT'"):
        _apply(params, [_harmonic()], cleaned)


def test_residue_count_mismatch_raises(tmp_path):
    params, cleaned = write_pdb2gmx_output(tmp_path, "single")
    lines = cleaned.read_text().splitlines(keepends=True)
    cleaned.write_text("".join(line for line in lines if line[21:26] != "B   2"))
    with pytest.raises(ValueError, match="has 4 residues but .* has 3"):
        _apply(params, [_harmonic()], cleaned)


# --- the builders copy the restraint files ------------------------------------------------


def test_proteinbox_copy_takes_the_restraint_itps(tmp_path):
    from mdfactory.build import _copy_protein_topology

    params, cleaned = write_pdb2gmx_output(tmp_path, "chains")
    params = _apply(params, [_harmonic(selection="chainID B")], cleaned)
    dest = tmp_path / "build"
    dest.mkdir()
    _copy_protein_topology(params, dest, "topology.top")
    assert (dest / "posre_core_Protein_chain_B.itp").is_file()
    assert ("POSRES_CORE", "posre_core_Protein_chain_B.itp") in _ifdef_blocks(
        dest / "topol_Protein_chain_B.itp"
    )


def test_protein_mixedbox_composite_topology_includes_restraints(tmp_path):
    from types import SimpleNamespace

    from mdfactory.build import _stage_protein_topology
    from mdfactory.models.species import SolutionSpecies

    params, cleaned = write_pdb2gmx_output(tmp_path, "single")
    params = _apply(params, [_harmonic(), _flat_bottom()], cleaned)
    ff_dir = tmp_path / "amber14sb.ff"
    ff_dir.mkdir()
    dest = tmp_path / "build"
    dest.mkdir()
    species = [SolutionSpecies(smiles="O", resname="SOL", count=10)]
    parameters = [SimpleNamespace(moleculetype="SOL", parameter_itp=None, topology_merge="native")]

    topology = _stage_protein_topology(params, ff_dir, dest, species, parameters, [10])

    for itp in ("posre_core_Protein_chain_A.itp", "posre_loop_Protein_chain_A.itp"):
        assert (dest / itp).is_file()
    text = topology.read_text()
    # Inside the protein moleculetype: after its [ atoms ], before the water include.
    first, last = (
        text.index(f'#include "posre_{n}_Protein_chain_A.itp"') for n in ("core", "loop")
    )
    assert text.index("[ atoms ]") < first < last < text.index("tip3p.itp")


# --- COM-distance restraints ----------------------------------------------------------------


def _com(name="dimer", selections=("chainID A", "chainID B"), **kwargs):
    return ComDistanceRestraint(
        name=name, type="com_distance", selections=selections, k=kwargs.pop("k", 1000.0), **kwargs
    )


def test_com_groups_resolve_chain_qualified_selections(tmp_path):
    params, cleaned = write_pdb2gmx_output(tmp_path, "chains")
    before = params.topology_file.read_text()
    result, groups = apply_restraints(params, [_com()], cleaned)
    first, second = groups["dimer"]
    n_a = sum(len(atoms) for chain, _, _, atoms in PDB2GMX_RESIDUES if chain == "A")
    assert first.tolist() == list(range(n_a))
    assert second.tolist() == list(range(n_a, n_a + 13))
    # COM restraints write no itp and leave the topology alone.
    assert result is params and params.topology_file.read_text() == before


@pytest.mark.parametrize(
    ("selections", "match"),
    [
        (("chainID A", "resid 1"), "share 5 atoms"),
        (("chainID A", "chainID C"), "matches no protein atom"),
    ],
)
def test_bad_com_groups_raise(tmp_path, selections, match):
    params, cleaned = write_pdb2gmx_output(tmp_path, "chains")
    with pytest.raises(ValueError, match=match):
        apply_restraints(params, [_com(selections=selections)], cleaned)


def test_measure_com_distance_is_mass_weighted_and_minimum_image():
    positions = np.array([[0.0, 0, 0], [3.0, 0, 0], [19.0, 0, 0]])
    masses = np.array([1.0, 2.0, 5.0])
    groups = (np.array([0, 1]), np.array([2]))
    # COM of the first group is at x = 2 Å, closest to atom 2 (1-based).
    open_box = measure_com_distance(positions, masses, np.array([100.0] * 3 + [90.0] * 3), groups)
    assert open_box.distance == pytest.approx(1.7)
    assert open_box.pbcatoms == (2, 3)
    periodic = measure_com_distance(positions, masses, np.array([20.0] * 3 + [90.0] * 3), groups)
    assert periodic.distance == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, 1.5),
        ({"width": 0.2}, 1.7),
        ({"distance": 1.0}, 1.0),
        ({"distance": 1.0, "width": 0.5}, 1.5),
    ],
)
def test_reference_distance(kwargs, expected):
    assert reference_distance(_com(**kwargs), measured=1.5) == pytest.approx(expected)


def test_index_groups_are_one_based_and_wrapped():
    text = format_index_groups({"COM_X_1": np.arange(17), "COM_X_2": np.array([20])})
    lines = text.splitlines()
    assert lines[0] == "[ COM_X_1 ]"
    assert lines[1].split() == [str(i) for i in range(1, 16)]
    assert lines[2:] == ["16 17", "[ COM_X_2 ]", "21"]


def _patch_make_ndx(monkeypatch, calls):
    def fake_run_gmx(name, cmd, cwd, stdin, timeout):
        calls.append((name, stdin))
        Path(cmd[cmd.index("-o") + 1]).write_text("[ System ]\n1 2 3\n")

    monkeypatch.setattr("mdfactory.setup.restraints._run_gmx", fake_run_gmx)
    monkeypatch.setattr("mdfactory.setup.restraints.check_gmx_available", lambda: Path("gmx"))


def test_write_com_restraint_files(tmp_path, monkeypatch):
    import MDAnalysis as mda

    calls = []
    _patch_make_ndx(monkeypatch, calls)
    params, cleaned = write_pdb2gmx_output(tmp_path, "chains")
    restraints = [
        _com(),
        _com(name="tail", selections=("chainID A and resid 1", "name OXT"), width=0.1),
    ]
    _, groups = apply_restraints(params, restraints, cleaned)

    index = write_com_restraint_files(
        tmp_path, restraints, groups, params.structure_file, params.topology_file
    )

    assert calls == [("make_ndx", "q\n")]
    text = index.read_text()
    assert text.startswith("[ System ]")
    headers = [line for line in text.splitlines() if line.startswith("[")]
    assert headers == [
        "[ System ]",
        "[ COM_DIMER_1 ]",
        "[ COM_DIMER_2 ]",
        "[ COM_TAIL_1 ]",
        "[ COM_TAIL_2 ]",
    ]
    entries = json.loads((tmp_path / "com_restraints.json").read_text())
    assert entries["dimer"]["groups"] == ["COM_DIMER_1", "COM_DIMER_2"]
    # Equal masses: the COMs are the plain centroids of the groups.
    x = mda.Universe(str(params.structure_file)).atoms.positions
    first, second = groups["dimer"]
    expected = np.linalg.norm(x[first].mean(axis=0) - x[second].mean(axis=0)) / 10.0
    assert entries["dimer"]["measured_distance"] == pytest.approx(expected, abs=1e-4)
    for group, pbcatom in zip(groups["dimer"], entries["dimer"]["pbcatoms"]):
        assert pbcatom - 1 in group


def test_com_distance_beyond_half_the_box_raises(tmp_path, monkeypatch):
    _patch_make_ndx(monkeypatch, [])
    params, cleaned = write_pdb2gmx_output(tmp_path, "chains")
    restraint = _com(distance=2.0, width=0.5)  # 2.5 nm in a 5 nm box
    _, groups = apply_restraints(params, [restraint], cleaned)
    with pytest.raises(ValueError, match="not below 0.49 x the shortest box edge"):
        write_com_restraint_files(
            tmp_path, [restraint], groups, params.structure_file, params.topology_file
        )
    assert not (tmp_path / "index.ndx").exists()


def test_com_structure_must_start_with_the_protein(tmp_path, monkeypatch):
    _patch_make_ndx(monkeypatch, [])
    params, cleaned = write_pdb2gmx_output(tmp_path, "chains")
    _, groups = apply_restraints(params, [_com()], cleaned)
    params.structure_file.write_text(params.structure_file.read_text().replace("  OXT", "  OT2"))
    with pytest.raises(ValueError, match="'OT2' in the protein of"):
        write_com_restraint_files(
            tmp_path, [_com()], groups, params.structure_file, params.topology_file
        )


def test_read_com_restraints_requires_the_build_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="rebuild the system"):
        read_com_restraints(tmp_path)
