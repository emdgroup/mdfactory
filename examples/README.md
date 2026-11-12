# mdfactory examples

Every build input here is validated by the test suite
(`mdfactory/tests/test_parametrization_schema.py` parses every YAML and CSV under
this directory). Paths such as `pdb_path` are resolved relative to the file that
contains them.

Structures for the protein examples are not tracked in git. Fetch them once:

```bash
bash examples/fetch_pdbs.sh    # 1aki.pdb (lysozyme), 1zni.pdb (insulin) -> examples/proteinbox/
```

## Build inputs by system type

| Directory | System type | Force-field family | What it shows |
|---|---|---|---|
| `mixedbox/` | `mixedbox` | `charmm`, `openff` | Water/ethanol; `water_ethanol_openff_explicit.yaml` spells out the OpenFF defaults; `water_ethanol_smirnoff_310K.yaml` carries inline `simulation_settings` |
| `bilayer/` | `bilayer` | `charmm`, `openff` | POPC bilayer; `popc_smirnoff_310K_short.yaml` shortens the 1 µs default |
| `lnp/` | `lnp` | `charmm`, `openff` | ALC-0315 lipid nanoparticle |
| `proteinbox/` | `proteinbox` | `charmm` (CHARMM36m), `amber` (ff14SB) | Solvated, neutralized protein; single and multi-chain (`insulin_amber14sb.yaml`); YAML and CSV |
| `protein_mixedbox/` | `protein_mixedbox` | `charmm`, `amber` | Protein in a water/co-solvent mixture: `fixed_box` and `count_density` sizing, concentration-based species, packed ions (`lysozyme_amber_packed_ions.yaml`), merged multi-chain protein (`insulin_amber_merged.yaml`) |

The `amber` family needs **GROMACS >= 2026** (pixi environments `gmx` / `gmx-cuda`) and no
CGenFF license. The `charmm` family with small molecules needs a SILCSBIO/CGenFF
installation (`mdfactory config init`).

## SimulationSettings

`simulation_settings/` shows every way to supply the run protocol:

| File | Use |
|---|---|
| `protocol_full.yaml` | Every field spelled out, for `--simulation-settings` |
| `protocol_short_test.yaml` | Minutes-long protocol for smoke tests |
| `apply_longer_production.yaml` | Input to `mdfactory apply-settings` for existing builds |
| `temperature_series.csv` | Dotted `simulation_settings.*` CSV columns (temperature series, per-row overrides) |
| `mixedbox/water_ethanol_smirnoff_310K.yaml`, `bilayer/popc_smirnoff_310K_short.yaml`, `pipeline/lysozyme_ethanol_amber.yaml` | Inline `simulation_settings:` sections |

## Full pipeline on SLURM

`pipeline/` runs CSV -> build -> simulate -> analysis for four lysozyme/ethanol systems
(two ethanol concentrations x two temperatures) with a shared protocol and a GPU SLURM config:

| File | Role |
|---|---|
| `systems.csv` | The four `protein_mixedbox` systems (amber family) |
| `protocol.yaml` | Shared `SimulationSettings` (`--simulation-settings`) |
| `slurm_gpu.yaml` | Executor config for build and simulate (pixi `gmx-cuda` workers) |
| `run_pipeline.sh` | `sbatch` controller script running all steps |
| `lysozyme_ethanol_amber.yaml` | The same kind of system as a single YAML with inline settings |

See `pipeline/README.md` and the tutorial
"Tutorial: Protein in a Co-solvent Box" (`docs/content/docs/user-guide/tutorial-protein-mixedbox.mdx`)
for a walkthrough.

## SLURM executor reference

`slurm_executor.yaml` is the annotated reference for every executor field
(`mdfactory config slurm` generates one interactively).
