# End-to-end pipeline: protein_mixedbox on SLURM

Four lysozyme/ethanol systems (0.5 M and 1 M ethanol, 300 K and 310 K) from one CSV, a shared
`SimulationSettings` protocol, and a GPU SLURM executor config. Open-source force fields only
(Amber ff14SB, OpenFF Sage, TIP3P): no CGenFF license needed, but GROMACS >= 2026.

The step-by-step explanation is in the tutorial
`docs/content/docs/user-guide/tutorial-protein-mixedbox.mdx`.

## Files

| File | Role |
|---|---|
| `systems.csv` | One row per system. Dotted columns: `parametrization.*`, `system.protein.*`, `system.species.<RESNAME>.*`, `system.sizing.*`, `simulation_settings.temperature` |
| `protocol.yaml` | Settings shared by all rows (`--simulation-settings`): 50 ns production, longer equilibration, `C-rescale` barostat in NPT |
| `slurm_gpu.yaml` | Parsl/SLURM executor for build **and** simulate, workers activate the pixi `gmx-cuda` env |
| `run_pipeline.sh` | `sbatch` controller: check-csv, build, simulate, analysis |
| `lysozyme_ethanol_amber.yaml` | Single-system variant with inline `simulation_settings:` |

## Run it

```bash
# 0. structures (once) and environment
bash examples/fetch_pdbs.sh
pixi install -e gmx-cuda

# 1. edit account / partition / gres and the manifest path in slurm_gpu.yaml
$EDITOR examples/pipeline/slurm_gpu.yaml

# 2. submit (from the repository root)
sbatch examples/pipeline/run_pipeline.sh
# smoke test with a minutes-long protocol:
PROTOCOL=examples/simulation_settings/protocol_short_test.yaml \
    sbatch examples/pipeline/run_pipeline.sh
```

The same steps by hand, with `MDF="pixi run -e gmx-cuda mdfactory"`:

```bash
$MDF check-csv examples/pipeline/systems.csv
$MDF build examples/pipeline/systems.csv --output runs_pipeline \
    --simulation-settings examples/pipeline/protocol.yaml \
    --slurm examples/pipeline/slurm_gpu.yaml
$MDF simulate runs_pipeline --slurm examples/pipeline/slurm_gpu.yaml --dry-run   # preview
$MDF simulate runs_pipeline --slurm examples/pipeline/slurm_gpu.yaml
$MDF analysis run runs_pipeline/systems.yaml --analysis system_chemistry
$MDF analysis artifacts run runs_pipeline/systems.yaml --artifact last_frame_pdb
```

## Without SLURM (CPU smoke test)

```bash
pixi run -e gmx mdfactory build examples/pipeline/systems.csv --output runs_local \
    --simulation-settings examples/simulation_settings/protocol_short_test.yaml
pixi run -e gmx mdfactory simulate runs_local
```

The local executor always runs without GPU, and the OpenMM compression stage of the build is slow
on CPUs (tens of minutes). Prefer the SLURM route for anything beyond a smoke test.

## Changing the protocol later

Hashes depend on the system and on temperature/pressure only, so a longer production run does not
create new systems. Re-render the mdp files of builds whose stages have not run yet:

```bash
$MDF apply-settings examples/simulation_settings/apply_longer_production.yaml runs_pipeline --dry-run
$MDF apply-settings examples/simulation_settings/apply_longer_production.yaml runs_pipeline
```
