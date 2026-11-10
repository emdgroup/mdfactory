#!/usr/bin/env bash
#SBATCH --job-name=mdfactory-pipeline
#SBATCH --partition=cpu
#SBATCH --account=my-account
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=mdfactory_pipeline_%j.out
#
# End-to-end pipeline: CSV + shared SimulationSettings -> build -> simulate ->
# analysis, with every heavy step submitted by Parsl as its own SLURM job.
# This script is only the *controller*: it needs a small CPU allocation and the
# same environment as the workers (pixi `gmx-cuda`), otherwise Parsl versions
# of controller and workers can disagree.
#
# Submit from the repository root, after `bash examples/fetch_pdbs.sh` and after
# editing the account/manifest path in examples/pipeline/slurm_gpu.yaml:
#
#   sbatch examples/pipeline/run_pipeline.sh
#
# Smoke test with a few minutes of MD instead of 50 ns:
#
#   PROTOCOL=examples/simulation_settings/protocol_short_test.yaml \
#       sbatch examples/pipeline/run_pipeline.sh
set -euo pipefail

PIPE="${PIPE:-examples/pipeline}"
CSV="${CSV:-$PIPE/systems.csv}"
PROTOCOL="${PROTOCOL:-$PIPE/protocol.yaml}"
SLURM_CONFIG="${SLURM_CONFIG:-$PIPE/slurm_gpu.yaml}"
RUNS="${RUNS:-runs_pipeline}"

# The controller node has no GPU; tell conda to solve/activate the CUDA build anyway.
export CONDA_OVERRIDE_CUDA=12.9
MDF="pixi run --as-is -e gmx-cuda mdfactory"

# 0. Validate every row (parametrization, sizing, protein options) before queueing anything.
$MDF check-csv "$CSV"

# 1. Build. Rows without simulation_settings.* cells take the protocol file;
#    each build directory records the resolved settings in <hash>.yaml and
#    gets em/nvt/npt/md.mdp rendered from them.
$MDF build "$CSV" --output "$RUNS" --simulation-settings "$PROTOCOL" --slurm "$SLURM_CONFIG"

# 2. Simulate: EM -> NVT -> NPT -> Production. Re-running resumes from checkpoints.
$MDF simulate "$RUNS" --slurm "$SLURM_CONFIG"

# 3. Analyze. Protein types register system_chemistry (species, counts, achieved
#    concentrations) and the last_frame_pdb artifact.
#    The summary YAML written by `build` is named after the CSV stem.
SUMMARY="$RUNS/$(basename "$CSV" .csv).yaml"
$MDF analysis run "$SUMMARY" --analysis system_chemistry
$MDF analysis artifacts run "$SUMMARY" --artifact last_frame_pdb

echo "[INFO] Pipeline completed: $RUNS"
