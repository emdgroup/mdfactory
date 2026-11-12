#!/usr/bin/env bash
# Download the PDB structures referenced by the protein examples from RCSB.
#
#   1AKI  hen egg-white lysozyme (single chain)    -> proteinbox/1aki.pdb
#   1ZNI  insulin (four chains A-D)                -> proteinbox/1zni.pdb
#
# The protein_mixedbox examples and pipeline/ use ../proteinbox/1aki.pdb, so
# this one script prepares every protein example. Structure files are not
# tracked in git (.gitignore: *.pdb).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$here/proteinbox"

for id in 1aki 1zni; do
  target="$here/proteinbox/$id.pdb"
  if [ -s "$target" ]; then
    echo "$target exists, skipping"
  else
    echo "downloading $id -> $target"
    curl -fsSL "https://files.rcsb.org/download/${id^^}.pdb" -o "$target"
  fi
done
