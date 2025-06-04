#!/usr/bin/env bash
# get_uci_dataset.sh – fetch the UCI Household Power dataset and unpack it
# into a local directory that is typically git‑ignored.
#
# Usage:
#   ./scripts/get_uci_dataset.sh               # default to data/raw/uci
#   ./scripts/get_uci_dataset.sh /custom/path  # custom destination

set -euo pipefail

OUTPUT_DIR=${1:-data/raw/uci}
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Create the target directory in case it is absent (e.g. because the
# entire `data/` tree is listed in .gitignore)
mkdir -p "$OUTPUT_DIR"

python "$SCRIPT_DIR/fetch_uci_dataset.py" --output_dir "$OUTPUT_DIR"

echo "UCI dataset downloaded and unpacked to $OUTPUT_DIR"