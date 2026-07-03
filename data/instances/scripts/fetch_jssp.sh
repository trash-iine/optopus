#!/usr/bin/env bash
# Regenerate the bundled JSSP instances in data/instances/jssp/orlib/.
# Downloads the OR-Library jobshop bundle (MIT; see data/instances/jssp/NOTICE)
# and splits it into per-instance files via split_jobshop.py.
set -u

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TMP_DIR=$( mktemp -d )
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Downloading jobshop1.txt"
curl -sf -o "$TMP_DIR/orlib_jobshop1.txt" \
	"https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/jobshop1.txt" || true

python3 "$SCRIPT_DIR/split_jobshop.py" "$TMP_DIR"
