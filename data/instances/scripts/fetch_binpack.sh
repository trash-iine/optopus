#!/usr/bin/env bash
# Regenerate the bundled one-dimensional bin packing instances in
# data/instances/bin_packing/falkenauer/. Downloads the OR-Library bundles
# binpack1.txt ... binpack8.txt (MIT; see data/instances/bin_packing/NOTICE)
# and splits them into per-instance files via split_binpack.py.
set -eu

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TMP_DIR=$( mktemp -d )
trap 'rm -rf "$TMP_DIR"' EXIT

for k in 1 2 3 4 5 6 7 8; do
	echo "Downloading binpack${k}.txt"
	curl -sf -o "$TMP_DIR/binpack${k}.txt" \
		"https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/binpack${k}.txt"
done

python3 "$SCRIPT_DIR/split_binpack.py" "$TMP_DIR"
