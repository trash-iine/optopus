#!/usr/bin/env bash
# Regenerate the bundled QUBO instances in data/instances/qubo/bqp/.
# Downloads the Beasley OR-Library bqp bundles (MIT; see data/instances/qubo/NOTICE)
# and splits them into per-instance files via split_bqp.py.
set -u

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TMP_DIR=$( mktemp -d )
trap 'rm -rf "$TMP_DIR"' EXIT

for n in 50 100 250 500 1000; do
	echo "Downloading bqp${n}.txt"
	curl -sf -o "$TMP_DIR/orlib_bqp${n}.txt" \
		"https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/bqp${n}.txt" || continue
done

python3 "$SCRIPT_DIR/split_bqp.py" "$TMP_DIR"
