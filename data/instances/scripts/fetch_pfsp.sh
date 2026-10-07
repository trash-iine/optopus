#!/usr/bin/env bash
# Regenerate the bundled permutation flow shop instances in
# data/instances/flow_shop/taillard/. Downloads OR-Library flowshop2.txt (MIT;
# see data/instances/flow_shop/NOTICE), which holds Taillard's generator and the
# time seed of ta001-ta120, and regenerates the instances via gen_taillard_pfsp.py.
set -eu

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TMP_DIR=$( mktemp -d )
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Downloading flowshop2.txt"
curl -sf -o "$TMP_DIR/flowshop2.txt" \
	"https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/flowshop2.txt"

python3 "$SCRIPT_DIR/gen_taillard_pfsp.py" "$TMP_DIR/flowshop2.txt"
