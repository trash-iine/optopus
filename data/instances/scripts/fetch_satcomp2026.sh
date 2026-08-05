#!/usr/bin/env bash
# Fetch a curated set of large-scale SAT Competition 2026 benchmark instances
# into data/instances/sat/satcomp2026/, for MaxSAT (maximize satisfied clauses).
#
# Source: SAT Competition 2026 benchmark selection
#   https://github.com/satcompetition/2026
#     downloads/benchmark-compilation-script/selected_benchmarks.csv
# The selection lists instances by md5 `hash`; the actual DIMACS `p cnf` files
# are served by the Global Benchmark Database (GBD):
#   https://benchmark-database.de/file/<hash>?context=cnf
# Files are distributed xz-compressed and decompressed to plain `.cnf` here
# (directly compatible with `Sat::load_file`, which reads unweighted `p cnf`).
#
# The seven instances below were chosen from crafted/combinatorial families
# spanning ~1.8e4 .. 2.5e6 clauses (see the table in data/instances/README.md).
set -u

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DEST=$SCRIPT_DIR/../sat/satcomp2026
GBD_BASE="https://benchmark-database.de/file"

# hash|local-name (without extension)|family
INSTANCES=(
	"14216e0f4c89982abcf5ad2cc1a23fd8|quasigroup_ukn006|quasigroup-completion"
	"cee485c7a301636a77d629b1897b476a|waerden_2-3-25_606|waerden"
	"6bf14262fd0dd58c3e4b913d0ad84698|sliding_puzzle57|sliding-puzzle"
	"fb621b2b4a54ec0f01ac36daaa7f0f5b|hamiltonian_hcp_470_105|hamiltonian-cycle"
	"79d2eae74f2b790d821578d9b7f675c0|mycielski10_hints6|coloring-mycielski-graph"
	"d189545a89cfa00bee42cfa0088f10dd|station_repacking_49|station-repacking"
	"63acdb1e4c4437b22b7ee3b405c1d096|coloring_4g_5color_170|coloring"
)

mkdir -p "$DEST"

for spec in "${INSTANCES[@]}"; do
	IFS='|' read -r hash name family <<< "$spec"
	out=$DEST/$name.cnf
	if [ -s "$out" ]; then
		echo "Present, skipping: $name.cnf ($family)"
		continue
	fi
	echo "Downloading $name.cnf  ($family, hash $hash)"
	tmp=$( mktemp )
	if ! curl -sfL --max-time 300 -o "$tmp" "$GBD_BASE/$hash?context=cnf"; then
		echo "  FAILED to download $hash" >&2
		rm -f "$tmp"
		continue
	fi
	# GBD serves xz-compressed CNF; decompress to plain text.
	if xz -t "$tmp" 2>/dev/null; then
		xz -dc "$tmp" > "$out"
	else
		# Already plain (or other): keep as-is.
		cp "$tmp" "$out"
	fi
	rm -f "$tmp"
	header=$( grep -m1 '^p ' "$out" 2>/dev/null )
	echo "  -> $name.cnf  [$header]"
done

echo "Done. Instances in $DEST"
