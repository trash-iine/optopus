#!/usr/bin/env bash
# Fetch SATLIB uniform random 3-SAT (uf*) instances into
# data/instances/sat/satlib/<family>/.
# SATLIB (Hoos & Stützle): https://www.cs.ubc.ca/~hoos/SATLIB/
# For each family the first 10 lexically-sorted .cnf files are kept.
set -u

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SAT_DIR=$SCRIPT_DIR/../sat
TMP_DIR=$( mktemp -d )
trap 'rm -rf "$TMP_DIR"' EXIT

FAMILIES=(
	uf50-218
	uf75-325
	uf100-430
	uf150-645
	uf200-860
)

for spec in ${FAMILIES[@]}; do
	DEST=$SAT_DIR/satlib/$spec
	if [ -d $DEST ] && [ -n "$(ls -A $DEST 2>/dev/null)" ]; then
		echo "Family $spec already present in $DEST"
		continue
	fi
	echo "Downloading $spec"
	curl -sf -o $TMP_DIR/$spec.tar.gz \
		https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/$spec.tar.gz || continue
	mkdir -p $TMP_DIR/extract/$spec
	tar -xzf $TMP_DIR/$spec.tar.gz -C $TMP_DIR/extract/$spec
	mkdir -p $DEST
	find $TMP_DIR/extract/$spec -name '*.cnf' | sort | head -10 | xargs -I{} cp {} $DEST/
done
