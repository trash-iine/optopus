#!/usr/bin/env bash
# Fetch GSET MaxCut graphs into data/instances/max_cut/.
# GSET (Y. Ye / Stanford): https://web.stanford.edu/~yyye/yyye/Gset/
# Files are served as plain text; missing numbers in 1..81 are skipped.
set -u

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATA_DIR=$SCRIPT_DIR/../max_cut
mkdir -p $DATA_DIR

for i in {1..81}; do
	FILEPATH=$DATA_DIR/G$i
	if [ -f $FILEPATH ]; then
		echo "File $FILEPATH already exists"
		continue
	fi
	echo "Downloading G$i"
	curl -sf -o $FILEPATH https://web.stanford.edu/~yyye/yyye/Gset/G$i || continue
done
