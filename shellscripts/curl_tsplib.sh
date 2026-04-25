#!bin/bash
set -u

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATA_DIR=$SCRIPT_DIR/../data/tsp
mkdir -p $DATA_DIR

FILES=(
	berlin52.tsp
	eil101.tsp
	ch150.tsp
	att48.tsp
	burma14.tsp
	eil51.tsp
	dsj1000.tsp
)

for f in ${FILES[@]}; do
	FILEPATH=$DATA_DIR/$f
	if [ -f $FILEPATH ]; then
		echo "File $FILEPATH already exists"
		continue
	fi
	echo "Downloading $f"
	curl -sf -o $FILEPATH https://raw.githubusercontent.com/mastqe/tsplib/master/$f || continue
done
