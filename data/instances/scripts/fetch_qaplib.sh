#!/usr/bin/env bash
# Fetch QAPLIB instances into data/instances/qap/.
# QAPLIB (Burkard, Karisch, Rendl; maintained by P. Hahn and M. Anjos) states
# no redistribution terms, so the files are fetched rather than bundled.
#   https://coral.ise.lehigh.edu/data-sets/qaplib/
set -u

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATA_DIR=$SCRIPT_DIR/../qap
mkdir -p $DATA_DIR

# Taillard's uniform random a-instances (the hard, flat family), and the
# Nugent and Skorin-Kapov grid instances.
FILES=(
	tai12a tai15a tai17a tai20a tai25a tai30a tai35a tai40a tai50a tai60a tai80a tai100a
	nug12 nug20 nug30
	sko42 sko56 sko72 sko100a
)

for f in ${FILES[@]}; do
	FILEPATH=$DATA_DIR/$f.dat
	if [ -f $FILEPATH ]; then
		echo "File $FILEPATH already exists"
		continue
	fi
	echo "Downloading $f.dat"
	curl -sfL -o $FILEPATH https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/$f.dat || continue
done
