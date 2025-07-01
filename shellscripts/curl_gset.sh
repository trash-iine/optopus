#!bin/bash
set -u

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for i in {1..81}; do
	FILEPATH=$SCRIPT_DIR/../data/G$i
	if [ -f $FILEPATH ]; then
		echo "File $FILEPATH already exists"
		continue
	fi
	echo "Downloading G$i"
	curl -sf -o $FILEPATH https://web.stanford.edu/~yyye/yyye/Gset/G$i || continue
done
