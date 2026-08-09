#!/usr/bin/env bash
# Fetch CVRPLIB "X" instances into data/instances/vrp/.
# CVRPLIB X set (Uchoa, Pecin, Pessoa, Poggi, Vidal & Subramanian, 2017):
# free for academic / non-commercial use; canonical source
#   http://vrp.atd-lab.inf.puc-rio.br/
# Downloaded here from the plain-text mirror github.com/PyVRP/Instances, which
# also ships the best-known solutions as *.sol.
#
# The X set is used because it parses as-is with `Vrp::load_file` (EUC_2D with
# integer rounding), spans n = 100..1000, and carries best-known values. The
# older Augerat/Christofides sets (A/B/E/F/M/P) have no stable plain-text mirror
# and must be fetched by hand from CVRPLIB if you want them.
#
# Note: these files carry no "No of trucks: K" comment, so the fleet size is
# computed rather than read — first-fit-decreasing over the demands plus a 10%
# margin (`default_fleet_size` in src/problem/vrp/problem.rs). It does not match
# the k in the file name, which is the best-known solution's vehicle count:
# first-fit-decreasing alone already needs 26 bins for X-n101-k25.
set -u

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATA_DIR=$SCRIPT_DIR/../vrp
mkdir -p $DATA_DIR

BASE=https://raw.githubusercontent.com/PyVRP/Instances/main/CVRP

# Roughly one instance per size decade in each benchmark band:
# small n <= 200, medium 200 < n <= 500, large n > 500.
FILES=(
	X-n101-k25
	X-n110-k13
	X-n125-k30
	X-n139-k10
	X-n153-k22
	X-n176-k26
	X-n195-k51
	X-n214-k11
	X-n242-k48
	X-n270-k35
	X-n303-k21
	X-n344-k43
	X-n401-k29
	X-n459-k26
	X-n502-k39
	X-n561-k42
	X-n627-k43
	X-n701-k44
	X-n801-k40
	X-n895-k37
	X-n1001-k43
)

for f in ${FILES[@]}; do
	for ext in vrp sol; do
		FILEPATH=$DATA_DIR/$f.$ext
		if [ -f $FILEPATH ]; then
			echo "File $FILEPATH already exists"
			continue
		fi
		echo "Downloading $f.$ext"
		curl -sf -o $FILEPATH $BASE/$f.$ext || continue
	done
done
