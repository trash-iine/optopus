# Benchmark Instance Inventory

This document catalogs every benchmark instance file under `data/` used by the
optopus benchmark pipeline (`src/benchmark.rs`), with sources and redistribution
notes. Instances are pure numerical descriptions of optimization problems
(adjacency lists, processing times, CNF clauses, city coordinates).

## QUBO — `data/instances/qubo/`

| Set | Files | Size (n vars) | Source |
|---|---|---|---|
| Beasley OR-Library bqp | `bqp/bqp{50,100,250,500,1000}_{1..10}.txt` (50 files) | 50 / 100 / 250 / 500 / 1000 | [OR-Library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/bqpinfo.html), J.E. Beasley |
| Ad-hoc samples | `sample.txt`, `test_data.txt` | tiny | repo-local |

Conversion: each `bqpN.txt` from OR-Library is a multi-instance text bundle
(first line `n_instances`, then per instance `n m` header + `m` lines of
`i j v`). Split into per-instance files by `data/scripts/split_bqp.py`. The
per-instance format matches `Qubo::load_file` directly.

## JSSP — `data/instances/jssp/`

| Set | Files | Size (jobs × machines) | Source |
|---|---|---|---|
| OR-Library jobshop1 | `orlib/{ft06,ft10,ft20,la01..la40,abz5..abz9,orb01..orb10,swv01..swv20,yn1..yn4,...}.txt` (82 files) | 6×6 up to 50×10 | [OR-Library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/jobshopinfo.html), compiled by D.C. Mattfeld & R.J.M. Vaessens |
| Existing | `ft06.txt` | 6×6 | duplicate of `orlib/ft06.txt`; kept for sample TOMLs |

Conversion: `data/scripts/split_jobshop.py` parses `instance NAME` markers and
extracts each block's `n m` dimensions + the following `n` operation rows.
Output format matches `JobShopScheduling::load_file` (machines 0-indexed,
operations as `(machine, time)` pairs per row).

## SAT — `data/instances/sat/`

| Set | Files | Variables × Clauses | Source |
|---|---|---|---|
| SATLIB uniform 3-SAT (sat) | `satlib/uf{50-218,75-325,100-430,150-645,200-860}/*.cnf` (10 each, 50 files) | 50/75/100/150/200 vars at phase-transition density | [SATLIB](https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html), H.H. Hoos & T. Stützle |
| Ad-hoc | `sample.cnf` | 20 vars | repo-local |

Conversion: tarballs from SATLIB extracted, the first 10 lexically-sorted
`.cnf` files per family copied. DIMACS CNF is directly compatible with
`Sat::load_file`. The `uuf` (unsatisfiable) families are deliberately omitted —
MaxSAT framing makes them equivalent to `uf` for this benchmark suite.

## TSP — `data/instances/tsp/`

| Set | Files | Cities | Source |
|---|---|---|---|
| TSPLIB | `att48.tsp`, `berlin52.tsp`, `burma14.tsp`, `ch150.tsp`, `dsj1000.tsp`, `eil51.tsp`, `eil101.tsp` (7 files) | 14–1000 | [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/), G. Reinelt |
| Ad-hoc | `sample.tsp`, `test_data.txt` | tiny | repo-local |

## MaxCut & VertexCover — `data/instances/max_cut/`

| Set | Files | Vertices | Source |
|---|---|---|---|
| GSET | `G1`..`G81` (varies, 73 present) | 800–20000 | [GSET](https://web.stanford.edu/~yyye/yyye/Gset/), Y. Ye / Stanford |
| Ad-hoc | `sample.txt`, `test_data.txt` | tiny | repo-local |

VertexCover reuses MaxCut graph files via `Graph::load_from_file`
(`src/benchmark.rs:50, 125`).

## Excluded

- `FormulaProblem` (`src/problem/binary_optimization/`) is not wired into
  `ProblemKind` in `src/benchmark.rs`, so no benchmark instance files are
  required. The problem is library-API only.
- OR-Library `bqp2500.txt` returned an HTML error page at fetch time and is
  excluded.

## Licensing / Redistribution

All instance files in this directory are publicly distributed academic
benchmark datasets containing numerical problem descriptions (no creative
content). Each source library encourages redistribution for research use.
Attribution is preserved in this file. When citing benchmark results, please
also cite the originating libraries (Beasley 1998 for OR-Library, Hoos &
Stützle 2000 for SATLIB, Reinelt 1991 for TSPLIB, Helmberg & Rendl 2000 era
for GSET).

## Reproducing the fetch

```sh
# QUBO
for n in 50 100 250 500 1000; do
  curl -L -o /tmp/orlib_bqp${n}.txt "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/bqp${n}.txt"
done
python3 data/scripts/split_bqp.py

# JSSP
curl -L -o /tmp/orlib_jobshop1.txt "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/jobshop1.txt"
python3 data/scripts/split_jobshop.py

# SAT
for spec in uf50-218 uf75-325 uf100-430 uf150-645 uf200-860; do
  curl -L -o /tmp/satlib_${spec}.tar.gz \
    "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/${spec}.tar.gz"
  mkdir -p /tmp/sat_extract/${spec} && tar -xzf /tmp/satlib_${spec}.tar.gz -C /tmp/sat_extract/${spec}
  mkdir -p data/instances/sat/satlib/${spec}
  find /tmp/sat_extract/${spec} -name '*.cnf' | sort | head -10 | xargs -I{} cp {} data/instances/sat/satlib/${spec}/
done
```
