# Benchmark Instance Inventory

This document catalogs every benchmark instance file in this directory
(`data/instances/`) used by the optopus benchmark pipeline (`src/benchmark/`),
with sources and licensing. Instances are pure numerical descriptions of
optimization problems (adjacency lists, processing times, CNF clauses, city
coordinates). Paths below are relative to this directory unless noted.

The OR-Library QUBO and JSSP sets are bundled directly in this repository. The
SATLIB, TSPLIB, and GSET sets are **not bundled**; fetch them locally with the
commands in [Obtaining instances not bundled](#obtaining-instances-not-bundled).

## QUBO — `qubo/`

**Bundled in this repository.**

| Set | Files | Size (n vars) | Source |
|---|---|---|---|
| Beasley OR-Library bqp | `bqp/bqp{50,100,250,500,1000}_{1..10}.txt` (50 files) | 50 / 100 / 250 / 500 / 1000 | [OR-Library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/bqpinfo.html), J.E. Beasley |
| Ad-hoc samples | `sample.txt`, `test_data.txt` | tiny | repo-local |

Conversion: each `bqpN.txt` from OR-Library is a multi-instance text bundle
(first line `n_instances`, then per instance `n m` header + `m` lines of
`i j v`). Split into per-instance files by `scripts/split_bqp.py`. The
per-instance format matches `Qubo::load_file` directly.

## JSSP — `jssp/`

**Bundled in this repository.**

| Set | Files | Size (jobs × machines) | Source |
|---|---|---|---|
| OR-Library jobshop1 | `orlib/{ft06,ft10,ft20,la01..la40,abz5..abz9,orb01..orb10,swv01..swv20,yn1..yn4,...}.txt` (82 files) | 6×6 up to 50×10 | [OR-Library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/jobshopinfo.html), compiled by D.C. Mattfeld & R.J.M. Vaessens |
| Existing | `ft06.txt` | 6×6 | duplicate of `orlib/ft06.txt`; kept for sample TOMLs |

Conversion: `scripts/split_jobshop.py` parses `instance NAME` markers and
extracts each block's `n m` dimensions + the following `n` operation rows.
Output format matches `JobShopScheduling::load_file` (machines 0-indexed,
operations as `(machine, time)` pairs per row).

## SAT — `sat/`

**Not bundled — fetch locally (see [Obtaining instances not bundled](#obtaining-instances-not-bundled)).**

| Set | Files | Variables × Clauses | Source |
|---|---|---|---|
| SATLIB uniform 3-SAT (sat) | `satlib/uf{50-218,75-325,100-430,150-645,200-860}/*.cnf` (10 each, 50 files) | 50/75/100/150/200 vars at phase-transition density | [SATLIB](https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html), H.H. Hoos & T. Stützle |
| SAT Competition 2026 (large) | `satcomp2026/*.cnf` (7 files) | 303–115k vars, 1.8e4–2.5e6 clauses | [SAT Competition 2026](https://github.com/satcompetition/2026) selection, files via [GBD](https://benchmark-database.de/) |
| Ad-hoc | `sample.cnf` | 20 vars | repo-local |

Conversion: tarballs from SATLIB extracted, the first 10 lexically-sorted
`.cnf` files per family copied. DIMACS CNF is directly compatible with
`Sat::load_file`. The `uuf` (unsatisfiable) families are deliberately omitted —
MaxSAT framing makes them equivalent to `uf` for this benchmark suite.

The `satcomp2026/` set is a curated large-scale MaxSAT sample. The SAT
Competition 2026 benchmark selection
(`downloads/benchmark-compilation-script/selected_benchmarks.csv` in the repo
above) lists instances by md5 hash; the actual `p cnf` files are fetched from
the Global Benchmark Database (GBD) at `https://benchmark-database.de/file/<hash>?context=cnf`
and xz-decompressed. Seven crafted/combinatorial instances spanning
~1.8e4–2.5e6 clauses were chosen (families: quasigroup-completion, waerden,
sliding-puzzle, hamiltonian-cycle, coloring-mycielski-graph, station-repacking,
coloring). These are unweighted CNF; under the MaxSAT framing the objective is
the number of satisfied clauses (an over-constrained/UNSAT instance simply has
an optimum below its clause count).

## TSP — `tsp/`

**Not bundled — fetch locally (see [Obtaining instances not bundled](#obtaining-instances-not-bundled)).**

| Set | Files | Cities | Source |
|---|---|---|---|
| TSPLIB | `att48.tsp`, `berlin52.tsp`, `burma14.tsp`, `ch150.tsp`, `dsj1000.tsp`, `eil51.tsp`, `eil101.tsp` (7 files) | 14–1000 | [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/), G. Reinelt |
| Ad-hoc | `sample.tsp`, `test_data.txt` | tiny | repo-local |

## MaxCut & VertexCover — `max_cut/`

**Not bundled — fetch locally (see [Obtaining instances not bundled](#obtaining-instances-not-bundled)).**

| Set | Files | Vertices | Source |
|---|---|---|---|
| GSET | `G1`..`G81` (varies, 73 present) | 800–20000 | [GSET](https://web.stanford.edu/~yyye/yyye/Gset/), Y. Ye / Stanford |
| Ad-hoc | `sample.txt`, `test_data.txt` | tiny | repo-local |

VertexCover reuses MaxCut graph files via `Graph::load_from_file`
(see `src/benchmark/problems.rs`).

## Excluded

- `FormulaProblem` (`src/problem/binary_optimization/`) is not wired into
  `ProblemKind` in `src/benchmark/config.rs`, so no benchmark instance files
  are required. The problem is library-API only.
- OR-Library `bqp2500.txt` returned an HTML error page at fetch time and is
  excluded.

## Licensing

- The OR-Library data bundled here (QUBO `bqp/`, JSSP `orlib/`) is redistributed
  under the **MIT License** (© 2010 J E Beasley). Full license text is included
  in `qubo/NOTICE` and `jssp/NOTICE`.
  Source: <https://people.brunel.ac.uk/~mastjjb/jeb/orlib/legal.html>
- The SATLIB, TSPLIB, and GSET instances are **not bundled** in this repository.
  Obtain them from their original sites (see below) and follow each site's own
  terms of use.
- The tiny `sample.*` / `test_data.*` files in each problem directory are
  original to this repository.
- When publishing results, please also cite the originating libraries:
  Beasley 1990 (OR-Library), Hoos & Stützle 2000 (SATLIB), Reinelt 1991
  (TSPLIB), Helmberg & Rendl 2000 (GSET).

## Obtaining instances not bundled

Each non-bundled set has a fetch script under `scripts/`. The scripts are
idempotent (existing files are skipped) and write into the matching data
directory. Run them from this directory (`data/instances/`):

```sh
bash scripts/fetch_sat.sh          # SATLIB uniform random 3-SAT (uf*)
bash scripts/fetch_satcomp2026.sh  # SAT Competition 2026 large instances (via GBD)
bash scripts/fetch_tsp.sh          # TSPLIB symmetric instances
bash scripts/fetch_maxcut.sh       # GSET graphs
```

The downloaded files stay out of version control via each directory's
`.gitignore`; the fetch scripts themselves are tracked.

### Regenerating the bundled OR-Library files (optional)

The QUBO and JSSP sets are already bundled; these scripts download the upstream
OR-Library bundles and split them back into the per-instance files in place.

```sh
bash scripts/fetch_qubo.sh  # → qubo/bqp/
bash scripts/fetch_jssp.sh  # → jssp/orlib/
```
