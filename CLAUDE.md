# Optopus — Codebase Guide

A metaheuristic optimization library for combinatorial problems, written in Rust.

Three orthogonal concerns are kept strictly separate. Problems say what to
optimize, heuristics say how to search, and `SearchState` carries the iteration
count, timing, RNG, and the current and best solutions. Any heuristic works with
any problem, and no problem-specific code lives in the heuristic layer. The CLI
entry `src/main.rs` turns a TOML config into a benchmark run and a TOML report.

## Where to look

This file is the map: where things live, what to touch when adding something,
and the conventions this repo holds itself to. It deliberately does not explain
the algorithms, the traits or the config format, because other places do that better.

| Question | Read |
|---|---|
| What does this heuristic do, and what does it need? | `docs/heuristics/README.md`, then the page it links |
| What does this problem look like, and what is its file format? | `docs/problems/` |
| What do the core traits require? | `docs/traits.md` and `src/trait_defs/` |
| How does `SearchState` work? | `docs/search_state.md` |
| How is a benchmark config written, and what does the report hold? | `docs/guide/benchmarking.md`, and each heuristic page's "Benchmark config" section for its fields |
| Where do the instances come from, and which are generated locally? | `data/instances/README.md` |
| Why is it built this way, and what has already been tried? | `decisions/README.md` |
| How do I use the library? | the rustdoc, and the site built from `docs/` |

`decisions/` holds one file per measured decision with an index. Read that index
before re-proposing anything that looks like an obvious improvement; several of
them were tried and priced already. `docs/` and the rustdoc are for users of the
library and carry none of that material.

## Extension recipes

### Add a new problem to the benchmark (one file, `src/benchmark/problems.rs`)

1. One row in the `problems!` table, `Variant => ProblemType, minimize;`. It makes the `ProblemKind` variant (the TOML spelling, also the name in error messages), the `with_problem` arm, the direction the report reads, and `load_instance`, which calls the type's inherent `load_file(path)`. A test checks that direction against the solution's `Evaluate`, and its `instance` match asks for a small instance of the new kind.
2. One impl block each: `BenchmarkProblem` (decode_indices, the inverse of encode) + `BenchmarkSolution` (encode_as_indices; best_objective_f64 defaults to the solution's `Evaluate` value; is_feasible on a problem whose solutions can break a constraint) + `ConfigurableProblem` (a `neighbors!` table of `NeighborKind => move type`, a `crossovers!` table of `Name => operator` whose first row is the default, optional `build_special_heuristic` / `build_branch_and_bound`, where `ruin_and_recreate` builds GRASP and ALNS on a `Ruinable` problem and `binary_branch_and_bound!()` the binary bound)

Plus the library side: `src/problem/<name>/{mod,problem,neighbor,crossover}.rs` (private mods + `pub use`), re-exports in `src/problem/mod.rs` (all types including the crossover) and `src/prelude.rs` (problem / solution / neighbor types; most crossovers are exported only from `problem/mod.rs`). A move derives `Rankable` with the solution's keys as its changes to them (none when the solution ranks by its objective) and the solution's `tolerance` if it has one, and its `move_to_be_better_than` is `is_better_after_move(self, src, other)`. A move's tabu policy is `#[derive(EnabledTabu)]` with `#[tabu(..)]` keys (by hand only when it checks other keys than it forbids), and its `MoveToNeighbor::tabu_policy`, which has no default, returns `Some(self)`. A built-in move with a policy that returned `None` would run with no tabu list, and `trait_defs/tabu.rs` has the test that pins every one.

The user-facing pages follow in the same PR: a page under `docs/problems/`, its ja page, the nav, and the neighbor table in `docs/guide/benchmarking.md`.

### Add a new base metaheuristic

Implement `Heuristic<P>` in `src/heuristic/<name>.rs`, re-export via `heuristic/mod.rs` + prelude, then add one `HeuristicConfig` variant in `src/benchmark/config.rs` with its name in the `heuristic_kinds!` list (by whether it carries a `neighbor`, `steps` or neither), and follow the compile errors to its arm in `build_heuristic` and `BaseBuilder::visit` in `src/benchmark/factory.rs`. Every match over the kinds names each one, so none of these is missed silently. The base-heuristic dispatch is written once, not per problem. Its page under `docs/heuristics/` carries a "Benchmark config" section, and `docs/guide/benchmarking.md` lists the kind.

## Module Map

Each entry says what a file is for, or where something lives that its name would
not suggest. Type and function names are left to grep and the rustdoc.

```
optopus-derive/               the proc-macro crate of the workspace: the derives of Rankable,
                              Evaluate, Distance and EnabledTabu, each re-exported from
                              optopus::trait_defs beside its trait
src/
├── lib.rs / main.rs / prelude.rs / error.rs
├── benchmark/                config → runs → report, the one layer that dispatches over every
│                             problem
│   ├── config.rs             the TOML schema, with the heuristic_kinds! list
│   ├── factory.rs            build_heuristic, the single generic factory, and the trait a
│   │                         problem registers its neighbors and special heuristics through
│   ├── problems.rs           the problems! table and all per-problem registration
│   ├── runner.rs             the run loop, per-run seeds, the anytime curve
│   └── report.rs
├── search_state/
├── trait_defs/               every trait the problem side implements and a heuristic calls;
│   │                         a trait both implemented and used inside one layer
│   │                         (Heuristic, PerturbationSchedule, ChangeDelta) stays by its type
│   ├── rankable.rs           the one place solutions and moves are compared:
│   │                         is_better_after_move (every move's move_to_be_better_than),
│   │                         key_cmp (the tolerant key comparison), and Distance
│   ├── evaluate.rs           the numeric surrogate, for where a size is needed
│   ├── tabu.rs               EnabledTabu, object safe on purpose, and the test that pins
│   │                         every built-in move's policy
│   ├── branch.rs             what BranchAndBound asks, and its integer and relaxation sides
│   └── problem.rs / neighbor.rs / crossover.rs / binary.rs / int_assignment.rs /
│       ruinable.rs / reduction.rs
├── building_blocks/          what problems and heuristics are built from, in three groups.
│   │                         New shared code goes in the group it belongs to, and code that
│   │                         fits none of them gets its place decided first. Items are
│   │                         reached through their group (building_blocks::instance::Graph)
│   ├── instance/             what an instance is built from and read with: the graph and
│   │                         its generators, the distance store Tsp and Vrp share, parsing
│   ├── representation/       helpers per kind of decision variable
│   │   ├── binary/           crossover, fixing (with the binary_branch_space! macro each
│   │   │                     problem's fix.rs calls) and the swap helpers
│   │   ├── integer.rs        integer variables, and the BranchSpace blanket impl over
│   │   │                     Branchable
│   │   └── permutation.rs    order crossover
│   └── search/               search machinery that knows no problem, plus MIN_IMPROVEMENT
│       ├── tabu.rs           the tabu keys and memory
│       ├── biased_fitness.rs the population ranking GA and HGS share
│       ├── ruin_recreate.rs  destroy and repair over any Ruinable, GRASP's construction too
│       ├── sampling.rs       the endpoints the position moves draw
│       ├── pair_scan.rs      two nested index ranges as plain loops, which the QAP, bin
│       │                     packing relocate and VRP swap / 2-opt scans list with
│       ├── adaptive_weights.rs  the roulette ALNS picks its operators by
│       ├── metropolis.rs     the Metropolis acceptance SA, PT and PA share, and their sweeps
│       └── anchored_sweep.rs the anchor widening and shuffled sweep every LocalRepair
│                             descent runs
├── heuristic/                one file per heuristic, named after it
│   ├── mod.rs                the Heuristic trait and StopCondition
│   ├── sequential.rs         Sequential and Iterated, so ILS lives here too
│   ├── simulated_annealing.rs  BangBangSimulatedAnnealing too
│   ├── crossover.rs          the crossover that solves a sub-problem
│   └── specific/             one directory per problem once it has several: the MaxCut
│                             parts of BLS, LKH, HGS, WalkSat, the ALNS presets of TSP,
│                             VRP and FlowShop, and FlowShop's IG
└── problem/                  one directory per problem with problem, neighbor and crossover.
    │                         The files beyond those:
    ├── max_cut/              kernel.rs (an exact reduction), eigen.rs (a relaxation
    │                         certified by Cholesky), planted.rs (an optimum known by
    │                         construction)
    ├── max_cut/ qubo/ sat/ vertex_cover/
    │                         fix.rs, fixing variables and the branch space
    ├── integer/              IntegerProblem, built from a closure, and formula.rs, the
    │                         expression model, with IntervalRelaxation in it too
    ├── vrp/                  one problem for one vehicle type or several (decisions/0014).
    │                         ops/pricing.rs is the one route-edit price, split.rs splits a
    │                         giant tour
    ├── tsp/ bin_packing/ flow_shop/ vrp/
    │                         ruin.rs, the Ruinable side, and its repair descent on all
    │                         but bin_packing, whose bins open and close instead
    └── flow_shop/            taillard.rs, the O(m) insertion and removal prices
```

## Benchmarking (`src/benchmark/`)

A TOML config becomes a `BenchmarkConfig`, each heuristic runs on each instance
N times (rayon-parallel), and a `BenchmarkReport` is written as timestamped TOML
under `result/`. `HeuristicConfig` is an internally-tagged enum
(`#[serde(tag = "kind")]`), so each `kind` declares its own required fields, and
missing fields or unknown kinds fail at parse time. The config format, every
kind with the problems it applies to, and the report's fields are documented for
users in `docs/guide/benchmarking.md`.

## Documentation site (`docs/` + `mkdocs.yml`)

`docs/` is the source of the Pages site (mkdocs-material). The **API reference is
part of that site**: `cargo doc --no-deps --lib` (the `--lib` avoids the lib/bin
output-filename collision) is copied to `docs/api/` — gitignored, generated —
so every page's `**API:**` line links it with a plain relative path that
`mkdocs build --strict` verifies. Both workflows build it that way; CI's `docs`
job adds `RUSTDOCFLAGS=-D warnings`, so a broken intra-doc link fails the PR.
To build the site locally:

```bash
cargo doc --no-deps --lib && rm -rf docs/api && cp -r target/doc docs/api
mkdocs build --strict --site-dir /tmp/site   # or `mkdocs serve`
```

Every page carries exactly one `**API:**` line, directly under the `# ` title,
holding exactly one link: the rustdoc item that page is about (a problem's
problem type, a heuristic's heuristic type; a module index only on the pages
that survey a whole module — `problems/README.md`, `heuristics/README.md`,
`traits.md`). Sibling types are reached from that item's module page, not from
a second link — a page that needs to point at one from its body links it
inline in the prose, never as another `**API:**` line. Adding a problem or
heuristic means adding that one line to the new page too — `--strict` catches a
wrong path, not a missing line.

`docs/ja/` is the Japanese edition: every page of `docs/` except `benchmarks/`,
at the same relative path, listed under the `日本語` section of the nav. The
rustdoc stays English and is shared, so a ja page's `**API:**` line is the
English one with one more `../`. A change to an English page updates its ja
counterpart in the same PR, and a new page gets a ja page and a nav entry in
both sections. Algorithm and problem names stay in English there. A translated
heading that something links to keeps the English anchor with `{ #id }`, so
links between pages read the same in both editions.
