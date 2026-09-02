# Optopus — Codebase Guide

A metaheuristic optimization library for combinatorial problems, written in Rust.

**Design philosophy:** three orthogonal concerns kept strictly separate:
- **Problems** — what to optimize (MaxCut, QUBO, MaxSAT, TSP, VertexCover, JobShop, VRP, custom formula)
- **Heuristics** — how to search (LocalSearch, SA, TabuSearch, GA, RlSearch, …)
- **SearchState** — iteration count, timing, RNG, current and best solutions

Any heuristic works with any problem; no problem-specific code lives in the heuristic layer.
CLI entry `src/main.rs`: TOML config → benchmark run → TOML output (via `BenchmarkReport::write_to_dir`).

## Library Concept (3 use cases)

1. **Existing problem × existing heuristic** — run `LocalSearch`, `SimulatedAnnealing`, `TabuSearch`, etc. on MaxCut / QUBO / SAT / TSP / VertexCover / JobShop / VRP in a few lines via `use optopus::prelude::*`.
2. **Apply existing heuristics to a new problem** — implement three traits (`ProblemTrait`, `Rankable` — on `Solution` **and** on the move, two separate impls — and `MoveToNeighbor`) and LocalSearch / RandomWalk / BeamSearch plus every meta-heuristic work as-is. The rest unlock one trait at a time: `Evaluate<f64>` for SA/LAHC/RlSearch, `EnabledTabu` (+ `Clone`, + the one-line `MoveToNeighbor::tabu_policy` override) for TabuSearch, `Distance` + a `Crossover` for GA, `SubProblemExtractable` for `SubProblemBasedCrossover`; `BinaryProblem` reuses the generic binary machinery in `src/common/`. Registering the problem with the **benchmark** is the strict case — `ConfigNeighbor` bundles `MoveToNeighbor + Rankable + Evaluate + EnabledTabu + Clone` and `ConfigurableProblem` requires `Solution: Distance`, so there is no partial registration.
3. **Combine heuristics and run benchmarks** — compose components with `Sequential` / `Iterated` / `Restart` / `GeneticAlgorithm`, write a TOML config, and get aggregated best/avg/worst/std/time results.

## Extension recipes

**Add a new problem to the benchmark (3 sites, all small):**
1. `ProblemKind` variant in `src/benchmark/config.rs`
2. `with_problem` arm in `src/benchmark/problems.rs`
3. One impl block in `src/benchmark/problems.rs`: `BenchmarkProblem` (load_instance) + `BenchmarkSolution` (objective/encode) + `ConfigurableProblem` (`NAME`, `MINIMIZE`, `VALID_NEIGHBORS`, `with_neighbor` registry, optional `build_special_heuristic`, `build_crossover`)

Plus the library side: `src/problem/<name>/{mod,problem,neighbor,crossover}.rs` (private mods + `pub use`), re-exports in `src/problem/mod.rs` (all types including the crossover) and `src/prelude.rs` (problem / solution / neighbor types; most crossovers are exported only from `problem/mod.rs`). Each move's `MoveToNeighbor` impl also needs `fn tabu_policy(&self) -> Option<&dyn EnabledTabu> { Some(self) }` next to its `EnabledTabu` impl — without it the move compiles and silently has no tabu list; `trait_defs/tabu.rs` has the test that pins every built-in move.

**Add a new base metaheuristic:** implement `Heuristic<P>` in `src/heuristic/<name>.rs`, re-export via `heuristic/mod.rs` + prelude, then add one `HeuristicConfig` variant in `src/benchmark/config.rs` and follow the compile errors (one arm in `BaseBuilder::visit` in `src/benchmark/factory.rs`). The base-heuristic dispatch is written once, not per problem.

## Module Map

```
src/
├── lib.rs / main.rs / prelude.rs / error.rs (OptError)
├── benchmark/
│   ├── mod.rs                public re-exports
│   ├── config.rs             ProblemKind, NeighborKind, HeuristicConfig (tagged enum),
│   │                         StopConditionConfig, BenchmarkConfig, validate_config
│   ├── factory.rs            ConfigNeighbor, NeighborVisitor, ConfigurableProblem trait,
│   │                         BaseBuilder, build_heuristic (the single generic factory)
│   ├── problems.rs           ALL per-problem registration + with_problem (ProblemVisitor)
│   ├── runner.rs             Benchmark::run_from_config, run loop, per-run seed derivation
│   └── report.rs             SingleRunResult / Summary / BenchmarkReport (+ write_to_dir)
├── search_state/
│   └── mod.rs                SearchState<'a, P>, SearchStateCloneType
├── trait_defs/               core traits (re-exported via search_state & prelude)
│   ├── rankable.rs           Rankable, rank_cmp, filter_best, Distance
│   ├── problem.rs            ProblemTrait
│   ├── neighbor.rs           MoveToNeighbor
│   ├── evaluate.rs           Evaluable, Evaluate
│   ├── crossover.rs          Crossover, SubProblemExtractable
│   ├── tabu.rs               EnabledTabu (object safe on purpose: the state holds
│   │                         a move's policy as &dyn EnabledTabu)
│   ├── binary.rs             BinaryProblem (unlocks the shared binary machinery)
│   └── reduction.rs          ProblemReduction: P1 -> P2 instance map + the
│                             solution map both ways. Pure — the crossing lives
│                             on SearchState (open_reduction / close_reduction)
├── common/                   shared data structures & helpers (put new shared code here)
│   ├── graph/                Graph (used by MaxCut / VertexCover); mod.rs = Graph
│   │                         + load_from_file / write_to_file, generator.rs =
│   │                         Graph::{erdos_renyi, barabasi_albert, watts_strogatz}
│   │                         + Graph::{grid_torus_2d, grid_torus_3d} (periodic
│   │                         lattices, degree 4 / 6, deterministic — the topology
│   │                         the planted tile instances live on)
│   │                         (unweighted) + .with_random_weights() + seeded_rng
│   ├── binary.rs             uniform_binary_crossover, hamming_distance,
│   │                         lift_binary_solution / lift_compact_binary_solution,
│   │                         apply_swap_as_two_flips
│   ├── tabu.rs               TabuKey (Var / Pair / Triple — the three key shapes
│   │                         every move uses) + TabuMemory (the one store
│   │                         SearchState holds: a dense Vec for Var, a HashMap
│   │                         for the rest, plus the tenure; is_enabled/forbid)
│   ├── epoch_marks.rs        EpochMarks (index set with an O(1) clear, for
│   │                         neighborhood walks that need a fresh "seen" set per call)
│   ├── permutation.rs        order_crossover (OX; shared by VRP + HGS)
│   ├── gain_index.rs         GainIndex (improving-move index)
│   └── parse.rs              InstanceLines (file-loader scaffold with FileLoad errors)
├── heuristic/
│   ├── mod.rs                Heuristic trait, StopCondition
│   ├── local_search.rs / simulated_annealing.rs (+BangBang) / tabu_search.rs
│   ├── late_acceptance.rs    LateAcceptanceHillClimbing<N>
│   ├── beam_search.rs / random_walk.rs
│   ├── sequential.rs         Sequential<P>, Iterated<P>  ← ILS lives here too
│   ├── variable_neighborhood_search.rs  VariableNeighborhoodSearch<P> (basic VNS)
│   ├── restart.rs            Restart<P>
│   ├── genetic_algorithm.rs  GeneticAlgorithm<P, C>, ParentSelection
│   ├── crossover.rs          SubProblemBasedCrossover<P>
│   ├── reinforcement_learning/  RlSearch<N> (REINFORCE policy over move features)
│   └── specific/            one directory per problem once it has several
│       ├── max_cut/
│       │   ├── ops/            the shared operators, one module per role, each a
│       │   │                    free fn over the SearchState's tabu memory:
│       │   │                    mod.rs (the keep_best tie rule),
│       │   │                    tabu_walk.rs, perturbation.rs
│       │   │                    (random_flips, best_swap); the descent is the
│       │   │                    generic LocalSearch, not an operator here
│       │   ├── bls.rs           BreakoutLocalSearchForMaxCut (+ its BlsSchedule);
│       │   │                    also exposes the round in halves — descend /
│       │   │                    kick / externally_driven — for a caller that
│       │   │                    replaces the schedule (examples/rl_bls.rs)
│       │   └── population_annealing.rs  PopulationAnnealingForMaxCut
│       ├── vrp/
│       │   ├── ops/            the shared route machinery: mod.rs = the pricing
│       │   │                    free fns (removal_gain / insertion_cost /
│       │   │                    segment_demand / route_loads / the three
│       │   │                    excess deltas), route_state.rs = RouteState
│       │   │                    (routes + loads + distance/excess + position
│       │   │                    indexes), granular.rs = build_neighbor_lists,
│       │   │                    descent.rs = Descent (candidate lists + sweep
│       │   │                    buffers; run / run_around)
│       │   ├── alns.rs          AdaptiveLargeNeighborhoodSearchForVrp
│       │   └── hgs/             HybridGeneticSearchForVrp (mod.rs = driver +
│       │                        adaptive penalty, population.rs = biased fitness)
│       ├── lkh_for_tsp.rs       LinKernighanHelsgaunForTsp
│       └── walksat_for_sat.rs   WalkSatForSat
└── problem/
    ├── max_cut/              MaxCut, MaxCutSolution, {Flip,Swap}Neighbor, UniformCrossover,
    │                         MaxCutKernel (kernel.rs: exact data reduction; it is
    │                         the library's one `ProblemReduction` impl, and is used
    │                         by crossing to it, not by a heuristic wrapper),
    │                         PlantedMaxCut (planted.rs: instances whose optimum is
    │                         exact by construction — tile / Wishart planting)
    ├── qubo/                 Qubo, QuboSolution, {Flip,Swap}Neighbor, UniformCrossover
    ├── sat/                  Sat, SatSolution, {Flip,Swap}Neighbor, UniformCrossover
    ├── tsp_2d/               TspWithCoordinates, TspSolution, {TwoOpt,Relocate}Neighbor, OrderCrossover
    ├── vertex_cover/         VertexCover, VertexCoverSolution, {Flip,Swap}Neighbor, UniformCrossover
    ├── job_shop_scheduling/  JobShopScheduling, JobShopSolution, {Swap,Relocate}Neighbor, PpxCrossover
    ├── vrp/                  Vrp, VrpSolution, {Relocate,Swap,TwoOpt}Neighbor, OrderCrossover,
    │                         split.rs = split_giant_tour (Prins' Split DP),
    │                         adjacency.rs = RouteAdjacency (who each customer is
    │                         served between; the broken-pairs count behind
    │                         `Distance for VrpSolution` and HGS's diversity rank)
    └── binary_optimization/  FormulaProblem, Expr, Formula{Flip,Swap}Neighbor, FormulaUniformCrossover
```

## Core Traits (`src/trait_defs/`)

These live in `src/trait_defs/` and are re-exported via `crate::search_state::*` and the `prelude`. Internal code imports them from `crate::trait_defs`; the `search_state` re-export is kept for the public API.

- **`Rankable`**: `is_better_than(&Self) -> bool`. Implemented **twice**: on every `Solution` (the optimization direction is baked into the problem) and on every move type, where it ranks candidate moves against each other — LocalSearch / BeamSearch / RandomWalk / TabuSearch select with `max_by(rank_cmp)` over the latter and do not compile without it. The `filter_best(iter)` helper returns the set of tied-best items.
- **`ProblemTrait`**: `type Solution: Clone + Rankable; fn new_solution(&self, rng) -> Solution`.
- **`MoveToNeighbor<P>`**: a single one-step move.
  ```rust
  fn iter(prob, sol) -> impl Iterator<Item = Self> + Send;     // lazy
  fn apply_to_solution(&self, prob, sol) -> Result<(), OptError>;
  fn move_to_be_better_than(&self, prob, src, other) -> bool;  // default: clone + apply
  fn random_neighbor(prob, sol, rng) -> Option<Self>;          // default: reservoir over iter()
  fn apply_to_iteration(&self, iter: u64) -> u64;              // default: iter + 1
  fn tabu_policy(&self) -> Option<&dyn EnabledTabu>;           // default: None = no tabu policy
  ```
  The two defaults are slow-but-correct and emit a one-shot `tracing::warn!` when hit;
  every built-in move overrides both (O(1) gain compare; direct O(1)/O(n) sampler used
  each step by SA / LAHC / RandomWalk).
  Every built-in move except `FormulaProblem`'s also has an inherent
  `new(prob, sol, indices…)` that computes the cached `gain` (and any other cached delta).
  It is the only correct way to build a move by hand — `apply_to_solution` trusts those
  caches and updates the solution's objective from them without recomputing — so
  `iter` / `random_neighbor` route through it, with two documented exceptions where a
  hot loop hoists shared work: TSP relocate (removal gain per `pos`) and JobShop
  (one scratch buffer for the whole neighborhood scan). Moves whose contract cannot be
  met are rejected with a panic (`# Panics`), not silently mis-evaluated: VRP relocate /
  swap are inter-route only, VRP 2-opt needs `p < q`, TSP relocate rejects the two
  no-op insertion points.
- **`Evaluable<T>` / `Evaluate<T>`** (default `T = f64`): `Maximize(T)` / `Minimize(T)` carries the direction of an objective delta. `Evaluable<f64>::worsening_amount()` normalizes both directions to "positive = worse" (used by `boltzmann_accept`). Required for SA / LAHC / RlSearch. QUBO also exposes `Evaluate<Coefficient = i32>` for integer gains.
- **`Crossover<P>`**: `crossover(&mut self, prob, sol1, sol2, rng) -> Result<Solution, OptError>` (exactly two parents; RNG passed in for reproducibility; `Err` only when the operator genuinely cannot produce an offspring, e.g. an inner sub-heuristic failed).
- **`EnabledTabu`**: `is_move_enabled(&TabuMemory, iter)`, `add_to_tabu_map(&mut TabuMemory, iter, rng)`. Required by TabuSearch. The move decides its **policy** — which keys must be free, which applying it forbids, and whether those are the same set (`VrpRelocateNeighbor`'s are deliberately not: it asks about the destination route and forbids the source, so a customer cannot be moved straight back). It does *not* decide where the prohibitions live: that is the one `TabuMemory` on the state. **The trait is deliberately object safe** — no associated type, no `Clone` supertrait — which is the whole reason there is one trait here and not three: the state is not generic over the move, so it reaches the policy as `&dyn EnabledTabu`. Implementing it is **half** the job; the move also overrides `MoveToNeighbor::tabu_policy` with `Some(self)` (one line). Leaving that default `None` means "no tabu policy": `apply` records nothing and still succeeds, while `state.record_tabu` / `require_tabu_policy` return `OptError::Unsupported` — which is what keeps tabu opt-in for a new problem, and what stops a `TabuSearch` silently degrading into a plain descent.
- **`SubProblemExtractable`**: `extract_sub_problem(sol1, sol2) -> Self`, `lift_solution(sol1, sol2, sub_sol)`. Variables that agree in both parents are fixed; the disagreeing variables form the sub-problem. Binary problems delegate lifting to `common::lift_binary_solution` (shared index space: MaxCut, VertexCover) or `common::lift_compact_binary_solution` (compacted indices: SAT, Formula); QUBO keeps its own bias-folding variant.
- **`ProblemReduction`**: a map from one problem instance to another, with solutions crossing both ways. **Pure — it never touches a `SearchState`.**
  ```rust
  type Source: ProblemTrait;  type Target: ProblemTrait;
  fn target(&self) -> &Self::Target;                                      // P1 -> P2
  //                                         P1.Solution -> P2.Solution
  fn project(&self, sol: &<Self::Source as ProblemTrait>::Solution)
      -> <Self::Target as ProblemTrait>::Solution;
  //                                         P2.Solution -> P1.Solution
  fn lift(&self, source: &Self::Source,
          base: &<Self::Source as ProblemTrait>::Solution,
          sol: &<Self::Target as ProblemTrait>::Solution)
      -> <Self::Source as ProblemTrait>::Solution;
  ```
  The qualified spelling is deliberate: an alias for it would be used at these
  five positions and nowhere else — impls and callers name concrete types — so
  it would be an exported name buying nothing.
  `lift` takes `base` because the target need not cover the source's whole index space — uncovered positions keep `base`'s value; it takes `source` because `MaxCutKernel` is stored in a heuristic that outlives any one `run_once` and so cannot borrow the instance. Implemented by `MaxCutKernel` (`Source = Target = MaxCut`); the two associated types are separate because a reduction need not stay inside its problem. Exactness is **not** required by the trait — it is required by each user, and stated on the implementation (`MaxCutKernel`'s `kernel_cut(y) + offset == original_cut(lift(y))` for every `y`).

  **Running a heuristic on the target and folding the result back is a SearchState operation**, and lives there (see below): `open_reduction` / `close_reduction`. There is deliberately **no heuristic wrapper** around that pair (see the kernelization entry below for why the one that existed was deleted); a caller writes the loop, and `tests/reduction_crossing.rs` is that loop with its trajectory pinned.
- **`BinaryProblem`**: `type Flip`, `variable_indices()`, `variable(sol, i)`, `flip_move(sol, i)` — implemented by all binary problems; unlocks the shared machinery in `src/common/binary.rs`.

## SearchState (`src/search_state/mod.rs`)

```rust
pub struct SearchState<'a, P: ProblemTrait> {
    pub instance: &'a P,
    pub solution: P::Solution,        // current
    pub best_solution: P::Solution,   // global best
    pub initial_solution: P::Solution,
    pub iteration: u64, pub best_iteration: u64,
    pub best_time: Instant,
    pub n_accepted: u64, pub n_rejected: u64, pub n_best_updates: u64,
    pub rng: SmallRng,                // ALL randomness flows through this
    pub trajectory: Vec<TrajectoryPoint>,  // anytime curve; empty unless a probe is set
    pub(crate) start_iteration/start_time: ...,  // where the phase began — hands off
    tabu: TabuMemory,                 // private: erased maps, reached through a move
}
```

**Key methods**: `new(problem)`, `new_with_seed(problem, seed)`, `with_solution(problem, sol)` / `with_solution_and_seed(problem, sol, seed)` (warm start), `apply(neighbor)` (apply + iter + best update), `apply_move_only(neighbor)` (defer best update), `update_best()`, `progress_iteration()`, `random_neighbor::<N>(context)` (uniform random move or `InvalidState` error), `clone_for_new_run(kind)` / `update_state(sub)` (the sub-run triad below), `is_neighbor_better_than_{current,best}(n)`, the tabu API below, `duration()` / `iterations_this_run()` (the elapsed-budget pair a heuristic normalizes progress by; both measure from *this* run's start, not the parent's), `set_objective_probe(fn(&Solution) -> f64)` (installs the probe that makes `update_best` append a `TrajectoryPoint`; off by default so `update_best` stays allocation-free — `benchmark/runner.rs` is the one caller, and it turns the result into the report's `trajectory`).

**Tabu memory** — `apply` / `apply_move_only` record the applied move in the state's tabu memory, at the iteration it was made on, before the counter advances. That is why no heuristic pairs `record` with `apply` by hand any more, and why operators that must respect each other's prohibitions get that from sharing a state rather than from being handed the same object. `set_tabu_tenure((min,max))` (default `(0,0)` = record but forbid nothing) / `tabu_tenure()` / `reset_tabu()` / `tabu_allows(&mv) -> bool` (typed on the move, so `is_move_enabled` inlines into a neighborhood scan and a move without tabu cannot compile a call) / `record_tabu(&mv) -> Result<()>` / `require_tabu_policy(&mv) -> Result<()>` (the two erased ones, where a missing `tabu_policy` becomes an error — `TabuSearch` spends one `require_tabu_policy` per iteration so that mistake cannot pass silently) / `reserve_tabu_vars(n)` (pre-allocation only). `TabuMemory` keys by `TabuKey` shape, so MaxCut's flip and swap (both `Var`) share prohibitions — the property BLS is built on — while JobShop's swap (`Var`) and relocate (`Pair`) do not. `clone_for_new_run` gives the child an **empty** memory under every clone type (a phase is its own tabu list), and `update_state` carries the sub-run's prohibitions *back* into the parent along with its solution — a replacement, not a merge, and sound only because parent and child share one iteration frame. `sub.inherit_tabu_from(&parent)` is the opposite direction, for a phase that should start from what the parent knows.

**Crossing to another instance** — a sub-run on a *different* problem instance (an exact kernel) cannot go through `update_state`, so it has its own pair. A caller that wants to search a reduction writes the loop itself (`tests/reduction_crossing.rs` is that loop, pinned); these two methods are the part that must not be hand-written:
- `open_reduction(&reduction) -> SearchState<R::Target>` — projects the incumbent as the warm start and draws the sub-state's seed from this state's RNG (exactly one draw). Any `ProblemTrait`.
- `close_reduction(&reduction, &sub)` — merges the sub-run's counters, then installs `reduction.lift(..., &sub.best_solution)` as the current solution and `update_best`. Also any `ProblemTrait`; neither half needs to know the variables are binary.

The two steps inside `close_reduction` are one method because their **order** is load-bearing and getting it wrong is invisible in the objective: merging the counters after installing the solution records a `best_iteration` that omits the sub-run's work. Installing costs nothing and is not charged — `lift` returns a complete `Solution`, caches included. (It used to *walk* there one flip per differing variable, on the theory that this avoided an `O(m)` rebuild and preserved the improving-move index; both were false — `lift` already rebuilds, and `new_from_assignment` starts the optional indexes disabled — so all the walk did was charge `iteration` for moves no search made.)

**Reproducibility**: all randomness (initial solutions, move selection, tabu tenures, BLS perturbations) flows through `state.rng`. `clone_for_new_run` forks the RNG so meta-heuristic composition stays deterministic under a fixed seed.

**Sub-run clone/merge pattern** (used by every meta-heuristic to isolate phases):
```rust
let mut sub = state.clone_for_new_run(SearchStateCloneType::ClearBest);
inner_heuristic.run(&mut sub)?;
state.update_state(sub);
// Simple    — keeps the best and the clocks (sets start_iteration to current)
// ClearBest — resets best and timers to current state (start_iteration = iteration)
// StartBest — restarts from best_solution              (start_iteration = iteration)
// All three keep the parent's iteration frame: the counter runs on and
// start_iteration is the anchor every budget is measured against. That is what
// lets tabu boundaries (absolute iterations) cross between parent and sub-run.
// The n_accepted / n_rejected / n_best_updates counters measure a phase instead,
// so *every* variant starts them at zero and update_state adds them straight
// on — which is why the state carries no start_n_* anchors.
```

## Heuristic Algorithms (`src/heuristic/`)

**`Heuristic<P>` trait**: `clear()`, `stop_condition() -> &StopCondition`, `run_once(state)`, `run(state)` (default: `clear` → loop `run_once`). `is_done(state)` has a default that delegates to `stop_condition()`; heuristics with extra termination logic (LocalSearch, LKH: "stop at a local optimum") override it on top.

**Conventions**: constructors take `stop_condition` first; invalid arguments panic with a `# Panics` doc section; `PhantomData` fields are named `_neighbor`.

**`StopCondition`** (builder; stops when *any* condition is met):
```rust
StopCondition::iterations(1_000_000)
    .with_duration(Duration::from_secs(30))
    .with_failed_updates(10_000)
// also: StopCondition::duration(d), StopCondition::failed_updates(n)
```

### Base
| Type | Description |
|---|---|
| `LocalSearch<N>` | Greedy best-improving; halts at a local optimum |
| `SimulatedAnnealing<N>` | Random neighbor, `exp(-Δ/T)` acceptance, multiplicative cooling (requires `Evaluate<f64>`) |
| `BangBangSimulatedAnnealing<N>` | Oscillating temperature between `min_wave_threshold` and `max_wave_threshold` |
| `LateAcceptanceHillClimbing<N>` | LAHC: accepts a move if it is no worse than the score `history_length` steps ago (requires `Evaluate<f64>`) |
| `TabuSearch<N>` | Best non-tabu neighbor; aspiration overrides tabu when global best is improved; tenure ∈ `(min, max)` sampled from `state.rng`. Holds the tenure only — the map is on the state, and `apply` records |
| `RandomWalk<N>` | Uniform random move with unconditional acceptance (useful as a perturbation) |
| `BeamSearch<P, N>` | Maintains top-`k` candidates; expands the full neighborhood of every beam member each iteration |
| `RlSearch<N>` | REINFORCE policy-gradient move selection over hand-crafted move features; weights persist across episodes (requires `Evaluate<f64>`) |

### Meta
| Type | Description |
|---|---|
| `Sequential<P>` | Runs a `Vec<Box<dyn Heuristic<P>>>` in order (each step on a `ClearBest` clone) |
| `Iterated<P>` | ILS: alternates `search` and `perturbation` (`Box<dyn Heuristic<P>>`); lives in `sequential.rs` |
| `VariableNeighborhoodSearch<P>` | Basic VNS: shake in `N_k` → local `search`; improvement resets `k`, failure restores the incumbent and advances `k` (wrap-around) |
| `Restart<P>` | Runs the inner heuristic; when `restart_condition` triggers, replaces `solution` with a fresh random one (best is preserved) |
| `GeneticAlgorithm<P, C>` | 2-parent selection (`Tournament` or `DistantTopK`) → `Crossover<P>` → mutation (`Heuristic<P>`) → worst-replacement; tracks `best_idx` incrementally |

### Crossover
- `SubProblemBasedCrossover<P>` (`crossover.rs`): builds a sub-problem from disagreeing variables, solves it with `sub_heuristic`, then lifts the result. Requires `P: SubProblemExtractable`.
- `*UniformCrossover`: per-variable random parent selection; all binary problems delegate to `common::uniform_binary_crossover`.
- `TspOrderCrossover` (OX) for TSP; `JobShopPpxCrossover` for JobShop.

### Problem-specific
MaxCut has its own directory (`specific/max_cut/`) because its heuristics share *operators* rather than merely a problem type. They live in `ops/` (private to that directory), one module per role, and each is a **free function over the `SearchState`** (which is where the prohibitions live): `tabu_walk` and the two kicks in `perturbation.rs` (`random_flips`, `best_swap`). **The descent is not among them** — BLS descends with the generic `LocalSearch`; `ops::descent` was folded into it at a measured cost of -40.0 average cut over a ten-instance G-set panel, recorded in `docs/heuristics/breakout_local_search.md`. `ops` names no kick and has **no dispatcher**: the selection vocabulary lives in `bls.rs` (`PerturbationType`, re-exported as `MaxCutPerturbation`) together with the one match that maps it to an operator, `BreakoutLocalSearch::kick` — the one place that takes the vocabulary from outside. That is what keeps each operator module independent of the others. `ops/mod.rs` holds only the `keep_best` tie rule they all select with.

There is deliberately no engine object. What the operators share is the tabu memory of the `SearchState` they are handed — recorded by `apply` itself, and shared between flips and swaps because both key on `TabuKey::Var`. In BLS the entries the descent writes are the ones the weak perturbations must not undo; a caller wanting them isolated runs the phases on separate states. Everything genuinely BLS-specific (the `omega`/`l` schedule, the Benlic & Hao selection rule) stays in `bls.rs`.

What no operator decides is what "tabu" *means*. Each marks and tests moves through `MaxCutFlipNeighbor`'s and `MaxCutSwapNeighbor`'s own `EnabledTabu` impls (via `state.tabu_allows` and the record inside `apply`), so they forbid exactly what a generic `TabuSearch` over the same neighborhood would; they only decide *which* moves to try. `TabuSearch` records through the same path.

- `BreakoutLocalSearchForMaxCut` (`specific/max_cut/bls.rs`): greedy local search plus adaptive perturbation (strong / weak flip / weak swap), with probabilities decaying via the non-improvement counter `omega`. It selects exactly the three operators Benlic & Hao define, which are also the only three `ops` offers. Reproduces Benlic & Hao (2013); `docs/heuristics/breakout_local_search.md` records where it does not and why.
- **The learned perturbation controller is an example, not a heuristic**: `examples/rl_bls.rs` drives `BreakoutLocalSearchForMaxCut` through `descend` / `kick` (constructed with `externally_driven`, which takes the tenure literally instead of as the paper's γ) and picks perturbation type (3 ops) × strength with a contextual softmax bandit (`reinforcement_learning::bandit::SoftmaxBandit`); weights persist across `Restart`/`Iterated` episodes. It was `RlBreakoutLocalSearchForMaxCut` with a benchmark `kind` until 2026-08-31: everything it added to BLS was policy (a feature vector and an action decode), so it moved out whole, bit-identical (verified by replaying a pre-move benchmark run's per-run seed through the example). Two objective-preserving *plateau* operators (flip connected clusters / an independent set of zero-gain vertices) used to be a fourth and fifth action, plus a `plateau_width` context feature. **They were removed knowing they paid**: the A/B at 30s x 5 runs costs it **-96.2 and -62.6 on G55 (two seeds), -110.4 on G60, -92.2 on G63**, against std 9-28, and is neutral on G70 (+5.8), G11 and G1 — with them it beat BLS on G55 (10200.4 against 10168.0), without them it does not. The removal was a deliberate trade of that objective for a smaller action space, one operator vocabulary and no second scratch structure; take it back if this controller becomes the thing that has to win. The mechanism itself survives as `PopulationAnnealingForMaxCut`'s non-local cluster move, which owns its own implementation and keeps the opt-in `zero_gain` index alive. See `docs/guide/learned_perturbation.md`.
- **Kernelization is not a heuristic** and has no wrapper type: `MaxCutKernel::new` (`problem/max_cut/kernel.rs`) produces an **exactly reduced** instance — isolated / pendant / degree-2-path / weight-domination rules (arXiv:1905.10902), to a fixpoint, with a trace that lifts a kernel solution back — and any `Heuristic<MaxCut>` searches it through `open_reduction` / `close_reduction`. `KernelizedSearchForMaxCut` used to wrap that loop and was deleted once `ProblemReduction` existed: it held nothing else, which `tests/reduction_crossing.rs` shows by reproducing its exact trajectory. Reduction is a property of the instance, not of tuning: sparse graphs shrink (G70 8646→2164, a tree to **0** vertices = exactly solved), regular and dense graphs do not shrink at all and `is_trivial()` says so in one comparison, so the crossing can be skipped. Searching the kernel beats searching the original on 9/9 sparse instances (BLS +5673 total); rules are validated by exhaustive brute force on `n ≤ 9`, not transcribed from the paper. See `docs/problems/max_cut_kernel.md`.
- `LinKernighanHelsgaunForTsp` (`specific/lkh_for_tsp.rs`): LK-style variable-depth moves with candidate lists; stops at a local optimum.
VRP has its own directory (`specific/vrp/`) for the same reason MaxCut does: what ALNS and HGS share is the *route machinery*, not merely the problem type. It lives in `ops/` — the pricing free functions (a route edit's distance and overload deltas), `RouteState`, the granular candidate lists, and `Descent`. `Descent` is the one thing there with a receiver, because it owns the caches (candidate lists + sweep buffers) both heuristics were otherwise keeping a private copy of; the pricing stays free functions because the two callers hold their routes in different containers. What no `ops` item decides is *policy*: when to descend, under which penalty, and over which customers is the caller's, which is exactly why one descent serves a fixed penalty and an adapted one.

- `AdaptiveLargeNeighborhoodSearchForVrp` (`specific/vrp/alns.rs`): ALNS ruin-and-recreate for CVRP. An `AlnsOps` bank (destroy: random / worst / Shaw removal; repair: greedy / regret-2 insertion) with adaptive roulette-wheel operator weights (segment-updated) and SA acceptance; operates directly on `state.solution` like LKH. Each recreated solution is then run through `ops::Descent::run_around`, **anchored at the re-inserted customers** (plus their five nearest partners) under `Vrp::penalty_weight()` — measured at 30s x 5 runs on ten X instances: **-0.38% mean objective, 8/10 improved**, and -0.97% on X-n701 at 60s. Anchoring is what makes it pay: a full sweep per iteration was +0.07% (a wash, and -2% on X-n459), because on a mid-sized instance the ruin's anchors widened by a full Γ=20 candidate list already cover everything. VRP-only, via `build_special_heuristic`.
- `HybridGeneticSearchForVrp` (`specific/vrp/hgs/`): giant-tour GA: OX crossover → `split_giant_tour` (optimal decode) → the shared granular descent (relocate / swap / 2-opt / 2-opt\*, restricted to the Γ nearest partners), over every customer of the offspring. Keeps **feasible and infeasible sub-populations** ranked by *biased fitness* (cost rank + broken-pairs diversity rank), with a capacity penalty retuned to hold the feasible share near `target_feasible`. It deliberately does **not** use the `Vrp*Neighbor` types or `Vrp::penalty_weight()` — those bake in a fixed enormous penalty, which is incompatible with a penalty the search adapts; `Vrp::solution_from_routes` converts back only when writing to the state. It used to beat ALNS by 0.6-2.5pp at equal budget and **no longer does**: at 30s the two are a wash (ALNS ahead on the largest instances, HGS on the mid-sized ones, all differences under 0.7%). The 600s band has not been re-measured since.

## Problem Types (`src/problem/`)

Binary solutions all name the assignment vector `x: Vec<bool>`.

| Problem | Direction | Solution | Neighbors | Crossover | Notes |
|---|---|---|---|---|---|
| **MaxCut** | Max | `x`, `gain: Vec<f32>`, `objective: f32` | Flip / Swap | Uniform | format: `N M / i j w` (1-indexed); optional `positive_gain` / `zero_gain` indexes (advanced; only `zero_gain` has a library consumer) |
| **QUBO** | Min | `x`, `gain: Vec<i32>`, `objective: i32` | Flip / Swap | Uniform | `Coefficient = i32`; `SubProblemExtractable` (bias folding); optional `negative_gain` index (advanced) |
| **MaxSAT** | Max | `x` (0-indexed), `n_satisfied: usize`, `gain: Vec<i64>` | Flip / Swap | Uniform | DIMACS CNF |
| **TSP 2D** | Min | `tour: Vec<usize>`, `objective: f64` | TwoOpt / Relocate | Order (OX) | TSPLIB (EUC_2D / CEIL_2D / ATT / GEO); lazy distance matrix for `n ≤ 2000` (`DIST_MATRIX_MAX_N`), move gains computed on the fly from it |
| **VertexCover** | Min | `x`, `gain: Vec<i32>`, `objective` (penalty-augmented), `cover_size`, `uncovered_edges` | Flip / Swap | Uniform | same edge-list format as MaxCut |
| **JobShop** | Min | `operations: Vec<usize>`, `objective` (makespan) | Swap / Relocate | Ppx | `n_jobs n_machines` header + one job per line |
| **VRP (CVRP)** | Min | `routes: Vec<Vec<usize>>` (fixed `num_vehicles`, depot implicit), `route_loads`, `distance`, `overload`, `objective` (penalty-augmented) | Relocate / Swap (inter-route) / TwoOpt (intra-route) | Order (OX + `split_giant_tour` decode) | CVRPLIB format (EUC_2D, `NODE_COORD`/`DEMAND`/`DEPOT` sections); capacity is soft via `penalty_weight` like VertexCover; lazy distance matrix for `nodes ≤ 2000`; `split_giant_tour` decodes a customer permutation into the distance-optimal route partition; an unspecified fleet is sized by first-fit-decreasing + 10% (**not** `ceil(demand/capacity)`, which is only a lower bound and often infeasible); `Distance` is the broken-pairs count of `RouteAdjacency` (relabeling- and orientation-invariant), symmetrized by taking the larger direction — HGS ranks diversity on the directional count Vidal defines biased fitness on |
| **FormulaProblem** | Configurable (`OptDirection`) | `x`, `score: f64` (always higher-is-better), `gain: Vec<f64>` | Flip / Swap | Uniform + `SubProblemExtractable` | see below; **library-only** (no instance file format, so intentionally absent from `ProblemKind`) |

**FormulaProblem details**: AST `Expr = Const(f64) | Var(usize) | Neg | Add(Vec) | Mul(Vec)` with `+ - * /` operators. Constraints: `Comparison { lhs, rel: ConstraintRel, rhs, penalty_weight }` (Lt / Gt / Le / Ge / Eq) or `Clamp { expr, lo, hi, penalty_weight }`. A pre-compiled polynomial (`CompiledPoly`) gives O(d) gain deltas; `interaction_neighbors` tracks which variables' gains may change on each flip.

## Benchmarking (`src/benchmark/`)

TOML config → `BenchmarkConfig` → run each heuristic on each instance N times (rayon-parallel) → `BenchmarkReport` → timestamped TOML in `result/`.

**MaxCut instance suites.** Besides the G-set, three suites are generated locally from fixed seeds (never committed — regenerating reproduces them byte for byte): `examples/generate_dense_maxcut.rs` (10–30% density, the band the G-set does not reach), `generate_sparse_maxcut.rs` (average degree 1–5, where `MaxCutKernel` fires), and `generate_hard_maxcut.rs` → `data/instances/max_cut/hard/`, whose **optimum is exact by construction** (`PlantedMaxCut`; tile planting on degree-4/6 lattice tori and Wishart planting on the complete graph, following `chook`, arXiv:2005.14344). The planted suite is the only one where a gap is measured against the answer rather than against another heuristic's best result; its optima live in `hard/manifest.toml`, which **nothing in the library reads** — same rule as best-known values, analysis only. Its hardness parameters were fixed by a sweep whose numbers are recorded in `generate_hard_maxcut.rs`; both families' peaks move with instance size, so published values do not transfer.

```toml
num_runs = 10
seed = 42                      # optional: makes every run bit-reproducible
[[instances]]
path = "data/instances/max_cut/G*"   # globs supported (Gset files have no extension)
problem = "MaxCut"             # MaxCut | Qubo | Sat | Tsp | VertexCover | JobShop | Vrp
[[heuristics]]
kind = "LocalSearch"           # see list below
neighbor = "Flip"              # Flip | Swap | TwoOpt | Relocate
[heuristics.stop_condition]
max_iteration = 100000         # max_duration_secs / max_failed_update also supported
```

`HeuristicConfig` is an internally-tagged enum (`#[serde(tag = "kind")]`), so each `kind` declares exactly its own required fields; missing fields and unknown kinds fail at parse time.

**Supported `kind` values**: `LocalSearch`, `TabuSearch` (`tabu_tenure = [min, max]`), `SimulatedAnnealing` (`initial_temperature`, `cooling_rate`), `LateAcceptanceHillClimbing` (`history_length`), `RandomWalk` (give it a `stop_condition` — an empty one never terminates), `RlSearch` (optional `learning_rate` / `softmax_temperature` / `reward_shaping` / `policy_weights` / `max_candidates`; `discount` is still parsed but ignored with a warning — single-step REINFORCE has none), `BreakoutLocalSearch` (MaxCut only; `tabu_tenure`, `t`, `l0`, `p0`, `q`), `PopulationAnnealingForMaxCut` (MaxCut only; `population_size`, optional `initial_beta` / `delta_beta` / `sweeps_per_step` / `reset_period` / `cluster_moves`), `LinKernighanHelsgaun` (TSP only; optional `num_neighbors`, `max_depth`), `WalkSat` (SAT only; optional `noise`, `adaptive_noise`), `AdaptiveLargeNeighborhoodSearch` (VRP only; optional `removal_fraction`, `cooling_rate`), `HybridGeneticSearch` (VRP only; optional `min_population_size` / `generation_size` / `granularity` / `target_feasible` / `restart_generations`), and the meta-heuristics `Sequential` / `Iterated` / `VariableNeighborhoodSearch` / `Restart` / `GeneticAlgorithm` (nested `steps` array; `Iterated` uses `steps[0] = search, steps[1] = perturbation`; `VariableNeighborhoodSearch` uses `steps[0] = search, steps[1..] = shakes N_1..N_kmax`; `Restart` also requires `restart_condition`; GA requires `population_size`, optional `crossover_kind` / `parent_selection` / `parent_top_k`).

**`Summary` fields**: `num_successful_runs`, `best/avg/worst/std_objective`, `best/avg_time_to_best_secs`, `avg_total_time_secs`, plus averaged `initial_objective` / `improvement` / acceptance counters. Each `SingleRunResult` carries `best_objective: f64`, `best_iteration: u64`, timing, the per-run `seed`, `solution: Vec<usize>` (0-indexed encoding), and `trajectory: Vec<(f64, f64)>` — the `(elapsed_secs, objective)` anytime curve, made monotone in the problem's direction, which is what the benchmark viewer plots.

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

## Key Design Patterns

1. **Gain-based incremental updates** — binary/formula solutions cache per-variable `gain`; applying a move only refreshes the affected neighbors in O(degree). MaxCut and QUBO additionally offer optional `positive_gain` / `negative_gain` indexes (advanced) to enumerate only improving moves — offered for callers writing their own descent, and **read by nothing in the library** since BLS's descent became the generic `LocalSearch`; MaxCut's `zero_gain` is the one index with a live consumer (`PopulationAnnealingForMaxCut`). TSP instead computes move gains on the fly from the lazily built distance matrix; JobShop re-decodes per candidate (and evaluates candidates with rayon on large instances, order-preserving so results are thread-count independent).
2. **Sub-run clone/merge** — every meta-heuristic isolates a phase with `clone_for_new_run(kind)` → run it → `update_state(sub)`. The global iteration counter advances monotonically across all phases. There is deliberately no `run_sub` wrapper around the triad: both halves are public, every user-facing doc teaches this form, and a wrapper would have to name `Heuristic`, which `search_state` must not — nothing about cloning and merging a state needs to know what a heuristic is.
3. **Seeded reproducibility** — all randomness flows through `state.rng` (`SmallRng`); `EnabledTabu::add_to_tabu_map` and `Crossover::crossover` take the RNG explicitly. With `seed` set in the benchmark config, reruns are bit-identical (enforced by e2e tests).
4. **Tabu abstraction via trait** — `TabuSearch` is generic over `N: EnabledTabu` and owns only the tenure; the prohibitions are the `SearchState`'s one `TabuMemory`, recorded by `apply` itself. Each move type owns its *policy* (which `TabuKey`s it reads and writes) and hands it over with `MoveToNeighbor::tabu_policy`; the storage is shared, which is what makes two move types over the same key shape see each other's entries.
5. **Always-higher-is-better in `FormulaProblem`** — for `Maximize`, `score = objective − penalty`; for `Minimize`, `score = −objective − penalty`. Heuristics only need the higher-is-better convention.
6. **`MoveToNeighbor::iter()` is lazy** — `LocalSearch` selects with `max_by` in O(n) without collecting; `TabuSearch` uses `max_by` plus aspiration; only `BeamSearch` materializes all candidates.
7. **Config factory is generic, per-problem code is registration-only** — `build_heuristic` + `BaseBuilder` (in `benchmark/factory.rs`) contain the only base-heuristic dispatch; each problem contributes a small `ConfigurableProblem` impl in `benchmark/problems.rs`. Shared code goes in `src/common/`, not in `problem/` or at the top level.
