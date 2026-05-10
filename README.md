# Optopus

Optopus is a metaheuristic optimization library for combinatorial problems.
It provides a uniform interface for applying local search, tabu search, simulated annealing, beam search, and other algorithms to MaxCut, QUBO, SAT (MaxSAT), TSP, and user-defined problems.

## Quick Start

If you want to confirm the library works before reading traits or internals:

```bash
cargo run --example max_cut
```

```rust
use optopus::prelude::*;

// 1. Create a problem instance (in-memory graph)
let mc = MaxCut::new(Graph::from_edges([
    (0, 1, 1.0),
    (0, 2, 1.0),
    (1, 2, 1.0),
]));
// Or load from file: let mc = MaxCut::new(Graph::load_from_file("data/max_cut/G1")?);

// 2. Initialize search state
let mut state = SearchState::new(&mc);

// 3. Configure and run a heuristic
let mut ls = LocalSearch::<MaxCutFlipNeighbor>::new(
    StopCondition::iterations(1_000_000)
);
ls.run(&mut state).unwrap();

// 4. Retrieve the best result
println!("best cut = {}", state.best_solution.objective);
```

QUBO instances can be loaded from a sparse Q-matrix file via `Qubo::load_file("path/to/file.qubo")?`
(format: header `N M`, then `i j v` lines with 1-indexed variables).

## Supported Problems

| Problem | Type | Neighbors |
|---|---|---|
| Max Cut | `MaxCut` | `MaxCutFlipNeighbor`, `MaxCutSwapNeighbor` |
| MaxSAT | `Sat` | `SatFlipNeighbor`, `SatSwapNeighbor` |
| TSP | `TspWithCoordinates` | `TspTwoOptNeighbor`, `TspRelocateNeighbor` |
| QUBO | `Qubo` | `QuboFlipNeighbor`, `QuboSwapNeighbor` |
| Vertex Cover | `VertexCover` | `VertexCoverFlipNeighbor`, `VertexCoverSwapNeighbor` |
| Job Shop Scheduling | `JobShopScheduling` | `JobShopSwapNeighbor`, `JobShopRelocateNeighbor` |
| Formula | `FormulaProblem` | `FormulaFlipNeighbor`, `FormulaSwapNeighbor` |

## Available Heuristics

| Algorithm | Type | Notes |
|---|---|---|
| Local Search | `LocalSearch<N>` | Greedy best-improving |
| Simulated Annealing | `SimulatedAnnealing<N>` | Exponential cooling |
| Bang-Bang SA | `BangBangSimulatedAnnealing<N>` | Two-temperature schedule |
| Late Acceptance Hill Climbing | `LateAcceptanceHillClimbing<N>` | Accept if no worse than score `history_length` steps ago |
| Tabu Search | `TabuSearch<N>` | Randomized tenure |
| Random Walk | `RandomWalk<N>` | Uniform random moves |
| Beam Search | `BeamSearch<P, N>` | Keeps top-k candidates |
| RL Search | `RLSearch<P, N>` | REINFORCE policy gradient over candidate moves |
| Sequential | `Sequential<P>` | Chains multiple heuristics |
| Iterated | `Iterated<P>` | Iterated Local Search (search + perturbation) |
| Restart | `Restart<P>` | Multi-start: restart from random when condition triggers |
| Genetic Algorithm | `GeneticAlgorithm<P, C>` | Population-based with crossover; optional `init_improvement` phase (HEA pattern) |
| Breakout Local Search | `BreakoutLocalSearchForMaxCut` | MaxCut-specific BLS |
| Lin-Kernighan-Helsgaun | `LinKernighanHelsgottForTsp` | TSP-specific LKH |

## Stop Conditions

`StopCondition` is configured via builder methods:

```rust
use optopus::heuristic::StopCondition;
use std::time::Duration;

// Stop after a fixed number of iterations
let sc = StopCondition::iterations(1_000_000);

// Stop after a time limit
let sc = StopCondition::duration(Duration::from_secs(30));

// Stop after too many iterations without improvement
let sc = StopCondition::failed_updates(10_000);

// Combine conditions (stops when any is met)
let sc = StopCondition::iterations(1_000_000)
    .with_duration(Duration::from_secs(30))
    .with_failed_updates(10_000);
```

## Custom Problems

The minimum contract for a new problem is:

- `ProblemTrait`: generate an initial solution
- `Rankable` on the solution: define better/worse
- `MoveToNeighbor`: enumerate and apply moves
  - `LocalSearch` additionally requires the neighbor type to implement `Rankable`
    (used to select the best move among candidates)

Additional traits are required only for some heuristics:

- `Evaluate<T>` (returns `Evaluable<T>`): required by `SimulatedAnnealing` and `LateAcceptanceHillClimbing`
- `EnabledTabu`: required by `TabuSearch`
- `Crossover`: required by `GeneticAlgorithm`
- `SubProblemExtractable`: required by `SubProblemBasedCrossover`

Implement these traits to plug a problem into every heuristic:

```rust
use optopus::search_state::{ProblemTrait, MoveToNeighbor, Rankable};

struct MyProblem { /* … */ }
struct MySolution { pub objective: f64 }
struct MyNeighbor { pub gain: f64 }

impl ProblemTrait for MyProblem {
    type Solution = MySolution;
    fn new_solution(&self, rng: &mut impl rand::Rng) -> MySolution { /* … */ }
}

impl Rankable for MySolution {
    fn is_better_than(&self, other: &Self) -> bool {
        self.objective > other.objective
    }
}

// Required by LocalSearch (selects the best-improving move)
impl Rankable for MyNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain > other.gain
    }
}

impl MoveToNeighbor<MyProblem> for MyNeighbor {
    fn apply_to_solution(&self, prob: &MyProblem, sol: &mut MySolution) -> Result<(), OptError> {
        sol.objective += self.gain;
        Ok(())
    }
    fn iter(prob: &MyProblem, sol: &MySolution) -> impl Iterator<Item = Self> + Send {
        /* enumerate neighbors … */
    }
    fn move_to_be_better_than(&self, _: &MyProblem, src: &MySolution, other: &MySolution) -> bool {
        src.objective + self.gain > other.objective
    }
}
```

See [examples/custom_problem.rs](examples/custom_problem.rs) for a complete worked example.

## Custom Heuristics

To add a new heuristic, implement `Heuristic<P>` and update `SearchState` inside `run_once`.
The smallest useful pattern is:

- read the current solution from `state.solution`
- choose a move from `N::iter(...)`
- call `state.apply(&neighbor)?`
- or call `state.progress_iteration()` when no move is applied

See [examples/custom_heuristic.rs](examples/custom_heuristic.rs) for a minimal first-improving search implementation.

## Composing Heuristics

Use `Sequential`, `Iterated`, or `Restart` to combine heuristics:

```rust
use optopus::prelude::*;

// Sequential: runs each heuristic in order, passing state along
let mut seq = Sequential::<MaxCut>::new(
    StopCondition::iterations(1_000_000),
    vec![
        Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
            StopCondition::iterations(100_000),
        )),
        Box::new(SimulatedAnnealing::<MaxCutFlipNeighbor>::new(
            StopCondition::iterations(500_000),
            100.0,
            0.999,
        )),
    ],
);
seq.run(&mut state).unwrap();

// Iterated Local Search: alternates search and perturbation phases
let mut ils = Iterated::<MaxCut>::new(
    StopCondition::iterations(1_000_000),
    Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
        StopCondition::failed_updates(1),
    )),
    Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(
        StopCondition::iterations(20),
    )),
);
ils.run(&mut state).unwrap();
```

## Benchmarking

Write a TOML config file and run the CLI to benchmark multiple heuristics across multiple instances:

```toml
# data/my_benchmark.toml
num_runs = 3

[[instances]]
path = "data/max_cut/G[1-2]"
problem = "MaxCut"   # MaxCut | Qubo | Sat | Tsp | VertexCover | JobShop

[[heuristics]]
kind = "LocalSearch"
neighbor = "Flip"    # Flip | Swap | TwoOpt | Relocate
[heuristics.stop_condition]
max_iteration = 100000

[[heuristics]]
kind = "SimulatedAnnealing"
neighbor = "Flip"
initial_temperature = 100.0
cooling_rate = 0.999
[heuristics.stop_condition]
max_duration_secs = 30.0
```

Supported `kind` values: `LocalSearch`, `TabuSearch`, `SimulatedAnnealing`,
`LateAcceptanceHillClimbing`, `RLSearch`, `BeamSearch`, `BreakoutLocalSearch`
(MaxCut only), `GeneticAlgorithm`, and the meta-heuristics `Sequential` /
`Iterated` / `Restart` (which take a nested `steps` array). `GeneticAlgorithm`
requires `population_size` and accepts optional `crossover_kind` (`"Uniform"`
default; TSP defaults to `"Order"`), `parent_selection` (`"Tournament"` default
or `"HammingTopK"`), and `parent_top_k`.

```bash
cargo run -- data/my_benchmark.toml
```

Results are written to `result/` as a timestamped TOML file with `best`, `avg`, `worst`, `std`, and timing statistics per heuristic-instance pair.
If a glob matches no files, or a heuristic is missing required fields, the CLI exits with a configuration error instead of silently producing an empty report.

See [`data/sample_setting.toml`](data/sample_setting.toml) for a multi-heuristic MaxCut comparison,
[`data/sample_ga_maxcut.toml`](data/sample_ga_maxcut.toml) for a `GeneticAlgorithm` setup,
and [`data/sample_jssp.toml`](data/sample_jssp.toml) for Job Shop Scheduling.

Per-heuristic performance summaries on standard instance sets are published under [`docs/benchmarks/`](docs/benchmarks/):

- [BreakoutLocalSearch on MaxCut (Gset)](docs/benchmarks/breakout_local_search.md)

Curated source TOMLs live in `docs/benchmarks/data/` and the Markdown is regenerated with `python3 docs/benchmarks/render.py`.

## Error Handling

All heuristic `run` / `run_once` methods return `Result<(), OptError>`:

```rust
use optopus::error::OptError;

match ls.run(&mut state) {
    Ok(()) => println!("done"),
    Err(OptError::Config(msg)) => eprintln!("config error: {msg}"),
    Err(OptError::InvalidState(msg)) => eprintln!("invalid state: {msg}"),
    Err(e) => eprintln!("error: {e}"),
}
```

## Examples

```bash
# MaxCut: compare LocalSearch and TabuSearch
cargo run --example max_cut

# Custom problem definition
cargo run --example custom_problem

# Custom heuristic definition
cargo run --example custom_heuristic
```
