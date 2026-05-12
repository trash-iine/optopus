# Quickstart

This page walks through the smallest possible end-to-end use of the library
and then shows how to load a problem from a file.

## Install

Add Optopus to your `Cargo.toml`:

```toml
[dependencies]
optopus = { git = "https://github.com/trash-iine/optopus" }
```

## Run an example

```bash
cargo run --example max_cut
```

## In-memory MaxCut

```rust
use optopus::prelude::*;

// 1. Build a problem instance from edges.
let mc = MaxCut::new(Graph::from_edges([
    (0, 1, 1.0),
    (0, 2, 1.0),
    (1, 2, 1.0),
]));

// 2. Initialise the search state (random initial solution).
let mut state = SearchState::new(&mc);

// 3. Configure and run a heuristic.
let mut ls = LocalSearch::<MaxCutFlipNeighbor>::new(
    StopCondition::iterations(1_000_000),
);
ls.run(&mut state).unwrap();

// 4. Read the best result.
println!("best cut = {}", state.best_solution.objective);
```

## Loading instances from files

Each problem ships with a loader that returns
`Result<Self, optopus::error::OptError>`:

```rust
use optopus::prelude::*;

// MaxCut / Vertex Cover use the shared Graph loader (format: `N M / i j w`).
let mc = MaxCut::new(Graph::load_from_file("data/max_cut/G1")?);

// QUBO loader (format: `N M / i j v`, 1-indexed):
let qubo = Qubo::load_file("data/qubo/sample.txt")?;

// MaxSAT loader (DIMACS CNF):
let sat = Sat::load_file("data/sat/example.cnf")?;

// TSP loader (TSPLIB):
let tsp = TspWithCoordinates::load_file("data/tsp/burma14.tsp")?;

// Job Shop Scheduling loader (Taillard / OR-Library):
let jssp = JobShopScheduling::load_file("data/jssp/ft06.txt")?;
# Ok::<(), optopus::error::OptError>(())
```

Each loader's file format is documented on the corresponding problem page.

## What to read next

- [Concepts](concepts.md) — the design philosophy and the small set of traits
  every problem and heuristic relies on.
- [Problems](problems/README.md) — what each built-in problem offers.
- [Heuristics](heuristics/README.md) — picking an algorithm.
- [Composing heuristics](guide/composing.md) — `Sequential`, `Iterated` (ILS),
  `Restart`, and `GeneticAlgorithm`.
- [Custom problem](guide/custom_problem.md) — plug your own problem into every
  built-in heuristic by implementing three traits.
