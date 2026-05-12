# Composing Heuristics

The meta-heuristics in [`heuristic::sequential`](../heuristics/meta.md) and
[`heuristic::genetic_algorithm`](../heuristics/genetic_algorithm.md) take other
`Heuristic<P>` values as building blocks. This page shows the four most common
compositions.

All examples assume:

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]));
let mut state = SearchState::new(&mc);
```

## Sequential

Run a list of heuristics one after another. Each step starts from the previous
step's solution; the global best is preserved.

```rust
let mut seq = Sequential::<MaxCut>::new(
    StopCondition::iterations(100_000),
    vec![
        Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
            StopCondition::failed_updates(1),
        )),
        Box::new(TabuSearch::<MaxCutFlipNeighbor>::new(
            StopCondition::failed_updates(500),
            (5, 10),
            None,
        )),
    ],
);
seq.run(&mut state)?;
# Ok::<(), optopus::error::OptError>(())
```

The outer `StopCondition` is checked between sub-heuristics; the cycle
re-runs from the top once it reaches the end of the list.

## Iterated Local Search (ILS)

Alternate a *search* phase with a *perturbation* phase.

```rust
let mut ils = Iterated::<MaxCut>::new(
    StopCondition::iterations(100_000),
    Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
        StopCondition::failed_updates(1),
    )),
    Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(
        StopCondition::iterations(5),
    )),
);
ils.run(&mut state)?;
# Ok::<(), optopus::error::OptError>(())
```

## Restart wrapping ILS

Whenever the inner heuristic stops improving for `restart_condition` iterations,
replace `state.solution` with a fresh random one. `state.best_solution` is
preserved.

```rust
let ils = Iterated::<MaxCut>::new(
    StopCondition::iterations(10_000),
    Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
        StopCondition::failed_updates(1),
    )),
    Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(
        StopCondition::iterations(5),
    )),
);

let mut solver = Restart::new(
    StopCondition::iterations(1_000_000),
    Box::new(ils),
    StopCondition::failed_updates(1_000),
);
solver.run(&mut state)?;
# Ok::<(), optopus::error::OptError>(())
```

## Hybrid Genetic Algorithm (HEA-style)

`SubProblemBasedCrossover` solves a sub-instance built from disagreeing
variables. Combined with `init_improvement` and a `TabuSearch` mutation, this
reproduces the Galinier–Hao Hybrid Evolutionary Algorithm.

```rust
let mut ga = GeneticAlgorithm::new_with_init(
    StopCondition::iterations(10_000),
    /* population_size  = */ 50,
    SubProblemBasedCrossover {
        sub_heuristic: Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
            StopCondition::failed_updates(1),
        )),
    },
    /* mutation         = */ Box::new(TabuSearch::<MaxCutFlipNeighbor>::new(
        StopCondition::failed_updates(100),
        (5, 10),
        None,
    )),
    /* init_improvement = */ Some(Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
        StopCondition::failed_updates(1),
    ))),
);
ga.run(&mut state)?;
# Ok::<(), optopus::error::OptError>(())
```

## How merging works

Every meta-heuristic uses the **sub-run clone/merge pattern**:

```text
let mut sub = state.clone_for_new_run(SearchStateCloneType::ClearBest);
inner.run(&mut sub)?;
state.update_state(sub);   // best merged back, global iteration accumulates
```

See [concepts.md](../concepts.md#sub-run-clonemerge-pattern) for the trait-level
details and the three [`SearchStateCloneType`] variants.
