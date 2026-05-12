# GeneticAlgorithm

Population-based search. Each iteration: select two parents → cross them with
operator `C` → mutate the offspring → insert into the population, evicting
the worst when at capacity.

## Constructor

```rust
GeneticAlgorithm::<P, C>::new(
    stop_condition: StopCondition,
    population_size: usize,
    crossover: C,
    mutation: Box<dyn Heuristic<P>>,
) -> Self
```

`C: Crossover<P>` and `P::Solution: Distance` (the distance impl is required
even when using `Tournament` selection because the type bound is on the
`Heuristic<P>` impl).

**Panics** if `population_size < 2`.

## Constructor with HEA-style init

```rust
GeneticAlgorithm::<P, C>::new_with_init(
    stop_condition: StopCondition,
    population_size: usize,
    crossover: C,
    mutation: Box<dyn Heuristic<P>>,
    init_improvement: Option<Box<dyn Heuristic<P>>>,
) -> Self
```

When `init_improvement = Some(op)`, every random initial individual is also
passed through `op` (using the sub-run clone/merge pattern). Pair this with a
`TabuSearch` mutation operator to reproduce the Galinier-Hao Hybrid
Evolutionary Algorithm (HEA).

## Parent selection

Builder method `with_parent_selection(strategy)` switches between:

```rust
pub enum ParentSelection {
    Tournament,                          // default — two binary tournaments
    HammingTopK { top_k: usize },        // pick A randomly, B from top-k by distance
}
```

`HammingTopK` requires `P::Solution: Distance` and promotes diversity by
preferring distant parents.

## Replacement

Worst-replacement: when the population is full, replace the worst member iff
the offspring is strictly better. `best_idx` is maintained incrementally —
no full population scan per iteration.

## Example: HEA-style hybrid GA

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]));
let mut state = SearchState::new(&mc);

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

## Crossover trait

```rust
pub trait Crossover<P: ProblemTrait> {
    fn crossover(&mut self, prob: &P, sol1: &P::Solution, sol2: &P::Solution) -> P::Solution;
    fn apply_to_crossover_count(&self, count: u64) -> u64 { count + 1 }
}
```

`&mut self` lets stateful operators (such as `SubProblemBasedCrossover`,
which runs an inner heuristic) hold mutable state across calls.

## SubProblemBasedCrossover

A generic crossover for any `P: SubProblemExtractable`:

1. `extract_sub_problem(sol1, sol2)` — variables that agree in both parents
   are fixed; the disagreeing variables form a sub-instance.
2. `sub_heuristic.run(...)` solves the sub-instance from scratch.
3. `lift_solution(sol1, sol2, sub_solution)` reconstructs the full solution.

```rust
let crossover = SubProblemBasedCrossover {
    sub_heuristic: Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
        StopCondition::failed_updates(1),
    )),
};
```

Implemented by MaxCut, QUBO, SAT, Vertex Cover, and Formula.
