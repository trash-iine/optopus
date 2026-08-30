# GeneticAlgorithm

**API:** [`GeneticAlgorithm`](../api/optopus/heuristic/struct.GeneticAlgorithm.html)

Population-based search: a population of solutions recombined pairwise by a
`Crossover<P>` operator, with a `Heuristic<P>` as the mutation operator.

## Example

An HEA-style hybrid GA: `SubProblemBasedCrossover` recombination with a
`TabuSearch` mutation operator, every random initial individual improved first.

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
println!("cut weight = {}", state.best_solution.objective);
# Ok::<(), optopus::error::OptError>(())
```

## Algorithm sketch

Each iteration:

1. Select two parents.
2. Cross them with operator `C`.
3. Mutate the offspring.
4. Insert into the population, evicting the worst when at capacity.

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

`clear()` drops the population and the cached `best_idx`; the population is
re-seeded on the first `run_once` after a `run`.

## Parent selection

Builder method `with_parent_selection(strategy)` switches between:

```rust
pub enum ParentSelection {
    Tournament,                          // default — two binary tournaments
    DistantTopK { top_k: usize },        // pick A randomly, B from top-k by distance
}
```

`DistantTopK` requires `P::Solution: Distance` and promotes diversity by
preferring distant parents.

## Replacement

Worst-replacement: when the population is full, replace the worst member iff
the offspring is strictly better. `best_idx` is maintained incrementally —
no full population scan per iteration.

## Crossover trait

```rust
pub trait Crossover<P: ProblemTrait> {
    fn crossover(
        &mut self,
        prob: &P,
        sol1: &P::Solution,
        sol2: &P::Solution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Result<P::Solution, OptError>;
}
```

`&mut self` lets stateful operators (such as
[`SubProblemBasedCrossover`](../api/optopus/heuristic/struct.SubProblemBasedCrossover.html),
which runs an inner heuristic) hold mutable state across calls. The RNG is
passed in explicitly so seeded runs stay reproducible.

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

## Benchmark config

```toml
[[heuristics]]
kind = "GeneticAlgorithm"
population_size = 20         # required; must be >= 2
crossover_kind = "Uniform"   # optional; default is per-problem (see below)
parent_selection = "Tournament"  # optional; Tournament (default) | DistantTopK
parent_top_k = 5             # required when parent_selection = "DistantTopK"
[heuristics.stop_condition]
max_duration_secs = 30.0

[[heuristics.steps]]         # steps[0] = mutation (required)
kind = "TabuSearch"
neighbor = "Flip"
tabu_tenure = [5, 150]
[heuristics.steps.stop_condition]
max_iteration = 2_000

[[heuristics.steps]]         # steps[1] = init_improvement (optional, HEA pattern)
kind = "LocalSearch"
neighbor = "Flip"
[heuristics.steps.stop_condition]
max_failed_update = 1
```

`crossover_kind` defaults to `"Uniform"`, except `"Order"` for TSP and CVRP
and `"Ppx"` for JobShop. MaxCut additionally accepts `"SubProblem"` — memetic
recombination that solves the sub-MaxCut of the disagreeing variables with an
internal bounded BLS (see
[SubProblemBasedCrossover](#subproblembasedcrossover)).

## References

- Holland, J. H. *Adaptation in Natural and Artificial Systems*. University of
  Michigan Press, 1975.
- Goldberg, D. E. *Genetic Algorithms in Search, Optimization, and Machine
  Learning*. Addison-Wesley, 1989.
- Galinier, P. and Hao, J.-K. "Hybrid Evolutionary Algorithms for Graph
  Coloring." *Journal of Combinatorial Optimization*, 3(4), 379-397, 1999.
