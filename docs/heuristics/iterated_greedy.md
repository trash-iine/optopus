# IteratedGreedy

**API:** [`IteratedGreedy`](../api/optopus/heuristic/struct.IteratedGreedy.html)

Iterated Greedy takes a few elements out of the incumbent at random, puts each
back where it costs least, runs a local search, and accepts the result on a
constant temperature. Ruiz and Stützle proposed it for the
[permutation flow shop](../problems/flow_shop.md), where this short loop is
still among the strongest methods known.

Runs on any problem implementing [`Ruinable`](../traits.md) whose solution
implements [`Evaluate`](../traits.md). In a config it is registered for the
flow shop.

## Example

```rust
use optopus::prelude::*;

let fs = FlowShop::load_file("data/instances/flow_shop/taillard/ta051.txt")?;
let mut state = SearchState::new_with_seed(&fs, 42);

iterated_greedy_for_flow_shop(
    StopCondition::duration(std::time::Duration::from_secs(10)),
    /* removal_count      = */ 4,
    /* temperature_factor = */ 0.4,
)
.run(&mut state)?;
println!("makespan = {}", state.best_solution.objective);
```

`iterated_greedy_for_flow_shop` is the constructor plus the two things the
flow shop hands it, the full insertion descent as its local search and the
paper's temperature. Written out, and on another problem, the same search is

```rust
use optopus::prelude::*;
use optopus::problem::tsp::AnchoredTourDescent;

let mut ig = IteratedGreedy::<Tsp>::new(
    StopCondition::iterations(10_000),
    /* removal_count  = */ 3,
    /* temperature_of = */ |_: &Tsp| 1.0,
)
.with_local_repair(Box::new(AnchoredTourDescent::new()));
```

## Algorithm sketch

Each `run_once` produces one candidate.

1. Destroy, take `removal_count` elements out of the incumbent uniformly at
   random.
2. Construct, put each back at its cheapest place, in random order.
3. Descend, run the `LocalRepair` if one was given.
4. Accept, if the candidate is no worse than the incumbent, or otherwise with
   probability `exp(−Δ / T)`. `T` stays the same for the whole run.

The global best is kept apart from the incumbent, so an accepted worse
solution loses nothing.

## How it differs from ALNS

Both are ruin and recreate over the same trait, and
[ALNS](alns.md) runs on the flow shop as well. ALNS chooses among three destroy
and two repair operators by a roulette, removes a fraction of the elements and
cools. Iterated Greedy fixes the one random destroy, the one greedy repair and
the number removed, and does not cool. On Taillard's 50 × 20 instances it is
the stronger of the two on the same time.

## Temperature

The temperature is in the units of the objective, so `IteratedGreedy::new`
takes it as a function of the instance, read from the instance being searched.
For the flow shop Ruiz and Stützle set it to a tenth of the mean processing
time scaled by a factor, `T = factor · Σp / (n · m · 10)`, which is what
`iterated_greedy_for_flow_shop` computes from `temperature_factor`.

## Constructor

```rust
IteratedGreedy::<P>::new(
    stop_condition: StopCondition,
    removal_count: usize,                       // elements taken out per iteration
    temperature_of: impl Fn(&P) -> f64 + 'static,
) -> Self
```

Panics if `removal_count` is zero, and in a run if the temperature is not
positive. `with_local_repair(Box<dyn LocalRepair<P>>)` sets the descent.

## Benchmark config

```toml
[[heuristics]]
kind = "IteratedGreedy"
removal_count = 4           # optional (default shown)
temperature_factor = 0.4    # optional (default shown)
[heuristics.stop_condition]
max_duration_secs = 30.0
```

The defaults are the values Ruiz and Stützle tuned.

## References

- Ruiz, R. and Stützle, T. "A simple and effective iterated greedy algorithm
  for the permutation flowshop scheduling problem." *European Journal of
  Operational Research*, 177(3), 2033-2049, 2007.
