# TabuSearch

**API:** [`TabuSearch`](../api/optopus/heuristic/struct.TabuSearch.html)

At each step, pick the strictly best move that is not currently tabu, then
mark it tabu for a tenure drawn uniformly from `tabu_tenure = (min, max)`.

A tabu move is still selectable when it satisfies the **aspiration criterion**:
the resulting solution would be strictly better than the current global best.

## Example

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]));
let mut state = SearchState::new(&mc);
let mut ts = TabuSearch::<MaxCutFlipNeighbor>::new(
    StopCondition::iterations(10_000),
    /* tabu_tenure = */ (5, 10),
);
ts.run(&mut state)?;
println!("cut weight = {}", state.best_solution.objective);
# Ok::<(), optopus::error::OptError>(())
```

## Constructor

```rust
TabuSearch::<N>::new(
    stop_condition: StopCondition,
    tabu_tenure: (u64, u64),
) -> Self
```

`N` must satisfy `MoveToNeighbor<P> + Clone + EnabledTabu + Rankable`.

**Panics** if `tabu_tenure.0 > tabu_tenure.1`.

## Where the tabu map lives

The map is on the [`SearchState`](../search_state.md), not on this heuristic.
The state is what applies a move, so the state is what records it: `apply` /
`apply_move_only` write the move into the tabu memory *before* the iteration
advances, and `TabuSearch` only installs the tenure it wants at the top of each
iteration.

That is why there is no `clear()` here, and no `borrow_tabu_map` /
`take_tabu_map` / `set_tabu_map`: a sub-run clone — how every meta-heuristic
starts a phase — already comes with an empty tabu memory, and
`state.reset_tabu()` drops the prohibitions on a state you are reusing.
`state.tabu_allows(&mv)` asks about a single move — what this heuristic's inner
loop calls per candidate — and `state.reserve_tabu_vars(n)` pre-grows the dense
key space.

## Tabu policy abstraction

Each neighbor type owns its *policy* — which keys have to be free, and which
applying the move forbids — via the `EnabledTabu` trait, and hands it to the
state by overriding `MoveToNeighbor::tabu_policy` with `Some(self)`, one line.
`TabuSearch` never knows what is keyed. This lets QUBO/MaxCut/SAT key by
variable index, TSP by edge pair, Job Shop by swap position, etc. The two are
not required to agree: a VRP relocate asks whether a customer may enter its
destination route and forbids the route it just left.

A move that leaves `tabu_policy` at its default `None` has no tabu policy at
all: applying it is fine and records nothing, while `state.record_tabu` and
`state.require_tabu_policy` report `OptError::Unsupported`. `TabuSearch` calls
`require_tabu_policy` once per iteration, on the move it is about to apply, so
a move type that implements `EnabledTabu` and forgets the one-line override
fails loudly instead of quietly running without a tabu list.

`common::TabuMemory` is the single store, split by `TabuKey` shape — `Var(i)`
for a dense index, `Pair` and `Triple` for the rest. Two move types over the
same shape share prohibitions (MaxCut's flip and swap are both `Var`, which is
what the operators in `src/heuristic/specific/max_cut/ops/` rely on), while
different shapes never collide (JobShop's swap is a `Var`, its relocate a
`Pair`).

## Benchmark config

```toml
[[heuristics]]
kind = "TabuSearch"
neighbor = "Flip"        # required; the valid values are per-problem
tabu_tenure = [5, 10]    # required; (min, max), drawn uniformly per move
[heuristics.stop_condition]
max_duration_secs = 30.0
```

The tenure is taken literally: a move stays forbidden for that many iterations.
(`BreakoutLocalSearch` reads the same key as the paper's `γ` and forbids for
`2γ`, so tuned values are not interchangeable between the two kinds.)

## References

- Glover, F. "Future Paths for Integer Programming and Links to Artificial
  Intelligence." *Computers & Operations Research*, 13(5), 533-549, 1986.
- Glover, F. "Tabu Search — Part I." *ORSA Journal on Computing*, 1(3),
  190-206, 1989.
