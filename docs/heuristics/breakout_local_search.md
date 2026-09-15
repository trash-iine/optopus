# BreakoutLocalSearch

**API:** [`BreakoutLocalSearch`](../api/optopus/heuristic/struct.BreakoutLocalSearch.html)

Alternates a greedy local search phase with an adaptive perturbation phase.

BLS is a framework rather than one algorithm. It is a descent, a bank of
perturbations, and a schedule that picks one of them and how far to go, and none
of the three needs the problem to be binary. `BreakoutLocalSearch<P, S>`
therefore takes any `P: ProblemTrait`. The paper that states the framework
introduces BLS on the vertex separator problem, which is not a binary problem.

```text
Algorithm 1  BLS general framework
1: p0 <- GenerateInitialSolution
2: L  <- L0
3: while stopping condition not reached do
4:     p  <- DescentBasedSearch(p0)
5:     L  <- DetermineJumpMagnitude(L, p, history)
6:     T  <- DeterminePerturbationType(p, history)
7:     p0 <- Perturb(L, T, p, history)
8: end while
```

Line 1 is `SearchState`'s initial solution, a library-wide convention rather
than something a heuristic owns. Line 2 is the schedule's `reset`, line 3 is
`Heuristic::run`'s loop, and `run_once` is lines 4 to 7, one line each. Lines 4
and 7 are ordinary heuristics, a `Box<dyn Heuristic<P>>` for the descent and a
bank of them for the kicks. Lines 5 and 6 are the
[`PerturbationSchedule`](../api/optopus/heuristic/trait.PerturbationSchedule.html)
trait, which returns an index into that bank.

The history the three procedures share is the tabu memory on the
`SearchState`. The descent writes it, Benlic & Hao's `H <- Iter + gamma` sitting
inside their descent loop, and the directed perturbations read it. That is what
stops a perturbation undoing the descent that just ran.

On [MaxCut](../problems/max_cut.md) the bank is a [`RandomWalk`](random_walk.md)
for the strong kick, a [`TabuSearch`](tabu_search.md) for the weak flip, and a
hand-written directed swap (`src/heuristic/specific/max_cut/best_swap.rs`),
which is the one operator with no generic equivalent, `M2` moving one vertex per
partition side in a single move.

## Example

```rust
use optopus::prelude::*;

let mut rng = seeded_rng(42);
let mc = MaxCut::new(Graph::erdos_renyi(800, 0.02, &mut rng));
let mut state = SearchState::new_with_seed(&mc, 42);

let mut bls = breakout_local_search_for_max_cut(
    StopCondition::iterations(100_000),
    /* tabu_tenure = */ (5, 150),
    /* t           = */ 1_000,
    /* l0          = */ 8,          // 0.01 * |V|
    /* p0          = */ 0.8,
    /* q           = */ 0.5,
);
bls.run(&mut state)?;
println!("cut weight = {}", state.best_solution.objective);
```

`l0` and `tabu_tenure` are instance-dependent, so they are derived from
`|V| = 800` here rather than left at a constant, see
[Benchmark config](#benchmark-config).

## Algorithm sketch

- Greedy phase: repeatedly apply the strictly best improving flip, updating
  a tabu map.
- Perturbation phase: `p = max(exp(−omega / t), p0)` is the probability of a
  directed (weak) perturbation, and it decays as the non-improvement counter
  `omega` grows:
  - `omega == 0`, the last descent improved the global best, or `omega` just
    passed `t` and was reset, a strong perturbation (random flips) runs.
  - `0 < omega <= t`: weak flip with probability `p * q`, weak swap with
    probability `p * (1 − q)`, strong (random flips) with probability
    `1 − p`. As `omega` grows `p` decays toward `p0`, so strong perturbations
    become steadily more likely.
  - `omega > t`: `omega` resets to 0, which the first branch then reads as a
    forced strong perturbation.
- Both weak perturbations take the highest-gain move that is not tabu, and
  admit a tabu move only when it would beat the global best (aspiration rule).
  These weak perturbations apply `l` best flip / swap moves.
- The perturbation length `l` increases by 1 whenever the descent lands on the
  same local optimum as the previous round, and resets to `l0` whenever it
  escapes.

## Differences from the original scheme

This implementation follows Benlic and Hao closely, and each departure below was
checked against the cut values they publish.

- The tenure parameter is doubled on the way in. The original tenure is
  added once when a vertex is recorded and once more in the eligibility test, so
  a vertex stays forbidden for twice it. `TabuMemory` stores a single tenure, so
  `paper_effective_tenure` doubles the caller's range and `tabu_tenure` keeps
  the original meaning, `rand[3, n/10]` on the G-set. Doubling only the upper
  bound does not reproduce it, the whole range has to scale.
- No bucket sort. The original buckets vertices by gain, so selecting a
  maximum-gain move is O(1) and a move costs only the O(degree(v)) rebucketing
  its gain update already implies. Here every selection is a linear scan over
  all n flip neighbours, O(n) per move, the descent included, since it is
  a plain `LocalSearch` (see [below](#what-a-round-is-built-from)).
  The gain update itself is O(degree(v)). The same move is selected either way,
  so this costs only speed.
- A swap advances the iteration counter by 2
  (`MaxCutSwapNeighbor::apply_to_iteration`), where BLS counts every move as
  one. That `+2` is a library-wide convention shared by every binary problem's
  swap, so it is not changed here for one heuristic's sake.

## What a round is built from

Three quarters of a round are the library's own heuristics. The descent is a
[`LocalSearch`](local_search.md), the strong perturbation a
[`RandomWalk`](random_walk.md), and the weak flip a
[`TabuSearch`](tabu_search.md). They select the same moves a dedicated operator
would, and because recording is a mode on the `SearchState`, their `apply`
writes the same prohibitions, so the tabu list update Benlic and Hao put inside
the descent loop still happens.

The weak swap is the one hand-written operator, because `M2` moves one vertex
per partition side in a single move and a pair of one-step searches cannot
express that. A side with nothing eligible would otherwise leave the other
vertex moved on its own.

## Constructors

MaxCut has a builder that fills the bank and applies the paper's reading of the
tenure:

```rust
breakout_local_search_for_max_cut(
    stop_condition: StopCondition,
    tabu_tenure: (u64, u64),
    t: u64,
    l0: u64,
    p0: f64,
    q: f64,
) -> BreakoutLocalSearchForMaxCut
```

Another problem builds one directly, supplying its own descent, bank and
schedule:

```rust
BreakoutLocalSearch::new(
    stop_condition: StopCondition,
    tabu_tenure: (u64, u64),
    descent: Box<dyn Heuristic<P>>,
    perturbations: Vec<Box<dyn Heuristic<P>>>,
    schedule: S,
) -> Self
```

`AdaptivePerturbation` indexes three perturbations, one random and two directed,
which is the shape Benlic & Hao's rule chooses between. A schedule of your own
may use any number.

| Parameter | Meaning |
|---|---|
| `tabu_tenure` | tabu tenure range `(min, max)` for the LS phase |
| `t` | period of the `omega` counter before it resets |
| `l0` | initial perturbation length |
| `p0` | minimum perturbation probability |
| `q` | fraction of weak perturbations using flip (vs. swap) |

`clear()` resets the schedule (`omega` to 0, `l` to `l0`, the remembered local
optimum dropped) and clears the descent and the bank. The prohibitions are not
its to clear, since they belong to the `SearchState`, and a sub-run clone starts
with an empty tabu memory anyway.
The remembered local optimum is dropped rather than kept because the same
schedule may be reused on a different instance, a meta-heuristic that rebuilds
its sub-problem every round does exactly that.

## Benchmark config

```toml
[[heuristics]]
kind = "BreakoutLocalSearch"
tabu_tenure = [3, 80]     # density-scaled; see docs/benchmarks
t = 1000
l0 = 80                   # 0.01 * |V|
p0 = 0.8
q = 0.5
[heuristics.stop_condition]
max_duration_secs = 30.0
```

`tabu_tenure` is read as original `γ`: a vertex stays forbidden for `2γ`
moves. This is the one kind that doubles the key, the same range under
[`TabuSearch`](tabu_search.md) prohibits for half as long, so tuned values do
not transfer between them.

## Replacing the schedule

There are two routes, and which fits depends on whether your rule is Algorithm
1's two decision procedures or something that has to act between them.

Implement `PerturbationSchedule<P>` and hand it to `BreakoutLocalSearch::new`.
You keep the framework's loop and `Heuristic`; what you replace is lines 5 and
6.

Or drive the round in halves yourself:

- `descend(state)`, the greedy descent, writing the prohibitions the kick reads.
- `kick(state, perturbation, l)`, `l` moves of the bank entry at index
  `perturbation`, followed by the round's single `update_best`. On MaxCut,
  [`MaxCutPerturbation`](../api/optopus/heuristic/enum.MaxCutPerturbation.html)
  names those indices.
- `externally_driven_bls_for_max_cut(stop_condition, tabu_tenure)`, the builder
  for that use. It takes the tenure literally, since the `2γ` doubling above
  belongs to the paper's schedule, which such a caller replaces. Its schedule
  slot is `()`, which does not implement `PerturbationSchedule`, so the result
  is not a `Heuristic` and `run_once` cannot be reached.

`run_once` is exactly `descend` + the schedule + `kick`.
[Driving BLS with a learned perturbation policy](../guide/learned_perturbation.md)
walks through a controller that decides between the two halves with a
contextual bandit.

## References

- Benlic, U. and Hao, J.-K. "Breakout Local Search for the Max-Cut problem."
  *Engineering Applications of Artificial Intelligence*, 26(3), 1162-1173,
  2013.
