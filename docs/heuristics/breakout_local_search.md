# BreakoutLocalSearch

**API:** [`BreakoutLocalSearch`](../api/optopus/heuristic/struct.BreakoutLocalSearch.html)

Alternates a greedy local search phase with an adaptive perturbation phase.

BLS is a framework rather than one algorithm. It is a descent, a bank of
perturbations, and a schedule that picks one of them and how far to go, and none
of the three needs the problem to be binary. `BreakoutLocalSearch<P, S>`
therefore takes any `P: ProblemTrait`. The paper that states the framework
introduces BLS on the vertex separator problem, which is not a binary problem.

Benlic & Hao state the framework as four procedures a concrete BLS supplies,
and they land in three places here. `DescentBasedSearch` and `Perturb` are ordinary
heuristics, a `Box<dyn Heuristic<P>>` for the descent and a bank of them for
the kicks. `DetermineJumpMagnitude` and `DeterminePerturbationType`, the two
decisions taken between a descent and the kick that follows it, are the
[`PerturbationSchedule`](../api/optopus/heuristic/trait.PerturbationSchedule.html)
trait, which returns an index into that bank.

What is left over is not BLS's to own. The initial solution comes from
`SearchState`, a library-wide convention, and the loop that repeats the round
until the stopping condition is `Heuristic::run`'s. One `run_once` is one
round, a descent, the two decisions, then the kick.

The history the three procedures share is the tabu memory on the
`SearchState`. The descent writes it, forbidding each vertex it moves for a
tenure's worth of iterations, which is where Benlic & Hao put the write too,
and the directed perturbations read it. That is what stops a perturbation
undoing the descent that just ran.

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

let mut bls = bls_for_max_cut(
    StopCondition::iterations(100_000),
    /* tabu_tenure = */ (10, 300),
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

- `tabu_tenure` is the prohibition length itself, as it is under
  [`TabuSearch`](tabu_search.md). The original tenure `γ` is added once when a
  vertex is recorded and once more in the eligibility test, so a vertex stays
  forbidden for `2γ`. `TabuMemory` stores that length directly, so the paper's
  `rand[3, n/10]` on the G-set is written `[6, n/5]` here. Doubling only the
  upper bound does not reproduce it, the whole range has to scale.
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

## Constructor

MaxCut has a builder that fills the schedule:

```rust
bls_for_max_cut(
    stop_condition: StopCondition,
    tabu_tenure: (u64, u64),
    t: u64,
    l0: u64,
    p0: f64,
    q: f64,
) -> BreakoutLocalSearchForMaxCut
```

Another problem builds one directly, supplying its own descent and schedule:

```rust
BreakoutLocalSearch::new(
    stop_condition: StopCondition,
    tabu_tenure: (u64, u64),
    descent: Box<dyn Heuristic<P>>,
    schedule: S,
) -> Self
```

The schedule owns the perturbations it chooses between, so there is no bank to
hand in and no order to agree on. `AdaptivePerturbation` takes the random one
and a list of directed ones, each with its share of the directed probability:

```rust
AdaptivePerturbation::<P>::new(
    t: u64,
    l0: u64,
    p0: f64,
    random: Box<dyn Heuristic<P>>,
    directed: Vec<(Box<dyn Heuristic<P>>, f64)>,
) -> Self
```

Benlic & Hao's `q` is the two-entry case, `[(flip, q), (swap, 1 - q)]`. Any
number of directed perturbations works, one included.

| Parameter | Meaning |
|---|---|
| `tabu_tenure` | tabu tenure range `(min, max)` for the LS phase |
| `t` | period of the `omega` counter before it resets |
| `l0` | initial perturbation length |
| `p0` | minimum perturbation probability |
| `q` | share of the directed probability the flip takes, the swap gets the rest |

`clear()` resets the schedule (`omega` to 0, `l` to `l0`, the remembered local
optimum dropped) and clears the descent and the schedule's perturbations. The prohibitions are not
its to clear, since they belong to the `SearchState`, and a sub-run clone starts
with an empty tabu memory anyway.
The remembered local optimum is dropped rather than kept because the same
schedule may be reused on a different instance, a meta-heuristic that rebuilds
its sub-problem every round does exactly that.

## Replacing the schedule

Implement `PerturbationSchedule<P>` and hand it to `BreakoutLocalSearch::new`.
You keep the framework's loop and `Heuristic`, and what you replace is the two
decisions plus the operators they choose between.

| Method | Called |
|---|---|
| `determine_jump_magnitude(state)` | after the descent, with the local optimum it reached |
| `select(state)` | straight after, returning the operator to apply |
| `round_ended(state)` | once the kick and its `update_best` have closed the round. Defaulted to nothing |
| `reset` / `clear_perturbations` | from `Heuristic::clear` |

Both decisions are handed the state, as `Heuristic::run_once` is, so an
operator may be picked from where the search has got to and not only from a
draw. Draw from `state.rng` rather than an RNG of your own, which is what keeps
a seeded run reproducible.

A policy whose two answers are one decision settles it in
`determine_jump_magnitude` and has `select` return what it chose, since the
length is asked for first. Reading what a descent alone did means remembering
what `round_ended` was given, which is the value the next descent starts from.

On MaxCut, `max_cut_descent()` and
`max_cut_perturbation(kind, tabu_tenure)` build the pieces, so a schedule of
your own reuses the operators rather than rebuilding them.

[Driving BLS with a learned perturbation policy](../guide/learned_perturbation.md)
walks through a contextual bandit written as one of these.

## Benchmark config

```toml
[[heuristics]]
kind = "BreakoutLocalSearch"
tabu_tenure = [6, 160]    # the paper's rand[3, |V|/10], counted twice
t = 1000
l0 = 80                   # 0.01 * |V|
p0 = 0.8
q = 0.5
[heuristics.stop_condition]
max_duration_secs = 30.0
```

`tabu_tenure` means the same thing under every kind, a move stays forbidden
for that many iterations, so a range tuned under [`TabuSearch`](tabu_search.md)
carries over as it is. A value quoted from the paper is doubled on the way into
the file, as above.

## References

- Benlic, U. and Hao, J.-K. "Breakout Local Search for the Max-Cut problem."
  *Engineering Applications of Artificial Intelligence*, 26(3), 1162-1173,
  2013.
