# BreakoutLocalSearchForMaxCut

**API:** [`BreakoutLocalSearchForMaxCut`](../api/optopus/heuristic/struct.BreakoutLocalSearchForMaxCut.html)

Problem-specific heuristic for [MaxCut](../problems/max_cut.md). Alternates a
greedy local search phase with an adaptive perturbation phase. What is BLS's own
is the schedule below: three of the four things a round does are the library's
generic heuristics — the descent is a [`LocalSearch`](local_search.md), the
strong perturbation a [`RandomWalk`](random_walk.md) and the weak flip a
[`TabuSearch`](tabu_search.md) — and only the weak swap is a hand-written
operator (`src/heuristic/specific/max_cut/best_swap.rs`). All of them record
into — and read — the tabu memory on the `SearchState` they are handed, which is
what stops a perturbation undoing the descent that just ran; the generic three
do it through `apply` on a state BLS has switched into recording mode.

## Example

```rust
use optopus::prelude::*;

let mut rng = seeded_rng(42);
let mc = MaxCut::new(Graph::erdos_renyi(800, 0.02, &mut rng));
let mut state = SearchState::new_with_seed(&mc, 42);

let mut bls = BreakoutLocalSearchForMaxCut::new(
    StopCondition::iterations(100_000),
    /* tabu_tenure = */ (5, 150),
    /* t           = */ 1_000,
    /* l0          = */ 8,          // 0.01 * |V|
    /* p0          = */ 0.8,
    /* q           = */ 0.5,
);
bls.run(&mut state)?;
println!("cut weight = {}", state.best_solution.objective);
# Ok::<(), optopus::error::OptError>(())
```

`l0` and `tabu_tenure` are instance-dependent, so they are derived from
`|V| = 800` here rather than left at a constant — see
[Benchmark config](#benchmark-config).

## Algorithm sketch

- **Greedy phase**: repeatedly apply the strictly best improving flip, updating
  a tabu map.
- **Perturbation phase**: `p = max(exp(−omega / t), p0)` is the probability of a
  *directed* (weak) perturbation, and it decays as the non-improvement counter
  `omega` grows:
  - `omega == 0` — the last descent improved the global best, or `omega` just
    passed `t` and was reset: a strong perturbation (random flips) runs.
  - `0 < omega <= t`: **weak flip** with probability `p * q`, **weak swap** with
    probability `p * (1 − q)`, **strong** (random flips) with probability
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

Everything here was measured against the cut values Benlic & Hao publish, on
G22 / G27 / G33 / G35 / G39 at one tenth of their budget, five runs each.

- **The tenure parameter is doubled on the way in.** The original tenure is
  added once when a vertex is recorded and once more in the eligibility test, so
  a vertex stays forbidden for twice it. `TabuMemory` stores a single tenure, so
  `paper_effective_tenure` doubles the caller's range and `tabu_tenure` keeps
  the original meaning, `rand[3, n/10]` on the G-set. Doubling only the upper
  bound does not reproduce it — the whole range has to scale.
- **No bucket sort.** The original buckets vertices by gain, so selecting a
  maximum-gain move is O(1) and a move costs only the O(degree(v)) rebucketing
  its gain update already implies. Here every selection is a linear scan over
  **all n** flip neighbours, O(n) per move — the descent included, since it is
  a plain `LocalSearch` (see [below](#why-three-quarters-of-a-round-are-generic-heuristics)).
  The gain update itself is O(degree(v)). The same move is selected either way,
  so this costs only speed.
- **A swap advances the iteration counter by 2**
  (`MaxCutSwapNeighbor::apply_to_iteration`), where BLS counts every move as
  one. That `+2` is a library-wide convention shared by every binary problem's
  swap, so it is not changed here for one heuristic's sake.

## Why three quarters of a round are generic heuristics

The descent, the strong perturbation and the weak flip are `LocalSearch`,
`RandomWalk` and `TabuSearch` rather than hand-written operators. They select the same moves a dedicated
operator would, and — since recording became a mode on the `SearchState` that
`prepare` arms — their `apply` writes the same prohibitions, so Benlic & Hao's
`H <- Iter + gamma` inside the descent loop holds either way. **This is not
free, and the price was measured before it was paid**, on
G1/G11/G22/G32/G43/G55/G60/G63/G70/G81 under a fixed iteration budget (one run
per instance, run sequentially so that parallel runs cannot land on efficiency
cores, three repetitions, minimum taken, timing only `Heuristic::run`):

- **The descent cost 1.15x the time for the same iterations** (1.03x on G63 to
  1.26x on G43, stable to within 0.03 across repetitions, against a machine
  noise floor of 1.01-1.06). It replaced a scan of an index of the improving
  flips — O(|improving|), which shrinks as the descent approaches its local
  optimum — with `LocalSearch`'s scan of all `n` flips, so the loss is worst on
  the small dense instances. `LocalSearch` also spends an iteration detecting
  the local optimum it has reached, where an empty index reported the same thing
  for free, which is 0.4-7.7% of the budget. That index (`MaxCutSolution`'s
  `positive_gain`) was deleted along with the descent: it was the last reader,
  and nothing outside the crate could ever read it. At a fixed 30s budget the two together are **−41.8 cut points in
  total** (3 instances better, 5 worse), concentrated on the large sparse
  instances: G81 −22.4, G63 −12.6, G70 −8.6, against G60 +8.0. `lto = "fat"`
  does not change the ratio — the same substitution cost −40.0 before it was
  enabled — because what is being paid for is the scan.
- **The strong perturbation cost 1-3%** and nothing else: at a fixed budget
  `RandomWalk` applies the same number of moves and reaches a **bit-identical
  best solution on all ten instances**. The overhead is one `Result`-returning
  `random_neighbor` call and a per-move `update_best` that `apply_move_only`
  deferred.

Reversing either one is a speed-only change; nothing about the schedule or the
tabu memory depends on which side of this line an operator sits.

One warning for anyone tidying `descend`: it hands `LocalSearch` an unreachable
iteration cap rather than the `StopCondition::new(None, None, None)` that says
the same thing, because the all-`None` spelling measured **7% slower** on every
instance of the timing suite while producing bit-identical solutions. Both
spellings leave `no_best_move` as the only thing that ends the run, so the
difference is in code generation, not in the search.

- **The weak flip was free in both directions, and then some.** `TabuSearch`
  selects by exactly the predicate the hand-written `tabu_walk` did —
  `tabu_allows(n) || is_neighbor_better_than_best(n)` expands to the same
  `gain + objective > best_objective` — and at 30s x 5 runs it is **+318.2 cut
  points in total, better on 7 of 10 instances and worse on none** (G81 +210.0,
  G70 +37.2, G63 +20.6, G55 +20.2, against standard deviations of 15-39). At a
  fixed iteration budget it is also **1.8x faster overall**, and 3.7x faster
  when every weak perturbation is a flip (`q = 1.0`).

  That speedup is **not explained**. Ruled out by measurement: the two run the
  same number of weak-flip iterations (693,099 vs 703,242 of a 900,000 budget)
  with the same perturbation length; the tie rule (`keep_best` keeps the first
  tied-best, `max_by` the last) does not account for it; neither does
  `apply_move_only` vs `apply` (15.14s vs 15.04s with that one line changed);
  and the two scan loops compile to the same 14 instructions per element. The
  one clue is that the ratio grows with instance size — 2.5x at n = 800, 4x at
  n = 20000 — so it behaves like a per-element memory effect rather than fixed
  overhead. Recorded as an open question rather than guessed at.

The weak swap stays a hand-written operator because it is not expressible as a
generic search at all: `M2` moves one vertex per partition side **in a single
move**, and a pair of one-step searches succeed or fail independently, so a side
with nothing eligible leaves the other vertex moved on its own.

## Constructor

```rust
BreakoutLocalSearchForMaxCut::new(
    stop_condition: StopCondition,
    tabu_tenure: (u64, u64),
    t: u64,
    l0: u64,
    p0: f64,
    q: f64,
) -> Self
```

| Parameter | Meaning |
|---|---|
| `tabu_tenure` | tabu tenure range `(min, max)` for the LS phase |
| `t` | period of the `omega` counter before it resets |
| `l0` | initial perturbation length |
| `p0` | minimum perturbation probability |
| `q` | fraction of weak perturbations using flip (vs. swap) |

`clear()` resets the schedule (`omega` to 0, `l` to `l0`, the remembered local
optimum dropped). The tabu are not its to clear, because they belong to the
`SearchState`. But the tabu is cleared in a sub-run clone.
The remembered local optimum is dropped rather than kept because the same
schedule may be reused on a different instance — a meta-heuristic that rebuilds
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
moves. This is the one kind that doubles the key — the same range under
[`TabuSearch`](tabu_search.md) prohibits for half as long, so tuned values do
not transfer between them.

## Driving it yourself

The round is also reachable in halves, for a caller that wants to keep the
descent and the operators but replace the schedule:

- `descend(state)` — the greedy descent, writing the prohibitions the kick reads.
- `kick(state, perturbation, l)` — one [`MaxCutPerturbation`](../api/optopus/heuristic/enum.MaxCutPerturbation.html)
  of length `l`, followed by the round's single `update_best`.
- `externally_driven(stop_condition, tabu_tenure)` — the constructor for that
  use, taking the tenure **literally**: the `2γ` doubling above belongs to the
  paper's schedule, which such a caller replaces.

`run_once` is exactly `descend` + the schedule + `kick`.
[Driving BLS with a learned perturbation policy](../guide/learned_perturbation.md)
walks through a controller that decides between the two halves with a
contextual bandit.

## References

- Benlic, U. and Hao, J.-K. "Breakout Local Search for the Max-Cut problem."
  *Engineering Applications of Artificial Intelligence*, 26(3), 1162-1173,
  2013.
