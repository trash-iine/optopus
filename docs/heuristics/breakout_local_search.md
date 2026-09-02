# BreakoutLocalSearchForMaxCut

**API:** [`BreakoutLocalSearchForMaxCut`](../api/optopus/heuristic/struct.BreakoutLocalSearchForMaxCut.html)

Problem-specific heuristic for [MaxCut](../problems/max_cut.md). Alternates a
greedy local search phase with an adaptive perturbation phase, using the
optional `positive_gain` index on `MaxCutSolution` to enumerate only improving
flips in O(|improving|). The descent, the tabu walk and the perturbations it
drives are free functions in `src/heuristic/specific/max_cut/ops/`, shared with
the other MaxCut heuristics; what is BLS's own is the schedule below. All of
them record into — and read — the tabu memory on the `SearchState` they are
handed, which is what stops a perturbation undoing the descent that just ran.

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
  its gain update already implies. Here selection is a linear scan: the descent
  narrows it with the optional `positive_gain` index on `MaxCutSolution`, so it
  costs O(|{v : gain(v) > 0}|) per move — bounded by n, and shrinking as the
  descent approaches a local optimum — while the tabu walk and the weak swap
  scan **all n** flip neighbours per move, O(n). The gain update itself is
  O(degree(v)). The same move is selected either way, so this costs
  only speed.
- **A swap advances the iteration counter by 2**
  (`MaxCutSwapNeighbor::apply_to_iteration`), where BLS counts every move as
  one. That `+2` is a library-wide convention shared by every binary problem's
  swap, so it is not changed here for one heuristic's sake.

## Why the operators are not the generic heuristics

Once `SearchState::apply` became what records a move into the tabu memory, the
operators stopped needing a shared object — and the question arose whether BLS
could simply drive `LocalSearch` and `RandomWalk` instead. It was measured: BLS
at 30s x 5 runs, seed 42, over G1 / G11 / G22 / G32 / G43 / G55 / G60 / G63 /
G70 / G81, with `l0 = 0.01|V|` and the density-scaled tenure.

Replacing the descent with `LocalSearch` costs **-40.0 total average cut**
(better on 3 instances, worse on 5): -21.2 on G81, -12.4 on G63, -8.4 on G70,
-6.4 on G55, against +8.2 on G60. Most of those sit inside one or two standard
deviations, but the reason behind them does not: in the same wall clock it
completed only **0.74-0.93x as many moves**, on every one of the ten. Both
select from the same candidate set — `is_neighbor_better_than_current` on a
flip is `gain > 0` — but `LocalSearch` rescans all `n` vertices per move where
the descent enumerates only the improving ones through `positive_gain`. Two
further differences ride along and are not separated by that number: `max_by`
breaks gain ties toward the last candidate where `keep_best` keeps the first,
and `LocalSearch` spends one extra `progress_iteration` per descent.

Replacing the strong perturbation with `RandomWalk` is, by the same measurement,
**free** — applied on top of the above it moved the total by +0.4 with
throughput unchanged. It was still not taken: it deletes no file (`best_swap`
keeps `perturbation.rs` alive regardless), and `RandomWalk` fails with
`InvalidState` on the edgeless sub-instances `SubProblemBasedCrossover`
produces, so the guard the operator holds would only move into the caller.

Two more differences are structural rather than measured. `best_swap` has no
generic counterpart at all: `TabuSearch<MaxCutSwapNeighbor>` enumerates O(n²)
vertex pairs where `best_swap` scans each partition side once, O(n) — 4·10⁸
against 2·10⁴ per step on G81. And `LocalSearch::new` forces
`max_failed_update = Some(1)`; after a kick the solution is worse than the
global best, so `Heuristic::run` returns before taking a single move unless that
field is cleared by hand.

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
optimum dropped). The prohibitions are not its to clear: they belong to the
`SearchState`, and a sub-run clone — how every meta-heuristic starts a phase —
already arrives with an empty tabu memory (`state.reset_tabu()` for a state you
are reusing directly).
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
