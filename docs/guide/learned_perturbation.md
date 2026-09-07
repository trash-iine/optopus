# Driving BLS with a learned perturbation policy

**API:** [`SoftmaxBandit`](../api/optopus/heuristic/reinforcement_learning/bandit/struct.SoftmaxBandit.html)

[BreakoutLocalSearchForMaxCut](../heuristics/breakout_local_search.md) exposes its
round as two halves, so a controller of your own can act at the point between
them:

```rust
let mut bls = BreakoutLocalSearchForMaxCut::externally_driven(
    StopCondition::iterations(100_000),
    /* tabu_tenure = */ (15, 300),
);

bls.descend(&mut state)?;                                   // greedy descent
let (kind, l) = my_policy.choose(&state);                   // your rule
bls.kick(&mut state, kind, l)?;                             // perturbation + update_best
```

`descend` and `kick` share the tabu memory of the `SearchState`.
The prohibitions which the descent writes are the ones the weak perturbations must not
undo.

`externally_driven` takes `tabu_tenure` literally, unlike
[`new`](../heuristics/breakout_local_search.md#benchmark-config), which reads
the same range as the paper's `γ` and forbids for `2γ`. The doubling belongs to
the paper's schedule, which an external controller replaces.

## The full example: a contextual bandit

The runnable example lives at
[`examples/rl_bls.rs`](https://github.com/trash-iine/optopus/blob/main/examples/rl_bls.rs)
(`cargo run --release --example rl_bls`). It swaps the hand-crafted rule and
the strength schedule for a contextual softmax gradient bandit
([`SoftmaxBandit`](../api/optopus/heuristic/reinforcement_learning/bandit/struct.SoftmaxBandit.html)),
and prints BLS and RL-BLS side by side on the same instance and seed.

Each outer iteration:

1. Greedy descent to a local optimum, `bls.descend`.
2. Reward observation for the previous decision, the change in local-optimum
   objective, normalized by an EMA of its own magnitude, clamped to `[−1, 1]`,
   plus a `+1` bonus when the global best improved. The bandit's per-action
   linear preferences are updated by one-step REINFORCE against an EMA
   baseline.
3. Action selection, from 7 context features,
   `[bias, min(ω/t, 1), exp(−ω/t), descent_improved_best, relative_gap,
   reward_ema, budget_progress]`, the bandit picks one of
   `3 × strength_bins.len()` actions: a perturbation type
   ([`MaxCutPerturbation`](../api/optopus/heuristic/enum.MaxCutPerturbation.html):
   `WeakFlip` / `WeakSwap` / `Strong`) together with a multiplier of `l0`.
4. Perturbation with the selected operator and strength, `bls.kick`.

`exp(−ω/t)` is exactly the probability BLS's hand-crafted rule thresholds on,
so a near-linear policy can imitate BLS quickly before improving on it.

Two library pieces make the example short: the bandit itself, and
[`SearchState::iterations_this_run`](../api/optopus/search_state/struct.SearchState.html#method.iterations_this_run),
which is what `budget_progress` normalizes by so that a sub-run inside
[`Restart`](../heuristics/meta.md#restart) does not read its parent's progress
as its own. Everything else, the deferred-reward bookkeeping, the feature
vector, the action decode, is policy, and lives in the example where you can
change it.

## Multi-episode learning

`clear()` resets the episode state (omega, the inner BLS, the pending decision,
the reward statistics) but preserves the bandit weights and baseline, so
the policy keeps improving across [`Restart`](../heuristics/meta.md#restart) /
[`Iterated`](../heuristics/meta.md#iterated) episodes. That is the same
contract [`RlSearch`](../heuristics/rl_search.md) keeps, and the reason the
weights are a field rather than a local.
