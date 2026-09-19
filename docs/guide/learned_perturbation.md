# Driving BLS with a learned perturbation policy

**API:** [`SoftmaxBandit`](../api/optopus/heuristic/reinforcement_learning/bandit/struct.SoftmaxBandit.html)

A rule of your own goes in where Benlic & Hao's does, as a
`PerturbationSchedule` handed to
[BreakoutLocalSearch](../heuristics/breakout_local_search.md). It owns the
operators it chooses between, decides in `determine_jump_magnitude`, and hands
back what it decided in `select`:

```rust
impl PerturbationSchedule<MaxCut> for MyPolicy {
    fn determine_jump_magnitude(&mut self, state: &mut SearchState<'_, MaxCut>) -> u64 {
        // the one point the schedule is handed the state, so decide here
        let (operator, l) = self.choose(state);
        self.chosen = operator;
        l
    }

    fn select<'s>(&'s mut self, _state: &mut SearchState<'_, MaxCut>)
        -> &'s mut dyn Heuristic<MaxCut>
    {
        self.operators[self.chosen].as_mut()
    }
    // reset / clear_perturbations, and round_ended when the rule needs it
}

let mut bls = BreakoutLocalSearch::new(
    StopCondition::iterations(100_000),
    /* tabu_tenure = */ (15, 300),
    max_cut_descent(),
    MyPolicy::new(...),
);
```

The descent and the perturbations share the tabu memory of the `SearchState`.
The prohibitions the descent writes are the ones the weak perturbations must
not undo.

`max_cut_perturbation`, which builds those operators, takes `tabu_tenure`
literally, unlike
[`bls_for_max_cut`](../heuristics/breakout_local_search.md#benchmark-config),
which reads the same range as the paper's `γ` and forbids for `2γ`. The
doubling belongs to the paper's schedule, which your rule replaces.

## The full example: a contextual bandit

The runnable example lives at
[`examples/rl_bls.rs`](https://github.com/trash-iine/optopus/blob/main/examples/rl_bls.rs)
(`cargo run --release --example rl_bls`). It swaps the hand-crafted rule and
the strength schedule for a contextual softmax gradient bandit
([`SoftmaxBandit`](../api/optopus/heuristic/reinforcement_learning/bandit/struct.SoftmaxBandit.html)),
and prints BLS and RL-BLS side by side on the same instance and seed.

Each round, with everything from step 2 to step 4 inside
`determine_jump_magnitude`:

1. Greedy descent to a local optimum, run by the framework.
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
4. The chosen strength is returned, `select` hands back the chosen operator,
   and the framework applies it and closes the round on `update_best`.
   `round_ended` then records the global best, which is what the next round
   compares against to fill `descent_improved_best`.

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

`clear()` resets the episode state (omega, the operators, the pending decision,
the reward statistics) but preserves the bandit weights and baseline, so
the policy keeps improving across [`Restart`](../heuristics/meta.md#restart) /
[`Iterated`](../heuristics/meta.md#iterated) episodes. That is the same
contract [`ReinforcementLearningSearch`](../heuristics/rl_search.md) keeps, and the reason the
weights are a field rather than a local.
