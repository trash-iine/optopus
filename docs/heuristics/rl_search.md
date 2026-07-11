# RlSearch

Online reinforcement learning over move features. At each step:

1. Enumerate all (or a subsample of) neighborhood moves and compute their
   worsening amounts.
2. Score each move with a linear policy over `NUM_FEATURES` (= 21)
   interaction features: 3 move-level features (normalized gain, is-improving,
   approximate rank) each on its own and multiplied by 6 state-level features
   (progress, stagnation, improvement ratio, neighborhood statistics), so the
   search state modulates the move preferences.
3. Sample one move from the resulting softmax distribution.
4. Apply the move, then update the policy via single-step REINFORCE with
   baseline subtraction.

## Constructor

```rust
RlSearch::<N>::new(
    stop_condition: StopCondition,
    learning_rate: f64,
    softmax_temperature: f64,
    reward_shaping: RewardShaping,
    max_candidates: Option<usize>,
) -> Self
```

`N` must satisfy `MoveToNeighbor<P> + Evaluate + Clone`.

`max_candidates`: if set, reservoir-samples this many moves from the lazy
neighborhood iterator **before** evaluating them, so per-step evaluation and
feature cost is `O(max_candidates)` instead of `O(neighborhood)`. Step
statistics (and therefore the neighborhood-level features) are computed over
the sample only.

`policy_weights` files trained before the feature-interaction change (9
elements) are rejected; retrain with the current 21-element layout. The old
`discount` parameter was removed: single-step REINFORCE has no discount
factor (the TOML key is still accepted but ignored, with a warning).

`with_policy_weights([f64; NUM_FEATURES])` lets you seed the policy with
pre-trained weights.

## Reward shaping

```rust
pub enum RewardShaping {
    Raw,                // -worsening
    Normalized,         // -worsening / step's max |worsening|
    BestImprovement,    // 1.0 if a new best was found, else 0.0
}
```

## Multi-episode learning

`clear()` resets per-episode state but **preserves `policy.weights` and the
running baseline**. Wrap `RlSearch` in [`Restart`](meta.md#restart) or
[`Iterated`](meta.md#iterated) to train across many episodes:

```rust
use optopus::prelude::*;

let rl = RlSearch::<MaxCutFlipNeighbor>::new(
    StopCondition::failed_updates(1_000),
    0.01, 1.0,
    RewardShaping::Normalized,
    Some(64),
);
let mut solver = Restart::new(
    StopCondition::iterations(1_000_000),
    Box::new(rl),
    StopCondition::failed_updates(10_000),
);
```

## References

- Williams, R. J. "Simple Statistical Gradient-Following Algorithms for
  Connectionist Reinforcement Learning." *Machine Learning*, 8(3-4), 229-256,
  1992.
