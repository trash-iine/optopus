# RlBreakoutLocalSearch vs BreakoutLocalSearch — Gset MaxCut

Paired comparison of the learned-perturbation controller
([RlBreakoutLocalSearchForMaxCut](../heuristics/rl_breakout_local_search.md))
against the tuned hand-crafted baseline
([BreakoutLocalSearchForMaxCut](../heuristics/breakout_local_search.md)).

**Protocol**: `num_runs = 10`, `seed = 42` (per-run seeds are derived from the
master seed, so BLS-vs-RL differences are paired per `run_index`),
`max_duration_secs = 120`. Both heuristics use the same density-scaled
`tabu_tenure` / `l0` per instance class (G1: `[3,80]` / 80, G11: `[15,300]` /
20, G22: `[3,200]` / 80, G43: `[3,100]` / 80). Run on 2026-07-04.

RL parameters: **default** = `strength_bins [1,2,4]`, `learning_rate 0.1`,
`softmax_temperature 1.0`, `exploration 0.05`; **variant D** =
`strength_bins [0.5,1,2,4]`, `learning_rate 0.2`, `softmax_temperature 0.5`.

## Results (120 s × 10 runs)

| Instance | Heuristic | best | avg | worst | std | paired vs BLS (win/tie/loss, Δavg) |
|---|---|---|---|---|---|---|
| G1 (800v dense) | BLS | 11624 | 11617.9 | 11602 | 9.4 | — |
| | RL default | 11624 | **11624.0** | **11624** | **0.0** | 3/7/0, **+6.1** |
| | RL variant D | 11624 | 11622.3 | 11607 | 5.1 | 3/7/0, +4.4 |
| G11 (800v sparse ±1) | BLS | 564 | 564.0 | 564 | 0.0 | — |
| | RL default | 564 | 564.0 | 564 | 0.0 | 0/10/0, ±0 |
| | RL variant D | 564 | 564.0 | 564 | 0.0 | 0/10/0, ±0 |
| G22 (2000v) | BLS | 13347 | 13320.6 | 13297 | 17.2 | — |
| | RL default | 13359 | 13322.9 | 13269 | 28.2 | 5/1/4, +2.3 |
| | RL variant D | **13359** | **13330.8** | **13313** | **13.6** | 8/0/2, **+10.2** |
| G43 (1000v) | BLS | 6660 | 6651.0 | 6618 | 13.5 | — |
| | RL default | 6660 | **6660.0** | **6660** | **0.0** | 4/6/0, **+9.0** |
| | RL variant D | 6660 | 6656.3 | 6640 | 7.4 | 4/5/1, +5.3 |

Positive Δavg = RL better (MaxCut maximizes). At 30 s the picture is the same
(RL default: G1 +2.7, G11 ±0, G22 +6.6, G43 +8.4) and the anytime trajectories
(recorded per run in the result TOML `trajectory` field) dominate BLS at the
1/3/10/30 s checkpoints on G1/G22/G43; on G11 RL is a second or two slower to
the (always reached) optimum.

## Takeaways

- The learned controller **never loses to the hand-tuned BLS rule on average**
  on any of the four instance classes, and hits the best-known values
  11624 (G1) and 6660 (G43) in **10/10 runs** at 120 s.
- **Defaults** (`[1,2,4]`, lr 0.1, τ 1.0, ε 0.05) are the best all-rounder and
  are what the config factory uses when the optional keys are omitted.
- On **larger instances (≥ 2000 vertices)** the sharper variant D
  (`[0.5,1,2,4]`, lr 0.2, τ 0.5) was clearly stronger (G22 +10.2, 8/0/2);
  worth trying when tuning for big graphs.
