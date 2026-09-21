# `tabu_tenure` means one thing, and the paper's doubling lives in the config

- Status: adopted
- Area: max_cut, benchmark
- Date: 2026-09-21
- Code: src/heuristic/specific/max_cut/bls.rs (`bls_for_max_cut`),
  data/benchmarks/maxcut/gset_bls/*.toml

## Decision

`tabu_tenure` is the prohibition length `TabuMemory` stores, under every kind
that takes the key. `bls_for_max_cut` used to double the range on the way in
(`paper_effective_tenure`), so that a config could quote Benlic & Hao's `γ`
verbatim and still get the `2γ` prohibition the paper's tabu list produces,
since it adds `γ` once when it records a vertex and once more in the eligibility
test. That doubling is gone. The G-set configs now say `[6, |V|/5]` where they
said `[3, |V|/10]`, and every in-tree caller, the SubProblem crossover's inner
BLS, the examples, the docs and the pinned trajectories, was doubled with them.

## Why

The same key under `TabuSearch` and under `BreakoutLocalSearch` prohibited for
lengths that differed by a factor of two, and nothing in the file said so. A
range tuned under one kind did not transfer to the other, and the reader had to
know which of the two kinds was the special one. A number that is transformed
between the file and the engine has to be documented at every place the number
appears, and this one was documented at four.

The alternative of keeping the paper's `γ` as the config's meaning and doubling
in code is what was in place. It buys the ability to copy a parameter out of the
paper unchanged, at the price above. Copying is a one-time act per config and
the note beside the value (`# the paper's rand[3, |V|/10], counted twice`)
carries it.

## Behaviour

Nothing changed in what the engine runs. Removing the doubling and doubling
every argument is an identity, and it is checked three ways. The pins in
`tests/bls_generalization.rs` pass unchanged with `(10, 30)` in place of
`(5, 15)`. The seeded e2e run in `tests/benchmark_e2e.rs` still reproduces
bit-identically. A `seed = 42` run of `gset_bls/n800.toml` restricted to G1 and
G11 gave the same `best_objective`, `best_iteration` and `solution` before the
change with `[3, 80]` and after it with `[6, 160]`.

The published records under `docs/benchmarks/data/maxcut/bls_gset_*.slim.toml`
were rewritten to the doubled ranges too, so what they record is the config
that reproduces them under the current reading of the key.

## Do not retry

Do not reintroduce a per-kind transformation of `tabu_tenure`, and do not add a
second key (`gamma`) that means the paper's value. One key with one meaning is
the point; a config that wants the paper's parameter writes the doubled range
and says so in a comment.
