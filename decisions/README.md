# Decision records

Measured decisions about this codebase, one file per decision. These are working
notes for whoever changes the code. User-facing documentation lives in `docs/`
and in the rustdoc comments, and deliberately carries none of this.

Read the table, open the one file you need.

| Record | Area | Status | Summary |
|---|---|---|---|
| [0001](0001-bls-round-uses-generic-heuristics.md) | max_cut | adopted | Three quarters of a Breakout Local Search round are generic heuristics |
| [0002](0002-auxiliary-gain-indexes-removed.md) | max_cut, qubo | adopted | The optional improving-move and plateau indexes were all removed |
| [0003](0003-maxcut-timings-follow-code-alignment.md) | max_cut | reference | A timing difference on MaxCut is not on its own evidence, and how to A/B here |
| [0004](0004-tabu-memory-stays-inline.md) | search_state | adopted | Fat LTO removes the allocation-layout cliff, so the sparse map is not boxed |
| [0005](0005-best-swap-keeps-the-first-tied-best.md) | max_cut | adopted | Ties keep the lowest vertex index; sampling them uniformly was rejected |
| [0006](0006-local-search-takes-its-stop-condition-as-given.md) | heuristic | adopted | `LocalSearch::new` no longer forces `max_failed_update` |
| [0007](0007-rl-bls-plateau-operators-removed.md) | max_cut | adopted | The learned controller gave up two plateau operators that were paying |
| [0008](0008-hgs-and-alns-are-level-at-30s.md) | vrp | reference | HGS no longer beats ALNS at equal budget |
| [0009](0009-maxcut-kernelization-pays-on-sparse-only.md) | max_cut | adopted | Exact reduction is worth crossing to on sparse graphs and nowhere else |
| [0010](0010-planted-instance-hardness.md) | max_cut | reference | Where the planted suites are hard, and why published parameters do not transfer |
| [0011](0011-vrp-distance-is-symmetrized.md) | vrp | adopted | The broken-pairs count is asymmetric at the depot, so `Distance` takes the larger direction |
| [0012](0012-alns-descent-is-anchored.md) | vrp | adopted | The descent after each ruin is anchored at the re-inserted customers |
| [0013](0013-no-run-sub-wrapper.md) | search_state | adopted | The clone/merge triad has no wrapper, and should not get one |

## Writing a new record

Copy the shape of an existing file and keep it under 40 lines.

```markdown
# One-line conclusion

- Status: adopted | rejected | reference
- Area: max_cut | qubo | vrp | search_state | heuristic | benchmark | all
- Date: YYYY-MM-DD
- Code: src/... (when there is a specific site)

## Decision
Two or three sentences.

## Measurement
Conditions and numbers.

## Do not retry
What should not be re-proposed, and why.
```
