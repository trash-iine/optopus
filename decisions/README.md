# Decision records

Measured decisions about this codebase, one file per decision. These are working
notes for whoever changes the code. User-facing documentation lives in `docs/`
and in the rustdoc comments, and deliberately carries none of this.

Read the table, open the one file you need.

| Record | Area | Status | Summary |
|---|---|---|---|
| [0001](0001-bls-round-uses-generic-heuristics-and-the-gain-indexes-are-gone.md) | max_cut, qubo | adopted | Three quarters of a Breakout Local Search round are generic heuristics, and the gain indexes they made idle are gone |
| [0002](0002-a-timing-difference-follows-layout-before-it-follows-the-change.md) | all | reference | A timing difference follows code and heap layout before it follows the change, and how to A/B here |
| [0003](0003-best-swap-keeps-the-first-tied-best.md) | max_cut | adopted | Ties keep the lowest vertex index; sampling them uniformly was rejected |
| [0004](0004-rl-bls-plateau-operators-removed.md) | max_cut | adopted | The learned controller gave up two plateau operators that were paying |
| [0005](0005-an-exact-reduction-is-crossed-to-not-wrapped-and-pays-only-where-the-instance-shrinks.md) | heuristic, max_cut | adopted | An exact reduction is crossed to by any heuristic rather than wrapped, and pays only where the instance shrinks, which on MaxCut means sparse graphs |
| [0006](0006-ruin-and-recreate-needs-an-anchored-local-repair-on-routes-and-tours.md) | vrp, tsp | adopted | Ruin-and-recreate needs an anchored local repair on routes and on tours, most of all on large instances |
| [0007](0007-no-run-sub-wrapper.md) | search_state | adopted | The clone/merge triad has no wrapper, and should not get one |
| [0008](0008-documentation-is-split-three-ways.md) | all | adopted | Documentation is split three ways, and each part has one audience |
| [0009](0009-an-abstraction-costs-nothing-except-where-a-trait-asks-finer-than-the-answer-changes.md) | heuristic, all | reference | An abstraction costs nothing, except where a trait asks at a finer granularity than its answer changes at |
| [0010](0010-population-annealing-drops-the-cluster-move.md) | heuristic, max_cut, qubo | adopted | Population annealing drops the non-local cluster move and runs on every problem |
| [0011](0011-a-test-for-returning-to-the-same-solution-compares-solutions-not-objectives.md) | heuristic, max_cut | adopted | A test for returning to the same solution compares solutions through `Distance`, not their objectives, as BLS's stagnation test does |
| [0012](0012-tabu-tenure-means-one-thing.md) | max_cut, benchmark | adopted | `tabu_tenure` means one thing under every kind, and the paper's `2γ` is written into the config |
| [0013](0013-integer-problems-have-two-layers-and-the-formula-problem-sits-on-them.md) | integer | adopted | `IntegerProblem` and `FormulaProblem` sit on `IntAssignment`, and a move-pricing cache lives in the user's solution, not the framework |
| [0014](0014-vrp-is-one-problem-with-a-fleet.md) | vrp | adopted | The VRP is one problem, homogeneous and heterogeneous fleet alike, and the unification cost nothing once the layout was fixed |
| [0015](0015-a-relaxation-solved-by-steps-certifies-before-it-prunes-and-floors-its-target.md) | heuristic, max_cut | adopted | A relaxation solved by steps certifies its bound before pruning and floors its incumbent target, and on MaxCut the eigenvalue bound takes the median G-set gap from 34% to 6% |
| [0016](0016-iterated-greedy-beats-alns-on-the-flow-shop.md) | heuristic, flow_shop | reference | Iterated Greedy is its own heuristic, and on the flow shop it beats ALNS on the same `Ruinable` |
| [0017](0017-grasp-is-a-composition-and-its-construction-is-only-as-good-as-the-insertion-cost.md) | heuristic, flow_shop, bin_packing | reference | GRASP is `Iterated` with a construction step, and it pays only where cheapest insertion is a good greedy rule |
| [0018](0018-parallel-tempering-is-level-with-population-annealing-on-the-g-set.md) | heuristic, max_cut | reference | Parallel tempering shares the Metropolis sweep with population annealing, and is level with it on the G-set |
| [0019](0019-alns-greedy-repair-is-the-papers-basic-greedy.md) | heuristic, vrp, tsp | adopted | The ALNS greedy repair is the paper's basic greedy, slower on large instances and no better per iteration |
| [0020](0020-a-profile-loop-found-the-cost-in-the-problems-pricing-and-in-the-shape-of-the-scans.md) | all | reference | A profile loop over every problem found the cost in the problems' pricing and in the shape of their scans, and what was left |

## Writing a new record

Copy the shape of an existing file. Everything above the appendix, the header,
the decision and what not to retry, stays within 40 lines. The appendix holds
the measurement, its conditions and tables, and does not count toward that.
When a new decision continues an existing record, extend that record rather
than starting another.
When the code a lesson guards is generic, write the lesson in its general form
and say which problem the evidence comes from, rather than tying it to that
problem or claiming it for problems nobody measured.

```markdown
# One-line conclusion

- Status: adopted | rejected | reference
- Area: max_cut | qubo | vrp | tsp | flow_shop | bin_packing | integer | search_state | heuristic | benchmark | all
- Date: YYYY-MM-DD
- Code: src/... (when there is a specific site)

## Decision
What was decided, and why, in a few paragraphs.

## Do not retry
What should not be re-proposed, and why.

## Appendix
The measurement, its conditions and numbers.
```
