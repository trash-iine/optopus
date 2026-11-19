# String removal with blinks runs on routes without a local search, and stays a VRP heuristic

- Status: adopted
- Area: heuristic, vrp
- Date: 2026-11-15
- Code: src/heuristic/specific/vrp/sisr.rs, src/problem/vrp/ruin.rs

## Decision

`SlackInductionByStringRemoval` is a ruin-and-recreate search for the VRP with
one ruin, one recreate and no local search. The ruin takes one string of
consecutive customers from each of a few routes near a seed customer, the
recreate inserts them greedily and skips each place with a small probability,
and simulated annealing accepts. The rule and its constants were read in the
original paper: an average of 10 removed customers, strings of at most 10,
split strings half of the time with a stop rate of 0.01, a blink rate of 0.01,
and a temperature falling from 100 to 1.

It lives under `heuristic/specific/vrp` and reads the route partition through
two accessors on `VrpPartial`. It was not written over `Ruinable`. A string is
a run of consecutive elements in one container among several, and two of the
four insertion orders read a demand and a depot. Of the `Ruinable` problems
only the VRP has all of that, and a trait method with one implementer is not
a generalization.

Three things differ from the paper.

- A place is priced with the problem's penalty, where the paper inserts only
  into a route with room and opens a route otherwise. The fleet here is fixed,
  so an idle vehicle is one more place, and with none left the search runs on
  the penalized objective.
- The search starts from the problem's own initial solution, not from one
  route per customer.
- The cooling has its own length, `cooling_iterations`, since a run stopped
  by time has no iteration count. Past it the temperature stays at its final
  value.

The fleet minimisation stage of the paper is not implemented.

## Do not retry

Adding the anchored route descent after the recreate was not measured, and is
the first thing that looks like an improvement. The search wins by making two
orders of magnitude more iterations than ALNS in the same time, so a descent
has to pay for the iterations it costs. Measure it at equal time before
keeping it.

Each iteration converts the solution to a partial and back, which is O(n) on
top of the ruin and the recreate. Keeping the partial across iterations and
undoing a rejected candidate is the known way to remove it, and is not done.

## Appendix

`data/benchmarks/vrp/sisr_medium.toml`, five runs, seed 42, 60 s, the five
runs in parallel on an Apple M4. Average objective, every run feasible.

| Instance | SlackInductionByStringRemoval | AdaptiveLargeNeighborhoodSearch | HybridGeneticSearch |
|---|---|---|---|
| X-n214-k11 | 10950.6 | 10967.4 | 11072.0 |
| X-n303-k21 | 21890.8 | 22045.4 | 22185.2 |
| X-n401-k29 | 66393.4 | 67890.2 | 67590.6 |

The lead on X-n214-k11 is inside the spread of the runs, 46 for the string
removal and 41 for ALNS. Iterations per run were 2.4, 1.9 and 1.6 million for
the string removal, against 40, 16 and 10 thousand for ALNS. The cooling
length was 2 million, so the run on X-n401-k29 ended at a temperature near 3
and has not finished cooling.
