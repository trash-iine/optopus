# Population Annealing drops the non-local cluster move

- Status: adopted
- Area: heuristic, max_cut, qubo
- Date: 2026-09-07
- Code: src/heuristic/population_annealing.rs

## Decision

`PopulationAnnealing` no longer applies a non-local cluster move. The move
flipped a maximal independent set of zero-gain variables in every replica at
every temperature step. That is the iso-site move of the augmented PAMC paper,
arXiv:2606.25203. Wang, Machta and Katzgraber state the annealing step as
resampling followed by `N_S` Metropolis sweeps per replica, and mention cluster
moves only as a possible direction. The move arrived because an operator of the
learned perturbation controller happened to fit, which is also why it ran
unconditionally rather than on that paper's trigger.

Dropping it lets the search run on every registered problem. Generalizing it
would have meant a trait asking which variables have zero gain, and a gain has
a different type in every problem, so the heuristic would have been pinned to
binary problems for an augmentation. Without it the search asks a problem for
an energy and a proposal and nothing else. The loop order is corrected in the
same change. It swept, applied the move and then resampled, and both papers
resample first and sweep at the new temperature.

## Do not retry

Do not put the iso-site move back inside `PopulationAnnealing`. If it is wanted,
it belongs outside the heuristic as a component the caller supplies, so that the
search stays the algorithm the paper states.

The cluster move worth trying instead is the isoenergetic cluster move of Zhu,
Ochoa and Katzgraber, arXiv:1501.05630. It grows a connected component inside
the region where two replicas disagree and exchanges their values on it. It
conserves the pair's energy, needs no gain and works on non-binary variables,
so it would not bring back the restriction removed here, and population
annealing already holds the population it needs. It is unimplemented because
this change returned to the stated algorithm rather than swapping augmentations.

## Appendix

120 seconds by five runs, same seeds, arms alternated, against the
implementation this replaces. MaxCut maximizes and QUBO minimizes, so the better
number is the larger one in the first three rows and the smaller one in the
fourth.

| Instance | Old best / avg / std | New best / avg / std |
|---|---|---|
| G55 | 10296 / 10293.4 / 1.4 | 10293 / 10292.4 / 0.5 |
| G63 | 27020 / 27009.0 / 7.4 | 27011 / 27007.2 / 3.2 |
| G70 | 9586 / 9584.4 / 1.7 | 9586 / 9582.8 / 2.5 |
| bqp1000_1 | -156690 / -156303.8 / 299.9 | -156560 / -156328.2 / 160.8 |

The move is worth a few cut and no more. It costs three on G55's best and nine
on G63's, is level on G70's best, and on QUBO the two arms trade: the old one
reaches a better best by 130 and the new one a better average by 24, both well
inside a standard deviation of 160 to 300. The same measurement was made when
the move was added and gave the same answer, two cut on G55 and five on G70.

Removing it also narrows the spread on three of the four instances. What the
move bought was an occasional better peak, not a better search.
