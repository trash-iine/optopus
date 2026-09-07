# Population Annealing drops the non-local cluster move

- Status: adopted
- Area: heuristic, max_cut, qubo
- Date: 2026-09-07
- Code: src/heuristic/population_annealing.rs

## Decision

`PopulationAnnealing` no longer applies a non-local cluster move.

The move flipped a maximal independent set of zero-gain variables in every
replica at every temperature step. That is the iso-site move of the augmented
PAMC paper, arXiv:2606.25203, not something population annealing requires.
Wang, Machta and Katzgraber state the whole annealing step as "in each annealing
step the population is resampled and then N_S sweeps of the Metropolis algorithm
are carried out on each replica", and mention cluster moves only as a direction
that might be useful. The move arrived here because an existing operator from
the learned perturbation controller happened to fit, which is also why it ran
unconditionally rather than on the trigger that paper describes.

Dropping it is what lets the search run on every registered problem rather than
on MaxCut and QUBO. Generalizing the move would have meant a trait asking the
problem which of its variables currently have zero gain, and a gain has a
different type in every problem, so that trait would have pinned the heuristic
to binary problems for the sake of an augmentation. Without the move the search
asks a problem for an energy and a proposal and nothing else.

The loop order is corrected in the same change. It used to sweep, then apply the
cluster move, then resample; both papers resample first and sweep at the new
temperature.

## Measurement

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

## Do not retry

Do not put the iso-site move back inside `PopulationAnnealing`. If it is wanted,
it belongs outside the heuristic as a component the caller supplies, so that the
search stays the algorithm the paper states.

The cluster move worth trying instead is the isoenergetic cluster move of Zhu,
Ochoa and Katzgraber, arXiv:1501.05630. It takes a pair of replicas, forms the
site overlap between them, grows a connected component inside the disagreeing
region and exchanges the two replicas' values on it. It conserves the pair's
total energy, needs no gain, and generalizes to non-binary variables unchanged,
so it would not bring back the restriction this record removes. Population
annealing also already holds the population that the original method needed
parallel tempering to supply. It is unimplemented because this change was about
returning to the stated algorithm, not about replacing one augmentation with
another.
