# Quadratic Assignment

**API:** [`Qap`](../api/optopus/problem/qap/struct.Qap.html)

The quadratic assignment problem places `n` facilities at `n` locations, one
each. Matrix `a` holds what passes between every pair of facilities and `b`
the distance between every pair of locations, and an assignment `p` costs

```text
minimize  Σ_i Σ_j a[i][j] · b[p(i)][p(j)]       p a permutation of 0..n
```

Neither matrix has to be symmetric. This is the Koopmans and Beckmann form,
with the matrices in the order QAPLIB lists them. The problem is NP-hard to
approximate, and instances of a few dozen facilities are already beyond exact
methods.

## Example

```rust
use optopus::prelude::*;

let qap = Qap::load_file("data/instances/qap/tai30a.dat")?;
let n = qap.n as u64;
let mut state = SearchState::new_with_seed(&qap, 42);
TabuSearch::<QapSwapNeighbor>::new(
    StopCondition::duration(std::time::Duration::from_secs(10)),
    /* tabu_tenure = */ (n * 9 / 10, n * 11 / 10),
)
.run(&mut state)?;
println!("cost = {}", state.best_solution.objective);
```

`TabuSearch` draws a tenure from its range for every move it makes, and
`QapSwapNeighbor` follows Taillard's tabu rule, so this is his robust tabu
search with nothing added. Wrapped in [`Iterated`](../heuristics/meta.md) with
a few random exchanges between runs it is the Iterated Tabu Search of
Misevicius, which
[`examples/integer_qap.rs`](https://github.com/trash-iine/optopus/blob/main/examples/integer_qap.rs)
runs next to it.

## Solution

[`QapSolution`](../api/optopus/problem/qap/struct.QapSolution.html) carries
`assignment`, where `assignment[i]` is the location of facility `i`, and
`objective`, its cost. `Qap::solution_from_assignment` wraps an assignment of
your own, and `Qap::cost` prices one without building a solution.

## Neighbors

| Type | Config | Move | Cost |
|---|---|---|---|
| `QapSwapNeighbor` | `Swap` | Exchange the locations of two facilities. | O(n) per move, O(n³) per scan |

The move is priced by the general formula for matrices that need not be
symmetric, `Qap::swap_delta`. Its tabu rule forbids each facility the
location it just left, and a move is tabu only when it would put both of its
facilities back, which is Taillard's rule.

## Crossover

`QapOrderCrossover` is Order Crossover (OX) over the assignment.

## File format (QAPLIB)

```text
n
a, n rows of n values
b, n rows of n values
```

All values are whitespace separated, and a row may wrap over several lines as
it does in QAPLIB's wide instances. QAPLIB states no terms for redistribution,
so its instances are not bundled. `data/instances/scripts/fetch_qaplib.sh`
downloads Taillard's `tai*a` family and a few classic ones into
`data/instances/qap/`.

## References

- Koopmans, T. C. and Beckmann, M. "Assignment Problems and the Location of
  Economic Activities." *Econometrica*, 25(1), 53-76, 1957.
- Taillard, E. "Robust taboo search for the quadratic assignment problem."
  *Parallel Computing*, 17(4-5), 443-455, 1991.
- Misevicius, A. "A tabu search algorithm for the quadratic assignment
  problem." *Computational Optimization and Applications*, 30(1), 95-111, 2005.
- Burkard, R. E., Karisch, S. E. and Rendl, F. "QAPLIB, a quadratic
  assignment problem library." *Journal of Global Optimization*, 10(4),
  391-403, 1997.
