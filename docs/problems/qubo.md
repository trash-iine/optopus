# QUBO

**API:** [`Qubo`](../api/optopus/problem/qubo/struct.Qubo.html)

Quadratic Unconstrained Binary Optimization (QUBO) minimizes a quadratic
polynomial over `n` binary variables `x ∈ {0,1}^n`, expressed through a
symmetric coefficient matrix `Q`. This crate stores only the upper triangle
(`i ≤ j`); the diagonal `Q[i][i]` is the linear term for `x[i]`, since
`x[i]² = x[i]` for a binary value:

```text
E(x) = Σ_{i ≤ j} Q[i][j] · x[i] · x[j]    (x ∈ {0,1}^n)
```

QUBO is the standard input format accepted by quantum and classical
Ising-machine annealers, and — being unconstrained — is also the form many
other combinatorial problems (MaxCut, graph coloring, ...) are reduced *to*.
This crate keeps `Qubo` as a standalone, generic problem type rather than
special-casing those reductions.

## Example

Running a search and reading back the minimizing assignment:

```rust
use optopus::prelude::*;

let qubo = Qubo::from_entries([
    (0, 0, -1), // diagonal = linear term
    (0, 1, 1),
    (1, 2, 2),
    (0, 2, 3),
]);
let mut state = SearchState::new(&qubo);
LocalSearch::<QuboFlipNeighbor>::new(StopCondition::iterations(10_000))
    .run(&mut state)
    .unwrap();

let sol = &state.best_solution;
println!("energy = {}", sol.objective);
println!("assignment = {:?}", sol.x); // sol.x[i] is the value of x[i] at the minimizer found
```

`Qubo::from_entries` is one way to build an instance; you can also start from
`Qubo::new()` and call `set_q` (overwrite) / `add_q` (accumulate)
incrementally.

## Solution

[`QuboSolution`](../api/optopus/problem/qubo/struct.QuboSolution.html) carries
the assignment `x` from the definition above (`x ∈ {0,1}^n`).

## Neighbors

| Type | TOML `neighbor` | Move |
|---|---|---|
| `QuboFlipNeighbor` | `"Flip"` | Flip one variable. `iter + 1`. |
| `QuboSwapNeighbor` | `"Swap"` | Swap two variables with different values. `iter + 2`. |

## Crossover

- `QuboUniformCrossover` — per-variable random parent selection.
- `Qubo` implements `SubProblemExtractable`. Variables that agree in both
  parents are fixed; their contributions are folded into the linear terms of
  the sub-QUBO so the sub-problem stays self-contained.

## File format

```text
N M
i j v
i j v
...
```

- `N` — variable count, `M` — number of entries.
- Indices are **1-indexed**; converted to 0-indexed internally.
- `i == j` lines store the linear (diagonal) coefficient.
- Duplicate entries follow `set_q` semantics: the last write wins.

```rust
use optopus::prelude::*;

let qubo = Qubo::load_file("data/instances/qubo/sample.qubo")?;
# Ok::<(), optopus::error::OptError>(())
```

## References

- Kochenberger, G., Hao, J.-K., Glover, F., Lewis, M., Lü, Z., Wang, H., and
  Wang, Y. "The Unconstrained Binary Quadratic Programming Problem: A Survey."
  *Journal of Combinatorial Optimization*, 28(1), 58-81, 2014.
- Beasley, J. E. "Obtaining Test Problems via Internet." *Journal of Global
  Optimization*, 8(4), 429-433, 1996. (OR-Library, source of the bundled
  `bqp` instance set.)

