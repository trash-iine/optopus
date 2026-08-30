# MaxSAT

**API:** [`Sat`](../api/optopus/problem/sat/struct.Sat.html)

Given a propositional formula over `n` Boolean variables `x ∈ {0,1}^n` in
Conjunctive Normal Form (CNF) — a conjunction of clauses `C_1, ..., C_m`, each
a disjunction of literals (a variable or its negation) — **maximize** the
number of clauses satisfied by `x`:

```text
maximize  Σ_{k=1}^{m} [C_k(x) = true]           (x ∈ {0,1}^n, clauses C_1..C_m)
```

MaxSAT relaxes the classic (decision) SAT problem: rather than requiring
every clause to hold, which may be impossible for an over-constrained
formula, it asks for the assignment that satisfies as many as it can. This
crate implements MaxSAT throughout — even an instance that happens to be
fully satisfiable is just solved by maximizing the satisfied-clause count to
`m`.

## Example

Running a search and reading back the assignment and how many clauses it satisfies:

```rust
use optopus::prelude::*;

let mut sat = Sat::new(3);
sat.add_clause([1, -2, 3]); // (x1 ∨ ¬x2 ∨ x3); literals are signed 1-indexed
sat.add_clause([-1, 2]);
sat.add_clause([1, 2, 3]);

let mut state = SearchState::new(&sat);
LocalSearch::<SatFlipNeighbor>::new(StopCondition::iterations(10_000))
    .run(&mut state)
    .unwrap();

let sol = &state.best_solution;
println!("{} / {} clauses satisfied", sol.n_satisfied, sat.n_clauses());
for (i, &v) in sol.x.iter().enumerate() {
    println!("x{} = {v}", i + 1); // 1-indexed to match the DIMACS/add_clause convention
}
```

## Solution

[`SatSolution`](../api/optopus/problem/sat/struct.SatSolution.html) carries
the assignment `x` from the definition above (`x ∈ {0,1}^n`), and 
`n_satisfied`, which is `Σ_{k=1}^{m} [C_k(x)=true]`.

## Neighbors

| Type | Move | Iteration cost |
|---|---|---|
| [`SatFlipNeighbor`](../api/optopus/problem/sat/struct.SatFlipNeighbor.html) | Flip one variable. | `iter + 1` |
| [`SatSwapNeighbor`](../api/optopus/problem/sat/struct.SatSwapNeighbor.html) | Swap two variables. | `iter + 2` |

## Crossover

- `SatUniformCrossover` — per-variable random parent selection.
- `Sat` implements `SubProblemExtractable` for `SubProblemBasedCrossover`.

## File format (DIMACS CNF)

Note the indexing convention: `add_clause` and the file format use **signed
1-indexed** literals (positive = positive literal, negative = negation).

```text
c optional comment lines
p cnf N M
1 -2 3 0
-1 2 0
...
```

- `N` — number of variables, `M` — number of clauses.
- Each clause line is a space-separated list of signed integers terminated by
  `0`; the sign carries the polarity, the magnitude is the variable index
  (1-indexed).

```rust
use optopus::prelude::*;

let sat = Sat::load_file("data/instances/sat/example.cnf")?;
# Ok::<(), optopus::error::OptError>(())
```

## References

- "Satisfiability Suggested Format." DIMACS Challenge technical report, 1993.
  (Defines the DIMACS CNF file format.)
- Hoos, H. H. and Stützle, T. "SATLIB: An Online Resource for Research on
  SAT." In *SAT 2000*, pp. 283-292. IOS Press, 2000. (Source of the `uf`
  instance sets.)

