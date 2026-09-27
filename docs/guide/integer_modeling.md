# Writing problems with integer variables

**API:** [`IntegerProblem`](../api/optopus/problem/integer/struct.IntegerProblem.html)

Every problem the library ships can also be written as integer variables and
an objective, with no move to define and no trait to implement. This page does
that for all eight of them. Each section shows the part that differs from
problem to problem, and links a runnable example that holds the rest.

Written this way, a problem is slower than its own type. The built-in problems
price a move from what it touches, while an
[`IntegerProblem`](../problems/integer.md) evaluates the whole objective for
every candidate. When that matters, give it a delta as
[Making it fast](../problems/integer.md#making-it-fast) shows, or use the
problem's own type.

## Three shapes of variables

| Variables | Declared with | Move | Used for |
|---|---|---|---|
| one binary variable per item | `IntVar::binary()` | `IntChangeNeighbor`, a flip | MaxCut, QUBO, MaxSAT, Vertex Cover |
| one variable per item, ranging over a few values | `IntVar::new(lower, upper)` | `IntChangeNeighbor` | Graph Coloring |
| a permutation | `IntVars::permutation(n)` | `IntSwapNeighbor`, `IntReverseNeighbor` | TSP, Job Shop Scheduling, CVRP |

## Running the search

Once the problem is built, every example ends the same way. The search is
picked by naming the move, and the best solution's values are read back with
`values()`.

```rust
let mut state = SearchState::new_with_seed(&prob, 42);
SimulatedAnnealing::<IntChangeNeighbor>::new(StopCondition::iterations(20_000), 2.0, 0.9995)
    .run(&mut state)
    .unwrap();
println!("{:?}", state.best_solution.values());
println!("{:?}", state.best_solution.evaluate());
```

The examples use `SimulatedAnnealing`, which prices one random candidate per
iteration. `LocalSearch` and `TabuSearch` price every candidate of the
neighborhood per iteration, which costs a full evaluation each when no delta
is given, so they suit small neighborhoods such as the MaxSAT one below.

## MaxCut

One binary variable per vertex says which side of the cut it is on. The
objective adds up the weight of every edge whose two ends differ.

```rust
let graph = Graph::erdos_renyi(100, 0.1, &mut seeded_rng(1));

let vars: IntVars = (0..graph.len()).map(|_| IntVar::binary()).collect();
let prob = IntegerProblem::maximize(vars, |x: &[i64]| {
    graph
        .edges()
        .filter(|&(i, j, _)| x[i] != x[j])
        .map(|(_, _, w)| w as f64)
        .sum()
});
```

[`examples/integer_max_cut.rs`](https://github.com/trash-iine/optopus/blob/main/examples/integer_max_cut.rs)
(`cargo run --example integer_max_cut`)

## QUBO

The energy `Σ Q[i][j] x_i x_j` is a polynomial, so it is written as an `Expr`
and handed to a [`FormulaProblem`](../problems/formula.md). That problem works
out what every flip changes from the expression, so it is fast with nothing
more written.

```rust
let qubo = Qubo::load_file("data/instances/qubo/bqp/bqp100_1.txt").unwrap();

let vars: IntVars = (0..qubo.len()).map(|_| IntVar::binary()).collect();
let energy = qubo.entries().fold(Expr::Const(0.0), |sum, (i, j, q)| {
    sum + q as f64 * Expr::Var(i) * Expr::Var(j)
});
let prob = FormulaProblem::minimize(vars, energy);
```

[`examples/integer_qubo.rs`](https://github.com/trash-iine/optopus/blob/main/examples/integer_qubo.rs)
(`cargo run --example integer_qubo`)

## MaxSAT

One binary variable per boolean variable, and the objective counts the clauses
that hold. A literal `l` asks variable `|l| - 1` to be `1` when `l` is
positive and `0` when it is negative.

```rust
let sat = Sat::load_file("data/instances/sat/sample.cnf").unwrap();

let vars: IntVars = (0..sat.n_vars()).map(|_| IntVar::binary()).collect();
let prob = IntegerProblem::maximize(vars, |x: &[i64]| {
    let holds = |l: i64| (x[l.unsigned_abs() as usize - 1] == 1) == (l > 0);
    sat.all_clauses()
        .filter(|clause| clause.iter().any(|&l| holds(l)))
        .count() as f64
});
```

With twenty variables the neighborhood is small, so the example runs
`TabuSearch::<IntChangeNeighbor>`.

[`examples/integer_max_sat.rs`](https://github.com/trash-iine/optopus/blob/main/examples/integer_max_sat.rs)
(`cargo run --example integer_max_sat`)

## TSP

The variables are a permutation, variable `p` being the city visited `p`th,
and the objective is the length of the closed tour. `IntReverseNeighbor`
reverses a stretch of the tour, which is a 2-opt move.

```rust
let tsp = Tsp::load_file("data/instances/tsp/eil51.tsp").unwrap();
let n = tsp.get_n();

let prob = IntegerProblem::minimize(IntVars::permutation(n), |tour: &[i64]| {
    (0..n)
        .map(|p| tsp.distance(tour[p] as usize, tour[(p + 1) % n] as usize))
        .sum()
});
```

[`examples/integer_tsp.rs`](https://github.com/trash-iine/optopus/blob/main/examples/integer_tsp.rs)
(`cargo run --example integer_tsp`)

## Vertex Cover

One binary variable per vertex says whether it is in the cover. The objective
counts the chosen vertices, and one `Constraint` per edge asks for at least
one of its ends. A broken constraint costs its `penalty_weight`, which is more
than the vertex that would mend it, so the best solution is a cover.

```rust
let graph = Graph::erdos_renyi(100, 0.05, &mut seeded_rng(1));

let vars: IntVars = (0..graph.len()).map(|_| IntVar::binary()).collect();
let size = (0..graph.len()).fold(Expr::Const(0.0), |sum, i| sum + Expr::Var(i));
let prob = graph
    .edges()
    .fold(FormulaProblem::minimize(vars, size), |prob, (i, j, _)| {
        prob.with_constraint(Constraint::Comparison {
            lhs: Expr::Var(i) + Expr::Var(j),
            rel: ConstraintRel::Ge,
            rhs: Expr::Const(1.0),
            penalty_weight: 2.0,
        })
    });
```

`prob.eval_objective(values)` and `prob.eval_penalty(values)` read the cover
size and the penalty apart.

[`examples/integer_vertex_cover.rs`](https://github.com/trash-iine/optopus/blob/main/examples/integer_vertex_cover.rs)
(`cargo run --example integer_vertex_cover`)

## Graph Coloring

One variable per vertex holds its color, one of `0..=k-1`, with `k` one more
than the largest degree so that a proper coloring always exists. The objective
is the number of colors in use plus a penalty for every edge whose ends share
a color. The penalty is larger than the number of vertices, so the best
solution is a proper coloring. `IntChangeNeighbor` recolors one vertex.

```rust
let graph = Graph::erdos_renyi(50, 0.2, &mut seeded_rng(1));
let n = graph.len();
let k = (0..n).map(|v| graph.degree(v)).max().unwrap() as i64 + 1;

let vars: IntVars = (0..n).map(|_| IntVar::new(0, k - 1)).collect();
let prob = IntegerProblem::minimize(vars, |color: &[i64]| {
    let mut used = vec![false; k as usize];
    color.iter().for_each(|&c| used[c as usize] = true);
    let colors = used.iter().filter(|&&u| u).count();
    let conflicts = graph
        .edges()
        .filter(|&(i, j, _)| color[i] == color[j])
        .count();
    (colors + (n + 1) * conflicts) as f64
});
```

[`examples/integer_graph_coloring.rs`](https://github.com/trash-iine/optopus/blob/main/examples/integer_graph_coloring.rs)
(`cargo run --example integer_graph_coloring`)

## Job Shop Scheduling

The variables are a permutation of `0..jobs * machines`, and value `v` is read
as job `v / machines`, so every job appears once per machine. That sequence of
jobs is the operation order `JobShopScheduling::decode` turns into a schedule,
and its makespan is the objective. A function the library already has can be
called inside the objective like any other. `IntSwapNeighbor` exchanges two
operations in the order.

```rust
let jssp = JobShopScheduling::load_file("data/instances/jssp/ft06.txt").unwrap();
let m = jssp.n_machines;

let as_jobs = |v: &[i64]| -> Vec<usize> { v.iter().map(|&p| p as usize / m).collect() };
let prob = IntegerProblem::minimize(IntVars::permutation(jssp.n_jobs * m), |v: &[i64]| {
    let (makespan, _) = jssp.decode(&as_jobs(v)).unwrap();
    makespan as f64
});
```

[`examples/integer_job_shop.rs`](https://github.com/trash-iine/optopus/blob/main/examples/integer_job_shop.rs)
(`cargo run --example integer_job_shop`)

## CVRP

The variables are a permutation of the customers, the order a single vehicle
would visit them in. `split_giant_tour` cuts that order into routes at the
best places, and the objective is what `Vrp::solution_from_routes` charges for
those routes, their distance plus a penalty for any load over capacity.
Customers are numbered from 1, the depot being 0, so a value is shifted by one.
`IntReverseNeighbor` reverses a stretch of the order.

```rust
use optopus::problem::vrp::split_giant_tour;

let vrp = Vrp::load_file("data/instances/vrp/demo16.vrp").unwrap();

let routes = |order: &[i64]| {
    let giant: Vec<usize> = order.iter().map(|&c| c as usize + 1).collect();
    split_giant_tour(&vrp, &giant, vrp.penalty_weight())
};
let prob = IntegerProblem::minimize(IntVars::permutation(vrp.get_n()), |order: &[i64]| {
    vrp.solution_from_routes(routes(order)).objective
});
```

[`examples/integer_vrp.rs`](https://github.com/trash-iine/optopus/blob/main/examples/integer_vrp.rs)
(`cargo run --example integer_vrp`)

## Beyond these examples

- A delta for each move makes an `IntegerProblem` fast, see
  [Making it fast](../problems/integer.md#making-it-fast).
- A problem that prices a move from something its solution keeps, such as the
  gain of each flip, implements `IntAssignment` with a solution of its own, see
  [A solution of your own](../problems/integer.md#a-solution-of-your-own).
- `GeneticAlgorithm` runs on any of these with `IntCrossover` as its crossover.
