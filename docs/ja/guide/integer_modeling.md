# 整数変数で問題を書く

**API:** [`IntegerProblem`](../../api/optopus/problem/integer/struct.IntegerProblem.html)

ライブラリに入っている問題は、どれも整数変数と目的関数だけでも書けます。近傍を定義することも、トレイトを実装することもありません。
このページでは 8 つの問題すべてをそのように書きます。各節には問題ごとに違う部分だけを載せ、残りは実行できる例へのリンクにあります。

こう書いた問題は、その問題専用の型より遅くなります。組み込みの問題は近傍が触れる部分から変化量を求めますが、
[`IntegerProblem`](../problems/integer.md) は候補ごとに目的関数全体を評価するからです。速さが要るときは
[高速化](../problems/integer.md#making-it-fast) のとおり変化量を渡すか、その問題専用の型を使ってください。

## 変数の三つの形 { #three-shapes-of-variables }

| 変数 | 宣言 | 近傍 | 使う問題 |
|---|---|---|---|
| 項目ごとに一つの二値変数 | `IntVar::binary()` | `IntChangeNeighbor` (Flip) | MaxCut, QUBO, MaxSAT, Vertex Cover |
| 項目ごとに一つの、少数の値をとる変数 | `IntVar::new(lower, upper)` | `IntChangeNeighbor` | Graph Coloring |
| 順列 | `IntVars::permutation(n)` | `IntSwapNeighbor`, `IntReverseNeighbor` | TSP, Job Shop Scheduling, CVRP |

## 探索を走らせる { #running-the-search }

問題を作ったあとは、どの例も同じ形で終わります。探索は近傍の型を指定して選び、最良解の値は `values()` で読み出します。

```rust
let mut state = SearchState::new_with_seed(&prob, 42);
SimulatedAnnealing::<IntChangeNeighbor>::new(StopCondition::iterations(20_000), 2.0, 0.9995)
    .run(&mut state)
    .unwrap();
println!("{:?}", state.best_solution.values());
println!("{:?}", state.best_solution.evaluate());
```

例では `SimulatedAnnealing` を使っています。これは 1 反復につき候補を一つだけランダムに評価します。`LocalSearch` と `TabuSearch` は
1 反復ごとに近傍のすべての候補を評価し、変化量を渡していなければそのたびに目的関数全体を評価するので、下の MaxSAT のような小さな近傍に向いています。

## MaxCut { #maxcut }

頂点ごとの二値変数が、その頂点がカットのどちら側にあるかを表します。目的関数は両端が異なる側にある辺の重みの和です。

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

## QUBO { #qubo }

エネルギー `Σ Q[i][j] x_i x_j` は多項式なので、`Expr` として書いて [`FormulaProblem`](../problems/formula.md) に渡します。
この問題は各 Flip による変化量を式から導くので、ほかに何も書かなくても速く動きます。

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

## MaxSAT { #maxsat }

論理変数ごとに一つの二値変数を置き、目的関数は成り立つ節の数を数えます。リテラル `l` は、`l` が正なら変数 `|l| - 1` が `1` であること、
負なら `0` であることを求めます。

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

変数が 20 個で近傍が小さいので、例では `TabuSearch::<IntChangeNeighbor>` を走らせます。

[`examples/integer_max_sat.rs`](https://github.com/trash-iine/optopus/blob/main/examples/integer_max_sat.rs)
(`cargo run --example integer_max_sat`)

## TSP { #tsp }

変数は順列で、変数 `p` は `p` 番目に訪れる都市です。目的関数は閉じた巡回路の長さです。`IntReverseNeighbor` は巡回路の一区間を反転し、
これは 2-opt の近傍です。

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

## Vertex Cover { #vertex-cover }

頂点ごとの二値変数が、その頂点を被覆に入れるかを表します。目的関数は選んだ頂点の数で、辺ごとに一つの `Constraint` が両端の少なくとも一方を求めます。
破れた制約は `penalty_weight` だけかかり、これはその制約を直す頂点一つより大きいので、最良解は被覆になります。

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

被覆の大きさと罰金は `prob.eval_objective(values)` と `prob.eval_penalty(values)` で別々に読めます。

[`examples/integer_vertex_cover.rs`](https://github.com/trash-iine/optopus/blob/main/examples/integer_vertex_cover.rs)
(`cargo run --example integer_vertex_cover`)

## Graph Coloring { #graph-coloring }

頂点ごとの変数がその色を `0..=k-1` のどれかで持ちます。`k` は最大次数に 1 を足したもので、これなら正しい彩色が必ず存在します。
目的関数は使っている色の数に、両端が同じ色の辺ごとの罰金を足したものです。罰金は頂点数より大きいので、最良解は正しい彩色になります。
`IntChangeNeighbor` は頂点を一つ塗り替えます。

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

## Job Shop Scheduling { #job-shop-scheduling }

変数は `0..jobs * machines` の順列で、値 `v` をジョブ `v / machines` と読むので、各ジョブが機械の数だけ現れます。
このジョブの並びが `JobShopScheduling::decode` がスケジュールに変換する作業順で、そのメイクスパンが目的関数です。
ライブラリにすでにある関数も、ほかの関数と同じように目的関数の中で呼べます。`IntSwapNeighbor` は作業順の二つの作業を交換します。

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

## CVRP { #cvrp }

変数は顧客の順列で、1 台の車両が顧客を訪れる順を表します。`split_giant_tour` がその順を最もよい位置で経路に切り分け、
目的関数はその経路に `Vrp::solution_from_routes` が付ける値、つまり距離と容量超過の罰金の和です。
顧客は 1 から番号が振られ、デポが 0 なので、値を 1 ずらします。`IntReverseNeighbor` は順の一区間を反転します。

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

## この先 { #beyond-these-examples }

- 近傍ごとに変化量を渡すと `IntegerProblem` は速くなります。[高速化](../problems/integer.md#making-it-fast) を参照してください。
- 各 Flip の利得のように、解が保持するものから近傍の変化量を求める問題は、自前の解を持つ `IntAssignment` を実装します。
  [自前の解を使う](../problems/integer.md#a-solution-of-your-own) を参照してください。
- `GeneticAlgorithm` は、交叉に `IntCrossover` を使えばこれらのどれでも動きます。
