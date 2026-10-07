# Quadratic Assignment

**API:** [`Qap`](../../api/optopus/problem/qap/struct.Qap.html)

Quadratic Assignment Problem は、`n` 個の施設を `n` 箇所の場所に一つずつ配置する問題です。
行列 `a` は施設の各組の間を流れる量、`b` は場所の各組の間の距離を表し、配置 `p` のコストは次のとおりです。

```text
minimize  Σ_i Σ_j a[i][j] · b[p(i)][p(j)]       p は 0..n の置換
```

どちらの行列も対称である必要はありません。これは Koopmans と Beckmann の形で、行列の順は QAPLIB の並びと同じです。
近似すら NP 困難で、施設が数十個のインスタンスですでに厳密解法の手に負えません。

## 例 { #example }

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

`TabuSearch` は移動のたびに範囲から tenure を引き、`QapSwapNeighbor` は Taillard のタブー規則に従うので、これは何も足さずに Taillard の robust tabu search になります。
[`Iterated`](../heuristics/meta.md) で包み、探索の合間にランダムな交換を数回入れれば Misevicius の Iterated Tabu Search で、
[`examples/integer_qap.rs`](https://github.com/trash-iine/optopus/blob/main/examples/integer_qap.rs) が両者を並べて実行します。

## 解 { #solution }

[`QapSolution`](../../api/optopus/problem/qap/struct.QapSolution.html) は、`assignment[i]` が施設 `i` の場所である `assignment` と、そのコスト `objective` を持ちます。
自前の配置は `Qap::solution_from_assignment` で解に包め、`Qap::cost` は解を作らずにコストを求めます。

## 近傍 { #neighbors }

| 型 | 設定 | 移動 | コスト |
|---|---|---|---|
| `QapSwapNeighbor` | `Swap` | 二つの施設の場所を入れ替える。 | 1 手 O(n)、走査 O(n³) |

移動は対称とは限らない行列に対する一般式 `Qap::swap_delta` で価格付けします。
タブー規則は各施設に直前までいた場所を禁じ、移動が tabu になるのは二つの施設の両方を元の場所に戻すときだけです。これが Taillard の規則です。

## 交叉 { #crossover }

`QapOrderCrossover` は配置に対する Order Crossover (OX) です。

## ファイル形式 (QAPLIB) { #file-format-qaplib }

```text
n
a, n 行 n 列
b, n 行 n 列
```

値はすべて空白区切りで、QAPLIB の幅の広いインスタンスのように一行が複数行に折り返されていても構いません。
QAPLIB は再配布の条件を示していないので、インスタンスは同梱していません。
`data/instances/scripts/fetch_qaplib.sh` が Taillard の `tai*a` 系といくつかの古典的なインスタンスを `data/instances/qap/` に取得します。

## 参考文献 { #references }

- Koopmans, T. C. and Beckmann, M. "Assignment Problems and the Location of
  Economic Activities." *Econometrica*, 25(1), 53-76, 1957.
- Taillard, E. "Robust taboo search for the quadratic assignment problem."
  *Parallel Computing*, 17(4-5), 443-455, 1991.
- Misevicius, A. "A tabu search algorithm for the quadratic assignment
  problem." *Computational Optimization and Applications*, 30(1), 95-111, 2005.
- Burkard, R. E., Karisch, S. E. and Rendl, F. "QAPLIB, a quadratic
  assignment problem library." *Journal of Global Optimization*, 10(4),
  391-403, 1997.
