# 整数変数

**API:** [`IntegerProblem`](../../api/optopus/problem/integer/struct.IntegerProblem.html)

`IntegerProblem` は、近傍を書かずに済ませたい問題のための型です。変数は整数で、それぞれ `lower..=upper` の範囲を動きます。
作るのに要るのはその範囲と、クロージャで書いた目的関数だけで、実装するものはありません。

- `ProblemTrait` はすべての `IntegerProblem` に付いてきて、解は `IntSolution` です。初期解は各変数をその範囲から一様に引いたもので、
  変数が順列ならば一様ランダムな順列です。
- 近傍が三つ付いてきます。どれも `LocalSearch`、`SimulatedAnnealing`、`LateAcceptanceHillClimbing`、`TabuSearch`、`RandomWalk`、
  `BeamSearch`、`ReinforcementLearningSearch` が近傍に求めるものをすべて実装しています。

| 近傍 | 何をするか | 向いているもの |
|---|---|---|
| `IntChangeNeighbor` | 一つの変数を範囲内の別の値にする。`0..=1` なら Flip | 範囲が独立した変数 |
| `IntSwapNeighbor` | 二つの変数の値を交換する | 割り当て、順列 |
| `IntReverseNeighbor` | 変数 `i..=j` の値の並びを反転する | 巡回路として読む順列 (2-opt) |

## 例 { #example }

```rust
use optopus::prelude::*;

// Minimize the sum of (x_i - 3)^2 over five variables in 0..=10.
let vars: IntVars = (0..5).map(|_| IntVar::new(0, 10)).collect();
let prob = IntegerProblem::minimize(vars, |x: &[i64]| {
    x.iter().map(|&v| ((v - 3) * (v - 3)) as f64).sum()
});

let mut state = SearchState::new(&prob);
LocalSearch::<IntChangeNeighbor>::new(StopCondition::iterations(100))
    .run(&mut state)
    .unwrap();
println!("{:?}", state.best_solution.values());
```

最適化の向きは `IntegerProblem::minimize` と `IntegerProblem::maximize` のどちらで作るかで決まり、クロージャは素の値を返します。

[`examples/integer_problem.rs`](https://github.com/trash-iine/optopus/blob/main/examples/integer_problem.rs)
はこの方法で有界ナップサックを解きます (`cargo run --example integer_problem`)。

## 順列 { #permutations }

`IntVars::permutation(n)` は、`0..=n-1` の値を互いに重ならずにとる `n` 個の変数を宣言します。
変数 `p` を `p` 番目に訪れる都市と読めば、解は巡回路です。

```rust
let tour_len = |v: &[i64]| {
    let n = v.len();
    (0..n).map(|p| dist[v[p] as usize][v[(p + 1) % n] as usize]).sum()
};
let prob = IntegerProblem::minimize(IntVars::permutation(n), tour_len);

LocalSearch::<IntReverseNeighbor>::new(StopCondition::iterations(1_000))
    .run(&mut SearchState::new(&prob))?;
```

順列の値を一つだけ変えると必ず別の値と重なるので、順列の上では `IntChangeNeighbor` の近傍は空です。
それを使った探索は順列を壊すのではなく、近傍が空であることを報告します。`solution_from` は値が重なる入力を受け付けません。

## 高速化 { #making-it-fast }

近傍の候補はどれも、それを適用すると何が変わるかを問題に問い合わせます。何も渡さなければ、問題は値を複製して目的関数全体を評価し直します。
どんな目的関数でも正しく、実行時に一度だけ警告します。使う近傍が何を変えるかを、その近傍が触れる部分だけから計算するクロージャとして渡してください。
返すのは素の目的値の差で、新しい値から古い値を引いたものです。

| 近傍 | ビルダー |
|---|---|
| `IntChangeNeighbor` | `with_delta(\|sol, i, value\| ...)` |
| `IntSwapNeighbor` | `with_swap_delta(\|sol, i, j\| ...)` |
| `IntReverseNeighbor` | `with_reverse_delta(\|sol, i, j\| ...)` |

上の巡回路なら、反転の差分は距離を四回引くだけで求まります。

```rust
let prob = IntegerProblem::minimize(IntVars::permutation(n), tour_len)
    .with_reverse_delta(|sol: &IntSolution, i, j| {
        let (v, n) = (sol.values(), sol.values().len());
        if j - i + 1 >= n - 1 {
            return 0.0; // reversing all but at most one city leaves the tour as it was
        }
        let d = |a: i64, b: i64| dist[a as usize][b as usize];
        let (prev, next) = (v[(i + n - 1) % n], v[(j + 1) % n]);
        d(prev, v[j]) + d(v[i], next) - d(prev, v[i]) - d(v[j], next)
    });
```

近傍を適用すると、目的値を評価し直すのではなく、その変化量を解の目的値に足します。

## 自前の解を使う { #a-solution-of-your-own }

近傍の差分を速く求めるのに、解が持っている何か、たとえば MaxCut の各変数を Flip したときの gain が要る問題があります。
`IntSolution` が持つのは値と目的値だけなので、そういう問題は代わりに自前の解の型でトレイト `IntAssignment` を実装します。

| メソッド | 必須 | 何をするか |
|---|---|---|
| `domains` | はい | 変数 |
| `get(sol, i)` | はい | 変数 `i` の値 |
| `assign(sol, i, value)` | はい | 値を設定し、目的値とキャッシュしているものを最新に保つ |
| `assign_delta(sol, i, value)` | いいえ | `assign` で何が変わるかを、キャッシュから読む |
| `assign_swap`、`assign_reverse` と、それぞれの `_delta` | いいえ | 残り二つの近傍について同じこと |

その `ProblemTrait` と解の `Evaluate` は、ほかの[独自の問題](../guide/custom_problem.md)と同じように書きます。
そうすれば三つの近傍は `IntegerProblem` のときと同じように動きます。`IntegerProblem` 自身も `IntAssignment` です。

## 与えた割り当てから始める { #starting-from-a-given-assignment }

`IntegerProblem::solution_from` は指定した値から解を作り、範囲外の値があれば失敗します。
それを `SearchState::with_solution` に渡すとそこから探索を始められます。

## できないこと { #what-it-does-not-do }

交叉と `Distance` がないので `GeneticAlgorithm` は `IntegerProblem` では動きません。
またインスタンスファイルから問題を読む CLI ベンチマークには登録できません。
