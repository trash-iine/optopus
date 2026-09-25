# 整数変数

**API:** [`IntegerProblem`](../../api/optopus/problem/integer/trait.IntegerProblem.html)

`IntegerProblem` は、近傍を書かずに済ませたい問題のためのトレイトです。変数は整数で、それぞれ `lower..=upper` の範囲を動きます。
トレイトを実装するとは、その範囲と目的関数を述べることで、残りはそこから決まります。

- `ProblemTrait` はすべての `IntegerProblem` に実装され、解は `IntSolution`、初期解は各変数をその範囲から一様に引いたものです。
- `IntChangeNeighbor` は一つの変数を範囲内の別の値にします。`0..=1` の変数ならこれは Flip です。
  `LocalSearch`、`SimulatedAnnealing`、`LateAcceptanceHillClimbing`、`TabuSearch`、`RandomWalk`、`BeamSearch`、
  `ReinforcementLearningSearch` が近傍に求めるものをすべて実装しているので、これらとこれらを組み合わせたヒューリスティクスがそのまま動きます。

解の近傍にはすべての変数のすべての値が入るので、100 万通りの値をとる変数が一つあると、`LocalSearch` や `TabuSearch` の走査ごとに候補が 100 万増えます。
一歩ごとに近傍を一つ無作為に引く `SimulatedAnnealing` などには影響しません。

## 例 { #example }

```rust
use optopus::prelude::*;

/// Minimize the sum of (x_i - 3)^2 over five variables in 0..=10.
struct Target { vars: IntVars }

impl IntegerProblem for Target {
    fn variables(&self) -> &IntVars {
        &self.vars
    }
    fn objective(&self, values: &[i64]) -> Evaluable<f64> {
        Evaluable::Minimize(values.iter().map(|&x| ((x - 3) * (x - 3)) as f64).sum())
    }
}

let prob = Target { vars: (0..5).map(|_| IntVar::new(0, 10)).collect() };
let mut state = SearchState::new(&prob);
LocalSearch::<IntChangeNeighbor>::new(StopCondition::iterations(100))
    .run(&mut state)
    .unwrap();
println!("{:?}", state.best_solution.values());
```

最適化の向きは目的関数が `Evaluable::Maximize` と `Evaluable::Minimize` のどちらを返すかで述べ、どの割り当てに対しても同じ方を返す必要があります。

[`examples/integer_problem.rs`](https://github.com/trash-iine/optopus/blob/main/examples/integer_problem.rs)
はこの方法で有界ナップサックを解きます (`cargo run --example integer_problem`)。

## 高速化 { #making-it-fast }

近傍の候補はどれも、目的値がどれだけ変わるかを `IntegerProblem::delta` に問い合わせます。既定の実装は値を複製して目的関数全体を評価し、
そうしていることを実行時に一度だけ警告します。変数 `i` が現れる項だけから変化量を計算するように `delta` を上書きしてください。
返すのは生の目的値の差で、新しい値から古い値を引いたものです。

```rust
fn delta(&self, sol: &IntSolution, i: usize, value: i64) -> f64 {
    let (old, new) = (sol.value(i), value);
    ((new - 3) * (new - 3) - (old - 3) * (old - 3)) as f64
}
```

近傍を適用すると、目的値を評価し直すのではなく、キャッシュした変化量を解の目的値に足します。

## 与えた割り当てから始める { #starting-from-a-given-assignment }

`IntegerProblem::solution_from` は指定した値から解を作り、範囲外の値があれば失敗します。
それを `SearchState::with_solution` に渡すとそこから探索を始められます。

## できないこと { #what-it-does-not-do }

交叉と `Distance` がないので `GeneticAlgorithm` は `IntegerProblem` では動きません。
またインスタンスファイルから問題を読む CLI ベンチマークには登録できません。
