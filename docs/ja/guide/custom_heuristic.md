# 独自のヒューリスティクスを定義する

**API:** [`Heuristic`](../../api/optopus/heuristic/trait.Heuristic.html)

`Heuristic<P>` を実装すると、自分のアルゴリズムをライブラリのほかの部分に組み込めます。
`SearchState`、メタヒューリスティクス (`Sequential`、`Iterated`、
`VariableNeighborhoodSearch`、`Restart`)、ベンチマークランナーはどれも変更なしでそれと一緒に動きます。

実行できる完全な例は
[`examples/custom_heuristic.rs`](https://github.com/trash-iine/optopus/blob/main/examples/custom_heuristic.rs)
にあります (`cargo run --example custom_heuristic`)。

## `Heuristic<P>` トレイト { #the-heuristicp-trait }

```rust
pub trait Heuristic<Problem: ProblemTrait> {
    fn clear(&mut self) {}
    fn stop_condition(&self) -> &StopCondition;
    fn run_once<'a>(&mut self, state: &mut SearchState<'a, Problem>) -> Result<(), OptError>;

    // 既定の `is_done` は `stop_condition()` に委ねる。
    fn is_done<'a>(&self, state: &SearchState<'a, Problem>) -> bool { … }

    // 既定の `run` は `clear()` を呼んでから、`!is_done` の間 `run_once` を繰り返す。
    fn run<'a>(&mut self, state: &mut SearchState<'a, Problem>) -> Result<(), OptError> { … }
}
```

実装するのは `stop_condition` と `run_once` で、`is_done` と `run` は用意されています。
`is_done` をオーバーライドするのは、停止条件では表せない終了規則を既定の判定に加えたいときだけです。
`LocalSearch` と `LinKernighanHelsgaunForTsp` の「局所最適で止まる」がその例です。
ヒューリスティクスが実行ごとの状態 (カウンタ、学習した重みなど) を持つなら `clear` をオーバーライドしてください。

## 最小の first-improving 探索 { #minimal-first-improving-search }

```rust
use optopus::error::OptError;
use optopus::prelude::*;

struct FirstImprovingSearch<N> {
    stop_condition: StopCondition,
    _neighbor: std::marker::PhantomData<N>,
}

impl<N> FirstImprovingSearch<N> {
    fn new(stop_condition: StopCondition) -> Self {
        Self {
            stop_condition,
            _neighbor: std::marker::PhantomData,
        }
    }
}

impl<P, N> Heuristic<P> for FirstImprovingSearch<N>
where
    P: ProblemTrait,
    N: MoveToNeighbor<P>,
{
    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        let instance = state.instance;
        let solution = &state.solution;
        let next_move = N::iter(instance, solution)
            .find(|neighbor| neighbor.move_to_be_better_than(instance, solution, solution));

        if let Some(neighbor) = next_move {
            state.apply(&neighbor)?;
        } else {
            state.progress_iteration();
        }

        Ok(())
    }
}
```

- [`MoveToNeighbor::iter`](../../api/optopus/trait_defs/trait.MoveToNeighbor.html#tymethod.iter)
- [`SearchState::apply`](../../api/optopus/search_state/struct.SearchState.html#method.apply)
- [`SearchState::progress_iteration`](../../api/optopus/search_state/struct.SearchState.html#method.progress_iteration)

主に触れる API は次のとおりです。

- [`state.apply(&neighbor)`](../../api/optopus/search_state/struct.SearchState.html#method.apply)。move を適用し、
  反復を一つ進め、改善していれば最良解を更新します。
- [`state.apply_move_only(&neighbor)`](../../api/optopus/search_state/struct.SearchState.html#method.apply_move_only)。同じですが、
  最良解の更新を後回しにします。複数の move からなる一歩の終わりに
  [`state.update_best()`](../../api/optopus/search_state/struct.SearchState.html#method.update_best) を呼ぶ必要があります。
- [`state.progress_iteration()`](../../api/optopus/search_state/struct.SearchState.html#method.progress_iteration)。何も適用せずに
  反復を一つ進めます (その一歩で前に進めないときに使います)。
- [`state.random_neighbor::<N>(context)`](../../api/optopus/search_state/struct.SearchState.html#method.random_neighbor)。一様ランダムな
  move を一つ引きます。近傍が空なら
  [`OptError::InvalidState`](../../api/optopus/error/enum.OptError.html#variant.InvalidState)
  を返します。SA、LAHC、
  [`RandomWalk`](../../api/optopus/heuristic/struct.RandomWalk.html) が毎ステップ呼ぶのはこれです。
- [`N::iter(prob, sol)`](../../api/optopus/trait_defs/trait.MoveToNeighbor.html#tymethod.iter)。move を遅延的に列挙するイテレータです。
  戦略に応じて `max_by`、`find`、
  [`filter_best`](../../api/optopus/trait_defs/fn.filter_best.html)、`.choose(&mut rng)` などと組み合わせます。

state のフィールドは公開されています。`state.instance` が問題、`state.solution` が現在の解、`state.best_solution` がこれまでの最良解、
`state.initial_solution` が実行を始めたときの解で、`state.iteration` と `state.best_iteration` が反復を数えます。
各解の `objective` などのフィールドは問題ごとのページに、各フィールドがいつ変わるかは [SearchState](../search_state.md) にあります。

## 差分で move を受理する { #accepting-a-move-by-its-delta }

Simulated Annealing、Late Acceptance、しきい値型の規則は近傍を走査しません。
move を一つ引き、それが生む変化から受理するかを決めます。
move はその変化を自身の `Evaluate` impl で返すので、こうしたヒューリスティクスは `N` に `Evaluate` を求め、何も適用せずに差分を読みます。

```rust
struct RandomDescent<N> {
    stop_condition: StopCondition,
    _neighbor: std::marker::PhantomData<N>,
}

impl<P, N> Heuristic<P> for RandomDescent<N>
where
    P: ProblemTrait,
    N: MoveToNeighbor<P> + Evaluate,
{
    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        let neighbor: N = state.random_neighbor("RandomDescent")?;
        let worsening = neighbor.evaluate().minimized();
        if worsening <= 0.0 {
            state.apply(&neighbor)?;
        } else {
            state.progress_iteration();
        }
        Ok(())
    }
}
```

[`neighbor.evaluate().minimized()`](../../api/optopus/trait_defs/enum.Evaluable.html#method.minimized)
は、その move を適用したときに目的関数がどれだけ悪化するかを、問題の向きを反映したうえで表します。
最大化問題でも最小化問題でも、悪くなる move なら正、良くなる move なら負なので、受理規則はこれをしきい値と比べるだけで、問題の向きを問う必要はありません。
受理は `state.apply` を通し、これは反復も数えて最良解も更新します。棄却は `state.progress_iteration` を通します。
`examples/custom_heuristic.rs` はこのヒューリスティクスを小さな MaxCut インスタンスで動かします。

## 並列評価 (任意) { #optional-parallel-evaluation }

実装すべき `Heuristic` の並列版はありません。並列化は近傍の型の役割です。`MoveToNeighbor::iter` は
`impl Iterator + Send` を返すので、候補ごとの計算が重い `iter` の実装は、rayon で候補を評価して結果を順番どおりに返せます。
[`JobShopSwapNeighbor`](../../api/optopus/problem/job_shop_scheduling/struct.JobShopSwapNeighbor.html)
はサイズがしきい値を超えるとまさにそうしています。これにより結果はスレッド数に依存せず、
すべてのヒューリスティクス (あなたのものも含めて) は逐次のままでいられます。

## ヒューリスティクスを組み合わせる { #composing-your-heuristic }

`Heuristic<P>` を実装すれば、あなたのアルゴリズムは次のように使えます。

- [`Restart`](../heuristics/meta.md#restart) で包み、停滞したらランダムな解にリセットする。
- [`Iterated`](../heuristics/meta.md#iterated) のフェーズとして使う。
- [`Sequential`](../heuristics/meta.md#sequential) の中に並べる。
- [`VariableNeighborhoodSearch`](../heuristics/meta.md#variableneighborhoodsearch)
  の局所探索や shake として使う。
- [`GeneticAlgorithm`](../heuristics/genetic_algorithm.md) の `mutation` 引数として渡す。

## 次に読むもの { #next-reading }

- [SearchState API](../search_state.md)
- [Meta-heuristics](../heuristics/meta.md)
