# 学習した摂動方策で BLS を動かす

**API:** [`SoftmaxBandit`](../../api/optopus/heuristic/reinforcement_learning/bandit/struct.SoftmaxBandit.html)

独自の規則は、Benlic と Hao の規則が入っている場所に、
[BreakoutLocalSearch](../heuristics/breakout_local_search.md) に渡す `PerturbationSchedule` として入れます。
規則は選択肢となるオペレータを持ち、`determine_jump_magnitude` で決定し、決めたものを `select` で返します。

```rust
impl PerturbationSchedule<MaxCut> for MyPolicy {
    fn determine_jump_magnitude(&mut self, state: &mut SearchState<'_, MaxCut>) -> u64 {
        // スケジュールが状態を受け取るのはここだけなので、ここで決める
        let (operator, l) = self.choose(state);
        self.chosen = operator;
        l
    }

    fn select<'s>(&'s mut self, _state: &mut SearchState<'_, MaxCut>)
        -> &'s mut dyn Heuristic<MaxCut>
    {
        self.operators[self.chosen].as_mut()
    }
    // reset / clear_perturbations、規則が必要とするなら round_ended も
}

let mut bls = BreakoutLocalSearch::new(
    StopCondition::iterations(100_000),
    /* tabu_tenure = */ (15, 300),
    max_cut_descent(),
    MyPolicy::new(...),
);
```

降下と摂動は `SearchState` のタブーメモリを共有します。
降下が書き込んだ禁止こそが、弱い摂動が元に戻してはならないものです。

それらのオペレータを作る `max_cut_perturbation` は、
[`bls_for_max_cut`](../heuristics/breakout_local_search.md#benchmark-config)
と同じく `tabu_tenure` を文字どおりに受け取ります。

## 完全な例 (文脈付きバンディット) { #the-full-example-a-contextual-bandit }

実行できる例は
[`examples/rl_bls.rs`](https://github.com/trash-iine/optopus/blob/main/examples/rl_bls.rs)
にあります (`cargo run --release --example rl_bls`)。手作りの規則と強さのスケジュールを、文脈付きの softmax 勾配バンディット
([`SoftmaxBandit`](../../api/optopus/heuristic/reinforcement_learning/bandit/struct.SoftmaxBandit.html))
に置き換え、同じインスタンスとシードで BLS と RL-BLS を並べて表示します。

各ラウンドは次のように進みます。ステップ 2 から 4 はすべて `determine_jump_magnitude` の中です。

1. フレームワークが貪欲な降下で局所最適まで進めます。
2. 直前の決定に対する報酬を観測します。報酬は局所最適の目的関数値の変化を、その大きさの EMA で正規化し、
   `[−1, 1]` に切り詰め、全体の最良解が改善していれば `+1` のボーナスを加えたものです。
   バンディットの行動ごとの線形な選好は、EMA のベースラインに対する1ステップの REINFORCE で更新されます。
3. 行動を選びます。7 個の文脈特徴
   `[bias, min(ω/t, 1), exp(−ω/t), descent_improved_best, relative_gap,
   reward_ema, budget_progress]` から、バンディットは
   `3 × strength_bins.len()` 個の行動の一つを選びます。行動は摂動の種類
   ([`MaxCutPerturbation`](../../api/optopus/heuristic/enum.MaxCutPerturbation.html) の
   `WeakFlip` / `WeakSwap` / `Strong`) と `l0` の倍率の組です。
4. 選んだ強さを返し、`select` が選んだオペレータを返します。フレームワークがそれを適用し、`update_best` でラウンドを閉じます。
   その後 `round_ended` が全体の最良値を記録し、次のラウンドはこれと比べて `descent_improved_best` を埋めます。

`exp(−ω/t)` は BLS の手作りの規則がしきい値判定に使う確率そのものです。そのため、ほぼ線形の方策でもまず BLS をすばやく真似て、
それから改善していけます。

例が短く済んでいるのは、ライブラリの二つの部品のおかげです。一つはバンディットそのもの、もう一つは
[`SearchState::iterations_this_run`](../../api/optopus/search_state/struct.SearchState.html#method.iterations_this_run)
です。`budget_progress` はこれで正規化するので、[`Restart`](../heuristics/meta.md#restart)
の中のサブランが親の進捗を自分のものとして読むことはありません。それ以外、つまり遅延報酬の管理、特徴ベクトル、行動の復号は方策の一部であり、
変更できるように例の中に置いてあります。

## 複数エピソードにわたる学習 { #multi-episode-learning }

`clear()` はエピソードの状態 (omega、オペレータ、保留中の決定、報酬の統計) をリセットしますが、
バンディットの重みとベースラインは保持します。そのため方策は [`Restart`](../heuristics/meta.md#restart) /
[`Iterated`](../heuristics/meta.md#iterated) のエピソードをまたいで改善し続けます。これは
[`ReinforcementLearningSearch`](../heuristics/rl_search.md) が守っているのと同じ契約で、
重みがローカル変数ではなくフィールドになっている理由でもあります。
