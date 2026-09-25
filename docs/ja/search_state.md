# `SearchState`

**API:** [`SearchState`](../api/optopus/search_state/struct.SearchState.html)

`SearchState<'a, P>` はすべてのヒューリスティクスが操作する作業領域です。現在の解、全体の最良解、
各種カウンタ、タイマー、そしてただ一つの RNG を持ちます。ヒューリスティクスは問題インスタンスに直接触れず、
この状態を読み書きします。

シグネチャと項目ごとの説明は上にリンクした rustdoc にあります。このページでは、どの場面で何を使うか、
各部品がどう組み合わさるかを説明します。

## 何を保持しているか { #what-it-holds }

| 役割 | フィールド | 書き込むもの |
|---|---|---|
| 探索 | `solution`, `best_solution`, `initial_solution` | 実行中のヒューリスティクス |
| 進捗 | `iteration`, `best_iteration`, `best_time` | `apply` / `progress_iteration` / `update_best` |
| 集計 | `n_accepted`, `n_rejected`, `n_best_updates` | 同じ三つが一つずつ |
| シード | `rng` | 乱数を引くすべて ([再現性](#reproducibility) を参照) |
| 記録 | `trajectory` | プローブを設定した後の `update_best` |
| タブー | 非公開のタブーメモリ | `apply` / `apply_move_only` ([タブー move の記憶](#remembering-tabu-moves) を参照) |
| 問題 | `instance` (`&'a P`) | 実行の間ずっと借用される |

## 状態を作る { #creating-a-state }

最初の解をどこから得るか、乱数をどこから得るかの二軸で、四つのコンストラクタがあります。

|  | ランダムな初期解 | 与えた初期解 |
|---|---|---|
| OS のエントロピー | `SearchState::new(problem)` | `SearchState::with_solution(problem, sol)` |
| 固定シード | `SearchState::new_with_seed(problem, seed)` | `SearchState::with_solution_and_seed(problem, sol, seed)` |

```rust
let mut state = SearchState::new_with_seed(&problem, 42);
```

ベンチマークはシード付きの組を使い、実行ごとに一つのシードを導出します。そのため再実行はビット単位で一致します。
`with_solution` の組はウォームスタート用です。与えた解が現在の解、最良解、`initial_solution` のすべてになり、
報告される改善量はその解から測られます。

## 一歩進める { #advancing-one-step }

`run_once` の中身は常に同じ三拍子です。move を一つ選び、判断し、適用するか、反復を消費するだけにするかです。

```rust
let m: N = state.random_neighbor("MyHeuristic")?;   // または N::iter(...) を走査する
if state.is_neighbor_better_than_current(&m) {
    state.apply(&m)?;                              // 適用し、数え、最良解を更新する
} else {
    state.progress_iteration();                    // 棄却として数え、move はしない
}
```

`random_neighbor` は一様ランダムに move を一つ引き、近傍が空のときは `InvalidState` エラーを返します。
引数の `context` 文字列はヒューリスティクス名で、どのヒューリスティクスで近傍が尽きたかがメッセージに出ます。

`apply` と `apply_move_only` の違いは、最良解を更新するかどうかです。悪化させることが目的の摂動の中では
`apply_move_only` を使い、フェーズの終わりに `update_best` を一度呼びます。どちらも受理として数えます。
`progress_iteration` は棄却として数えます。

## タブー move の記憶 { #remembering-tabu-moves }

記録はモードになっていて、新しい状態でもすべてのサブランでも最初はオフです。
`start_record_tabu((min, max))` でオンにすると、テニュアとモードが同時に設定されます。
オンの間は、`apply` と `apply_move_only` が適用した move を、それが行われた反復番号とともに記録します。

```rust
state.start_record_tabu((5, 10));          // テニュアとモードを一度に設定する
let free = state.tabu_allows(&m);          // この move は今禁止されているか
state.apply(&m)?;                          // 適用し、記録する
state.reset_tabu();                        // すべての禁止を解除する
```

モードの設定は、テニュアを決めるのと同じ頻度で行います。`TabuSearch::run_once` では反復ごと、
`BreakoutLocalSearch::run_once` では 1 ラウンドに 2 回 (降下の前とキックの前) です。
設定するまでテニュアは `(0, 0)` で、move は記録されますが次の反復で再び許可されます。
`stop_record_tabu()` はテニュアを残したままにするので、`record_tabu` に渡したものだけを禁止したい探索に使えます。

move 型は [`EnabledTabu`](traits.md#core-trait-reference) を実装し、
`MoveToNeighbor::tabu_policy` を `Some(self)` でオーバーライドすることで対応します。
**トレイトを実装しても1行のオーバーライドがなければ、その move にはタブー方策がありません**。
その場合、適用しても何も記録されず、`TabuSearch` は空のリストのまま何も言わずに動きます。
`trait_defs/tabu.rs` はすべての組み込み move についてこれを検査しています。
`tabu_allows` と `record_tabu` は `EnabledTabu` を境界に持つので、方策のない move について問い合わせるとコンパイルエラーになります。

move が禁止する対象は `TabuKey` です。密なインデックスには `Var(i)`、それ以外には `Pair` と `Triple` を使います。
形ごとに別の空間なので、同じ形を使う二つの move 型は禁止を共有します。MaxCut の flip と swap が互いのエントリを見るのはこの仕組みによります。
形が違えば衝突することはありません。move が読むキーと書くキーは一致している必要はありません。
VRP の relocate は、ある顧客が移動先のルートに入ってよいかを問い合わせ、出てきたルートを禁止します。
これで顧客がすぐに元へ戻されることを防ぎます。インスタンスの大きさが事前に分かっていれば、
`state.reserve_tabu_vars(n)` で密な空間をあらかじめ確保できます。

## サブランを隔離する { #isolating-a-sub-run }

どのメタヒューリスティクスも、各フェーズをクローンの上で走らせ、結果をマージして戻します。

```rust
let mut sub = state.clone_for_new_run(SearchStateCloneType::ClearBest);
inner_heuristic.run(&mut sub)?;
state.update_state(sub);
```

`update_state` はサブランの現在の解を取り込み、各カウンタの差分を親に加え、最良解は改善しているときだけ採用します。
`initial_solution` は上書きされないので、親は報告用に自分の基準点を保ちます。
サブ状態が別の問題インスタンスを借用していると panic します。その場合は [縮約をまたぐ](#crossing-a-reduction) 方法を使います。

### [`SearchStateCloneType`](../api/optopus/search_state/enum.SearchStateCloneType.html) の種類 { #searchstateclonetype-variants }

| 種類 | 解 | 最良解 | 時計と基準点 |
|---|---|---|---|
| `Simple` | 現在の解 | 保持 | `start_iteration = iteration`、時計はそのまま |
| `ClearBest` | 現在の解 | 現在の解にリセット | `start_iteration = best_iteration = iteration`、時計をリセット |
| `StartBest` | 最良解 | 保持 | `start_iteration = best_iteration = iteration`、時計をリセット |

普段使うのは `ClearBest` です。フェーズには局所的な新しい「最良」を与え、親は全体の最良を保ちます。
`StartBest` は、直前のフェーズが流れ着いた場所ではなく現時点の最良解からフェーズをやり直します。
`Simple` は履歴全体をそのまま引き継ぎます。`ClearBest` と `StartBest` は `initial_solution`
をフェーズの開始点に置き直し、`Simple` はそれを引き継ぎます。

三つとも親の反復の座標系を保ちます。`iteration` は現在のフェーズがどこにいるか、`start_iteration`
はどこで始まったかを表し、予算に関わるものはすべてその基準点から測られます。
そのため、フェーズは自分の予算の 0 から始まりつつ、反復番号はマージの前後で同じ意味を持ちます。
一方 `n_accepted` / `n_rejected` / `n_best_updates` はフェーズ単位で数えるので、どの種類でも 0 から始まります。

どの種類も RNG をフォークして子に独立した乱数列を与えます。`clone_for_new_run` が `&mut self`
を取るのはこのためです。どの種類でも子のタブーメモリは空で始まり、`update_state` が子の学んだ内容を親に戻します。
親の禁止を引き継いで始めたいフェーズは `sub.inherit_tabu_from(&state)` を呼びます。

## 縮約をまたぐ { #crossing-a-reduction }

[`ProblemReduction`](traits.md#problemreduction) はあるインスタンスを別のインスタンスへ写します。カーネルがその例です。
写した先のインスタンスでのサブランは、同じインスタンスを要求する `update_state` を通せないので、次の組でまたぎます。

```rust
let mut sub = state.open_reduction(&kernel);   // ウォームスタート、シードは state.rng から引く
heuristic.run(&mut sub)?;
state.close_reduction(&kernel, &sub);          // カウンタを先に、次に持ち上げた最良解
```

`open_reduction` は現時点の最良解を射影してサブランの開始解とし、この状態の RNG からちょうど1回だけシードを引きます。
これにより、シード付きの実行は縮約を通っても再現可能なままです。`close_reduction` は持ち上げた解を取り込む前にカウンタをマージするので、
`best_iteration` にはサブランの仕事量が反映されます。またぐのはサブランの最良解であり、たまたま止まった位置の解ではありません。

この組を囲むループは呼び出し側が書きます。`tests/reduction_crossing.rs` はそのループを trajectory ごと固定したテストで、
[MaxCut kernelization](problems/max_cut_kernel.md) が実例です。

## anytime 曲線を記録する { #recording-the-anytime-curve }

目的関数のプローブを設定するまで `trajectory` は空のままです。そのため既定では `update_best` は何も確保しません。

```rust
state.set_objective_probe(|sol| sol.objective as f64);
```

以後、最良解が更新されるたびに、時刻、反復番号、目的関数値からなる `TrajectoryPoint` が追加されます。
サブランのクローンはプローブを引き継ぎ、`update_state` がその反復番号を親の座標系に写し直します。
ライブラリ内で呼んでいるのは `benchmark/runner.rs` だけで、そこで集めたものがレポートの `trajectory`
になり、ベンチマークビューアが描く曲線になります。

`duration()` は現在のサブランの経過時間で、`ClearBest` と `StartBest` がリセットする `start_time`
から測ります。停止条件が比較するのはこの値です。

## 再現性 { #reproducibility }

乱数の出どころはすべて `state.rng` に集まり、それ以外はありません。初期解、`random_neighbor`、
タブーテニュア、交叉、摂動、そしてサブランと縮約のシードです。構築時にシードを固定すれば、
入れ子のメタヒューリスティクスを含む組み合わせ全体がビット単位で再現されます。
