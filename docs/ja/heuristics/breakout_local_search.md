# BreakoutLocalSearch

**API:** [`BreakoutLocalSearch`](../../api/optopus/heuristic/struct.BreakoutLocalSearch.html)

貪欲な局所探索のフェーズと適応的な摂動のフェーズを交互に行います。

BLS は一つのアルゴリズムというよりフレームワークです。降下、摂動の集まり、そしてそのうちどれをどこまで行うかを選ぶスケジュールからなり、
三つのどれも問題が二値であることを必要としません。そのため `BreakoutLocalSearch<P, S>` は任意の `P: ProblemTrait` を受け取ります。
このフレームワークを述べた論文も、BLS を二値問題ではない vertex separator problem で導入しています。

Benlic と Hao はこのフレームワークを、具体的な BLS が与える四つの手続きとして述べていて、ここではそれが三か所に収まります。
`DescentBasedSearch` と `Perturb` は普通のヒューリスティクスで、降下は `Box<dyn Heuristic<P>>`、キックはその集まりです。
`DetermineJumpMagnitude` と `DeterminePerturbationType` は降下とそれに続くキックの間で行う二つの決定で、
その集まりへのインデックスを返す
[`PerturbationSchedule`](../../api/optopus/heuristic/trait.PerturbationSchedule.html)
トレイトになっています。

残りは BLS が持つべきものではありません。初期解はライブラリ全体の慣習どおり `SearchState` から来て、停止条件までラウンドを繰り返すループは
`Heuristic::run` のものです。一回の `run_once` が一ラウンドで、降下、二つの決定、キックの順に進みます。

三つの手続きが共有する履歴は `SearchState` のタブーメモリです。降下がそれを書き込み、動かした頂点をテニュア分の反復の間禁止します。
Benlic と Hao が書き込みを置いた場所も同じです。そして方向付きの摂動がそれを読みます。
これによって、摂動が直前に走った降下を元に戻してしまうことを防いでいます。

[MaxCut](../problems/max_cut.md) では、強いキックに [`RandomWalk`](random_walk.md)、weak flip に
[`TabuSearch`](tabu_search.md)、そして手書きの方向付き swap (`src/heuristic/specific/max_cut/best_swap.rs`)
を使います。最後のものは汎用の相当物がない唯一のオペレータで、`M2` は一回の move で分割の両側から一頂点ずつ動かします。

## 例 { #example }

```rust
use optopus::prelude::*;

let mut rng = seeded_rng(42);
let mc = MaxCut::new(Graph::erdos_renyi(800, 0.02, &mut rng));
let mut state = SearchState::new_with_seed(&mc, 42);

let mut bls = bls_for_max_cut(
    StopCondition::iterations(100_000),
    /* tabu_tenure = */ (10, 300),
    /* t           = */ 1_000,
    /* l0          = */ 8,          // 0.01 * |V|
    /* p0          = */ 0.8,
    /* q           = */ 0.5,
);
bls.run(&mut state)?;
println!("cut weight = {}", state.best_solution.objective);
```

`l0` と `tabu_tenure` はインスタンスに依存するので、ここでは定数のままにせず `|V| = 800` から導いています。
[ベンチマーク設定](#benchmark-config) を参照してください。

## アルゴリズムの概要 { #algorithm-sketch }

- 貪欲フェーズ。厳密に最良の改善 flip を繰り返し適用し、タブーマップを更新します。
- 摂動フェーズ。`p = max(exp(−omega / t), p0)` が方向付き (weak) 摂動の確率で、改善しなかった回数のカウンタ
  `omega` が増えるにつれて減衰します。
  - `omega == 0`。直前の降下が全体の最良解を改善したか、`omega` が `t` を超えてリセットされたばかりのときで、強い摂動 (ランダムな flip) を行います。
  - `0 < omega <= t`。確率 `p * q` で weak flip、確率 `p * (1 − q)` で weak swap、確率 `1 − p` で強い摂動 (ランダムな flip) を行います。
    `omega` が増えると `p` は `p0` に向かって減衰するので、強い摂動が次第に起こりやすくなります。
  - `omega > t`。`omega` を 0 に戻し、最初の分岐がそれを強い摂動の強制として読みます。
- どちらの weak 摂動も、タブーでない move のうち gain が最も大きいものを選び、タブーの move は全体の最良解を上回る場合だけ認めます (aspiration 規則)。
  これらの weak 摂動は最良の flip / swap の move を `l` 回適用します。
- 摂動の長さ `l` は、降下が前のラウンドと同じ局所最適に着地するたびに 1 増え、抜け出すたびに `l0` に戻ります。

## 原論文の方式との違い { #differences-from-the-original-scheme }

この実装は Benlic と Hao にかなり忠実に従っていて、以下の相違点はそれぞれ論文が公表しているカット値と照らし合わせて確認しています。

- `tabu_tenure` は、[`TabuSearch`](tabu_search.md) と同じく禁止の長さそのものです。原論文のテニュア `γ` は、頂点を記録するときに一度、
  適格性の判定でもう一度加えられるので、頂点は `2γ` の間禁止されます。`TabuMemory` はその長さを直接保持するので、
  論文の G-set での `rand[3, n/10]` はここでは `[6, n/5]` と書きます。上限だけを二倍にしても再現できず、範囲全体を拡大する必要があります。
- バケットソートはありません。原論文は頂点を gain でバケットに分けるので、最大 gain の move の選択は O(1) で、
  move のコストは gain の更新がもともと必要とする O(degree(v)) の再バケット化だけです。ここではどの選択も n 個の flip 近傍すべての線形走査で、
  move 一回あたり O(n) です。降下も素の `LocalSearch` なので同じです ([下](#what-a-round-is-built-from) を参照)。
  gain の更新自体は O(degree(v)) です。どちらの方法でも選ばれる move は同じなので、犠牲になるのは速度だけです。
- swap は反復カウンタを 2 進めます (`MaxCutSwapNeighbor::apply_to_iteration`)。BLS はどの move も 1 と数えます。
  この `+2` はすべての二値問題の swap に共通するライブラリ全体の慣習なので、一つのヒューリスティクスのためにここで変えることはしていません。

## 一ラウンドの構成 { #what-a-round-is-built-from }

一ラウンドの四分の三はライブラリ自身のヒューリスティクスです。降下は [`LocalSearch`](local_search.md)、強い摂動は
[`RandomWalk`](random_walk.md)、weak flip は [`TabuSearch`](tabu_search.md) です。
これらは専用のオペレータと同じ move を選び、記録は `SearchState` のモードなので、その `apply` は同じ禁止を書き込みます。
そのため、Benlic と Hao が降下のループの中に置いたタブーリストの更新はここでも行われます。

weak swap だけが手書きのオペレータです。`M2` は一回の move で分割の両側から一頂点ずつ動かし、一歩ずつの探索を二つ組み合わせてもそれは表せないからです。
片側に適格な頂点がないと、もう片方の頂点だけが動いてしまうことになります。

## コンストラクタ { #constructor }

MaxCut にはスケジュールを埋める builder があります。

```rust
bls_for_max_cut(
    stop_condition: StopCondition,
    tabu_tenure: (u64, u64),
    t: u64,
    l0: u64,
    p0: f64,
    q: f64,
) -> BreakoutLocalSearchForMaxCut
```

ほかの問題では、自前の降下とスケジュールを与えて直接作ります。

```rust
BreakoutLocalSearch::new(
    stop_condition: StopCondition,
    tabu_tenure: (u64, u64),
    descent: Box<dyn Heuristic<P>>,
    schedule: S,
) -> Self
```

スケジュールは選択肢となる摂動を自分で持つので、集まりを渡したり順序を取り決めたりする必要はありません。
`AdaptivePerturbation` はランダムな摂動一つと方向付きの摂動のリストを受け取り、後者はそれぞれ方向付き確率の中での取り分を持ちます。

```rust
AdaptivePerturbation::<P>::new(
    t: u64,
    l0: u64,
    p0: f64,
    random: Box<dyn Heuristic<P>>,
    directed: Vec<(Box<dyn Heuristic<P>>, f64)>,
) -> Self
```

Benlic と Hao の `q` は要素が二つの場合で、`[(flip, q), (swap, 1 - q)]` です。方向付きの摂動はいくつでも (一つでも) かまいません。

| パラメータ | 意味 |
|---|---|
| `tabu_tenure` | LS フェーズのタブーテニュアの範囲 `(min, max)` |
| `t` | `omega` カウンタがリセットされるまでの周期 |
| `l0` | 摂動の長さの初期値 |
| `p0` | 摂動確率の最小値 |
| `q` | 方向付き確率のうち flip が取る割合。残りは swap |

`clear()` はスケジュールをリセットし (`omega` を 0 に、`l` を `l0` に、記憶していた局所最適を捨てる)、降下とスケジュールの摂動をクリアします。
禁止は `SearchState` のものなのでここではクリアしません。いずれにせよ、サブランのクローンは空のタブーメモリで始まります。
記憶していた局所最適を残さずに捨てるのは、同じスケジュールが別のインスタンスで再利用されることがあるからです。
ラウンドごとに部分問題を作り直すメタヒューリスティクスはまさにそうします。

## スケジュールを置き換える { #replacing-the-schedule }

`PerturbationSchedule<P>` を実装して `BreakoutLocalSearch::new` に渡します。フレームワークのループと `Heuristic` はそのまま使い、
置き換えるのは二つの決定と、それが選ぶオペレータです。

| メソッド | 呼ばれるタイミング |
|---|---|
| `determine_jump_magnitude(state)` | 降下の後、降下が到達した局所最適とともに |
| `select(state)` | その直後。適用するオペレータを返す |
| `round_ended(state)` | キックとその `update_best` がラウンドを閉じた後。既定では何もしない |
| `reset` / `clear_perturbations` | `Heuristic::clear` から |

`Heuristic::run_once` と同じく、どちらの決定にも状態が渡されるので、乱数を引くだけでなく探索の現在地をもとにオペレータを選べます。
乱数は自前の RNG ではなく `state.rng` から引いてください。これがシード付きの実行を再現可能に保ちます。

二つの答えが一つの決定である方策は、長さが先に問われるので、`determine_jump_magnitude` で決めて `select` では選んだものを返します。
降下だけで何が起きたかを読むには、`round_ended` に渡された値を覚えておきます。それが次の降下の出発点の値です。

MaxCut では `max_cut_descent()` と `max_cut_perturbation(kind, tabu_tenure)` が部品を作るので、
自前のスケジュールでもオペレータを作り直さずに再利用できます。

[学習した摂動方策で BLS を動かす](../guide/learned_perturbation.md)
では、文脈付きバンディットをこのスケジュールとして書く例を順に説明しています。

## ベンチマーク設定 { #benchmark-config }

```toml
[[heuristics]]
kind = "BreakoutLocalSearch"
tabu_tenure = [6, 160]    # 論文の rand[3, |V|/10] を二回数えたもの
t = 1000
l0 = 80                   # 0.01 * |V|
p0 = 0.8
q = 0.5
[heuristics.stop_condition]
max_duration_secs = 30.0
```

`tabu_tenure` はどの kind でも同じ意味で、move はその反復回数だけ禁止されたままになります。そのため
[`TabuSearch`](tabu_search.md) で調整した範囲はそのまま持ち込めます。論文から引用した値は、上のようにファイルに書くときに二倍にします。

## 参考文献 { #references }

- Benlic, U. and Hao, J.-K. "Breakout Local Search for the Max-Cut problem."
  *Engineering Applications of Artificial Intelligence*, 26(3), 1162-1173,
  2013.
