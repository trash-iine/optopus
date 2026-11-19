# EdgeAssemblyCrossoverForTsp

**API:** [`EdgeAssemblyCrossoverForTsp`](../../api/optopus/heuristic/struct.EdgeAssemblyCrossoverForTsp.html)

距離が対称な [TSP](../problems/tsp.md) 専用のヒューリスティクスです。遺伝的アルゴリズムで、その交叉である edge assembly crossover は、子をほぼ両親の辺だけから組み立てます。
置き換えの規則は、集団が長いあいだ多様なまま保たれるように作られています。

## 例 { #example }

```rust
use optopus::prelude::*;

let tsp = Tsp::load_file("data/instances/tsp/eil101.tsp")?;
let mut state = SearchState::new(&tsp);

let mut eax = EdgeAssemblyCrossoverForTsp::new(
    StopCondition::duration(std::time::Duration::from_secs(30)),
    /* population_size    = */ 300,
    /* offspring_per_pair = */ 30,
);
eax.run(&mut state)?;
println!("tour length = {}", state.best_solution.objective);
```

近傍を自前で持つので、`neighbor` の型引数は取りません。実行はふつう時間制限より前に終わります。[止まるとき](#when-it-stops)を参照してください。

## アルゴリズムの概要 { #algorithm-sketch }

集団は `population_size` 本の巡回路から始まります。1 本目は現在解、残りはランダムで、どれも
[`AnchoredTourDescent`](../../api/optopus/problem/tsp/struct.AnchoredTourDescent.html)
(近傍リスト上の Or-opt と 2-opt の降下)で局所最適まで下ろします。

1 世代は、集団のランダムな順序を引き、各個体を 1 人目の親、その次の個体を 2 人目の親として交配します。
`run_once` 1 回は交配 1 回、集団を作っているあいだは個体 1 つの構築です。

### 交叉 { #crossover }

1. 両親の片方だけが持つ辺を取り出します。どの都市でも、一方の親から来る辺と他方から来る辺は同じ本数です。
2. それらを AB-cycle に分けます。AB-cycle は、1 人目の親の辺と 2 人目の親の辺を交互にたどる閉路です。歩き始めの都市はランダムで、選択肢があるところもランダムに選びます。
3. 子は、1 人目の親で AB-cycle を 1 つ入れ替えたものです。その cycle にある 1 人目の親の辺を外し、2 人目の親の辺を入れます。どの都市も辺を 2 本持ったままなので、結果は 1 本の巡回路か、複数の部分巡回路です。
4. 部分巡回路を、小さいものから順につなぎます。最小のものについて、その辺 1 本と別の部分巡回路の辺 1 本を外し、4 つの端をつなぐ 2 本の辺を入れます。入れ替えは最も安いものを選びます。相手の辺は、部分巡回路の各都市の近傍 10 都市から探し、見つからなければ近傍 50 都市まで広げます。

1 回の交配は、重複なしで引いた最大 `offspring_per_pair` 個の AB-cycle から、子を 1 つずつ作ります。

子が 1 人目の親と違うのは、cycle 1 つと数回のつなぎ替えだけです。交配は両親を 1 回ずつ読み、子 1 つを作る時間は cycle と、つなぐ部分巡回路の長さに比例します。cycle が巡回路を大きく分けない限り、巡回路全体の長さより短く済みます。

### 選択と置き換え { #selection-and-replacement }

選択が選んだ子は、1 人目の親より短ければ、その親を置き換えます。集団のほかの個体は変わりません。
個体は、自分の辺を大部分残した巡回路にしか置き換えられないので、一方が良いというだけで 2 つの個体が同じ巡回路になることはありません。

| `EaxSelection` | 選ぶもの |
|---|---|
| `Greedy` | 最も短い子 |
| `Entropy` (既定) | 短くなる量を、それを採ることで集団が失う辺のエントロピーで割った値が最大の子 |

辺のエントロピーは、各辺を持つ個体の割合 `p` についての `−Σ p ln p` です。
集団にあまりない辺で巡回路を短くする子は、ほとんどの個体がすでに持っている辺を写すだけの子より高く評価され、エントロピーをまったく失わない子が最も高くなります。

### 止まるとき { #when-it-stops }

停止条件のほかに、実行は次の 2 つの場合に自分から終わります。

- 集団のすべての巡回路が同じ長さになった。
- 最良の巡回路が停滞した。最良が `1500 / offspring_per_pair` 世代のあいだ更新されなかった時点で、それまでの世代数の 10 分の 1 を上限とし、最良がその世代数だけ更新されなければ終わります。

### 入っていないもの { #what-is-not-here }

これは公開されているアルゴリズムの第 1 段階です。公開版は第 1 段階が停滞すると第 2 段階に移り、隣り合う複数の AB-cycle を一度に入れ替えます。どの cycle を採るかは tabu search で探し、巡回路の残りの部分と接する都市が少ないまとまりを選びます。
第 2 段階は実装していないので、実行は公開版が段階を切り替えるところで終わります。

## コンストラクタ { #constructor }

```rust
EdgeAssemblyCrossoverForTsp::new(
    stop_condition: StopCondition,
    population_size: usize,      // 2 以上
    offspring_per_pair: usize,   // 1 以上
) -> Self
```

`stop_condition` は実行の終わりを決めます。[停止条件](../guide/stop_conditions.md)を参照してください。交配 1 回が 1 反復です。
公開されている値は `300` と `30` で、関連定数 `POPULATION_SIZE` と `OFFSPRING_PER_PAIR` です。

| Builder | 設定するもの | 既定値 |
|---|---|---|
| `with_selection(EaxSelection)` | どの子が 1 人目の親を置き換えるか | `EaxSelection::Entropy` |

集団の構築には、個体 1 つにつきランダムな巡回路からの降下 1 回がかかります。メモリは、個体 1 つにつき都市ごとに隣 2 つ分です。

`clear()` は集団を捨てるので、2 回目の `run` は新しい集団を作ります。

## ベンチマーク設定 { #benchmark-config }

```toml
[[heuristics]]
kind = "EdgeAssemblyCrossover"
population_size = 300      # 省略可(既定値)
offspring_per_pair = 30    # 省略可(既定値)
selection = "Entropy"      # 省略可、Entropy(既定) | Greedy
[heuristics.stop_condition]
max_duration_secs = 60.0
```

距離が非対称なインスタンスでは、各実行がエラーで失敗します。

## 参考文献 { #references }

- Nagata, Y. and Kobayashi, S. "A Powerful Genetic Algorithm Using Edge
  Assembly Crossover for the Traveling Salesman Problem." *INFORMS Journal on
  Computing*, 25(2), 346-363, 2013.
