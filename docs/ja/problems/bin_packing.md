# Bin Packing

**API:** [`BinPacking`](../../api/optopus/problem/bin_packing/struct.BinPacking.html)

1 次元の Bin Packing は、大きさの決まった品目を、同じ容量の bin に容量を超えないように詰め、使う bin の数を最小にする問題です。
bin の数だけでは目的関数が平らで、同じ数の詰め方はすべて同点になります。そこで探索は次の値を最小化します。

```text
num_bins − (1 / num_bins) Σ_b (load_b / capacity)²
```

bin の数から Falkenauer の充填項を引いたものです。この項は `(0, 1]` に収まるので bin が少ない詰め方が必ず勝ち、
同じ数の中では bin がより満杯の詰め方、つまり bin を一つ空にできる詰め方を好みます。
`BinPacking::lower_bound` は `⌈Σ size / capacity⌉` を返します。

## 例 { #example }

```rust
use optopus::prelude::*;

let bp = BinPacking::load_file("data/instances/bin_packing/falkenauer/u250_00.txt")?;
let mut state = SearchState::new_with_seed(&bp, 42);
AdaptiveLargeNeighborhoodSearch::<BinPacking>::new(
    StopCondition::duration(std::time::Duration::from_secs(10)),
    /* removal_fraction = */ 0.15,
    /* cooling_rate     = */ 0.9995,
)
.run(&mut state)?;
println!("{} bins, at least {}", state.best_solution.num_bins(), bp.lower_bound());
```

## 解 { #solution }

[`BinPackingSolution`](../../api/optopus/problem/bin_packing/struct.BinPackingSolution.html) は、各品目の bin `bin_of`、各 bin の負荷 `loads`、`objective` を持ちます。
bin は常に空きのない `0..num_bins()` で番号付けされ、bin を空にする移動は残りの番号を付け直します。
`Distance` は bin の番号ではなく、どの品目が同じ bin にあるかを比べます。

## 近傍 { #neighbors }

| 型 | 設定 | 移動 |
|---|---|---|
| `BinPackingRelocateNeighbor` | `Relocate` | 品目を一つ、空きのある別の bin に移す。bin を空にすることで数が減る。 |
| `BinPackingSwapNeighbor` | `Swap` | 別々の bin にある大きさの違う二つの品目を、両方が収まるときに交換する。 |

どちらも負荷から O(1) で価格付けし、bin を容量超過にすることはありません。

## Ruin and recreate { #ruin-and-recreate }

`BinPacking` は品目を要素、bin をコンテナとして [`Ruinable`](../traits.md) を実装しています。このトレイトが想定していたそのものの形です。
開いている bin がそれぞれ bucket で、その後ろに常に空の bin を一つ用意するので、修復で置き場所がなくなることはありません。
収まらない bin への挿入コストは無限大で、regret、つまり品目の最良の bin と次善の bin の差に意味があります。
[ALNS](../heuristics/alns.md) は局所修復なしでこの上で動き、[GreedyRandomizedConstruction](../heuristics/greedy_randomized_construction.md) も動きます。

## 交叉 { #crossover }

`BinPackingGroupCrossover` は Falkenauer のグルーピング交叉の考え方で、品目ではなく bin を丸ごと受け継ぎます。
子は二つの親から、満杯の bin から順に交互に bin を受け取り、すでに配置した品目を含む bin は飛ばし、残りを First Fit Decreasing で詰めます。

## ファイル形式 { #file-format }

```text
n capacity
size_0
size_1
...
```

値はすべて空白区切りの整数です。Falkenauer の 160 インスタンスを `data/instances/bin_packing/falkenauer/` に同梱しています。
一様分布の `u120` から `u1000` は容量 150、triplet の `t60` から `t501` は大きさが整数になるよう 10 倍して容量 1000 にしています。
triplet のインスタンスは作り方から最適値がちょうど `n / 3` bin です。

## 参考文献 { #references }

- Falkenauer, E. "A hybrid grouping genetic algorithm for bin packing."
  *Journal of Heuristics*, 2(1), 5-30, 1996.
- インスタンスの出典とライセンスは [`data/instances/README.md`](https://github.com/trash-iine/optopus/blob/main/data/instances/README.md) を参照してください。
