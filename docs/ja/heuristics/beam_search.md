# BeamSearch

**API:** [`BeamSearch`](../../api/optopus/heuristic/struct.BeamSearch.html)

`beam_width` 個の候補解からなるビームを並行して保持します。

## 例 { #example }

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]));
let mut state = SearchState::new(&mc);
let mut bs = BeamSearch::<MaxCut, MaxCutFlipNeighbor>::new(
    StopCondition::iterations(1_000),
    /* beam_width = */ 5,
);
bs.run(&mut state)?;
println!("cut weight = {}", state.best_solution.objective);
```

## アルゴリズムの概要 { #algorithm-sketch }

各ステップで次を行います。

1. ビームの各メンバーのすべての近傍を展開します。
2. `state.solution` を最良の候補にします (`best_solution` も更新します)。
3. 上位 `beam_width` 個の候補を残し、残りを捨てます。

ステップ 3 は `select_nth_unstable_by` (期待 O(n)) を使います。生き残ったビームの中の順序は関係ないからです。

## コンストラクタ { #constructor }

```rust
BeamSearch::<P, N>::new(
    stop_condition: StopCondition,
    beam_width: usize,
) -> Self
```

`N` は `MoveToNeighbor<P> + Rankable` を満たす必要があります。

`beam_width == 0` なら panic します。

`clear()` はビームを空にします。ビームは `run` の後の最初の `run_once` で `state.solution` から再び作られます。

## コスト { #cost }

`LocalSearch` や `TabuSearch` と違い、BeamSearch はすべての近傍を実体化します。
各ステップは O(beam_width × |近傍|) のメモリを使い、その回数だけ `apply_to_solution` を呼びます。
近傍が大きい問題では `beam_width` を控えめにしてください。

## ベンチマーク設定 { #benchmark-config }

```toml
[[heuristics]]
kind = "BeamSearch"
neighbor = "Flip"        # 必須。使える値は問題ごとに決まる
beam_width = 5           # 必須。1 以上
[heuristics.stop_condition]
max_iteration = 1_000
```

`beam_width` が 0 の設定は読み込み時に拒否されます。

## 参考文献 { #references }

- Ow, P. S. and Morton, T. E. "Filtered Beam Search in Scheduling."
  *International Journal of Production Research*, 26(1), 35-62, 1988.
