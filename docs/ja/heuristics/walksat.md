# WalkSatForSat

**API:** [`WalkSatForSat`](../../api/optopus/heuristic/struct.WalkSatForSat.html)

[MaxSAT](../problems/sat.md) 専用のヒューリスティクスです。WalkSAT は探索を集中させます。ステップごとに `n` 個の変数すべてを走査する代わりに、
現在充足されていない節を一つ引き、その中の変数を一つ反転します。

## 例 { #example }

```rust
use optopus::prelude::*;

let mut sat = Sat::new(3);
sat.add_clause([1, -2, 3]); // (x1 ∨ ¬x2 ∨ x3)。リテラルは符号付きの 1 始まり
sat.add_clause([-1, 2]);
sat.add_clause([1, 2, 3]);

let mut state = SearchState::new(&sat);
let mut ws = WalkSatForSat::new(
    StopCondition::iterations(100_000),
    /* noise    = */ 0.3,
    /* adaptive = */ false,
);
ws.run(&mut state)?;

let sol = &state.best_solution;
println!("{} / {} clauses satisfied", sol.n_satisfied, sat.n_clauses());
println!("assignment = {:?}", sol.x);
```

三つの節のインスタンスは呼び出し方を示すだけです。このヒューリスティクスを使う価値があるのは、ステップごとのコストのおかげで、
汎用の `LocalSearch` や `TabuSearch` では走査しきれないインスタンスを扱えるところにあります。

## アルゴリズムの概要 { #algorithm-sketch }

充足されていない節のリテラルはすべて偽なので、そのどの変数を反転してもその節は充足されます。どれを選ぶかが Selman–Kautz–Cohen の規則です。
各 `run_once` で次を行います。

1. 充足されていない節を一様ランダムに一つ引きます。
2. その節の各変数を break 数、つまりその反転で現在充足されている節がいくつ壊れるかでスコア付けします。
3. 選択します。break 数が `0` の変数があればそれを選びます (コストのない move です)。
   そうでなければ、確率 `noise` で節の中のランダムな変数を、それ以外では break 数が最小の変数を反転します。
4. 反転を確定し、作業用の状態を O(degree) で更新します。

ステップごとのコストは `O(節の長さ × 変数の次数)` で、変数の総数に依存しません。汎用の `LocalSearch` や `TabuSearch`
(move ごとに O(n)) では届かないインスタンスにこのヒューリスティクスが届くのはこのためです。

`is_done` はすべての節が充足された時点で早めに止まります。MaxSAT ではそれが大域最適で、改善の余地がないからです。

### 独自の flip { #its-own-flip }

この move は意図的に `SatFlipNeighbor::apply` を通しません。それはすべての隣接変数についてキャッシュした `gain[]` を更新し、
これが `O(degree²)` で密なインスタンスでの支配的なコストになります。WalkSAT は `gain[]` を読まず、自前の充足リテラル数から選びます。
そのため `x`、`n_satisfied` と作業用の状態だけを O(degree) で更新し、実行の終わりに一度だけ正しい `gain[]` を復元します。
速さの利点はこれに尽きます。

作業用の状態は、実行ごとに一度作り直して差分で維持する三つの構造です。節ごとの充足リテラル数、O(1) で所属判定できる充足されていない節の密なリスト、
そして変数から節への出現インデックスです。

### 適応的なノイズ { #adaptive-noise }

`adaptive = true` のとき、`noise` は初期値にすぎず、Hoos のスケジュールに従います。充足されていない節の数が最小を更新するたびに係数 `φ/2` (φ = 0.2) だけ下げ、
改善のないまま `n_clauses / 6` 回反転すると `φ` だけ 1 に向けて上げます。

## コンストラクタ { #constructor }

```rust
WalkSatForSat::new(
    stop_condition: StopCondition,
    noise: f64,        // 節の中でランダムウォークのステップを行う確率
    adaptive: bool,    // Hoos の自動ノイズ調整
) -> Self
```

`noise` が `[0.0, 1.0]` の外にあれば panic します。

`clear()` は作業用の状態を捨て、作業中のノイズを `noise` に戻すので、新しいエピソードはまっさらな状態で始まります。
複数回のリスタートは外側で組み合わせます。`WalkSat` を [`Restart`](meta.md#restart) で囲むのが普通の形です。

## ベンチマーク設定 { #benchmark-config }

```toml
[[heuristics]]
kind = "WalkSat"
noise = 0.3              # 任意 (値は既定値)
adaptive_noise = false   # 任意 (値は既定値)
[heuristics.stop_condition]
max_duration_secs = 30.0
```

## 参考文献 { #references }

- Selman, B., Kautz, H. A., and Cohen, B. "Noise Strategies for Improving Local
  Search." Proc. AAAI-94, 337-343, 1994.
- Hoos, H. H. "An Adaptive Noise Mechanism for WalkSAT." Proc. AAAI-02,
  655-660, 2002.
