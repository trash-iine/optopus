# 停止条件

**API:** [`StopCondition`](../../api/optopus/heuristic/struct.StopCondition.html)

どのヒューリスティクスも、いつ止まるかを決める [`StopCondition`](#builder-api) を受け取ります。
条件は各反復の先頭で調べられ、設定した上限のどれか一つに達した時点で発火します。

## Builder API

```rust
use optopus::prelude::*;
use std::time::Duration;

// 一つの基準だけを持つコンストラクタ:
StopCondition::iterations(1_000_000);
StopCondition::duration(Duration::from_secs(30));
StopCondition::failed_updates(10_000);

// `with_*` で基準を追加する:
StopCondition::iterations(1_000_000)
    .with_duration(Duration::from_secs(30))
    .with_failed_updates(10_000);
```

- [`StopCondition::iterations`](../../api/optopus/heuristic/struct.StopCondition.html#method.iterations) 
- [`StopCondition::duration`](../../api/optopus/heuristic/struct.StopCondition.html#method.duration) 
- [`StopCondition::failed_updates`](../../api/optopus/heuristic/struct.StopCondition.html#method.failed_updates)

| 基準 | 意味 |
|---|---|
| `max_iteration` | 現在の実行でこの回数の反復が経過したら止まる。 |
| `max_duration` | 実行開始からの実時間がこの長さに達したら止まる。 |
| `max_failed_update` | `best_solution` を改善しない反復がこの回数続いたら止まる。 |

`new(max_iteration, max_duration, max_failed_update)` を使えば、`Option` のフィールドから直接
`StopCondition` を作ることもできます (設定ファイルからデシリアライズするときに便利です)。

## サブラン { #sub-runs }

サブランのクローンとマージのパターン ([基本概念](../concepts.md#sub-run-clonemerge-pattern) を参照) の中では、
反復回数は全体の反復ではなくサブランの開始から数えます。外側の条件は引き続き全体のカウンタを使うので、
内側の `failed_updates(100)` は内側のフェーズの進み具合で発火し、実行全体は外側の `iterations(10_000)` の予算で管理されます。

## ヒント { #tips }

- `iterations` は最も再現性が高い基準です (時計に依存しません)。決定的に止めたいテストやベンチマークで使ってください。
- `duration` はマシンをまたいで時間予算をそろえて比較するのに向いています。
- `failed_updates` は「改善が見つからなくなるまで走らせる」ための自然な条件です。
  `LocalSearch` (これを `1` に強制します) や `Iterated` の内側のフェーズと組み合わせます。
- 緩い目標と厳しい上限を両方持たせたいときは基準を組み合わせます。たとえば
  `failed_updates(1_000).with_duration(Duration::from_secs(60))` です。
