# SlackInductionByStringRemovalForVrp

**API:** [`SlackInductionByStringRemovalForVrp`](../../api/optopus/heuristic/struct.SlackInductionByStringRemovalForVrp.html)

[VRP](../problems/vrp.md) 専用のヒューリスティクスです。Slack Induction by String Removals は、ruin と recreate を 1 つずつしか持たない ruin-and-recreate の探索です。
互いに近くを通る数本のルートから連続した顧客の列(string)を抜き、ときどき挿入位置を読み飛ばしながら貪欲に戻し、焼きなましの判定で採否を決めます。
局所探索を走らせず、オペレータを選ぶこともないので、1 反復が軽く、1 回の実行で多くの反復を回します。

## 例 { #example }

```rust
use optopus::prelude::*;

let vrp = Vrp::load_file("data/instances/vrp/demo16.vrp")?;
let mut state = SearchState::new(&vrp);

let mut sisr = SlackInductionByStringRemovalForVrp::new(
    StopCondition::iterations(100_000),
    /* cooling_iterations = */ 100_000,
);
sisr.run(&mut state)?;

let sol = &state.best_solution;
println!("total distance = {}", sol.total_distance());
for (vehicle, route) in sol.routes.iter().enumerate() {
    println!("vehicle {vehicle}: depot -> {route:?} -> depot");
}
```

近傍を自前で持つので、`neighbor` の型引数は取りません。

## アルゴリズムの概要 { #algorithm-sketch }

`run_once` は、現在の解から候補を 1 つ作ります。

### Ruin { #ruin }

1. 抜く string の本数を引きます。`max_string_length` と、使用中のルート 1 本あたりの平均顧客数の小さいほうを `L` として、本数は `1` から `4 · average_removal / (1 + L) − 1` までの範囲から引きます。
2. 種になる顧客を 1 人引き、そこから近い順に顧客をたどります。
3. まだ壊していないルートに乗っている顧客に当たったら、そのルートを壊します。長さを `1` から、`L` とそのルートの長さの小さいほうまでの範囲で引き、その顧客を含む長さ分の string を抜きます。
4. 引いた本数のルートを壊したら終わりです。

string の半分はそのまま抜きます。残りの半分は分割します。長めの string を選び、その内側の一続きの顧客をルートに残します。
残す区間は顧客 1 人から始まり、ルートに余りがある限り、確率 `1 − split_stop_rate` で 1 人ずつ伸びます。
既定の率ではほぼ伸び切るので、ルートから抜けるのはその両端になります。

近くを通るルートから 1 本ずつ string を抜くと、1 か所のまわりで複数の車両に同時に空きができます。名前にある slack はこの空きのことです。

### Recreate { #recreate }

抜いた顧客を並べる順序を、反復ごとに引きます。

| 順序 | 重み |
|---|---|
| ランダム | 4 |
| 需要の大きい順 | 4 |
| デポから遠い順 | 2 |
| デポから近い順 | 1 |

各顧客は、すべてのルートのすべての位置で価格付けされ、最も安い位置に挿入されます。ただし、各位置は確率 `blink_rate` で読み飛ばされます。
最も安い位置が飛ばされると顧客は 2 番目に安い位置に入り、以下同様です。recreate のばらつきはここからしか生まれません。

位置の価格には、容量やルート時間の違反に問題が課すペナルティが含まれます。そのため空きのあるルートは空きのないルートに必ず勝ち、待機中の車両も位置の 1 つとして数えられます。
すべての車両が使用中でどこにも空きがないときは、違反が最も安く済む位置に入り、探索はペナルティ込みの目的関数の上で続きます。

### 受理 { #acceptance }

コスト `c'` の候補は、次の条件でコスト `c` の現在解を置き換えます。

```text
c' < c − T · ln(u),   u は (0, 1] の一様乱数
```

温度 `T` は `initial_temperature` から始まり、反復ごとに定数を掛けて、`cooling_iterations` 回の反復で `final_temperature` に届きます。
それを過ぎても実行が続く場合は、最終温度のままです。

2 つの温度は目的関数と同じ単位です。既定の `100` と `1` は、ルート 1 本が数千の長さになる CVRPLIB のインスタンスの距離に合わせた値です。
座標が単位正方形に収まるインスタンスでは、数桁小さい温度が必要です。

## コンストラクタ { #constructor }

```rust
SlackInductionByStringRemovalForVrp::new(
    stop_condition: StopCondition,
    cooling_iterations: u64,
) -> Self
```

`stop_condition` は実行の終わりを決めます。[停止条件](../guide/stop_conditions.md)を参照してください。
`cooling_iterations` は温度が最終値まで下がるのにかける反復数です。時間で止まる実行には読み取れる反復数がないので、独立した引数になっています。
反復数で止める実行では、ふつう同じ数を 2 回渡します。

`cooling_iterations` が 0 なら panic します。

それ以外はすべて既定値つきの builder で、既定値は型の関連定数です。

| Builder | 設定するもの | 既定値 |
|---|---|---|
| `with_temperatures(initial, last)` | 探索の開始時と終了時の温度 | `100.0` / `1.0` |
| `with_removal(average_removal, max_string_length)` | 1 反復で抜く顧客数の平均と、1 本のルートから抜く string の最大長 | `10.0` / `10` |
| `with_split_stop_rate(rate)` | 分割した string が残す区間の伸びが止まる確率 | `0.01` |
| `with_blink_rate(rate)` | recreate が位置を読み飛ばす確率 | `0.01` |

`clear()` は温度を初期値に戻し、builder で設定した値は保ちます。

## ベンチマーク設定 { #benchmark-config }

```toml
[[heuristics]]
kind = "SlackInductionByStringRemoval"
cooling_iterations = 2000000   # 必須
initial_temperature = 100.0    # 省略可(既定値)
final_temperature = 1.0        # 省略可(既定値)
average_removal = 10.0         # 省略可(既定値)
max_string_length = 10         # 省略可(既定値)
split_stop_rate = 0.01         # 省略可(既定値)
blink_rate = 0.01              # 省略可(既定値)
[heuristics.stop_condition]
max_duration_secs = 60.0
```

時間制限のもとでは、実行が届く `cooling_iterations` を選んでください。届かない数にすると、探索は最後まで高い温度のままになります。
1 回目の実行のレポートにある `n_accepted + n_rejected` が、その制限で回せる反復数です。

## 参考文献 { #references }

- Christiaens, J. and Vanden Berghe, G. "Slack Induction by String Removals
  for Vehicle Routing Problems." *Transportation Science*, 54(2), 417-433,
  2020.
