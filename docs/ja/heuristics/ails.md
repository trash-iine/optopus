# AdaptiveIteratedLocalSearchForVrp

**API:** [`AdaptiveIteratedLocalSearchForVrp`](../../api/optopus/heuristic/struct.AdaptiveIteratedLocalSearchForVrp.html)

[VRP](../problems/vrp.md) 専用のヒューリスティクスです。Adaptive Iterated Local Search の大規模インスタンス向けの変種で、参照解を摂動し、降下し、その結果を新しい参照解にするかどうかを決めます。
1 回の摂動で外す顧客の数と、結果に求める良さの両方が実行中に適応し、どちらも実行の終わりに向けて厳しくなります。

## 例 { #example }

```rust
use optopus::prelude::*;

let vrp = Vrp::load_file("data/instances/vrp/demo16.vrp")?;
let mut state = SearchState::new(&vrp);

let mut ails = AdaptiveIteratedLocalSearchForVrp::new(StopCondition::iterations(20_000));
ails.run(&mut state)?;

let sol = &state.best_solution;
println!("total distance = {}", sol.total_distance());
for (vehicle, route) in sol.routes.iter().enumerate() {
    println!("vehicle {vehicle}: depot -> {route:?} -> depot");
}
```

近傍を自前で持つので、`neighbor` の型引数は取りません。

## アルゴリズムの概要 { #algorithm-sketch }

最初の `run_once` は現在解を局所最適まで下ろし、それが参照解になります。その後の 1 回が 1 反復です。

1. 除去と挿入を、それぞれ 2 種類から 1 つ引きます。
2. 参照解から、その除去の degree が示す数だけ顧客を外します。
3. 外した顧客を、ランダムな順序で、引いた挿入に従って戻します。
4. 動かした顧客をアンカーにして granular な経路降下を行い、1 パスで何も変わらなくなるまで続けます。
5. 結果と参照解の距離を測り、それで除去の degree を調整します。
6. 結果が閾値以下なら、新しい参照解として受理します。

### 摂動 { #perturbation }

| 除去 | 外すもの |
|---|---|
| Concentric | 引いた顧客 1 人に最も近い顧客たち |
| Sequential | 引いた顧客 1 人の後ろに、そのルートを回る順で続く顧客の列 |

| 挿入 | 顧客を入れる場所 |
|---|---|
| 距離による挿入 | 配置済みで最も近い顧客の隣 |
| コストによる挿入 | 配置済みで近い `neighbors` 人のうち、遠回りが最も小さくなる顧客の隣 |

デポも顧客の近傍の 1 つとして順番が回ってきて、そのときはすべてのルートの先頭を表します。「隣」は前か後ろで、遠回りが小さいほうです。
どちらの挿入も容量を見ません。続く降下は、過積載を問題のペナルティで価格付けします。このペナルティはどんな距離よりも重いので、降下はまず実行可能性を回復し、そのあとでルートを短くします。

### 摂動の degree { #perturbation-degree }

除去はそれぞれ自分の degree、つまり外す顧客の数を持ち、`distance_max` から始まります。結果と参照解の距離は、結果の辺のうち参照解にないものの数です。
除去を `gamma` 回使うごとに、その degree に次の値を掛けます。

```text
理想の距離 / その除去の結果が落ち着いた距離の平均
```

degree は `1` から `n − 2` の範囲に保たれます。結果が参照解に戻ってしまう除去は次からより多く外し、結果が遠くに行く除去はより少なく外します。
理想の距離は、実行を通じて `distance_max` から `distance_min` へ幾何的に下がります。

### 受理 { #acceptance }

コスト `f` の実行可能な結果は、次の条件で参照解になります。

```text
f ≤ low + eta · (mean − low)
```

`mean` は降下が落ち着いたコストの移動平均、`low` は直近 `gamma` から `2 · gamma` 反復の中での最良です。
`eta` は実行を通じて `1` から `eta_min` へ幾何的に下がります。始めのうちは平均的な結果が受理され、終わりには最近の最良に並ぶものだけがほぼ受理されます。
制約を破る結果は平均に入れず、参照解がそれ以上に制約を破っているときだけ置き換えます。

### 実行の長さ { #the-length-of-the-run }

2 つのスケジュールは、実行が行う反復数の上に配置されます。その数は次の順で決まります。

1. `with_expected_iterations` に渡した数
2. 停止条件の反復数の上限
3. 停止条件の時間制限のうち経過した割合からの推定。`gamma` 反復ごとに更新します。

3 つのどれもない実行はエラーです。更新失敗回数だけの停止条件からは、実行の長さが分かりません。

## コンストラクタ { #constructor }

```rust
AdaptiveIteratedLocalSearchForVrp::new(stop_condition: StopCondition) -> Self
```

`stop_condition` は実行の終わりを決めます。[停止条件](../guide/stop_conditions.md)を参照してください。

それ以外はすべて既定値つきの builder で、既定値は型の関連定数です。

| Builder | 設定するもの | 既定値 |
|---|---|---|
| `with_gamma(gamma)` | 除去の degree を調整する間隔(反復数)。2 つの平均の窓でもある | `30` |
| `with_distances(min, max)` | 実行の終わりと始めでの理想の距離 | `15.0` / `30.0` |
| `with_eta_min(eta_min)` | `eta` の下がる先 | `0.01` |
| `with_neighbors(neighbors)` | コストによる挿入と降下の move が見る近傍の顧客数 | `40` |
| `with_expected_iterations(n)` | スケジュールを配置する長さ | 停止条件から |

`clear()` はスケジュールを実行の始めに戻します。

## ベンチマーク設定 { #benchmark-config }

```toml
[[heuristics]]
kind = "AdaptiveIteratedLocalSearch"
gamma = 30               # 省略可(既定値)
distance_min = 15.0      # 省略可(既定値)
distance_max = 30.0      # 省略可(既定値)
eta_min = 0.01           # 省略可(既定値)
neighbors = 40           # 省略可(既定値)
# expected_iterations = 1000000   # 省略可、下を参照
[heuristics.stop_condition]
max_duration_secs = 60.0
```

`expected_iterations` が必要なのは、停止条件が `max_iteration` も `max_duration_secs` も設定していないときだけです。
時間制限のもとではスケジュールが時計に従うので、同じシードの 2 回の実行は一致しません。これは時間で止まる実行すべてに共通です。

## 公開されている探索との違い { #what-differs-from-the-published-search }

摂動、degree、受理は公開されている探索に従っています。降下は従っていません。
公開されている探索は、解消する過積載 1 単位あたりのコストで move を順位付けする実行可能化の段階と、SHIFT、SWAP、2-opt、SWAP\*、2-opt\* の move を持つ局所探索を備えています。
ここでは 1 つの降下が両方の役目を担います。[Hybrid Genetic Search](hgs.md) も使う granular な経路降下を、問題のペナルティのもとで走らせます。SWAP\* はありません。

## 参考文献 { #references }

- Máximo, V. R., Cordeau, J.-F. and Nascimento, M. C. V. "AILS-II: An
  Adaptive Iterated Local Search Heuristic for the Large-Scale Capacitated
  Vehicle Routing Problem." *INFORMS Journal on Computing*, 36(4), 974-986,
  2024.
