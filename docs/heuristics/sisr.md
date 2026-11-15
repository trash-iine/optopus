# SlackInductionByStringRemovalForVrp

**API:** [`SlackInductionByStringRemovalForVrp`](../api/optopus/heuristic/struct.SlackInductionByStringRemovalForVrp.html)

Problem-specific heuristic for the [VRP](../problems/vrp.md). Slack Induction
by String Removals is a ruin-and-recreate search with one ruin and one
recreate. It takes strings of consecutive customers out of a few routes that
pass near each other, puts them back greedily while skipping a place now and
then, and keeps the result by a simulated annealing test. It runs no local
search and chooses between no operators, so an iteration is cheap and a run
makes many of them.

## Example

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

Takes no `neighbor` type parameter, since it owns its move set.

## Algorithm sketch

Each `run_once` produces one candidate from the incumbent.

### Ruin

1. Draw the number of strings. With `L` the smaller of `max_string_length`
   and the average number of customers on a route in use, the number is drawn
   from `1` to `4 · average_removal / (1 + L) − 1`.
2. Draw a seed customer and walk the customers from the nearest to it outward.
3. Each customer found on a route not yet ruined marks that route. A length
   is drawn from `1` to the smaller of `L` and the route's length, and a
   string of that length containing the customer is taken out.
4. Stop when the drawn number of routes are ruined.

Half of the strings are taken whole. The other half are split. A longer
string is chosen, and a run of customers inside it stays on the route. The run
starts at one customer and grows by one with probability
`1 − split_stop_rate` until the route has no more customers to give, so with
the default rate it is usually as long as it can be and what leaves the route
is its two ends.

A string per route, from routes near each other, frees room in several
vehicles around one place at once. That is the slack the name refers to.

### Recreate

The removed customers are put in an order drawn each iteration.

| Order | Weight |
|---|---|
| random | 4 |
| largest demand first | 4 |
| farthest from the depot first | 2 |
| closest to the depot first | 1 |

Each customer is then priced at every place of every route and inserted at
the cheapest, except that each place is skipped with probability
`blink_rate`. When the cheapest place is skipped the customer lands on the
second cheapest, and so on, which is the only source of variety in the
recreate.

A place is priced with the penalty the problem charges a violated capacity or
route time at, so a route with room always wins over one without, and an idle
vehicle counts as one more place. When every vehicle is in use and none has
room, the customer goes where the violation costs least and the search
continues on the penalized objective.

### Acceptance

A candidate of cost `c'` replaces the incumbent of cost `c` when

```text
c' < c − T · ln(u),   u uniform on (0, 1]
```

The temperature `T` starts at `initial_temperature` and is multiplied by a
constant each iteration, so that it reaches `final_temperature` after
`cooling_iterations` iterations. A run that goes on past that stays at the
final temperature.

Both temperatures are in the units of the objective. The defaults of `100`
and `1` suit distances of the size CVRPLIB instances have, where a route is
some thousands long. An instance with coordinates in a unit square needs
temperatures a few orders of magnitude smaller.

## Constructor

```rust
SlackInductionByStringRemovalForVrp::new(
    stop_condition: StopCondition,
    cooling_iterations: u64,
) -> Self
```

`stop_condition` decides when the run ends, see [Stop
conditions](../guide/stop_conditions.md). `cooling_iterations` is how many
iterations the temperature takes to fall to its final value, and is a number
of its own because a run stopped by time has no iteration count to read it
from. A run stopped by iterations usually passes the same number twice.

Panics if `cooling_iterations` is zero.

Everything else is a builder with a default, each an associated constant of
the type.

| Builder | Sets | Default |
|---|---|---|
| `with_temperatures(initial, last)` | the temperatures the search starts and ends at | `100.0` / `1.0` |
| `with_removal(average_removal, max_string_length)` | the average number of customers an iteration removes and the longest string taken from one route | `10.0` / `10` |
| `with_split_stop_rate(rate)` | the probability that the run a split string keeps stops growing | `0.01` |
| `with_blink_rate(rate)` | the probability that the recreate skips a place | `0.01` |

`clear()` resets the temperature to its initial value and keeps what the
builders set.

## Benchmark config

```toml
[[heuristics]]
kind = "SlackInductionByStringRemoval"
cooling_iterations = 2000000   # optional when max_iteration is set
initial_temperature = 100.0    # optional (default shown)
final_temperature = 1.0        # optional (default shown)
average_removal = 10.0         # optional (default shown)
max_string_length = 10         # optional (default shown)
split_stop_rate = 0.01         # optional (default shown)
blink_rate = 0.01              # optional (default shown)
[heuristics.stop_condition]
max_duration_secs = 60.0
```

`cooling_iterations` defaults to the stop condition's `max_iteration`. A
config that sets neither is rejected, since the cooling would have no length.
Under a time limit, pick a count the run reaches. A count it never reaches
leaves the search at a high temperature to the end. The report's
`n_accepted + n_rejected` of a first run is how many iterations the limit
allows.

## References

- Christiaens, J. and Vanden Berghe, G. "Slack Induction by String Removals
  for Vehicle Routing Problems." *Transportation Science*, 54(2), 417-433,
  2020.
