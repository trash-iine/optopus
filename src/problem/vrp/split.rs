//! Optimal route splitting for the CVRP "giant tour" encoding (Prins' Split).
//!
//! A giant tour is a permutation of the customers with no route boundaries. Split
//! turns it into a route partition *without reordering the customers*: it only
//! chooses where to cut. Under that restriction the optimal cut positions are
//! found exactly by dynamic programming, which is what makes the giant-tour
//! encoding usable as a genetic representation — the decoder is optimal, so the
//! search only has to get the customer order right.

use super::problem::{Vrp, overload_of};

/// Splits a giant customer tour into exactly `prob.num_vehicles` routes,
/// minimizing `distance + penalty · overload` over all ways of cutting it.
///
/// The customer order within `giant` is preserved; only the cut positions are
/// chosen. Capacity is soft: a route may overflow, paying `penalty` per unit.
/// This matters because CVRPLIB fleets are sized for the *best* tour, so an
/// arbitrary tour order frequently admits no feasible split at all — returning
/// an overloaded partition is the useful answer there, not a failure.
///
/// `penalty` is the cost per unit of capacity overflow. Pass
/// [`Vrp::penalty_weight`] to reproduce the objective of [`super::VrpSolution`]
/// (which makes any feasible split beat every infeasible one), or an adaptive
/// value when the caller manages its own penalty, as Hybrid Genetic Search does.
///
/// The result always has length `prob.num_vehicles` (padded with empty routes)
/// and visits every customer of `giant` exactly once.
pub fn split_giant_tour(prob: &Vrp, giant: &[usize], penalty: f64) -> Vec<Vec<usize>> {
    let fleet = prob.num_vehicles;
    if giant.is_empty() {
        return vec![Vec::new(); fleet];
    }
    let mut routes = split_dp(prob, giant, penalty);
    debug_assert!(
        routes.len() <= fleet,
        "split produced more routes than vehicles"
    );
    routes.resize_with(fleet, Vec::new);
    routes
}

/// `cost[k][j]`: cheapest way to serve the first `j` customers of `giant` with at
/// most `k` routes, counting `penalty` per unit of overload.
///
/// Arcs `i → j` represent the route `giant[i..j]`. Capacity-respecting arcs are
/// always available, so every feasible split is reachable. Overloaded arcs are
/// capped at `max_overloaded_len`, which keeps the DP at O(fleet · n · route
/// length) instead of O(fleet · n²) while staying above the `n / fleet` needed to
/// guarantee the fleet can always cover the tour. Splits that pile most of the
/// tour onto one grossly overloaded vehicle are therefore out of reach — no loss,
/// since they are never worth keeping.
fn split_dp(prob: &Vrp, giant: &[usize], penalty: f64) -> Vec<Vec<usize>> {
    let n = giant.len();
    let fleet = prob.num_vehicles;
    let width = n + 1;
    let max_overloaded_len = 2 * n.div_ceil(fleet).max(1);

    let mut cost = vec![f64::INFINITY; (fleet + 1) * width];
    // `Some(i)`: the k-th route is `giant[i..j]`. `None`: this state is reached
    // with fewer than `k` routes (the "at most" relaxation).
    let mut pred: Vec<Option<usize>> = vec![None; (fleet + 1) * width];
    cost[0] = 0.0;

    for k in 1..=fleet {
        let row = k * width;
        let prev = (k - 1) * width;

        // Carry over states reachable with fewer routes.
        for j in 0..=n {
            if cost[prev + j] < cost[row + j] {
                cost[row + j] = cost[prev + j];
                pred[row + j] = None;
            }
        }

        for i in 0..n {
            let base = cost[prev + i];
            if !base.is_finite() {
                continue;
            }
            let mut load = 0i64;
            let mut route_distance = 0.0;
            for j in i..n {
                let c = giant[j];
                load += prob.demands[c];
                if load > prob.capacity && j - i + 1 > max_overloaded_len {
                    break;
                }
                route_distance = extend_route(prob, route_distance, giant, i, j);
                let candidate =
                    base + route_distance + penalty * overload_of(load, prob.capacity) as f64;
                if candidate < cost[row + j + 1] {
                    cost[row + j + 1] = candidate;
                    pred[row + j + 1] = Some(i);
                }
            }
        }
    }

    debug_assert!(
        cost[fleet * width + n].is_finite(),
        "max_overloaded_len must keep the whole tour reachable"
    );
    reconstruct(giant, &pred, width, fleet, n)
}

/// Distance of `depot → giant[i..=j] → depot`, updated in O(1) from the distance
/// of `depot → giant[i..=j-1] → depot`.
#[inline]
fn extend_route(prob: &Vrp, current: f64, giant: &[usize], i: usize, j: usize) -> f64 {
    let c = giant[j];
    if j == i {
        return prob.distance(0, c) + prob.distance(c, 0);
    }
    let prev = giant[j - 1];
    current - prob.distance(prev, 0) + prob.distance(prev, c) + prob.distance(c, 0)
}

/// Walks the predecessor table back from `cost[fleet][n]` to the empty prefix.
fn reconstruct(
    giant: &[usize],
    pred: &[Option<usize>],
    width: usize,
    fleet: usize,
    n: usize,
) -> Vec<Vec<usize>> {
    let mut routes = Vec::new();
    let mut k = fleet;
    let mut j = n;
    while j > 0 && k > 0 {
        match pred[k * width + j] {
            // State reached with fewer routes: drop down a level, same prefix.
            None => k -= 1,
            Some(i) => {
                routes.push(giant[i..j].to_vec());
                j = i;
                k -= 1;
            }
        }
    }
    debug_assert_eq!(j, 0, "reconstruction did not consume the whole tour");
    routes.reverse();
    routes
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    /// Depot at the origin plus `n` customers on a ring, unit demands.
    fn ring_vrp(n: usize, capacity: i64, fleet: usize) -> Vrp {
        let mut coordinates = vec![(0.0, 0.0)];
        let mut demands = vec![0i64];
        for i in 0..n {
            let theta = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            coordinates.push((10.0 * theta.cos(), 10.0 * theta.sin()));
            demands.push(1);
        }
        Vrp::new("ring", coordinates, demands, capacity, fleet)
    }

    fn random_vrp(rng: &mut SmallRng, n: usize, capacity: i64, fleet: usize) -> Vrp {
        let mut coordinates = vec![(0.0, 0.0)];
        let mut demands = vec![0i64];
        for _ in 0..n {
            coordinates.push((rng.random_range(-50.0..50.0), rng.random_range(-50.0..50.0)));
            demands.push(rng.random_range(1..=3));
        }
        Vrp::new("rnd", coordinates, demands, capacity, fleet)
    }

    fn random_tour(rng: &mut SmallRng, n: usize) -> Vec<usize> {
        let mut giant: Vec<usize> = (1..=n).collect();
        for i in (1..n).rev() {
            giant.swap(i, rng.random_range(0..=i));
        }
        giant
    }

    fn total_distance(prob: &Vrp, routes: &[Vec<usize>]) -> f64 {
        routes.iter().map(|r| prob.route_distance(r)).sum()
    }

    /// Enumerates every way of cutting `giant` into at most `fleet` contiguous
    /// capacity-respecting routes and returns the cheapest total distance.
    fn brute_force_optimum(prob: &Vrp, giant: &[usize], fleet: usize) -> Option<f64> {
        let n = giant.len();
        let mut best: Option<f64> = None;
        // Bit `g` set = cut after position `g`.
        for mask in 0u32..(1u32 << (n - 1)) {
            let mut routes: Vec<Vec<usize>> = Vec::new();
            let mut current: Vec<usize> = Vec::new();
            for (g, &c) in giant.iter().enumerate() {
                current.push(c);
                if g + 1 < n && mask & (1 << g) != 0 {
                    routes.push(std::mem::take(&mut current));
                }
            }
            routes.push(current);
            if routes.len() > fleet {
                continue;
            }
            let feasible = routes.iter().all(|r| {
                let load: i64 = r.iter().map(|&c| prob.demands[c]).sum();
                load <= prob.capacity
            });
            if !feasible {
                continue;
            }
            let d = total_distance(prob, &routes);
            if best.is_none_or(|b| d < b) {
                best = Some(d);
            }
        }
        best
    }

    /// With `penalty_weight` (larger than any possible tour), a feasible split is
    /// always preferred, so Split must reproduce the brute-force optimum whenever
    /// one exists — and must report overload when none does.
    #[test]
    fn split_matches_brute_force_optimum() {
        let mut rng = SmallRng::seed_from_u64(20260807);
        for seed in 0..50u64 {
            let n = 4 + (seed as usize % 6); // 4..=9
            let fleet = 2 + (seed as usize % 3); // 2..=4
            let prob = random_vrp(&mut rng, n, 4, fleet);
            let giant = random_tour(&mut rng, n);

            let routes = split_giant_tour(&prob, &giant, prob.penalty_weight());
            let sol = prob.solution_from_routes(routes);

            match brute_force_optimum(&prob, &giant, fleet) {
                Some(want) => {
                    assert_eq!(sol.overload, 0, "seed {seed}: a feasible split exists");
                    assert!(
                        (sol.distance - want).abs() < 1e-9,
                        "seed {seed}: split gave {}, brute force {want}",
                        sol.distance
                    );
                }
                None => assert!(
                    sol.overload > 0,
                    "seed {seed}: no feasible split exists, yet Split reported none"
                ),
            }
        }
    }

    #[test]
    fn split_is_a_valid_partition() {
        let mut rng = SmallRng::seed_from_u64(7);
        for _ in 0..30 {
            let n = 12;
            let prob = random_vrp(&mut rng, n, 5, 4);
            let giant = random_tour(&mut rng, n);
            let routes = split_giant_tour(&prob, &giant, prob.penalty_weight());
            assert_eq!(routes.len(), prob.num_vehicles);
            prob.validate_routes(&routes).unwrap();
        }
    }

    #[test]
    fn split_respects_capacity_when_possible() {
        // 8 unit-demand customers, capacity 2, 4 vehicles: exactly feasible.
        let prob = ring_vrp(8, 2, 4);
        let giant: Vec<usize> = (1..=8).collect();
        let sol = prob.solution_from_routes(split_giant_tour(&prob, &giant, prob.penalty_weight()));
        assert_eq!(sol.overload, 0);
        assert!(sol.route_loads.iter().all(|&l| l <= prob.capacity));
    }

    #[test]
    fn split_reports_overload_when_fleet_too_small() {
        // Three demand-3 customers, capacity 4, only 2 vehicles: infeasible.
        let prob = Vrp::new(
            "tight",
            vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
            vec![0, 3, 3, 3],
            4,
            2,
        );
        let routes = split_giant_tour(&prob, &[1, 2, 3], prob.penalty_weight());
        assert_eq!(routes.len(), 2);
        prob.validate_routes(&routes).unwrap();
        let sol = prob.solution_from_routes(routes);
        assert_eq!(sol.overload, 2, "9 demand over 2 × capacity 4");
    }

    /// A small penalty lets Split trade overload for distance; a large one does not.
    #[test]
    fn penalty_controls_the_overload_tradeoff() {
        // Two far-apart clusters of demand 3 each, capacity 4, 2 vehicles.
        let prob = Vrp::new(
            "tradeoff",
            vec![
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (100.0, 0.0),
                (100.0, 1.0),
            ],
            vec![0, 3, 3, 3, 3],
            4,
            2,
        );
        let giant = vec![1, 2, 3, 4];
        let cheap = prob.solution_from_routes(split_giant_tour(&prob, &giant, 0.01));
        let dear = prob.solution_from_routes(split_giant_tour(&prob, &giant, 1e6));
        assert!(
            cheap.overload >= dear.overload,
            "a cheap penalty must not buy less overload than an expensive one"
        );
        assert!(cheap.distance <= dear.distance + 1e-9);
    }

    #[test]
    fn split_pads_empty_routes() {
        // 3 customers, 6 vehicles: the fleet is larger than the tour needs.
        let prob = ring_vrp(3, 10, 6);
        let routes = split_giant_tour(&prob, &[1, 2, 3], prob.penalty_weight());
        assert_eq!(routes.len(), 6);
        assert!(routes.iter().any(|r| r.is_empty()));
        prob.validate_routes(&routes).unwrap();
    }

    #[test]
    fn empty_tour_yields_an_idle_fleet() {
        let prob = ring_vrp(3, 10, 4);
        let routes = split_giant_tour(&prob, &[], prob.penalty_weight());
        assert_eq!(routes.len(), 4);
        assert!(routes.iter().all(|r| r.is_empty()));
    }
}
