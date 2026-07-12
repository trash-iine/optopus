use std::collections::HashMap;

use rand::Rng;
use rand::rngs::SmallRng;

use super::problem::{Vrp, VrpSolution, overload_of};
use crate::error::OptError;
use crate::search_state::{EnabledTabu, Evaluable, Evaluate, MoveToNeighbor, Rankable};

/// Applies the cached `(gain, overload_delta)` of a move to a solution's numeric
/// caches. `distance` is derived from `gain` and `overload_delta` so the three
/// caches stay mutually consistent (`objective == distance + weight * overload`).
#[inline]
fn apply_deltas(prob: &Vrp, sol: &mut VrpSolution, gain: f64, overload_delta: i64) {
    sol.overload += overload_delta;
    sol.distance += gain - prob.penalty_weight() * overload_delta as f64;
    sol.objective += gain;
}

// ---------------------------------------------------------------------------
// Relocate (inter-route shift)
// ---------------------------------------------------------------------------

/// Moves one customer from `(from_r, from_i)` to position `to_i` in a *different*
/// route `to_r` (`to_r != from_r`). Inserting into an empty route (`to_i == 0`)
/// puts an idle vehicle to use. `gain` is the change in objective (distance plus
/// penalty), negative = improvement.
#[derive(Debug, Clone)]
pub struct VrpRelocateNeighbor {
    /// Source route index.
    pub from_r: usize,
    /// Position of the customer within the source route.
    pub from_i: usize,
    /// Destination route index (`!= from_r`).
    pub to_r: usize,
    /// Insertion position within the destination route (`0..=len`).
    pub to_i: usize,
    /// The relocated customer (cached for `apply` / tabu keying).
    pub customer: usize,
    /// Change in objective (negative = improvement).
    pub gain: f64,
    /// Change in total capacity overflow.
    pub overload_delta: i64,
}

/// Computes `(gain, overload_delta, customer)` for a relocate move.
fn relocate_gain(
    prob: &Vrp,
    sol: &VrpSolution,
    from_r: usize,
    from_i: usize,
    to_r: usize,
    to_i: usize,
) -> (f64, i64, usize) {
    let from = &sol.routes[from_r];
    let c = from[from_i];
    let prev = if from_i == 0 { 0 } else { from[from_i - 1] };
    let next = if from_i + 1 == from.len() {
        0
    } else {
        from[from_i + 1]
    };
    let removal = prob.distance(prev, next) - prob.distance(prev, c) - prob.distance(c, next);

    let to = &sol.routes[to_r];
    let a = if to_i == 0 { 0 } else { to[to_i - 1] };
    let b = if to_i == to.len() { 0 } else { to[to_i] };
    let insertion = prob.distance(a, c) + prob.distance(c, b) - prob.distance(a, b);

    let dc = prob.demands[c];
    let cap = prob.capacity;
    let lf = sol.route_loads[from_r];
    let lt = sol.route_loads[to_r];
    let od = (overload_of(lf - dc, cap) - overload_of(lf, cap))
        + (overload_of(lt + dc, cap) - overload_of(lt, cap));

    let gain = removal + insertion + prob.penalty_weight() * od as f64;
    (gain, od, c)
}

impl Rankable for VrpRelocateNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain < other.gain
    }
}

impl Evaluate for VrpRelocateNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain)
    }
}

impl EnabledTabu for VrpRelocateNeighbor {
    /// Keyed by `(customer, route)`: forbids returning a moved customer to the
    /// route it just left.
    type TabuMap = HashMap<(usize, usize), u64>;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        tabu_map
            .get(&(self.customer, self.to_r))
            .is_none_or(|&t| iteration > t)
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
        rng: &mut SmallRng,
    ) {
        let d = rng.random_range(tabu_tenure.0..=tabu_tenure.1);
        tabu_map.insert((self.customer, self.from_r), iteration + d);
    }
}

impl MoveToNeighbor<Vrp> for VrpRelocateNeighbor {
    fn apply_to_solution(&self, prob: &Vrp, sol: &mut VrpSolution) -> Result<(), OptError> {
        let dc = prob.demands[self.customer];
        sol.routes[self.from_r].remove(self.from_i);
        sol.route_loads[self.from_r] -= dc;
        sol.routes[self.to_r].insert(self.to_i, self.customer);
        sol.route_loads[self.to_r] += dc;
        apply_deltas(prob, sol, self.gain, self.overload_delta);
        Ok(())
    }

    fn iter(prob: &Vrp, sol: &VrpSolution) -> impl Iterator<Item = Self> + Send {
        let v = sol.routes.len();
        (0..v).flat_map(move |from_r| {
            (0..sol.routes[from_r].len()).flat_map(move |from_i| {
                (0..v)
                    .filter(move |&to_r| to_r != from_r)
                    .flat_map(move |to_r| {
                        (0..=sol.routes[to_r].len()).map(move |to_i| {
                            let (gain, od, c) =
                                relocate_gain(prob, sol, from_r, from_i, to_r, to_i);
                            VrpRelocateNeighbor {
                                from_r,
                                from_i,
                                to_r,
                                to_i,
                                customer: c,
                                gain,
                                overload_delta: od,
                            }
                        })
                    })
            })
        })
    }

    fn move_to_be_better_than(&self, _: &Vrp, src: &VrpSolution, other: &VrpSolution) -> bool {
        self.gain + src.objective < other.objective
    }

    fn random_neighbor(prob: &Vrp, sol: &VrpSolution, rng: &mut SmallRng) -> Option<Self> {
        let v = sol.routes.len();
        if v < 2 {
            return None;
        }
        for _ in 0..64 {
            let from_r = rng.random_range(0..v);
            let len_from = sol.routes[from_r].len();
            if len_from == 0 {
                continue;
            }
            let mut to_r = rng.random_range(0..v - 1);
            if to_r >= from_r {
                to_r += 1;
            }
            let from_i = rng.random_range(0..len_from);
            let to_i = rng.random_range(0..=sol.routes[to_r].len());
            let (gain, od, c) = relocate_gain(prob, sol, from_r, from_i, to_r, to_i);
            return Some(Self {
                from_r,
                from_i,
                to_r,
                to_i,
                customer: c,
                gain,
                overload_delta: od,
            });
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Swap (inter-route customer exchange)
// ---------------------------------------------------------------------------

/// Exchanges the customer at `(r1, i1)` with the one at `(r2, i2)` in a
/// *different* route (`r1 != r2`). `gain` is the change in objective.
#[derive(Debug, Clone)]
pub struct VrpSwapNeighbor {
    pub r1: usize,
    pub i1: usize,
    pub r2: usize,
    pub i2: usize,
    /// Customer originally at `(r1, i1)`.
    pub c1: usize,
    /// Customer originally at `(r2, i2)`.
    pub c2: usize,
    pub gain: f64,
    pub overload_delta: i64,
}

/// Computes `(gain, overload_delta, c1, c2)` for a swap move (`r1 != r2`).
fn swap_gain(
    prob: &Vrp,
    sol: &VrpSolution,
    r1: usize,
    i1: usize,
    r2: usize,
    i2: usize,
) -> (f64, i64, usize, usize) {
    let route1 = &sol.routes[r1];
    let route2 = &sol.routes[r2];
    let c1 = route1[i1];
    let c2 = route2[i2];

    let prev1 = if i1 == 0 { 0 } else { route1[i1 - 1] };
    let next1 = if i1 + 1 == route1.len() {
        0
    } else {
        route1[i1 + 1]
    };
    let prev2 = if i2 == 0 { 0 } else { route2[i2 - 1] };
    let next2 = if i2 + 1 == route2.len() {
        0
    } else {
        route2[i2 + 1]
    };

    let old = prob.distance(prev1, c1)
        + prob.distance(c1, next1)
        + prob.distance(prev2, c2)
        + prob.distance(c2, next2);
    let new = prob.distance(prev1, c2)
        + prob.distance(c2, next1)
        + prob.distance(prev2, c1)
        + prob.distance(c1, next2);
    let dist_delta = new - old;

    let d1 = prob.demands[c1];
    let d2 = prob.demands[c2];
    let cap = prob.capacity;
    let l1 = sol.route_loads[r1];
    let l2 = sol.route_loads[r2];
    let od = (overload_of(l1 - d1 + d2, cap) - overload_of(l1, cap))
        + (overload_of(l2 - d2 + d1, cap) - overload_of(l2, cap));

    let gain = dist_delta + prob.penalty_weight() * od as f64;
    (gain, od, c1, c2)
}

impl Rankable for VrpSwapNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain < other.gain
    }
}

impl Evaluate for VrpSwapNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain)
    }
}

impl EnabledTabu for VrpSwapNeighbor {
    /// Keyed by the unordered customer pair, forbidding an immediate re-swap.
    type TabuMap = HashMap<(usize, usize), u64>;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        let key = (self.c1.min(self.c2), self.c1.max(self.c2));
        tabu_map.get(&key).is_none_or(|&t| iteration > t)
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
        rng: &mut SmallRng,
    ) {
        let d = rng.random_range(tabu_tenure.0..=tabu_tenure.1);
        let key = (self.c1.min(self.c2), self.c1.max(self.c2));
        tabu_map.insert(key, iteration + d);
    }
}

impl MoveToNeighbor<Vrp> for VrpSwapNeighbor {
    fn apply_to_solution(&self, prob: &Vrp, sol: &mut VrpSolution) -> Result<(), OptError> {
        let d1 = prob.demands[self.c1];
        let d2 = prob.demands[self.c2];
        sol.routes[self.r1][self.i1] = self.c2;
        sol.routes[self.r2][self.i2] = self.c1;
        sol.route_loads[self.r1] += d2 - d1;
        sol.route_loads[self.r2] += d1 - d2;
        apply_deltas(prob, sol, self.gain, self.overload_delta);
        Ok(())
    }

    fn iter(prob: &Vrp, sol: &VrpSolution) -> impl Iterator<Item = Self> + Send {
        let v = sol.routes.len();
        (0..v).flat_map(move |r1| {
            ((r1 + 1)..v).flat_map(move |r2| {
                (0..sol.routes[r1].len()).flat_map(move |i1| {
                    (0..sol.routes[r2].len()).map(move |i2| {
                        let (gain, od, c1, c2) = swap_gain(prob, sol, r1, i1, r2, i2);
                        VrpSwapNeighbor {
                            r1,
                            i1,
                            r2,
                            i2,
                            c1,
                            c2,
                            gain,
                            overload_delta: od,
                        }
                    })
                })
            })
        })
    }

    fn move_to_be_better_than(&self, _: &Vrp, src: &VrpSolution, other: &VrpSolution) -> bool {
        self.gain + src.objective < other.objective
    }

    fn random_neighbor(prob: &Vrp, sol: &VrpSolution, rng: &mut SmallRng) -> Option<Self> {
        let v = sol.routes.len();
        if v < 2 {
            return None;
        }
        for _ in 0..64 {
            let r1 = rng.random_range(0..v);
            let mut r2 = rng.random_range(0..v - 1);
            if r2 >= r1 {
                r2 += 1;
            }
            let (len1, len2) = (sol.routes[r1].len(), sol.routes[r2].len());
            if len1 == 0 || len2 == 0 {
                continue;
            }
            let i1 = rng.random_range(0..len1);
            let i2 = rng.random_range(0..len2);
            let (gain, od, c1, c2) = swap_gain(prob, sol, r1, i1, r2, i2);
            return Some(Self {
                r1,
                i1,
                r2,
                i2,
                c1,
                c2,
                gain,
                overload_delta: od,
            });
        }
        None
    }
}

// ---------------------------------------------------------------------------
// 2-opt (intra-route segment reversal)
// ---------------------------------------------------------------------------

/// Reverses the segment `route[p..=q]` of route `r` (`p < q`). Load is unchanged,
/// so `gain` is a pure distance delta (negative = improvement).
#[derive(Debug, Clone)]
pub struct VrpTwoOptNeighbor {
    pub r: usize,
    pub p: usize,
    pub q: usize,
    pub gain: f64,
}

/// Computes the distance delta of reversing `route[p..=q]` in route `r`.
fn two_opt_gain(prob: &Vrp, sol: &VrpSolution, r: usize, p: usize, q: usize) -> f64 {
    let route = &sol.routes[r];
    let before = if p == 0 { 0 } else { route[p - 1] };
    let after = if q + 1 == route.len() {
        0
    } else {
        route[q + 1]
    };
    prob.distance(before, route[q]) + prob.distance(route[p], after)
        - prob.distance(before, route[p])
        - prob.distance(route[q], after)
}

impl Rankable for VrpTwoOptNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain < other.gain
    }
}

impl Evaluate for VrpTwoOptNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain)
    }
}

impl EnabledTabu for VrpTwoOptNeighbor {
    type TabuMap = HashMap<(usize, usize, usize), u64>;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        tabu_map
            .get(&(self.r, self.p, self.q))
            .is_none_or(|&t| iteration > t)
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
        rng: &mut SmallRng,
    ) {
        let d = rng.random_range(tabu_tenure.0..=tabu_tenure.1);
        tabu_map.insert((self.r, self.p, self.q), iteration + d);
    }
}

impl MoveToNeighbor<Vrp> for VrpTwoOptNeighbor {
    fn apply_to_solution(&self, _prob: &Vrp, sol: &mut VrpSolution) -> Result<(), OptError> {
        sol.routes[self.r][self.p..=self.q].reverse();
        sol.distance += self.gain;
        sol.objective += self.gain;
        Ok(())
    }

    fn iter(prob: &Vrp, sol: &VrpSolution) -> impl Iterator<Item = Self> + Send {
        let v = sol.routes.len();
        (0..v).flat_map(move |r| {
            let len = sol.routes[r].len();
            (0..len).flat_map(move |p| {
                ((p + 1)..len).map(move |q| VrpTwoOptNeighbor {
                    r,
                    p,
                    q,
                    gain: two_opt_gain(prob, sol, r, p, q),
                })
            })
        })
    }

    fn move_to_be_better_than(&self, _: &Vrp, src: &VrpSolution, other: &VrpSolution) -> bool {
        self.gain + src.objective < other.objective
    }

    fn random_neighbor(prob: &Vrp, sol: &VrpSolution, rng: &mut SmallRng) -> Option<Self> {
        let v = sol.routes.len();
        // Collect route indices with at least two customers.
        let candidates: Vec<usize> = (0..v).filter(|&r| sol.routes[r].len() >= 2).collect();
        if candidates.is_empty() {
            return None;
        }
        let r = candidates[rng.random_range(0..candidates.len())];
        let len = sol.routes[r].len();
        let a = rng.random_range(0..len);
        let b = rng.random_range(0..len);
        if a == b {
            return None;
        }
        let (p, q) = (a.min(b), a.max(b));
        Some(Self {
            r,
            p,
            q,
            gain: two_opt_gain(prob, sol, r, p, q),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// 6 customers around the depot, capacity 3, 3 vehicles.
    fn vrp() -> Vrp {
        Vrp::new(
            "t",
            vec![
                (0.0, 0.0),
                (1.0, 1.0),
                (2.0, 2.0),
                (-1.0, 1.0),
                (-2.0, 2.0),
                (1.0, -1.0),
                (2.0, -2.0),
            ],
            vec![0, 1, 1, 1, 1, 1, 1],
            3,
            3,
        )
    }

    fn assert_caches_consistent(prob: &Vrp, sol: &VrpSolution) {
        let recomputed = prob.solution_from_routes(sol.routes.clone());
        assert!(
            (sol.distance - recomputed.distance).abs() < 1e-6,
            "distance drift: {} vs {}",
            sol.distance,
            recomputed.distance
        );
        assert_eq!(sol.overload, recomputed.overload, "overload drift");
        assert!(
            (sol.objective - recomputed.objective).abs() < 1e-6,
            "objective drift: {} vs {}",
            sol.objective,
            recomputed.objective
        );
        assert_eq!(sol.route_loads, recomputed.route_loads, "load drift");
        prob.validate_routes(&sol.routes).unwrap();
    }

    #[test]
    fn relocate_apply_matches_recompute() {
        let prob = vrp();
        let sol = prob.solution_from_routes(vec![vec![1, 2], vec![3, 4], vec![5, 6]]);
        for nb in VrpRelocateNeighbor::iter(&prob, &sol) {
            let mut s = sol.clone();
            nb.apply_to_solution(&prob, &mut s).unwrap();
            assert_caches_consistent(&prob, &s);
            assert!(
                (s.objective - (sol.objective + nb.gain)).abs() < 1e-6,
                "gain mismatch"
            );
        }
    }

    #[test]
    fn swap_apply_matches_recompute() {
        let prob = vrp();
        let sol = prob.solution_from_routes(vec![vec![1, 2], vec![3, 4], vec![5, 6]]);
        for nb in VrpSwapNeighbor::iter(&prob, &sol) {
            let mut s = sol.clone();
            nb.apply_to_solution(&prob, &mut s).unwrap();
            assert_caches_consistent(&prob, &s);
            assert!((s.objective - (sol.objective + nb.gain)).abs() < 1e-6);
        }
    }

    #[test]
    fn two_opt_apply_matches_recompute() {
        let prob = vrp();
        let sol = prob.solution_from_routes(vec![vec![1, 2, 5], vec![3, 4, 6], vec![]]);
        for nb in VrpTwoOptNeighbor::iter(&prob, &sol) {
            let mut s = sol.clone();
            nb.apply_to_solution(&prob, &mut s).unwrap();
            assert_caches_consistent(&prob, &s);
            assert!((s.objective - (sol.objective + nb.gain)).abs() < 1e-6);
        }
    }

    #[test]
    fn random_neighbors_are_members_of_iter() {
        let prob = vrp();
        let sol = prob.solution_from_routes(vec![vec![1, 2], vec![3, 4], vec![5, 6]]);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(3);

        let relocs: Vec<_> = VrpRelocateNeighbor::iter(&prob, &sol).collect();
        for _ in 0..30 {
            if let Some(m) = VrpRelocateNeighbor::random_neighbor(&prob, &sol, &mut rng) {
                assert!(relocs.iter().any(|r| r.from_r == m.from_r
                    && r.from_i == m.from_i
                    && r.to_r == m.to_r
                    && r.to_i == m.to_i));
            }
        }

        let swaps: Vec<_> = VrpSwapNeighbor::iter(&prob, &sol).collect();
        for _ in 0..30 {
            if let Some(m) = VrpSwapNeighbor::random_neighbor(&prob, &sol, &mut rng) {
                assert!(swaps.iter().any(|s| {
                    (s.r1 == m.r1 && s.i1 == m.i1 && s.r2 == m.r2 && s.i2 == m.i2)
                        || (s.r1 == m.r2 && s.i1 == m.i2 && s.r2 == m.r1 && s.i2 == m.i1)
                }));
            }
        }
    }

    #[test]
    fn relocate_into_empty_route_uses_idle_vehicle() {
        let prob = vrp();
        let sol = prob.solution_from_routes(vec![vec![1, 2, 3], vec![4, 5, 6], vec![]]);
        // Some relocate move must target the empty route (index 2, position 0).
        let has_empty_target =
            VrpRelocateNeighbor::iter(&prob, &sol).any(|m| m.to_r == 2 && m.to_i == 0);
        assert!(has_empty_target);
    }
}
