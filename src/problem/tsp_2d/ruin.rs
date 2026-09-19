//! TSP's answer to [`Ruinable`]: cities are the elements and the one tour is
//! the only container.
//!
//! This is the single-container case the trait describes. Ruin-and-recreate
//! still works, since removing a set of cities and putting each back where it
//! is cheapest is a large-neighbourhood move on a tour as much as on a route.
//! What the single container takes away is regret. With no second container
//! the second-best placement is undefined, so regret-2 reduces to inserting in
//! pool order, and the operators that carry the search are the three destroys
//! and greedy insertion, followed by the anchored descent at the bottom.

use rand::rngs::SmallRng;

use super::problem::{TspSolution, TspWithCoordinates};
use crate::common::anchored_sweep::{AnchoredSweep, MIN_IMPROVEMENT};
use crate::trait_defs::{LocalRepair, Ruinable};

/// Marks a city that is not currently on the tour in [`TspPartial::pos`].
const NOT_PLACED: usize = usize::MAX;

/// Nearest partners considered per city by the post-repair descent.
///
/// Twice Lin-Kernighan's default of 5, since Or-opt and 2-opt have no depth
/// to make up for a short list, and half VRP's 20, since a tour has no second
/// route for a partner to be in. Unmeasured beyond that.
const GRANULARITY: usize = 10;

/// Descent passes over a recreated tour. Matches VRP's.
const MAX_LS_PASSES: usize = 4;

/// A tour mid-ruin, with the cities taken out of it.
///
/// The tour is kept as the sequence it will be handed back as, plus the
/// position index that makes [`Ruinable::removal_gain`] an O(1) question and
/// the running length that [`Ruinable::partial_energy`] reads. Every edit
/// through the trait and through [`AnchoredTourDescent`] keeps the three in
/// step.
pub struct TspPartial {
    /// The cities currently placed, in tour order.
    tour: Vec<usize>,
    /// City to index in `tour`, or [`NOT_PLACED`].
    pos: Vec<usize>,
    /// The cyclic length of `tour`.
    length: f64,
}

impl TspPartial {
    /// Rewrites the index for `tour[from..]`, which is every entry an edit at
    /// `from` can have shifted.
    fn reindex_from(&mut self, from: usize) {
        for i in from..self.tour.len() {
            self.pos[self.tour[i]] = i;
        }
    }

    /// The index before `i`, cyclically. Written out rather than as `% n`,
    /// since a variable modulus is an integer division and this sits inside
    /// the insertion scan.
    fn before(&self, i: usize) -> usize {
        if i == 0 { self.tour.len() - 1 } else { i - 1 }
    }

    /// The index after `i`, cyclically.
    fn after(&self, i: usize) -> usize {
        if i + 1 == self.tour.len() { 0 } else { i + 1 }
    }

    fn pred(&self, c: usize) -> usize {
        self.tour[self.before(self.pos[c])]
    }

    fn succ(&self, c: usize) -> usize {
        self.tour[self.after(self.pos[c])]
    }

    /// The cyclic length of `tour`, measured from scratch.
    fn measure(&self, prob: &TspWithCoordinates) -> f64 {
        (0..self.tour.len())
            .map(|i| prob.distance(self.tour[i], self.tour[self.after(i)]))
            .sum()
    }

    /// What inserting `c` at `place` costs. Place `p` is between `tour[p-1]`
    /// and `tour[p]`, cyclically, so an empty tour has the one place `0`
    /// whose cost is the loop `c` to itself, which is what
    /// [`TspWithCoordinates::calculate_tour_length`] charges a one-city tour.
    fn insertion_cost_at(&self, prob: &TspWithCoordinates, place: usize, c: usize) -> f64 {
        if self.tour.is_empty() {
            return prob.distance(c, c);
        }
        let prev = self.tour[self.before(place)];
        let next = self.tour[place];
        prob.distance(prev, c) + prob.distance(c, next) - prob.distance(prev, next)
    }

    /// Reverses `tour[i+1..=j]`, the 2-opt move that removes the edges after
    /// `i` and after `j` and reconnects `tour[i]` to `tour[j]`.
    fn reverse_after(&mut self, i: usize, j: usize) {
        self.tour[i + 1..=j].reverse();
        self.reindex_range(i + 1, j);
    }

    /// Rewrites the index for `tour[lo..=hi]`.
    fn reindex_range(&mut self, lo: usize, hi: usize) {
        for k in lo..=hi {
            self.pos[self.tour[k]] = k;
        }
    }

    #[cfg(debug_assertions)]
    fn assert_caches_consistent(&self, prob: &TspWithCoordinates) {
        for (i, &c) in self.tour.iter().enumerate() {
            debug_assert_eq!(self.pos[c], i, "position index is stale");
        }
        let length = self.measure(prob);
        debug_assert!(
            (length - self.length).abs() < 1e-6 * length.abs().max(1.0),
            "running length {} drifted from the tour's {length}",
            self.length
        );
    }
}

impl Ruinable for TspWithCoordinates {
    type Element = usize;
    type Partial = TspPartial;

    fn to_partial(&self, sol: &TspSolution) -> TspPartial {
        let mut partial = TspPartial {
            tour: sol.tour.clone(),
            pos: vec![NOT_PLACED; self.get_n()],
            length: sol.objective,
        };
        partial.reindex_from(0);
        partial
    }

    /// Recomputes the length rather than handing back the running total, so a
    /// run's float drift is cut at every accepted candidate.
    fn finish(&self, partial: &TspPartial) -> TspSolution {
        let tour = partial.tour.clone();
        let objective = self
            .calculate_tour_length(&tour)
            .expect("a ruined-and-recreated tour visits every city once");
        TspSolution { tour, objective }
    }

    fn elements(&self, partial: &TspPartial, out: &mut Vec<usize>) {
        out.clear();
        out.extend_from_slice(&partial.tour);
    }

    fn num_elements(&self, partial: &TspPartial) -> usize {
        partial.tour.len()
    }

    /// One compaction and one re-measurement, both O(n), rather than a
    /// shift and a gain per city. The gains of adjacent removals do not add
    /// up, so the running total is remeasured instead of adjusted.
    fn remove_all(&self, partial: &mut TspPartial, set: &[usize]) {
        for &c in set {
            partial.pos[c] = NOT_PLACED;
        }
        let pos = &partial.pos;
        partial.tour.retain(|&c| pos[c] != NOT_PLACED);
        partial.reindex_from(0);
        partial.length = partial.measure(self);
    }

    fn removal_gain(&self, partial: &TspPartial, element: usize) -> f64 {
        let (a, b) = (partial.pred(element), partial.succ(element));
        self.distance(a, element) + self.distance(element, b) - self.distance(a, b)
    }

    /// Shaw's relatedness with nothing but geometry to read, so it is the
    /// distance.
    fn relatedness(&self, a: usize, b: usize) -> f64 {
        self.distance(a, b)
    }

    fn num_buckets(&self, _partial: &TspPartial) -> usize {
        1
    }

    /// A tour is cyclic, so inserting after the last city is inserting before
    /// the first, and there are `len` places rather than `len + 1`.
    fn num_places(&self, partial: &TspPartial, _bucket: usize) -> usize {
        partial.tour.len().max(1)
    }

    fn insertion_cost(
        &self,
        partial: &TspPartial,
        _bucket: usize,
        place: usize,
        element: usize,
    ) -> f64 {
        partial.insertion_cost_at(self, place, element)
    }

    fn insert(&self, partial: &mut TspPartial, _bucket: usize, place: usize, element: usize) {
        partial.length += partial.insertion_cost_at(self, place, element);
        partial.tour.insert(place, element);
        partial.reindex_from(place);
    }

    fn partial_energy(&self, partial: &TspPartial) -> f64 {
        partial.length
    }
}

/// Or-opt and 2-opt, anchored at the cities a ruin just re-inserted.
///
/// Holds the nearest-neighbour lists because they are instance-derived and
/// worth keeping across iterations, restarts and clears. The moves are the
/// two a tour has that price in O(1) from their endpoints, relocating one city
/// next to a near one and reversing the path between two near ones, tried
/// first-improvement over the granular pairs, the same shape as VRP's
/// [`AnchoredRouteDescent`](crate::problem::vrp::AnchoredRouteDescent).
#[derive(Debug, Default)]
pub struct AnchoredTourDescent {
    neighbors: Vec<Vec<usize>>,
    sweep: AnchoredSweep,
}

impl AnchoredTourDescent {
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure(&mut self, prob: &TspWithCoordinates) {
        if self.neighbors.len() != prob.get_n() {
            self.neighbors = prob.nearest_neighbors(GRANULARITY);
            self.sweep.ensure(prob.get_n());
        }
    }
}

/// Tries every granular move anchored at `u`, applying the first improving
/// one.
fn improve_around(
    partial: &mut TspPartial,
    prob: &TspWithCoordinates,
    neighbors: &[usize],
    u: usize,
) -> bool {
    for &v in neighbors {
        if partial.pos[v] == NOT_PLACED {
            continue;
        }
        // Read before any move is tried. A move that applies returns early,
        // so the pair is only used against the tour it was read from. Moving
        // `u` after `pred(v)` is moving it before `v`, and the 2-opt on the
        // predecessors is the one that makes the edge `(u, v)` from the other
        // side.
        let (pu, pv) = (partial.pred(u), partial.pred(v));
        if try_relocate(partial, prob, u, v)
            || try_relocate(partial, prob, u, pv)
            || try_two_opt(partial, prob, u, v)
            || try_two_opt(partial, prob, pu, pv)
        {
            return true;
        }
    }
    false
}

impl LocalRepair<TspWithCoordinates> for AnchoredTourDescent {
    fn repair_around(
        &mut self,
        prob: &TspWithCoordinates,
        partial: &mut TspPartial,
        anchors: &[usize],
        rng: &mut SmallRng,
    ) {
        // Below four cities every tour is the same cycle up to rotation and
        // reflection, and both moves below assume four distinct endpoints.
        if partial.tour.len() < 4 || anchors.is_empty() {
            return;
        }
        self.ensure(prob);
        self.sweep
            .collect_around(anchors, &self.neighbors, |v| partial.pos[v] != NOT_PLACED);
        let neighbors = &self.neighbors;
        self.sweep.sweep(rng, MAX_LS_PASSES, |u| {
            improve_around(partial, prob, &neighbors[u], u)
        });
        #[cfg(debug_assertions)]
        partial.assert_caches_consistent(prob);
    }
}

/// Moves `u` to sit directly after `v`, if that shortens the tour.
fn try_relocate(partial: &mut TspPartial, prob: &TspWithCoordinates, u: usize, v: usize) -> bool {
    let (pu, su) = (partial.pred(u), partial.succ(u));
    let sv = partial.succ(v);
    if v == u || sv == u {
        return false;
    }
    let gain = prob.distance(pu, su) - prob.distance(pu, u) - prob.distance(u, su)
        + prob.distance(v, u)
        + prob.distance(u, sv)
        - prob.distance(v, sv);
    if gain >= -MIN_IMPROVEMENT {
        return false;
    }
    // Rotating the span between the two slots by one is the move, and only
    // that span's index changes. `v` is a near neighbour of `u`, so on a
    // settled tour the span is short.
    let (i, j) = (partial.pos[u], partial.pos[v]);
    if i < j {
        partial.tour[i..=j].rotate_left(1);
        partial.reindex_range(i, j);
    } else {
        partial.tour[j + 1..=i].rotate_right(1);
        partial.reindex_range(j + 1, i);
    }
    partial.length += gain;
    true
}

/// Removes the edges after `x` and after `y` and reconnects `x` to `y`, if
/// that shortens the tour.
fn try_two_opt(partial: &mut TspPartial, prob: &TspWithCoordinates, x: usize, y: usize) -> bool {
    if x == y {
        return false;
    }
    let (sx, sy) = (partial.succ(x), partial.succ(y));
    // Adjacent endpoints make the move a no-op, and the gain reads zero for
    // them anyway.
    let gain = prob.calc_2opt_gain_cities((x, sx), (y, sy));
    if gain >= -MIN_IMPROVEMENT {
        return false;
    }
    let (i, j) = (partial.pos[x], partial.pos[y]);
    let (i, j) = if i < j { (i, j) } else { (j, i) };
    partial.reverse_after(i, j);
    partial.length += gain;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::tsp_2d::EdgeWeightType;
    use crate::search_state::ProblemTrait;
    use rand::SeedableRng;

    fn ring(n: usize) -> TspWithCoordinates {
        let coords = (0..n)
            .map(|i| {
                let theta = std::f64::consts::TAU * i as f64 / n as f64;
                (theta.cos() * 10.0, theta.sin() * 10.0)
            })
            .collect();
        TspWithCoordinates::new("ring".into(), coords)
    }

    fn assert_length_matches(prob: &TspWithCoordinates, partial: &TspPartial) {
        let expected = partial.measure(prob);
        assert!(
            (partial.length - expected).abs() < 1e-9,
            "running length {} but the tour measures {expected}",
            partial.length
        );
    }

    #[test]
    fn to_partial_and_finish_round_trip() {
        let tsp = ring(12);
        let mut rng = SmallRng::seed_from_u64(3);
        let sol = tsp.new_solution(&mut rng);
        let partial = tsp.to_partial(&sol);
        let back = tsp.finish(&partial);
        assert_eq!(back.tour, sol.tour);
        assert_eq!(back.objective, sol.objective);
        assert_eq!(tsp.num_elements(&partial), 12);
        assert_eq!(tsp.num_buckets(&partial), 1);
        assert_eq!(tsp.num_places(&partial, 0), 12);
    }

    /// The running length has to agree with a re-measurement through every
    /// edit, including adjacent removals and the one- and two-city tours a
    /// deep ruin leaves behind. GEO charges a city a distance to itself, so
    /// it is the type that would expose a formula that skipped that loop.
    #[test]
    fn edits_keep_the_running_length_exact() {
        for ewt in [EdgeWeightType::Continuous, EdgeWeightType::Geo] {
            let coords: Vec<(f64, f64)> =
                (0..6).map(|i| (i as f64 * 1.5, (i % 2) as f64)).collect();
            let tsp = TspWithCoordinates::with_edge_weight_type("t".into(), coords, ewt);
            let mut rng = SmallRng::seed_from_u64(9);
            let sol = tsp.new_solution(&mut rng);
            let mut partial = tsp.to_partial(&sol);

            // Two adjacent cities and one more, all at once.
            let removed = vec![partial.tour[2], partial.tour[3], partial.tour[0]];
            tsp.remove_all(&mut partial, &removed);
            assert_eq!(tsp.num_elements(&partial), 3);
            assert_length_matches(&tsp, &partial);

            // Down to one city, then to none.
            let rest: Vec<usize> = partial.tour[1..].to_vec();
            tsp.remove_all(&mut partial, &rest);
            assert_eq!(tsp.num_elements(&partial), 1);
            assert_length_matches(&tsp, &partial);
            let last = vec![partial.tour[0]];
            tsp.remove_all(&mut partial, &last);
            assert_eq!(tsp.num_elements(&partial), 0);
            assert_eq!(tsp.num_places(&partial, 0), 1);

            // Back up, cheapest place each time.
            for c in 0..6 {
                let best = (0..tsp.num_places(&partial, 0))
                    .min_by(|&p, &q| {
                        tsp.insertion_cost(&partial, 0, p, c)
                            .total_cmp(&tsp.insertion_cost(&partial, 0, q, c))
                    })
                    .unwrap();
                tsp.insert(&mut partial, 0, best, c);
                assert_length_matches(&tsp, &partial);
            }
            let sol = tsp.finish(&partial);
            assert!((sol.objective - tsp.partial_energy(&partial)).abs() < 1e-9);
        }
    }

    #[test]
    fn removal_gain_is_the_finish_delta() {
        let tsp = ring(9);
        let mut rng = SmallRng::seed_from_u64(1);
        let sol = tsp.new_solution(&mut rng);
        let mut partial = tsp.to_partial(&sol);
        let c = sol.tour[4];
        let gain = tsp.removal_gain(&partial, c);
        tsp.remove_all(&mut partial, &[c]);
        assert_length_matches(&tsp, &partial);
        assert!((sol.objective - partial.length - gain).abs() < 1e-9);
    }

    /// The descent may only shorten the tour, and has to leave it a
    /// permutation with the caches in step.
    #[test]
    fn the_descent_never_worsens_and_keeps_a_valid_tour() {
        let tsp = ring(40);
        let mut rng = SmallRng::seed_from_u64(5);
        let sol = tsp.new_solution(&mut rng);
        let mut partial = tsp.to_partial(&sol);
        let anchors: Vec<usize> = sol.tour[..8].to_vec();
        let mut descent = AnchoredTourDescent::new();
        descent.repair_around(&tsp, &mut partial, &anchors, &mut rng);
        assert!(
            partial.length < sol.objective,
            "a random ring tour has improving moves"
        );
        assert_length_matches(&tsp, &partial);
        let back = tsp.finish(&partial);
        assert!((back.objective - partial.length).abs() < 1e-9);
    }

    #[test]
    fn two_opt_uncrosses_a_square() {
        let tsp = TspWithCoordinates::new(
            "square".into(),
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        );
        let tour = vec![0, 1, 3, 2];
        let objective = tsp.calculate_tour_length(&tour).unwrap();
        let mut partial = tsp.to_partial(&TspSolution { tour, objective });
        // The crossing edges are the ones after cities 1 and 2, `1 -> 3` and
        // `2 -> 0`. Reconnecting 1 to 2 and 3 to 0 is the square's perimeter.
        assert!(try_two_opt(&mut partial, &tsp, 1, 2));
        assert_length_matches(&tsp, &partial);
        assert!((partial.length - 4.0).abs() < 1e-9);
    }
}
