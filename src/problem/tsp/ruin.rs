//! TSP's answer to [`Ruinable`]: cities are the elements and the one tour is
//! the only container.
//!
//! This is the single-container case the trait describes. Ruin-and-recreate
//! still works, since removing a set of cities and putting each back where it
//! is cheapest is a large-neighbourhood move on a tour as much as on a route.
//! What the single container takes away is regret. With no second container
//! the second-best placement is undefined, so regret-2 inserts the pool in a
//! fixed order that ranks nothing, and the operators that carry the search are
//! the three destroys and greedy insertion, followed by the anchored descent at
//! the bottom.

use rand::rngs::SmallRng;

use super::problem::{Tsp, TspSolution};
use crate::common::{AnchoredSweep, MIN_IMPROVEMENT};
use crate::trait_defs::{LocalRepair, Ruinable};

/// Marks a city that is not currently on the tour in [`TspPartial::pos`].
const NOT_PLACED: usize = usize::MAX;

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
    /// Rewrites the index for `tour[lo..=hi]`, the entries an edit there can
    /// have shifted.
    fn reindex_range(&mut self, lo: usize, hi: usize) {
        for k in lo..=hi {
            self.pos[self.tour[k]] = k;
        }
    }

    /// Rewrites the index for `tour[from..]`.
    fn reindex_from(&mut self, from: usize) {
        if from < self.tour.len() {
            self.reindex_range(from, self.tour.len() - 1);
        }
    }

    /// Cyclic distance forward from index `from` to index `to`.
    fn steps_forward(&self, from: usize, to: usize) -> usize {
        if to >= from {
            to - from
        } else {
            to + self.tour.len() - from
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
    fn measure(&self, prob: &Tsp) -> f64 {
        (0..self.tour.len())
            .map(|i| prob.distance(self.tour[i], self.tour[self.after(i)]))
            .sum()
    }

    /// What inserting `c` at `place` costs. Place `p` is between `tour[p-1]`
    /// and `tour[p]`, cyclically, so an empty tour has the one place `0`
    /// whose cost is the loop `c` to itself, which is what
    /// [`Tsp::calculate_tour_length`] charges a one-city tour.
    fn insertion_cost_at(&self, prob: &Tsp, place: usize, c: usize) -> f64 {
        if self.tour.is_empty() {
            return prob.distance(c, c);
        }
        let prev = self.tour[self.before(place)];
        let next = self.tour[place];
        prob.distance(prev, c) + prob.distance(c, next) - prob.distance(prev, next)
    }

    /// Reverses the cyclic segment from index `start` forward to index
    /// `end`, inclusive, wrapping past the array's end if it has to.
    ///
    /// The three cyclic edits here walk the segment they touch rather than
    /// slicing the array, so a pair of cities that are near on the cycle but
    /// sit on opposite ends of the array costs what the short arc costs, not
    /// what the array's long side does.
    fn reverse_cyclic(&mut self, start: usize, end: usize) {
        let len = self.steps_forward(start, end) + 1;
        let (mut p, mut q) = (start, end);
        for _ in 0..len / 2 {
            self.tour.swap(p, q);
            self.pos[self.tour[p]] = p;
            self.pos[self.tour[q]] = q;
            p = self.after(p);
            q = self.before(q);
        }
    }

    /// Moves the city at `start` to `end`, shifting the cyclic segment
    /// between them back by one.
    fn shift_left_cyclic(&mut self, start: usize, end: usize) {
        let first = self.tour[start];
        let mut p = start;
        while p != end {
            let q = self.after(p);
            self.tour[p] = self.tour[q];
            self.pos[self.tour[p]] = p;
            p = q;
        }
        self.tour[end] = first;
        self.pos[first] = end;
    }

    /// Moves the city at `end` to `start`, shifting the cyclic segment
    /// between them forward by one.
    fn shift_right_cyclic(&mut self, start: usize, end: usize) {
        let last = self.tour[end];
        let mut p = end;
        while p != start {
            let q = self.before(p);
            self.tour[p] = self.tour[q];
            self.pos[self.tour[p]] = p;
            p = q;
        }
        self.tour[start] = last;
        self.pos[last] = start;
    }

    #[cfg(debug_assertions)]
    fn assert_caches_consistent(&self, prob: &Tsp) {
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

impl Ruinable for Tsp {
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

    /// Re-measures the length rather than handing back the running total, so
    /// a run's float drift is cut at every accepted candidate. The
    /// permutation check is a debug assertion, since every edit above keeps
    /// the tour one and a release build should not pay a hash set per accept.
    fn finish(&self, partial: &TspPartial) -> TspSolution {
        let tour = partial.tour.clone();
        let objective = partial.measure(self);
        debug_assert_eq!(
            self.calculate_tour_length(&tour).ok(),
            Some(objective),
            "a ruined-and-recreated tour visits every city once"
        );
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
/// The moves are the two a tour has that price in O(1) from their endpoints,
/// relocating one city next to a near one and reversing the path between two
/// near ones, tried first-improvement over the granular pairs, the same shape
/// as VRP's [`AnchoredRouteDescent`](crate::problem::vrp::AnchoredRouteDescent).
/// The nearest-neighbour lists it reads are cached on the instance, so this
/// holds only its sweep buffers and its three tunables, each behind a builder
/// with a published default.
#[derive(Debug)]
pub struct AnchoredTourDescent {
    sweep: AnchoredSweep,
    granularity: usize,
    max_passes: usize,
}

impl Default for AnchoredTourDescent {
    fn default() -> Self {
        Self::new()
    }
}

impl AnchoredTourDescent {
    /// Nearest partners considered per city, unless
    /// [`with_granularity`](Self::with_granularity) says otherwise.
    ///
    /// Twice Lin-Kernighan's default of 5, since Or-opt and 2-opt have no
    /// depth to make up for a short list, and half VRP's 20, since a tour has
    /// no second route for a partner to be in. Unmeasured beyond that.
    pub const DEFAULT_GRANULARITY: usize = 10;

    /// Descent passes over a recreated tour, unless
    /// [`with_max_passes`](Self::with_max_passes) says otherwise. Matches
    /// VRP's.
    pub const DEFAULT_MAX_PASSES: usize = 4;

    /// A descent with the published defaults.
    pub fn new() -> Self {
        Self {
            sweep: AnchoredSweep::new(),
            granularity: Self::DEFAULT_GRANULARITY,
            max_passes: Self::DEFAULT_MAX_PASSES,
        }
    }

    /// Builder-style: how many nearest partners of each city the moves
    /// consider. Defaults to [`DEFAULT_GRANULARITY`](Self::DEFAULT_GRANULARITY).
    ///
    /// # Panics
    ///
    /// Panics if `granularity` is zero, since a descent with no partners
    /// can make no move.
    pub fn with_granularity(mut self, granularity: usize) -> Self {
        assert!(granularity >= 1, "granularity must be at least 1");
        self.granularity = granularity;
        self
    }

    /// Builder-style: how many nearest partners of each anchor the sweep
    /// visits along with it. Defaults to [`AnchoredSweep::DEFAULT_RING`], and
    /// zero sweeps the anchors alone.
    pub fn with_ring(mut self, ring: usize) -> Self {
        self.sweep = std::mem::take(&mut self.sweep).with_ring(ring);
        self
    }

    /// Builder-style: how many passes over the sweep list a repair may
    /// spend. Defaults to [`DEFAULT_MAX_PASSES`](Self::DEFAULT_MAX_PASSES).
    ///
    /// # Panics
    ///
    /// Panics if `max_passes` is zero, since a repair with no pass repairs
    /// nothing.
    pub fn with_max_passes(mut self, max_passes: usize) -> Self {
        assert!(max_passes >= 1, "max_passes must be at least 1");
        self.max_passes = max_passes;
        self
    }
}

/// Tries every granular move anchored at `u`, applying the first improving
/// one.
fn improve_around(partial: &mut TspPartial, prob: &Tsp, neighbors: &[usize], u: usize) -> bool {
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

impl LocalRepair<Tsp> for AnchoredTourDescent {
    fn repair_around(
        &mut self,
        prob: &Tsp,
        partial: &mut TspPartial,
        anchors: &[usize],
        rng: &mut SmallRng,
    ) {
        // Below four cities every tour is the same cycle up to rotation and
        // reflection, and both moves below assume four distinct endpoints.
        if partial.tour.len() < 4 || anchors.is_empty() {
            return;
        }
        // Cached on the instance, so this is a lookup after the first call
        // and can never be another instance's lists.
        let neighbors = prob.nearest_neighbors(self.granularity);
        self.sweep.ensure(prob.get_n());
        self.sweep
            .collect_around(anchors, &neighbors, |v| partial.pos[v] != NOT_PLACED);
        self.sweep.sweep(rng, self.max_passes, |u| {
            improve_around(partial, prob, &neighbors[u], u)
        });
        #[cfg(debug_assertions)]
        partial.assert_caches_consistent(prob);
    }
}

/// Moves `u` to sit directly after `v`, if that shortens the tour.
fn try_relocate(partial: &mut TspPartial, prob: &Tsp, u: usize, v: usize) -> bool {
    if v == u || partial.succ(v) == u {
        return false;
    }
    let (i, j) = (partial.pos[u], partial.pos[v]);
    // The slot after `v`, priced by the same function the repair prices it
    // with, less what taking `u` out saves.
    let gain = partial.insertion_cost_at(prob, partial.after(j), u) - prob.removal_gain(partial, u);
    if gain >= -MIN_IMPROVEMENT {
        return false;
    }
    // `u` can travel forward to the slot or the slot's contents can travel
    // back to `u`. Either shifts one arc of the cycle by one city, and the
    // two arcs add up to `n`, so the shorter one is taken.
    let forward = partial.steps_forward(i, j);
    if forward <= partial.tour.len() - forward {
        partial.shift_left_cyclic(i, j);
    } else {
        partial.shift_right_cyclic(partial.after(j), i);
    }
    partial.length += gain;
    true
}

/// Removes the edges after `x` and after `y` and reconnects `x` to `y`, if
/// that shortens the tour.
fn try_two_opt(partial: &mut TspPartial, prob: &Tsp, x: usize, y: usize) -> bool {
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
    // Reversing the path from `sx` to `y` and reversing its complement, the
    // path from `sy` to `x`, give the same cycle, so the shorter is taken.
    let (px, py) = (partial.pos[x], partial.pos[y]);
    let path = partial.steps_forward(px, py);
    if path <= partial.tour.len() - path {
        partial.reverse_cyclic(partial.after(px), py);
    } else {
        partial.reverse_cyclic(partial.after(py), px);
    }
    partial.length += gain;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::tsp::EdgeWeightType;
    use crate::search_state::ProblemTrait;
    use rand::SeedableRng;

    fn ring(n: usize) -> Tsp {
        let coords = (0..n)
            .map(|i| {
                let theta = std::f64::consts::TAU * i as f64 / n as f64;
                (theta.cos() * 10.0, theta.sin() * 10.0)
            })
            .collect();
        Tsp::new("ring".into(), coords)
    }

    fn assert_length_matches(prob: &Tsp, partial: &TspPartial) {
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
            let tsp = Tsp::with_edge_weight_type("t".into(), coords, ewt);
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

    /// The builders have to reach the sweep, not merely be stored. The
    /// narrowest descent, one partner, no ring, one pass, still repairs a
    /// random ring tour and leaves it valid.
    #[test]
    fn the_narrowest_descent_still_repairs() {
        let tsp = ring(40);
        let mut rng = SmallRng::seed_from_u64(5);
        let sol = tsp.new_solution(&mut rng);
        let mut partial = tsp.to_partial(&sol);
        let anchors: Vec<usize> = sol.tour[..8].to_vec();
        let mut descent = AnchoredTourDescent::new()
            .with_granularity(1)
            .with_ring(0)
            .with_max_passes(1);
        assert_eq!(descent.sweep.ring(), 0);
        descent.repair_around(&tsp, &mut partial, &anchors, &mut rng);
        assert!(partial.length < sol.objective);
        assert_length_matches(&tsp, &partial);
    }

    #[test]
    #[should_panic(expected = "granularity must be at least 1")]
    fn zero_granularity_is_rejected() {
        let _ = AnchoredTourDescent::new().with_granularity(0);
    }

    #[test]
    #[should_panic(expected = "max_passes must be at least 1")]
    fn zero_passes_are_rejected() {
        let _ = AnchoredTourDescent::new().with_max_passes(0);
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

    /// One descent object serving two instances of the same size has to read
    /// each instance's own neighbours. The second ring is the first one
    /// relabelled, so lists carried over from the first would point the moves
    /// at cities that are far apart on the second.
    #[test]
    fn one_descent_serves_two_instances_of_one_size() {
        let n = 40;
        let first = ring(n);
        let mut coords = first.coordinates().unwrap().to_vec();
        coords.rotate_left(n / 2);
        coords.swap(1, 21);
        let second = Tsp::new("relabelled".into(), coords);
        let mut descent = AnchoredTourDescent::new();
        let mut rng = SmallRng::seed_from_u64(8);
        for tsp in [&first, &second] {
            let sol = tsp.new_solution(&mut rng);
            let mut partial = tsp.to_partial(&sol);
            let anchors: Vec<usize> = sol.tour[..10].to_vec();
            descent.repair_around(tsp, &mut partial, &anchors, &mut rng);
            assert!(partial.length < sol.objective, "{}", tsp.name);
            assert_length_matches(tsp, &partial);
        }
    }

    /// The three cyclic edits are checked against the plain array
    /// operations they replace, on every pair of positions of a small tour,
    /// so the wrap-around cases are all covered. A reversal is compared as
    /// a cycle, since the complement reversal is the same cycle read the
    /// other way round.
    #[test]
    fn cyclic_edits_match_their_array_forms() {
        fn canonical(tour: &[usize]) -> Vec<usize> {
            let n = tour.len();
            let start = tour.iter().position(|&c| c == 0).unwrap();
            let forward: Vec<usize> = (0..n).map(|k| tour[(start + k) % n]).collect();
            let backward: Vec<usize> = (0..n).map(|k| tour[(start + n - k) % n]).collect();
            forward.min(backward)
        }
        let tsp = ring(7);
        let n = 7;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let base: Vec<usize> = (0..n).collect();
                let sol = TspSolution {
                    tour: base.clone(),
                    objective: tsp.calculate_tour_length(&base).unwrap(),
                };

                // Relocate city `i` to sit after city `j`, both ways round.
                let mut expected = base.clone();
                expected.remove(i);
                let at = expected.iter().position(|&c| c == j).unwrap();
                expected.insert(at + 1, i);
                let mut partial = tsp.to_partial(&sol);
                partial.shift_left_cyclic(i, j);
                assert_eq!(
                    canonical(&partial.tour),
                    canonical(&expected),
                    "left {i} {j}"
                );
                partial.length = partial.measure(&tsp);
                partial.assert_caches_consistent(&tsp);
                let mut partial = tsp.to_partial(&sol);
                partial.shift_right_cyclic(partial.after(j), i);
                assert_eq!(
                    canonical(&partial.tour),
                    canonical(&expected),
                    "right {i} {j}"
                );
                partial.length = partial.measure(&tsp);
                partial.assert_caches_consistent(&tsp);

                // Reverse the path after `i` up to `j`, and its complement.
                let mut expected = base.clone();
                if i < j {
                    expected[i + 1..=j].reverse();
                } else {
                    let mut rotated: Vec<usize> = (0..n).map(|k| base[(i + 1 + k) % n]).collect();
                    let len = (j + n - i) % n;
                    rotated[..len].reverse();
                    expected = rotated;
                }
                let mut partial = tsp.to_partial(&sol);
                partial.reverse_cyclic(partial.after(i), j);
                assert_eq!(
                    canonical(&partial.tour),
                    canonical(&expected),
                    "reverse {i} {j}"
                );
                partial.length = partial.measure(&tsp);
                partial.assert_caches_consistent(&tsp);
                let mut partial = tsp.to_partial(&sol);
                partial.reverse_cyclic(partial.after(j), i);
                assert_eq!(
                    canonical(&partial.tour),
                    canonical(&expected),
                    "complement {i} {j}"
                );
                partial.length = partial.measure(&tsp);
                partial.assert_caches_consistent(&tsp);
            }
        }
    }

    #[test]
    fn two_opt_uncrosses_a_square() {
        let tsp = Tsp::new(
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
