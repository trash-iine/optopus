//! Lin-Kernighan-Helsgott (LKH) variable-depth search for TSP.
//!
//! Each iteration picks a starting city, performs a variable-depth
//! edge-exchange search (up to `max_depth`-opt) guided by candidate lists,
//! and applies the first improving move found. The search terminates when
//! no improving move exists for any starting city (local optimum) or
//! when the stop condition is met.

use super::super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::problem::tsp_2d::{TspSolution, TspWithCoordinates};
use crate::search_state::SearchState;

/// Describes an improving LK move: which tour edges to remove and which to add.
struct LkMove {
    removed: Vec<(usize, usize)>,
    added: Vec<(usize, usize)>,
    gain: f64,
}

/// Lin-Kernighan-Helsgott heuristic for the Travelling Salesman Problem.
///
/// Performs a variable-depth edge-exchange search starting from each city.
/// At each depth level the algorithm:
/// 1. Selects a candidate city near the current chain endpoint
/// 2. Attempts to close the move (checking if the resulting tour is shorter)
/// 3. If closure fails, extends the chain to deeper levels
///
/// The search is pruned by:
/// - **Candidate lists**: only the `num_neighbors` nearest cities are considered
/// - **Positive gain criterion**: partial gain must remain positive at each step
/// - **Maximum depth**: search stops after `max_depth` levels (k in k-opt)
///
/// # References
///
/// - Lin, S. and Kernighan, B. W. "An Effective Heuristic Algorithm for the
///   Traveling-Salesman Problem." *Operations Research*, 21(2), 498-516, 1973.
///   [DOI](https://doi.org/10.1287/opre.21.2.498)
/// - Helsgaun, K. "An Effective Implementation of the Lin-Kernighan Traveling
///   Salesman Heuristic." *European Journal of Operational Research*, 126(1),
///   106-130, 2000. [DOI](https://doi.org/10.1016/S0377-2217(99)00284-2)
///
/// # Parameters
///
/// - `stop_condition` — overall stopping criterion
/// - `num_neighbors` — number of nearest neighbors in candidate lists (default: 5)
/// - `max_depth` — maximum LK search depth (default: 5)
pub struct LinKernighanHelsgott {
    stop_condition: StopCondition,
    num_neighbors: usize,
    max_depth: usize,
    candidates: Vec<Vec<usize>>,
    position: Vec<usize>,
    no_improvement: bool,
}

impl LinKernighanHelsgott {
    pub fn new(stop_condition: StopCondition, num_neighbors: usize, max_depth: usize) -> Self {
        Self {
            stop_condition,
            num_neighbors,
            max_depth,
            candidates: Vec::new(),
            position: Vec::new(),
            no_improvement: false,
        }
    }

    fn ensure_candidates(&mut self, prob: &TspWithCoordinates) {
        if !self.candidates.is_empty() {
            return;
        }
        let n = prob.get_n();
        self.candidates = (0..n)
            .map(|i| {
                let mut nbrs: Vec<(f64, usize)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (prob.distance(i, j), j))
                    .collect();
                nbrs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                nbrs.truncate(self.num_neighbors);
                nbrs.into_iter().map(|(_, j)| j).collect()
            })
            .collect();
    }

    fn build_position(&mut self, tour: &[usize]) {
        let n = tour.len();
        self.position.resize(n, 0);
        for (idx, &city) in tour.iter().enumerate() {
            self.position[city] = idx;
        }
    }

    #[inline]
    fn succ(&self, tour: &[usize], city: usize) -> usize {
        tour[(self.position[city] + 1) % tour.len()]
    }

    #[inline]
    fn pred(&self, tour: &[usize], city: usize) -> usize {
        tour[(self.position[city] + tour.len() - 1) % tour.len()]
    }

    // -- LK search -------------------------------------------------------

    /// Tries to find an improving LK move starting from city `t1`.
    fn find_lk_move(&self, prob: &TspWithCoordinates, tour: &[usize], t1: usize) -> Option<LkMove> {
        let n = tour.len();
        let t2_options = [self.succ(tour, t1), self.pred(tour, t1)];

        for &t2 in &t2_options {
            let d_x1 = prob.distance(t1, t2);

            for &t3 in &self.candidates[t2] {
                if t3 == t1 || t3 == t2 {
                    continue;
                }
                // y1 = (t2, t3) must not already be a tour edge
                if t3 == self.succ(tour, t2) || t3 == self.pred(tour, t2) {
                    continue;
                }

                let d_y1 = prob.distance(t2, t3);
                let g1 = d_x1 - d_y1;
                if g1 <= 0.0 {
                    continue;
                }

                // Try both tour-neighbors of t3 as t4
                let t4_options = [self.succ(tour, t3), self.pred(tour, t3)];
                for &t4 in &t4_options {
                    if t4 == t2 {
                        continue;
                    }

                    let d_x2 = prob.distance(t3, t4);

                    // -- closure test (depth 1 = 2-opt) --
                    let d_close = prob.distance(t4, t1);
                    let g_close = g1 + d_x2 - d_close;
                    if g_close > 1e-10 {
                        let removed = vec![(t1, t2), (t3, t4)];
                        let added = vec![(t2, t3), (t4, t1)];
                        if is_valid_move(n, tour, &removed, &added) {
                            return Some(LkMove {
                                removed,
                                added,
                                gain: g_close,
                            });
                        }
                    }

                    // -- extend to deeper levels --
                    if self.max_depth >= 2 && t4 != t1 {
                        let broken = vec![(t1, t2), (t3, t4)];
                        let added_so_far = vec![(t2, t3)];
                        let mut in_chain = vec![false; n];
                        in_chain[t1] = true;
                        in_chain[t2] = true;
                        in_chain[t3] = true;
                        in_chain[t4] = true;

                        if let Some(mv) = self.extend_search(
                            prob,
                            tour,
                            t1,
                            t4,
                            g1 + d_x2,
                            2,
                            broken,
                            added_so_far,
                            &mut in_chain,
                        ) {
                            return Some(mv);
                        }
                    }
                }
            }
        }
        None
    }

    /// Recursively extends the LK chain to deeper levels.
    ///
    /// * `t1`         – starting city (for the closing edge)
    /// * `t_last`     – current dangling endpoint of the chain
    /// * `g_sum`      – Σ(broken costs) − Σ(added costs) so far
    /// * `depth`      – current depth (2-indexed: 2 means trying a 3-opt, etc.)
    /// * `broken`     – tour edges broken so far
    /// * `added_so_far` – non-closing edges added so far
    /// * `in_chain`   – cities already part of the chain (for cycle avoidance)
    #[allow(clippy::too_many_arguments)]
    fn extend_search(
        &self,
        prob: &TspWithCoordinates,
        tour: &[usize],
        t1: usize,
        t_last: usize,
        g_sum: f64,
        depth: usize,
        broken: Vec<(usize, usize)>,
        added_so_far: Vec<(usize, usize)>,
        in_chain: &mut Vec<bool>,
    ) -> Option<LkMove> {
        let n = tour.len();

        for &t_next in &self.candidates[t_last] {
            if t_next == t1 || in_chain[t_next] {
                continue;
            }
            // Added edge must not be a tour edge
            if t_next == self.succ(tour, t_last) || t_next == self.pred(tour, t_last) {
                continue;
            }
            // Must not re-add a previously broken edge
            if is_edge_in(&broken, t_last, t_next) {
                continue;
            }

            let d_y = prob.distance(t_last, t_next);
            let g_partial = g_sum - d_y;
            if g_partial <= 0.0 {
                continue;
            }

            let t_break_options = [self.succ(tour, t_next), self.pred(tour, t_next)];
            for &t_break in &t_break_options {
                if t_break == t_last {
                    continue;
                }
                // Don't break an already-added edge
                if is_edge_in(&added_so_far, t_next, t_break) {
                    continue;
                }
                // Don't close with an already-broken edge
                if is_edge_in(&broken, t_break, t1) {
                    continue;
                }

                let d_x = prob.distance(t_next, t_break);

                // -- closure test --
                let d_close = prob.distance(t_break, t1);
                let g_close = g_partial + d_x - d_close;
                if g_close > 1e-10 {
                    let mut removed = broken.clone();
                    removed.push((t_next, t_break));
                    let mut added = added_so_far.clone();
                    added.push((t_last, t_next));
                    added.push((t_break, t1));

                    if is_valid_move(n, tour, &removed, &added) {
                        return Some(LkMove {
                            removed,
                            added,
                            gain: g_close,
                        });
                    }
                }

                // -- extend further --
                if depth < self.max_depth && t_break != t1 {
                    let mut new_broken = broken.clone();
                    new_broken.push((t_next, t_break));
                    let mut new_added = added_so_far.clone();
                    new_added.push((t_last, t_next));

                    in_chain[t_next] = true;
                    in_chain[t_break] = true;

                    let result = self.extend_search(
                        prob,
                        tour,
                        t1,
                        t_break,
                        g_partial + d_x,
                        depth + 1,
                        new_broken,
                        new_added,
                        in_chain,
                    );

                    in_chain[t_next] = false;
                    in_chain[t_break] = false;

                    if result.is_some() {
                        return result;
                    }
                }
            }
        }
        None
    }

    // -- move application -------------------------------------------------

    fn apply_lk_move(
        &mut self,
        prob: &TspWithCoordinates,
        sol: &mut TspSolution,
        lk_move: &LkMove,
    ) {
        let n = sol.tour.len();

        // Build adjacency from the current tour, then patch it.
        let mut adj = vec![[usize::MAX; 2]; n];
        for i in 0..n {
            let a = sol.tour[i];
            let b = sol.tour[(i + 1) % n];
            let s = if adj[a][0] == usize::MAX { 0 } else { 1 };
            adj[a][s] = b;
            let s = if adj[b][0] == usize::MAX { 0 } else { 1 };
            adj[b][s] = a;
        }
        for &(a, b) in &lk_move.removed {
            if adj[a][0] == b {
                adj[a][0] = usize::MAX;
            } else {
                adj[a][1] = usize::MAX;
            }
            if adj[b][0] == a {
                adj[b][0] = usize::MAX;
            } else {
                adj[b][1] = usize::MAX;
            }
        }
        for &(a, b) in &lk_move.added {
            let s = if adj[a][0] == usize::MAX { 0 } else { 1 };
            adj[a][s] = b;
            let s = if adj[b][0] == usize::MAX { 0 } else { 1 };
            adj[b][s] = a;
        }

        // Traverse the modified adjacency to build the new tour.
        let mut new_tour = Vec::with_capacity(n);
        let mut current = sol.tour[0];
        let mut prev = usize::MAX;
        for _ in 0..n {
            new_tour.push(current);
            let next = if adj[current][0] != prev {
                adj[current][0]
            } else {
                adj[current][1]
            };
            prev = current;
            current = next;
        }

        sol.tour = new_tour;
        sol.objective -= lk_move.gain;
        sol.gain = prob.compute_all_gains(&sol.tour);
    }
}

// -- free helpers --------------------------------------------------------

#[inline]
fn is_edge_in(edges: &[(usize, usize)], a: usize, b: usize) -> bool {
    edges
        .iter()
        .any(|&(x, y)| (x == a && y == b) || (x == b && y == a))
}

/// Checks whether replacing `removed` edges with `added` edges in the tour
/// produces a valid Hamiltonian cycle.
fn is_valid_move(
    n: usize,
    tour: &[usize],
    removed: &[(usize, usize)],
    added: &[(usize, usize)],
) -> bool {
    let mut adj = vec![[usize::MAX; 2]; n];

    // Original tour edges
    for i in 0..n {
        let a = tour[i];
        let b = tour[(i + 1) % n];
        let s = if adj[a][0] == usize::MAX { 0 } else { 1 };
        adj[a][s] = b;
        let s = if adj[b][0] == usize::MAX { 0 } else { 1 };
        adj[b][s] = a;
    }

    // Remove
    for &(a, b) in removed {
        if adj[a][0] == b {
            adj[a][0] = usize::MAX;
        } else if adj[a][1] == b {
            adj[a][1] = usize::MAX;
        } else {
            return false;
        }
        if adj[b][0] == a {
            adj[b][0] = usize::MAX;
        } else if adj[b][1] == a {
            adj[b][1] = usize::MAX;
        } else {
            return false;
        }
    }

    // Add
    for &(a, b) in added {
        if adj[a][0] == usize::MAX {
            adj[a][0] = b;
        } else if adj[a][1] == usize::MAX {
            adj[a][1] = b;
        } else {
            return false;
        }
        if adj[b][0] == usize::MAX {
            adj[b][0] = a;
        } else if adj[b][1] == usize::MAX {
            adj[b][1] = a;
        } else {
            return false;
        }
    }

    // Every vertex must have degree 2
    for entry in adj.iter().take(n) {
        if entry[0] == usize::MAX || entry[1] == usize::MAX {
            return false;
        }
    }

    // Traverse and check single cycle of length n
    let mut count = 0usize;
    let mut current = tour[0];
    let mut prev = usize::MAX;
    loop {
        count += 1;
        if count > n {
            return false;
        }
        let next = if adj[current][0] != prev {
            adj[current][0]
        } else {
            adj[current][1]
        };
        prev = current;
        current = next;
        if current == tour[0] {
            break;
        }
    }
    count == n
}

// -- Heuristic impl ------------------------------------------------------

impl Heuristic<TspWithCoordinates> for LinKernighanHelsgott {
    fn clear(&mut self) {
        self.no_improvement = false;
    }

    fn is_done<'a>(&self, state: &SearchState<'a, TspWithCoordinates>) -> bool {
        self.stop_condition.is_done(state) || self.no_improvement
    }

    fn run_once<'a>(
        &mut self,
        state: &mut SearchState<'a, TspWithCoordinates>,
    ) -> Result<(), OptError> {
        self.ensure_candidates(state.instance);
        self.build_position(&state.solution.tour);

        let n = state.instance.get_n();
        if n < 4 {
            self.no_improvement = true;
            state.progress_iteration();
            return Ok(());
        }

        for tour_idx in 0..n {
            let t1 = state.solution.tour[tour_idx];

            if let Some(lk_move) = self.find_lk_move(state.instance, &state.solution.tour, t1) {
                tracing::debug!(
                    t1 = t1,
                    gain = lk_move.gain,
                    depth = lk_move.removed.len(),
                    "LKH: improving move found"
                );
                self.apply_lk_move(state.instance, &mut state.solution, &lk_move);
                self.build_position(&state.solution.tour);
                state.iteration += 1;
                state.update_best();
                return Ok(());
            }
        }

        // No improving move found for any starting city — local optimum.
        self.no_improvement = true;
        state.progress_iteration();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic::Heuristic;
    use crate::search_state::SearchState;

    fn make_square_tsp() -> TspWithCoordinates {
        TspWithCoordinates::new(
            "square".to_string(),
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        )
    }

    fn make_solution(
        prob: &TspWithCoordinates,
        tour: Vec<usize>,
    ) -> crate::problem::tsp_2d::TspSolution {
        let objective = prob.calculate_tour_length(&tour).unwrap();
        let gain = prob.compute_all_gains(&tour);
        crate::problem::tsp_2d::TspSolution {
            tour,
            objective,
            gain,
        }
    }

    #[test]
    fn test_lkh_improves_suboptimal_tour() {
        let prob = make_square_tsp();
        // [0,1,3,2] has a crossing (length ≈ 2 + 2*sqrt(2))
        let sol = make_solution(&prob, vec![0, 1, 3, 2]);
        let initial_obj = sol.objective;

        let mut state = SearchState::with_solution(&prob, sol);
        let mut lkh = LinKernighanHelsgott::new(StopCondition::iterations(100), 3, 5);
        lkh.run(&mut state).unwrap();

        assert!(
            state.best_solution.objective < initial_obj,
            "LKH should improve a suboptimal tour: {} -> {}",
            initial_obj,
            state.best_solution.objective,
        );
    }

    #[test]
    fn test_lkh_produces_valid_tour() {
        let prob = make_square_tsp();
        let sol = make_solution(&prob, vec![0, 1, 3, 2]);

        let mut state = SearchState::with_solution(&prob, sol);
        let mut lkh = LinKernighanHelsgott::new(StopCondition::iterations(100), 3, 5);
        lkh.run(&mut state).unwrap();

        let tour = &state.best_solution.tour;
        assert_eq!(tour.len(), 4);

        // Check valid permutation
        let mut sorted = tour.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3]);

        // Check objective matches calculated length
        let expected = prob.calculate_tour_length(tour).unwrap();
        assert!(
            (state.best_solution.objective - expected).abs() < 1e-9,
            "objective {} != calculated {}",
            state.best_solution.objective,
            expected,
        );
    }

    #[test]
    fn test_lkh_gain_cache_consistent() {
        let prob = make_square_tsp();
        let sol = make_solution(&prob, vec![0, 1, 3, 2]);

        let mut state = SearchState::with_solution(&prob, sol);
        let mut lkh = LinKernighanHelsgott::new(StopCondition::iterations(100), 3, 5);
        lkh.run(&mut state).unwrap();

        let expected_gains = prob.compute_all_gains(&state.solution.tour);
        for (key, &val) in &state.solution.gain {
            let exp = expected_gains[key];
            assert!(
                (val - exp).abs() < 1e-9,
                "gain[{:?}]: {} expected {}",
                key,
                val,
                exp,
            );
        }
        assert_eq!(state.solution.gain.len(), expected_gains.len());
    }

    #[test]
    fn test_lkh_stops_at_local_optimum() {
        let prob = make_square_tsp();
        // [0,1,2,3] is the optimal tour for the unit square (length = 4.0)
        let sol = make_solution(&prob, vec![0, 1, 2, 3]);
        let initial_obj = sol.objective;

        let mut state = SearchState::with_solution(&prob, sol);
        let mut lkh = LinKernighanHelsgott::new(StopCondition::iterations(100), 3, 5);
        lkh.run(&mut state).unwrap();

        assert!(
            lkh.no_improvement,
            "LKH should detect local optimum on the optimal tour",
        );
        assert!(
            (state.best_solution.objective - initial_obj).abs() < 1e-9,
            "optimal tour should not change",
        );
    }

    #[test]
    fn test_lkh_on_larger_instance() {
        let prob = TspWithCoordinates::load_file("data/tsp/eil51.tsp").unwrap();
        let mut state = SearchState::new(&prob);
        let initial_obj = state.solution.objective;

        let mut lkh = LinKernighanHelsgott::new(StopCondition::iterations(1000), 3, 5);
        lkh.run(&mut state).unwrap();

        // Should find at least as good a solution
        assert!(state.best_solution.objective <= initial_obj);

        // Verify valid tour
        let tour = &state.best_solution.tour;
        let mut sorted = tour.clone();
        sorted.sort();
        let expected: Vec<usize> = (0..prob.get_n()).collect();
        assert_eq!(sorted, expected);

        let expected_obj = prob.calculate_tour_length(tour).unwrap();
        assert!(
            (state.best_solution.objective - expected_obj).abs() < 1e-9,
            "objective {} != calculated {}",
            state.best_solution.objective,
            expected_obj,
        );
    }

    #[test]
    fn test_is_valid_move_basic() {
        let tour = vec![0, 1, 2, 3];
        // Valid 2-opt: remove (0,1) and (2,3), add (0,2) and (1,3)
        // New tour: [0, 2, 1, 3]
        assert!(is_valid_move(
            4,
            &tour,
            &[(0, 1), (2, 3)],
            &[(0, 2), (1, 3)]
        ));
        // Invalid: remove one edge without replacement → broken cycle
        assert!(!is_valid_move(4, &tour, &[(0, 1)], &[]));
    }
}
