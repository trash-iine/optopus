use super::problem::{TspSolution, TspWithCoordinates};
use crate::{
    error::OptError,
    search_state::{EnabledTabu, Evaluable, Evaluate, MoveToNeighbor, Rankable},
};
use rand::Rng;

/// A 2-opt move that removes edges `(tour[i], tour[i+1])` and `(tour[j], tour[(j+1)%n])`,
/// then reconnects as `(tour[i], tour[j])` and `(tour[i+1], tour[(j+1)%n])`.
///
/// Implemented by reversing the sub-segment `tour[i+1..=j]`.
/// `gain` is the change in tour length after the move (negative = improvement).
#[derive(Debug, Clone)]
pub struct TspTwoOptNeighbor {
    /// Smallest index of the two removed edges.
    pub i: usize,
    /// Largest index of the two removed edges.
    pub j: usize,
    /// Change in tour length (negative = improvement).
    pub gain: f64,
}

impl TspTwoOptNeighbor {
    /// Builds the 2-opt move on `(i, j)`, computing its gain from the four
    /// endpoint distances.
    ///
    /// Every construction site goes through here so the gain can never be
    /// filled in by hand — a wrong value would silently corrupt
    /// [`TspSolution::objective`], which [`apply_to_solution`](MoveToNeighbor::apply_to_solution)
    /// updates from it without re-measuring the tour.
    ///
    /// `i < j` with `j >= i + 2` is what makes the move non-trivial;
    /// degenerate pairs (`j == i + 1`, or `(0, n - 1)` which reverses the whole
    /// tour) are consistently evaluated as zero-gain no-ops rather than
    /// rejected, so a caller that builds one is not silently misled.
    ///
    /// # Panics
    ///
    /// Panics if `i` or `j` is out of range for `sol.tour`.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// // Unit square; the tour 0 → 1 → 3 → 2 crosses itself.
    /// let tsp = TspWithCoordinates::new(
    ///     "square".to_string(),
    ///     vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    /// );
    /// let tour = vec![0, 1, 3, 2];
    /// let objective = tsp.calculate_tour_length(&tour).unwrap();
    /// let sol = TspSolution { tour, objective };
    ///
    /// // Reversing tour[2..=3] uncrosses it.
    /// let m = TspTwoOptNeighbor::new(&tsp, &sol, 1, 3);
    /// assert!(m.gain < 0.0);
    ///
    /// let mut after = sol.clone();
    /// m.apply_to_solution(&tsp, &mut after).unwrap();
    /// assert_eq!(after.tour, vec![0, 1, 2, 3]);
    /// assert!((after.objective - (sol.objective + m.gain)).abs() < 1e-9);
    /// ```
    pub fn new(prob: &TspWithCoordinates, sol: &TspSolution, i: usize, j: usize) -> Self {
        let e1 = prob.get_edge_from(&sol.tour, i);
        let e2 = prob.get_edge_from(&sol.tour, j);
        Self {
            i,
            j,
            gain: prob.calc_2opt_gain_cities(e1, e2),
        }
    }
}

impl Rankable for TspTwoOptNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain < other.gain
    }
}

impl Evaluate for TspTwoOptNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain)
    }
}

impl EnabledTabu for TspTwoOptNeighbor {
    // Keyed by the (i, j) pair (normalized so i < j)
    type TabuMap = std::collections::HashMap<(usize, usize), u64>;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        let key = (self.i.min(self.j), self.i.max(self.j));
        tabu_map.get(&key).is_none_or(|&t| iteration > t)
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
        rng: &mut rand::rngs::SmallRng,
    ) {
        let d = rng.random_range(tabu_tenure.0..=tabu_tenure.1);
        let key = (self.i.min(self.j), self.i.max(self.j));
        tabu_map.insert(key, iteration + d);
    }
}

impl MoveToNeighbor<TspWithCoordinates> for TspTwoOptNeighbor {
    fn apply_to_solution(
        &self,
        _prob: &TspWithCoordinates,
        sol: &mut TspSolution,
    ) -> Result<(), OptError> {
        sol.tour[self.i + 1..=self.j].reverse();
        sol.objective += self.gain;
        Ok(())
    }

    fn iter(prob: &TspWithCoordinates, sol: &TspSolution) -> impl Iterator<Item = Self> + Send {
        let n = sol.tour.len();
        (0..n - 1).flat_map(move |i| {
            // When i=0, j=n-1 would reverse the entire tour (trivially equivalent for undirected),
            // so exclude that case to avoid redundant moves
            let max_j = if i == 0 { n - 1 } else { n };
            (i + 2..max_j).map(move |j| Self::new(prob, sol, i, j))
        })
    }

    fn move_to_be_better_than(
        &self,
        _: &TspWithCoordinates,
        src: &TspSolution,
        other: &TspSolution,
    ) -> bool {
        self.gain + src.objective < other.objective
    }

    /// O(1) expected: rejection-samples a uniformly random valid `(i, j)`
    /// pair (`j ≥ i + 2`, excluding the full-tour reversal `(0, n-1)`) and
    /// computes its gain from four distance lookups.
    fn random_neighbor(
        prob: &TspWithCoordinates,
        sol: &TspSolution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Option<Self> {
        let n = sol.tour.len();
        if n < 4 {
            return None;
        }
        // Acceptance ratio is ~1/2, so 64 attempts fail with probability
        // ~2^-64; fall back to reservoir sampling for guaranteed termination.
        for _ in 0..64 {
            let a = rng.random_range(0..n);
            let b = rng.random_range(0..n);
            let (i, j) = (a.min(b), a.max(b));
            if j < i + 2 || (i == 0 && j == n - 1) {
                continue;
            }
            return Some(Self::new(prob, sol, i, j));
        }
        use rand::seq::IteratorRandom;
        Self::iter(prob, sol).choose(rng)
    }
}

/// A relocate move that removes city `tour[pos]` from its current position
/// and inserts it immediately after `tour[ins]`.
///
/// `pos` and `ins` are indices in the original tour before the move is applied.
/// `gain` is the change in tour length (negative = improvement).
#[derive(Debug, Clone)]
pub struct TspRelocateNeighbor {
    /// Index of the city to be relocated.
    pub pos: usize,
    /// Index of the city after which the relocated city will be inserted.
    pub ins: usize,
    /// Change in tour length (negative = improvement).
    pub gain: f64,
}

/// Cost saved by splicing `tour[pos]` out from between its two tour neighbors.
///
/// Depends only on `pos`, so the neighborhood scan hoists it out of the inner
/// loop over insertion points.
fn removal_gain(prob: &TspWithCoordinates, sol: &TspSolution, pos: usize) -> f64 {
    let n = sol.tour.len();
    let prev = (pos + n - 1) % n;
    let next = (pos + 1) % n;
    prob.distance(sol.tour[prev], sol.tour[pos]) + prob.distance(sol.tour[pos], sol.tour[next])
        - prob.distance(sol.tour[prev], sol.tour[next])
}

/// Cost of splicing `tour[pos]` back in between `tour[ins]` and its successor.
fn insertion_cost(prob: &TspWithCoordinates, sol: &TspSolution, pos: usize, ins: usize) -> f64 {
    let n = sol.tour.len();
    let ins_next = (ins + 1) % n;
    prob.distance(sol.tour[ins], sol.tour[pos]) + prob.distance(sol.tour[pos], sol.tour[ins_next])
        - prob.distance(sol.tour[ins], sol.tour[ins_next])
}

impl TspRelocateNeighbor {
    /// Builds the relocation of `tour[pos]` to just after `tour[ins]`,
    /// computing its gain as `insertion_cost - removal_gain`.
    ///
    /// Every construction site goes through the same two gain helpers, so the
    /// two halves of the delta can never be paired up wrongly — a bad `gain`
    /// would silently corrupt [`TspSolution::objective`], which
    /// [`apply_to_solution`](MoveToNeighbor::apply_to_solution) updates from it
    /// without re-measuring the tour.
    ///
    /// # Panics
    ///
    /// Panics if `pos` or `ins` is out of range for `sol.tour`, or if the move
    /// is one of the two that leave the tour unchanged (`ins == pos` or `ins`
    /// is the predecessor of `pos`). Those are rejected rather than evaluated:
    /// the tour would not move but the gain would come out non-zero, so
    /// applying one would desynchronize `objective` from the actual tour.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// // Unit square; the tour 0 → 1 → 3 → 2 crosses itself.
    /// let tsp = TspWithCoordinates::new(
    ///     "square".to_string(),
    ///     vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    /// );
    /// let tour = vec![0, 1, 3, 2];
    /// let objective = tsp.calculate_tour_length(&tour).unwrap();
    /// let sol = TspSolution { tour, objective };
    ///
    /// // Move city 2 (at index 3) to just after city 1 (at index 1).
    /// let m = TspRelocateNeighbor::new(&tsp, &sol, 3, 1);
    /// assert!(m.gain < 0.0);
    ///
    /// let mut after = sol.clone();
    /// m.apply_to_solution(&tsp, &mut after).unwrap();
    /// assert_eq!(after.tour, vec![0, 1, 2, 3]);
    /// assert!((after.objective - (sol.objective + m.gain)).abs() < 1e-9);
    /// ```
    pub fn new(prob: &TspWithCoordinates, sol: &TspSolution, pos: usize, ins: usize) -> Self {
        let n = sol.tour.len();
        assert!(
            pos < n && ins < n,
            "relocate indices out of range: pos={pos}, ins={ins}, n={n}"
        );
        assert!(
            ins != pos && ins != (pos + n - 1) % n,
            "relocate would leave the tour unchanged: pos={pos}, ins={ins}"
        );
        Self {
            pos,
            ins,
            gain: insertion_cost(prob, sol, pos, ins) - removal_gain(prob, sol, pos),
        }
    }
}

impl Rankable for TspRelocateNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain < other.gain
    }
}

impl Evaluate for TspRelocateNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain)
    }
}

impl EnabledTabu for TspRelocateNeighbor {
    // Keyed by the (pos, ins) pair
    type TabuMap = std::collections::HashMap<(usize, usize), u64>;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        tabu_map
            .get(&(self.pos, self.ins))
            .is_none_or(|&t| iteration > t)
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
        rng: &mut rand::rngs::SmallRng,
    ) {
        let d = rng.random_range(tabu_tenure.0..=tabu_tenure.1);
        tabu_map.insert((self.pos, self.ins), iteration + d);
    }
}

impl MoveToNeighbor<TspWithCoordinates> for TspRelocateNeighbor {
    fn apply_to_solution(
        &self,
        _prob: &TspWithCoordinates,
        sol: &mut TspSolution,
    ) -> Result<(), OptError> {
        let city = sol.tour.remove(self.pos);
        let insert_at = if self.ins < self.pos {
            self.ins + 1
        } else {
            self.ins
        };
        sol.tour.insert(insert_at, city);
        sol.objective += self.gain;
        Ok(())
    }

    fn iter(prob: &TspWithCoordinates, sol: &TspSolution) -> impl Iterator<Item = Self> + Send {
        let n = prob.get_n();
        (0..n).flat_map(move |pos| {
            let prev = (pos + n - 1) % n;
            // Hoisted out of the inner loop: the removal side depends only on pos.
            let removal = removal_gain(prob, sol, pos);

            (0..n).filter_map(move |ins| {
                // Inserting after pos itself or after its predecessor (prev) would
                // leave the tour unchanged, so skip these cases
                if ins == pos || ins == prev {
                    return None;
                }

                Some(TspRelocateNeighbor {
                    pos,
                    ins,
                    gain: insertion_cost(prob, sol, pos, ins) - removal,
                })
            })
        })
    }

    fn move_to_be_better_than(
        &self,
        _: &TspWithCoordinates,
        src: &TspSolution,
        other: &TspSolution,
    ) -> bool {
        self.gain + src.objective < other.objective
    }

    /// O(1): samples a uniformly random `(pos, ins)` pair with
    /// `ins ∉ {pos, prev(pos)}` and computes its gain from six distance
    /// lookups.
    fn random_neighbor(
        prob: &TspWithCoordinates,
        sol: &TspSolution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Option<Self> {
        let n = sol.tour.len();
        if n < 3 {
            return None;
        }
        let pos = rng.random_range(0..n);
        let prev = (pos + n - 1) % n;
        // Map a draw from the (n - 2)-element valid set onto 0..n skipping
        // the two excluded indices in ascending order.
        let (e1, e2) = (pos.min(prev), pos.max(prev));
        let mut ins = rng.random_range(0..n - 2);
        if ins >= e1 {
            ins += 1;
        }
        if ins >= e2 {
            ins += 1;
        }

        Some(Self::new(prob, sol, pos, ins))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_square_tsp() -> TspWithCoordinates {
        // 4 cities forming a unit square: (0,0), (1,0), (1,1), (0,1)
        // Edge length = 1, diagonal = sqrt(2)
        TspWithCoordinates::new(
            "square".to_string(),
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        )
    }

    fn make_sol(tsp: &TspWithCoordinates, tour: Vec<usize>) -> TspSolution {
        let objective = tsp.calculate_tour_length(&tour).unwrap();
        TspSolution { tour, objective }
    }

    #[test]
    fn test_two_opt_gain_calculation() {
        let tsp = make_square_tsp();
        // Initial tour [0,1,3,2] has a crossing (length = 1 + sqrt(2) + 1 + sqrt(2))
        let sol = make_sol(&tsp, vec![0, 1, 3, 2]);

        // 2-opt (i=1, j=2): replace edges (1,3) and (2,0) with (1,2) and (3,0)
        // → tour [0,1,2,3] (length = 4.0)
        let neighbors: Vec<_> = TspTwoOptNeighbor::iter(&tsp, &sol).collect();
        let best = neighbors
            .iter()
            .min_by(|a, b| a.gain.partial_cmp(&b.gain).unwrap())
            .unwrap();
        assert!(
            best.gain < 0.0,
            "the best 2-opt move should be an improvement"
        );

        let mut sol2 = sol.clone();
        best.apply_to_solution(&tsp, &mut sol2).unwrap();
        let expected = tsp.calculate_tour_length(&sol2.tour).unwrap();
        assert!((sol2.objective - expected).abs() < 1e-9);
    }

    #[test]
    fn test_two_opt_apply_consistency() {
        let tsp = make_square_tsp();
        let sol = make_sol(&tsp, vec![0, 2, 1, 3]);

        for neighbor in TspTwoOptNeighbor::iter(&tsp, &sol) {
            let mut s = sol.clone();
            neighbor.apply_to_solution(&tsp, &mut s).unwrap();
            let expected = tsp.calculate_tour_length(&s.tour).unwrap();
            assert!(
                (s.objective - expected).abs() < 1e-9,
                "2-opt ({},{}) after: objective={} expected={}",
                neighbor.i,
                neighbor.j,
                s.objective,
                expected
            );
        }
    }

    #[test]
    fn test_relocate_apply_consistency() {
        let tsp = make_square_tsp();
        let sol = make_sol(&tsp, vec![0, 1, 2, 3]);

        for neighbor in TspRelocateNeighbor::iter(&tsp, &sol) {
            let mut s = sol.clone();
            neighbor.apply_to_solution(&tsp, &mut s).unwrap();
            let expected = tsp.calculate_tour_length(&s.tour).unwrap();
            assert!(
                (s.objective - expected).abs() < 1e-9,
                "relocate (pos={}, ins={}) after: objective={} expected={}",
                neighbor.pos,
                neighbor.ins,
                s.objective,
                expected
            );
        }
    }

    #[test]
    fn test_relocate_tour_is_valid_permutation() {
        let tsp = make_square_tsp();
        let sol = make_sol(&tsp, vec![0, 1, 2, 3]);

        for neighbor in TspRelocateNeighbor::iter(&tsp, &sol) {
            let mut s = sol.clone();
            neighbor.apply_to_solution(&tsp, &mut s).unwrap();
            let mut sorted = s.tour.clone();
            sorted.sort();
            assert_eq!(
                sorted,
                vec![0, 1, 2, 3],
                "tour should remain a valid permutation"
            );
        }
    }

    /// 6-city instance with irregular coordinates (no gain ties), for
    /// random-sampling membership checks.
    fn make_hex_tsp() -> TspWithCoordinates {
        TspWithCoordinates::new(
            "hex".to_string(),
            vec![
                (0.0, 0.0),
                (3.0, 0.5),
                (4.0, 2.5),
                (2.5, 4.0),
                (0.5, 3.5),
                (-1.0, 1.5),
            ],
        )
    }

    #[test]
    fn test_random_neighbor_samples_member_of_iter() {
        use rand::SeedableRng;
        let tsp = make_hex_tsp();
        let sol = make_sol(&tsp, vec![0, 2, 4, 1, 5, 3]);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);

        let two_opts: Vec<_> = TspTwoOptNeighbor::iter(&tsp, &sol).collect();
        for _ in 0..40 {
            let m = <TspTwoOptNeighbor as MoveToNeighbor<TspWithCoordinates>>::random_neighbor(
                &tsp, &sol, &mut rng,
            )
            .unwrap();
            assert!(
                two_opts
                    .iter()
                    .any(|t| t.i == m.i && t.j == m.j && t.gain == m.gain)
            );
        }

        let relocs: Vec<_> = TspRelocateNeighbor::iter(&tsp, &sol).collect();
        for _ in 0..40 {
            let m = <TspRelocateNeighbor as MoveToNeighbor<TspWithCoordinates>>::random_neighbor(
                &tsp, &sol, &mut rng,
            )
            .unwrap();
            assert!(
                relocs
                    .iter()
                    .any(|r| r.pos == m.pos && r.ins == m.ins && r.gain == m.gain)
            );
        }
    }

    #[test]
    fn test_new_matches_iter() {
        let tsp = make_hex_tsp();
        let sol = make_sol(&tsp, vec![0, 2, 4, 1, 5, 3]);

        for m in TspTwoOptNeighbor::iter(&tsp, &sol) {
            let rebuilt = TspTwoOptNeighbor::new(&tsp, &sol, m.i, m.j);
            assert_eq!(rebuilt.gain, m.gain, "2-opt ({}, {})", m.i, m.j);
        }
        for m in TspRelocateNeighbor::iter(&tsp, &sol) {
            let rebuilt = TspRelocateNeighbor::new(&tsp, &sol, m.pos, m.ins);
            assert_eq!(
                rebuilt.gain, m.gain,
                "relocate (pos={}, ins={})",
                m.pos, m.ins
            );
        }
    }

    #[test]
    #[should_panic(expected = "leave the tour unchanged")]
    fn test_relocate_new_rejects_identity_move() {
        let tsp = make_hex_tsp();
        let sol = make_sol(&tsp, vec![0, 2, 4, 1, 5, 3]);
        // ins is the predecessor of pos: the tour would not move, but the gain
        // would come out non-zero.
        let _ = TspRelocateNeighbor::new(&tsp, &sol, 2, 1);
    }

    #[test]
    fn test_random_neighbor_none_when_too_small() {
        use rand::SeedableRng;
        let tsp =
            TspWithCoordinates::new("tiny".to_string(), vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]);
        let sol = make_sol(&tsp, vec![0, 1, 2]);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
        // n=3: the 2-opt neighborhood is empty (iter yields nothing).
        assert!(TspTwoOptNeighbor::iter(&tsp, &sol).next().is_none());
        assert!(
            <TspTwoOptNeighbor as MoveToNeighbor<TspWithCoordinates>>::random_neighbor(
                &tsp, &sol, &mut rng
            )
            .is_none()
        );
    }
}
