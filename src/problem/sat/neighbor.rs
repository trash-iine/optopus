use super::problem::{Sat, SatSolution};
use crate::{
    common::{VarTabuMap, add_var_to_tabu, is_var_enabled},
    error::OptError,
    search_state::{EnabledTabu, Evaluable, Evaluate, MoveToNeighbor, Rankable},
};
use rand::Rng;

/// A flip move that toggles a single variable `i`.
///
/// `gain` is the change in the number of satisfied clauses after the flip
/// (positive = improvement, since MaxSAT is maximized).
#[derive(Debug, Clone)]
pub struct SatFlipNeighbor {
    /// Index of the variable to flip.
    pub i: usize,
    /// Change in satisfied-clause count when this variable is flipped (positive = improvement).
    pub gain: i64,
}

impl Rankable for SatFlipNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain > other.gain
    }
}

impl EnabledTabu for SatFlipNeighbor {
    type TabuMap = VarTabuMap;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        is_var_enabled(tabu_map, self.i, iteration)
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
        rng: &mut rand::rngs::SmallRng,
    ) {
        add_var_to_tabu(tabu_map, self.i, iteration, tabu_tenure, rng);
    }
}

impl Evaluate for SatFlipNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Maximize(self.gain as f64)
    }
}

impl MoveToNeighbor<Sat> for SatFlipNeighbor {
    fn apply_to_solution(&self, prob: &Sat, sol: &mut SatSolution) -> Result<(), OptError> {
        // Flip x[i]
        sol.x[self.i] = !sol.x[self.i];

        // Update n_satisfied
        sol.n_satisfied = (sol.n_satisfied as i64 + self.gain) as usize;

        // Update gain[i]: flipping again reverts to the original (sign flip)
        sol.gain[self.i] = -self.gain;

        // Recompute gain for variables sharing a clause with i.
        // `prob.var_neighbors(i)` is precomputed at problem-construction time.
        for &j in prob.var_neighbors(self.i) {
            sol.gain[j] = prob.calc_gain(&sol.x, j);
        }
        Ok(())
    }

    fn iter(prob: &Sat, sol: &SatSolution) -> impl Iterator<Item = Self> + Send {
        let n = prob.n_vars();
        (0..n).map(move |i| SatFlipNeighbor {
            i,
            gain: sol.gain[i],
        })
    }

    fn move_to_be_better_than(&self, _: &Sat, src: &SatSolution, other: &SatSolution) -> bool {
        (src.n_satisfied as i64 + self.gain) > other.n_satisfied as i64
    }

    /// O(1): picks a uniformly random variable.
    fn random_neighbor(
        prob: &Sat,
        sol: &SatSolution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Option<Self> {
        let n = prob.n_vars();
        if n == 0 {
            return None;
        }
        let i = rng.random_range(0..n);
        Some(Self {
            i,
            gain: sol.gain[i],
        })
    }
}

/// A swap move that simultaneously flips variables `i` and `j`.
///
/// Only pairs that appear together in at least one clause are enumerated,
/// which reduces the search space relative to all O(n²) pairs.
/// `gain` is the combined change in satisfied-clause count (positive = improvement).
#[derive(Debug, Clone)]
pub struct SatSwapNeighbor {
    /// Index of the first variable to flip.
    pub i: usize,
    /// Index of the second variable to flip.
    pub j: usize,
    /// Combined change in satisfied-clause count (positive = improvement).
    pub gain: i64,
}

impl Rankable for SatSwapNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain > other.gain
    }
}

impl EnabledTabu for SatSwapNeighbor {
    type TabuMap = VarTabuMap;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        is_var_enabled(tabu_map, self.i, iteration) && is_var_enabled(tabu_map, self.j, iteration)
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
        rng: &mut rand::rngs::SmallRng,
    ) {
        add_var_to_tabu(tabu_map, self.i, iteration, tabu_tenure, rng);
        add_var_to_tabu(tabu_map, self.j, iteration, tabu_tenure, rng);
    }
}

impl Evaluate for SatSwapNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Maximize(self.gain as f64)
    }
}

impl MoveToNeighbor<Sat> for SatSwapNeighbor {
    fn apply_to_iteration(&self, iter: u64) -> u64 {
        iter + 2
    }

    fn apply_to_solution(&self, prob: &Sat, sol: &mut SatSolution) -> Result<(), OptError> {
        // Flip i first, then flip j using the updated gain[j] after the first flip
        crate::common::apply_swap_as_two_flips(prob, sol, self.i, self.j)
    }

    /// Iterates the precomputed clause-sharing pairs lazily (no per-call
    /// allocation); the pair list itself is cached on the problem.
    fn iter(prob: &Sat, sol: &SatSolution) -> impl Iterator<Item = Self> + Send {
        prob.clause_sharing_pairs().iter().map(move |&(i, j)| {
            // gain_swap = gain_i + gain_j_after_flip_i
            let gain_j_after_flip_i = prob.calc_gain_with_virtual_flip(&sol.x, i, j);
            SatSwapNeighbor {
                i,
                j,
                gain: sol.gain[i] + gain_j_after_flip_i,
            }
        })
    }

    fn move_to_be_better_than(&self, _: &Sat, src: &SatSolution, other: &SatSolution) -> bool {
        (src.n_satisfied as i64 + self.gain) > other.n_satisfied as i64
    }

    /// O(clause): picks a uniformly random pair from the precomputed
    /// clause-sharing pair list and computes its gain.
    fn random_neighbor(
        prob: &Sat,
        sol: &SatSolution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Option<Self> {
        let pairs = prob.clause_sharing_pairs();
        if pairs.is_empty() {
            return None;
        }
        let (i, j) = pairs[rng.random_range(0..pairs.len())];
        let gain_j_after_flip_i = prob.calc_gain_with_virtual_flip(&sol.x, i, j);
        Some(Self {
            i,
            j,
            gain: sol.gain[i] + gain_j_after_flip_i,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search_state::SearchState;

    /// (x1 ∨ x2), (¬x1 ∨ x3), (¬x2 ∨ ¬x3)
    fn make_sat() -> Sat {
        let mut sat = Sat::new(3);
        sat.add_clause([1, 2]);
        sat.add_clause([-1, 3]);
        sat.add_clause([-2, -3]);
        sat
    }

    fn make_solution(sat: &Sat, x: Vec<bool>) -> SatSolution {
        let gain: Vec<i64> = (0..sat.n_vars()).map(|i| sat.calc_gain(&x, i)).collect();
        let n_satisfied = sat.calc_satisfied(&x);
        SatSolution {
            x,
            gain,
            n_satisfied,
        }
    }

    #[test]
    fn test_search_state_new() {
        let sat = make_sat();
        let _state = SearchState::new(&sat);
    }

    #[test]
    fn test_flip_gain_matches_energy_delta() {
        let sat = make_sat();
        let sol = make_solution(&sat, vec![true, false, true]);

        for neighbor in SatFlipNeighbor::iter(&sat, &sol) {
            let mut x2 = sol.x.clone();
            x2[neighbor.i] = !x2[neighbor.i];
            let expected_delta = sat.calc_satisfied(&x2) as i64 - sol.n_satisfied as i64;
            assert_eq!(
                neighbor.gain, expected_delta,
                "flip {}: gain={} expected={}",
                neighbor.i, neighbor.gain, expected_delta
            );
        }
    }

    #[test]
    fn test_flip_apply_consistency() {
        let sat = make_sat();
        let sol = make_solution(&sat, vec![true, false, true]);

        for neighbor in SatFlipNeighbor::iter(&sat, &sol) {
            let mut s = sol.clone();
            neighbor.apply_to_solution(&sat, &mut s).unwrap();

            let expected_n_sat = sat.calc_satisfied(&s.x);
            assert_eq!(
                s.n_satisfied, expected_n_sat,
                "flip {}: n_satisfied={} expected={}",
                neighbor.i, s.n_satisfied, expected_n_sat
            );

            for i in 0..sat.n_vars() {
                let expected_gain = sat.calc_gain(&s.x, i);
                assert_eq!(
                    s.gain[i], expected_gain,
                    "flip {}: gain[{}]={} expected={}",
                    neighbor.i, i, s.gain[i], expected_gain
                );
            }
        }
    }

    #[test]
    fn test_swap_gain_matches_energy_delta() {
        let sat = make_sat();
        let sol = make_solution(&sat, vec![true, false, true]);

        for neighbor in SatSwapNeighbor::iter(&sat, &sol) {
            let mut x2 = sol.x.clone();
            x2[neighbor.i] = !x2[neighbor.i];
            x2[neighbor.j] = !x2[neighbor.j];
            let expected_delta = sat.calc_satisfied(&x2) as i64 - sol.n_satisfied as i64;
            assert_eq!(
                neighbor.gain, expected_delta,
                "swap ({},{}): gain={} expected={}",
                neighbor.i, neighbor.j, neighbor.gain, expected_delta
            );
        }
    }

    #[test]
    fn test_swap_apply_consistency() {
        let sat = make_sat();
        let sol = make_solution(&sat, vec![true, false, true]);

        for neighbor in SatSwapNeighbor::iter(&sat, &sol) {
            let mut s = sol.clone();
            neighbor.apply_to_solution(&sat, &mut s).unwrap();

            let expected_n_sat = sat.calc_satisfied(&s.x);
            assert_eq!(
                s.n_satisfied, expected_n_sat,
                "swap ({},{}): n_satisfied={} expected={}",
                neighbor.i, neighbor.j, s.n_satisfied, expected_n_sat
            );
        }
    }

    #[test]
    fn test_random_neighbor_samples_member_of_iter() {
        use rand::SeedableRng;
        let sat = make_sat();
        let sol = make_solution(&sat, vec![true, false, true]);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);

        let flips: Vec<_> = SatFlipNeighbor::iter(&sat, &sol).collect();
        for _ in 0..20 {
            let m = <SatFlipNeighbor as MoveToNeighbor<Sat>>::random_neighbor(&sat, &sol, &mut rng)
                .unwrap();
            assert!(flips.iter().any(|f| f.i == m.i && f.gain == m.gain));
        }

        let swaps: Vec<_> = SatSwapNeighbor::iter(&sat, &sol).collect();
        for _ in 0..20 {
            let m = <SatSwapNeighbor as MoveToNeighbor<Sat>>::random_neighbor(&sat, &sol, &mut rng)
                .unwrap();
            assert!(
                swaps
                    .iter()
                    .any(|s| s.i == m.i && s.j == m.j && s.gain == m.gain)
            );
        }
    }

    #[test]
    fn test_random_swap_neighbor_none_without_shared_clauses() {
        use rand::SeedableRng;
        // Two variables that never share a clause: no swap pairs exist.
        let mut sat = Sat::new(2);
        sat.add_clause([1]);
        sat.add_clause([-2]);
        let sol = make_solution(&sat, vec![true, false]);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
        assert!(SatSwapNeighbor::iter(&sat, &sol).next().is_none());
        assert!(
            <SatSwapNeighbor as MoveToNeighbor<Sat>>::random_neighbor(&sat, &sol, &mut rng)
                .is_none()
        );
    }

    #[test]
    fn test_clause_sharing_pairs_reset_by_add_clause() {
        let mut sat = make_sat();
        assert_eq!(sat.clause_sharing_pairs(), &[(0, 1), (0, 2), (1, 2)]);
        // A new clause introducing no new pair keeps the set; one with a new
        // pair (via a 4th variable) must appear after the cache reset.
        let mut sat4 = Sat::new(4);
        sat4.add_clause([1, 2]);
        assert_eq!(sat4.clause_sharing_pairs(), &[(0, 1)]);
        sat4.add_clause([3, 4]);
        assert_eq!(sat4.clause_sharing_pairs(), &[(0, 1), (2, 3)]);
        // make_sat-based instance: adding a clause resets its cache too.
        sat.add_clause([1, 2, 3]);
        assert_eq!(sat.clause_sharing_pairs(), &[(0, 1), (0, 2), (1, 2)]);
    }
}
