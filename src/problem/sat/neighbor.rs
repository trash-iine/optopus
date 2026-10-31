use super::problem::{Sat, SatSolution};
use crate::{
    building_blocks::search::{TabuKey, TabuMemory},
    error::OptError,
    search_state::{EnabledTabu, Evaluable, Evaluate, MoveToNeighbor},
};
use rand::Rng;
use rand::rngs::SmallRng;

/// A flip move that toggles a single variable `i`.
///
/// `gain` is the change in the number of satisfied clauses after the flip
/// (positive = improvement, since MaxSAT is maximized). Applying the move
/// updates the gains through the clauses that hold `i` and no others, in
/// O(total length of those clauses).
#[derive(Debug, Clone)]
pub struct SatFlipNeighbor {
    /// Index of the variable to flip.
    pub i: usize,
    /// Change in satisfied-clause count when this variable is flipped (positive = improvement).
    pub gain: i64,
}

impl SatFlipNeighbor {
    /// Builds the flip of variable `i`, reading its cached gain.
    ///
    /// A flip's gain needs no correction. It is exactly the value the solution
    /// already maintains, so this is [`BinaryProblem::flip_move`](crate::trait_defs::BinaryProblem::flip_move)
    /// spelled like the other constructors, keeping every construction site
    /// on one path, the way [`SatSwapNeighbor::new`] does. `prob` is unused for
    /// that reason and taken only so the two constructors read alike at the call
    /// site.
    pub fn new(_prob: &Sat, sol: &SatSolution, i: usize) -> Self {
        <Sat as crate::trait_defs::BinaryProblem>::flip_move(sol, i)
    }
}

impl EnabledTabu for SatFlipNeighbor {
    /// The move is tabu while variable `i` is still blocked at the current iteration.
    fn is_move_enabled(&self, tabu: &TabuMemory, iteration: u64) -> bool {
        tabu.is_enabled(TabuKey::DenseVar(self.i), iteration)
    }

    /// Applying the move forbids variable `i` for a tenure the memory draws.
    fn add_to_tabu_map(&self, tabu: &mut TabuMemory, iteration: u64, rng: &mut SmallRng) {
        tabu.forbid(TabuKey::DenseVar(self.i), iteration, rng);
    }
}

impl Evaluate for SatFlipNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Maximize(self.gain as f64)
    }
}
crate::rank_by_objective!(SatFlipNeighbor);

impl MoveToNeighbor<Sat> for SatFlipNeighbor {
    /// Hands this move's [`EnabledTabu`] policy to the search state, which is
    /// what holds the tabu map.
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }

    fn apply_to_solution(&self, prob: &Sat, sol: &mut SatSolution) -> Result<(), OptError> {
        // Only the clauses holding i change, so their shares of the gains are
        // taken out under the old value and put back under the new one.
        for &c in prob.clauses_of(self.i) {
            prob.add_clause_shares(c, &sol.x, &mut sol.gain, -1);
        }
        sol.x[self.i] = !sol.x[self.i];
        for &c in prob.clauses_of(self.i) {
            prob.add_clause_shares(c, &sol.x, &mut sol.gain, 1);
        }
        sol.n_satisfied = (sol.n_satisfied as i64 + self.gain) as usize;
        Ok(())
    }

    fn iter(prob: &Sat, sol: &SatSolution) -> impl Iterator<Item = Self> + Send {
        let n = prob.n_vars();
        (0..n).map(move |i| SatFlipNeighbor::new(prob, sol, i))
    }

    fn move_to_be_better_than(&self, _: &Sat, src: &SatSolution, other: &SatSolution) -> bool {
        self.evaluate()
            .improves_over(src.evaluate(), other.evaluate())
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
        Some(Self::new(prob, sol, i))
    }
}

/// A swap move that simultaneously flips variables `i` and `j`.
///
/// Only pairs that appear together in at least one clause are enumerated,
/// which reduces the search space relative to all O(n²) pairs.
/// `gain` is the combined change in satisfied-clause count (positive = improvement).
/// Applying the move performs two flips in sequence (`apply_to_iteration`
/// returns `iter + 2`), each refreshing its own O(clauses containing that
/// variable) set of neighbor gains.
#[derive(Debug, Clone)]
pub struct SatSwapNeighbor {
    /// Index of the first variable to flip.
    pub i: usize,
    /// Index of the second variable to flip.
    pub j: usize,
    /// Combined change in satisfied-clause count (positive = improvement).
    pub gain: i64,
}

impl SatSwapNeighbor {
    /// Builds the swap of `i` and `j`, computing the combined gain.
    ///
    /// The gain is `gain[i]` plus `j`'s gain after `i` has been flipped, not
    /// the two standalone gains: the swap applies the flips in order, and when
    /// the two variables share a clause the second flip sees a different
    /// clause state. Every construction site goes through here so the
    /// virtual flip cannot be skipped at one of them.
    pub fn new(prob: &Sat, sol: &SatSolution, i: usize, j: usize) -> Self {
        let gain_j_after_flip_i = prob.calc_gain_with_virtual_flip(&sol.x, i, j);
        Self {
            i,
            j,
            gain: sol.gain[i] + gain_j_after_flip_i,
        }
    }
}

impl EnabledTabu for SatSwapNeighbor {
    /// A swap is tabu unless both variables it moves are free.
    fn is_move_enabled(&self, tabu: &TabuMemory, iteration: u64) -> bool {
        tabu.is_enabled(TabuKey::DenseVar(self.i), iteration)
            && tabu.is_enabled(TabuKey::DenseVar(self.j), iteration)
    }

    /// Applying the swap forbids both variables, each for its own drawn tenure.
    fn add_to_tabu_map(&self, tabu: &mut TabuMemory, iteration: u64, rng: &mut SmallRng) {
        tabu.forbid(TabuKey::DenseVar(self.i), iteration, rng);
        tabu.forbid(TabuKey::DenseVar(self.j), iteration, rng);
    }
}

impl Evaluate for SatSwapNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Maximize(self.gain as f64)
    }
}
crate::rank_by_objective!(SatSwapNeighbor);

/// Whether two clause-sharing pairs have the same `i`, which is what splits
/// the pair list into the runs [`SwapScan`] prices one at a time.
fn same_i(a: &(usize, usize), b: &(usize, usize)) -> bool {
    a.0 == b.0
}

type Runs<'a> =
    std::slice::ChunkBy<'a, (usize, usize), fn(&(usize, usize), &(usize, usize)) -> bool>;

/// The swap scan, a run of clause-sharing pairs with the same `i` at a time.
/// `shift` holds the prices of `run`'s `i`, set when the run starts and
/// cleared when it ends.
///
/// `fold`, which is how `max_by` and `filter` drive a scan, runs as two plain
/// loops.
struct SwapScan<'a, 's> {
    prob: &'a Sat,
    sol: &'s SatSolution,
    x: Vec<bool>,
    shift: Vec<i64>,
    runs: Runs<'a>,
    run: &'a [(usize, usize)],
}

impl SwapScan<'_, '_> {
    /// The change each clause of `i` makes to the gains of the variables it
    /// holds when `i` flips.
    fn price(&mut self, i: usize) {
        let prob = self.prob;
        for &c in prob.clauses_of(i) {
            prob.add_clause_shares(c, &self.x, &mut self.shift, -1);
        }
        self.x[i] = !self.x[i];
        for &c in prob.clauses_of(i) {
            prob.add_clause_shares(c, &self.x, &mut self.shift, 1);
        }
        self.x[i] = !self.x[i];
    }

    fn clear(&mut self, i: usize) {
        for &c in self.prob.clauses_of(i) {
            for &lit in self.prob.clause(c) {
                self.shift[lit.unsigned_abs() as usize - 1] = 0;
            }
        }
    }
}

impl Iterator for SwapScan<'_, '_> {
    type Item = SatSwapNeighbor;

    fn next(&mut self) -> Option<SatSwapNeighbor> {
        loop {
            if let Some((&(i, j), rest)) = self.run.split_first() {
                self.run = rest;
                let gain = self.sol.gain[i] + self.sol.gain[j] + self.shift[j];
                if rest.is_empty() {
                    self.clear(i);
                }
                return Some(SatSwapNeighbor { i, j, gain });
            }
            self.run = self.runs.next()?;
            self.price(self.run[0].0);
        }
    }

    fn fold<B, F: FnMut(B, Self::Item) -> B>(mut self, init: B, mut f: F) -> B {
        let mut acc = init;
        if let Some(&(i, _)) = self.run.first() {
            for &(i, j) in self.run {
                let gain = self.sol.gain[i] + self.sol.gain[j] + self.shift[j];
                acc = f(acc, SatSwapNeighbor { i, j, gain });
            }
            self.clear(i);
        }
        while let Some(run) = self.runs.next() {
            let i = run[0].0;
            self.price(i);
            for &(i, j) in run {
                let gain = self.sol.gain[i] + self.sol.gain[j] + self.shift[j];
                acc = f(acc, SatSwapNeighbor { i, j, gain });
            }
            self.clear(i);
        }
        acc
    }
}

impl MoveToNeighbor<Sat> for SatSwapNeighbor {
    /// Hands this move's [`EnabledTabu`] policy to the search state, which is
    /// what holds the tabu map.
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }

    fn apply_to_iteration(&self, iter: u64) -> u64 {
        iter + 2
    }

    fn apply_to_solution(&self, prob: &Sat, sol: &mut SatSolution) -> Result<(), OptError> {
        // Flip i first, then flip j using the updated gain[j] after the first flip
        crate::building_blocks::representation::apply_swap_as_two_flips(prob, sol, self.i, self.j)
    }

    /// Iterates the precomputed clause-sharing pairs, the list cached on the
    /// problem, a run of pairs with the same `i` at a time.
    ///
    /// Flipping `i` changes the gains of the variables sharing a clause with
    /// it through those clauses only, so one pass over the clauses of `i`
    /// prices `j`'s gain after the flip for every `j` of the run, the same
    /// value [`SatSwapNeighbor::new`] reads with one walk of `j`'s clauses per
    /// pair.
    fn iter(prob: &Sat, sol: &SatSolution) -> impl Iterator<Item = Self> + Send {
        SwapScan {
            prob,
            sol,
            x: sol.x.clone(),
            shift: vec![0i64; prob.n_vars()],
            runs: prob.clause_sharing_pairs().chunk_by(same_i),
            run: &[],
        }
    }

    fn move_to_be_better_than(&self, _: &Sat, src: &SatSolution, other: &SatSolution) -> bool {
        self.evaluate()
            .improves_over(src.evaluate(), other.evaluate())
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
        Some(Self::new(prob, sol, i, j))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// (x1 ∨ x2), (¬x1 ∨ x3), (¬x2 ∨ ¬x3)
    fn make_sat() -> Sat {
        let mut sat = Sat::new(3);
        sat.add_clause([1, 2]);
        sat.add_clause([-1, 3]);
        sat.add_clause([-2, -3]);
        sat
    }

    fn make_solution(sat: &Sat, x: Vec<bool>) -> SatSolution {
        SatSolution::new_from_assignment(sat, x)
    }

    /// Clauses that hold a variable together with its negation, which are
    /// always satisfied, and clauses that repeat a literal.
    fn make_degenerate_sat() -> Sat {
        let mut sat = Sat::new(3);
        sat.add_clause([1, -1]);
        sat.add_clause([2, 2]);
        sat.add_clause([1, -1, 2]);
        sat.add_clause([-3, 3, -3]);
        sat.add_clause([1, 2, 3]);
        sat.add_clause([-2, -2, 3]);
        sat
    }

    #[test]
    fn flips_and_swaps_keep_counts_with_tautologies_and_repeated_literals() {
        use rand::{Rng, SeedableRng};
        let sat = make_degenerate_sat();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let mut sol = make_solution(&sat, vec![false; 3]);
        for _ in 0..200 {
            let i = rng.random_range(0..3);
            if rng.random_bool(0.5) {
                SatFlipNeighbor::new(&sat, &sol, i)
                    .apply_to_solution(&sat, &mut sol)
                    .unwrap();
            } else {
                SatSwapNeighbor::new(&sat, &sol, i, (i + 1) % 3)
                    .apply_to_solution(&sat, &mut sol)
                    .unwrap();
            }
            assert_eq!(sol.n_satisfied, sat.calc_satisfied(&sol.x));
            for k in 0..3 {
                let mut flipped = sol.x.clone();
                flipped[k] = !flipped[k];
                let gain = sat.calc_satisfied(&flipped) as i64 - sol.n_satisfied as i64;
                assert_eq!(sol.gain[k], gain, "gain of variable {k} at {:?}", sol.x);
            }
        }
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

    /// The scan that prices a run of pairs from one pass over the clauses of
    /// their first variable agrees with pricing every pair on its own, on a
    /// random formula and on one with tautologies and repeated literals,
    /// whether it is read by `next`, by `fold`, or by both in turn.
    #[test]
    fn swap_scan_prices_every_pair_as_new_does() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(8);
        let mut random = Sat::new(12);
        for _ in 0..40 {
            let lits: Vec<i64> = (0..3)
                .map(|_| {
                    let v = rng.random_range(1..=12);
                    if rng.random_bool(0.5) { v } else { -v }
                })
                .collect();
            random.add_clause(lits);
        }
        for sat in [random, make_degenerate_sat()] {
            for _ in 0..10 {
                let x = (0..sat.n_vars()).map(|_| rng.random_bool(0.5)).collect();
                let sol = make_solution(&sat, x);
                let scanned: Vec<_> = SatSwapNeighbor::iter(&sat, &sol)
                    .map(|m| (m.i, m.j, m.gain))
                    .collect();
                let priced: Vec<_> = sat
                    .clause_sharing_pairs()
                    .iter()
                    .map(|&(i, j)| (i, j, SatSwapNeighbor::new(&sat, &sol, i, j).gain))
                    .collect();
                assert_eq!(scanned, priced);
                // `collect` reads by `next`, `fold` by its own loops, and a
                // scan may switch from the first to the second at any pair.
                for k in 0..=priced.len() {
                    let mut it = SatSwapNeighbor::iter(&sat, &sol);
                    let head: Vec<_> = it.by_ref().take(k).map(|m| (m.i, m.j, m.gain)).collect();
                    let all = it.fold(head, |mut acc, m| {
                        acc.push((m.i, m.j, m.gain));
                        acc
                    });
                    assert_eq!(all, priced, "switched after {k}");
                }
            }
        }
    }
}
