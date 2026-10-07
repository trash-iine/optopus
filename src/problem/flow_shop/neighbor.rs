use std::iter::once;

use rand::rngs::SmallRng;

use super::problem::{FlowShop, FlowShopSolution};
use super::taillard::HeadsTails;
use crate::building_blocks::search::{TabuKey, TabuMemory, random_distinct_pair};
use crate::error::OptError;
use crate::search_state::{EnabledTabu, Evaluable, Evaluate, MoveToNeighbor};

/// A makespan moved by a move's gain. Makespans are integers and a gain is
/// the difference of two, so the sum is exact.
fn shifted(objective: u32, gain: f64) -> u32 {
    (f64::from(objective) + gain) as u32
}

/// Moves one job to another place in the sequence, the insertion move.
///
/// The move takes the job at `from` out and puts it back at `to`, counted in
/// the sequence without it, so `to == from` would leave it where it was and is
/// never generated. Insertion is the neighborhood the flow shop literature
/// relies on. [`iter`](MoveToNeighbor::iter) prices all `n (n - 1)`
/// insertions in O(n²m) with Taillard's heads and tails, one O(nm)
/// tabulation per job taken out.
#[derive(Debug, Clone)]
pub struct FlowShopInsertNeighbor {
    /// The job that moves.
    pub job: usize,
    /// Its position before the move.
    pub from: usize,
    /// Its position after the move.
    pub to: usize,
    /// Change in makespan (negative = improvement).
    pub gain: f64,
}

impl FlowShopInsertNeighbor {
    /// Builds the insertion of the job at `from` into place `to`, pricing it
    /// with one O(nm) schedule.
    ///
    /// # Panics
    ///
    /// Panics if `from` or `to` is outside the sequence.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// let fs = FlowShop::new("tiny", vec![vec![3, 2, 4], vec![2, 5, 1]]);
    /// let sol = fs.solution_from_sequence(vec![0, 1, 2]);
    /// let m = FlowShopInsertNeighbor::new(&fs, &sol, 1, 0);
    ///
    /// let mut after = sol.clone();
    /// m.apply_to_solution(&fs, &mut after).unwrap();
    /// assert_eq!(after.sequence, vec![1, 0, 2]);
    /// assert_eq!(after.objective as f64, sol.objective as f64 + m.gain);
    /// ```
    pub fn new(prob: &FlowShop, sol: &FlowShopSolution, from: usize, to: usize) -> Self {
        let mut moved = sol.sequence.clone();
        let job = moved.remove(from);
        moved.insert(to, job);
        Self {
            job,
            from,
            to,
            gain: f64::from(prob.makespan(&moved)) - f64::from(sol.objective),
        }
    }
}

impl Evaluate for FlowShopInsertNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain)
    }
}

impl EnabledTabu for FlowShopInsertNeighbor {
    /// The move is tabu while the job it moves is.
    fn is_move_enabled(&self, tabu: &TabuMemory, iteration: u64) -> bool {
        tabu.is_enabled(TabuKey::DenseVar(self.job), iteration)
    }

    /// Moving a job forbids moving it again for a tenure the memory draws.
    fn add_to_tabu_map(&self, tabu: &mut TabuMemory, iteration: u64, rng: &mut SmallRng) {
        tabu.forbid(TabuKey::DenseVar(self.job), iteration, rng);
    }
}

impl MoveToNeighbor<FlowShop> for FlowShopInsertNeighbor {
    /// Hands this move's [`EnabledTabu`] policy to the search state, which is
    /// what holds the tabu map.
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }

    /// Compares through the cached `gain`, exact since the costs are
    /// integers, relative to the solution the move was built from (`src`).
    fn move_to_be_better_than(
        &self,
        _prob: &FlowShop,
        src: &FlowShopSolution,
        other: &FlowShopSolution,
    ) -> bool {
        self.evaluate()
            .improves_over(src.evaluate(), other.evaluate())
    }

    fn apply_to_solution(
        &self,
        prob: &FlowShop,
        sol: &mut FlowShopSolution,
    ) -> Result<(), OptError> {
        let job = sol.sequence.remove(self.from);
        sol.sequence.insert(self.to, job);
        sol.objective = shifted(sol.objective, self.gain);
        debug_assert_eq!(sol.objective, prob.makespan(&sol.sequence));
        Ok(())
    }

    fn iter(prob: &FlowShop, sol: &FlowShopSolution) -> impl Iterator<Item = Self> + Send {
        let n = sol.sequence.len();
        let base = f64::from(sol.objective);
        let mut moves = Vec::with_capacity(n * n.saturating_sub(1));
        let mut rest = Vec::with_capacity(n);
        let mut ht = HeadsTails::default();
        for from in 0..n {
            let job = sol.sequence[from];
            rest.clear();
            rest.extend_from_slice(&sol.sequence[..from]);
            rest.extend_from_slice(&sol.sequence[from + 1..]);
            ht.rebuild(prob, &rest);
            moves.extend((0..n).filter(|&to| to != from).map(|to| Self {
                job,
                from,
                to,
                gain: f64::from(ht.makespan_with(prob, to, job)) - base,
            }));
        }
        moves.into_iter()
    }

    /// An ordered pair of distinct positions, uniform over the `n (n - 1)`
    /// moves [`iter`](MoveToNeighbor::iter) yields.
    fn random_neighbor(
        prob: &FlowShop,
        sol: &FlowShopSolution,
        rng: &mut SmallRng,
    ) -> Option<Self> {
        let (from, to) = random_distinct_pair(sol.sequence.len(), rng)?;
        Some(Self::new(prob, sol, from, to))
    }
}

/// Exchanges the jobs at two positions.
///
/// Priced from the heads before the first position and the tails after the
/// second, so only the stretch between them is rescheduled.
#[derive(Debug, Clone)]
pub struct FlowShopSwapNeighbor {
    /// The earlier position.
    pub i: usize,
    /// The later position.
    pub j: usize,
    /// Change in makespan (negative = improvement).
    pub gain: f64,
}

impl FlowShopSwapNeighbor {
    /// Builds the exchange of positions `i < j`, pricing it with one O(nm)
    /// schedule.
    ///
    /// # Panics
    ///
    /// Panics if `i` or `j` is outside the sequence.
    pub fn new(prob: &FlowShop, sol: &FlowShopSolution, i: usize, j: usize) -> Self {
        let (i, j) = (i.min(j), i.max(j));
        let mut swapped = sol.sequence.clone();
        swapped.swap(i, j);
        Self {
            i,
            j,
            gain: f64::from(prob.makespan(&swapped)) - f64::from(sol.objective),
        }
    }
}

impl Evaluate for FlowShopSwapNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain)
    }
}

impl EnabledTabu for FlowShopSwapNeighbor {
    /// Keyed by the two positions, so swapping them back is what is blocked.
    fn is_move_enabled(&self, tabu: &TabuMemory, iteration: u64) -> bool {
        tabu.is_enabled((self.i, self.j), iteration)
    }

    /// Applying it forbids that pair of positions.
    fn add_to_tabu_map(&self, tabu: &mut TabuMemory, iteration: u64, rng: &mut SmallRng) {
        tabu.forbid((self.i, self.j), iteration, rng);
    }
}

impl MoveToNeighbor<FlowShop> for FlowShopSwapNeighbor {
    /// Hands this move's [`EnabledTabu`] policy to the search state, which is
    /// what holds the tabu map.
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }

    /// Compares through the cached `gain`, exact since the costs are
    /// integers, relative to the solution the move was built from (`src`).
    fn move_to_be_better_than(
        &self,
        _prob: &FlowShop,
        src: &FlowShopSolution,
        other: &FlowShopSolution,
    ) -> bool {
        self.evaluate()
            .improves_over(src.evaluate(), other.evaluate())
    }

    fn apply_to_solution(
        &self,
        prob: &FlowShop,
        sol: &mut FlowShopSolution,
    ) -> Result<(), OptError> {
        sol.sequence.swap(self.i, self.j);
        sol.objective = shifted(sol.objective, self.gain);
        debug_assert_eq!(sol.objective, prob.makespan(&sol.sequence));
        Ok(())
    }

    fn iter(prob: &FlowShop, sol: &FlowShopSolution) -> impl Iterator<Item = Self> + Send {
        let seq = &sol.sequence;
        let n = seq.len();
        let base = f64::from(sol.objective);
        let mut ht = HeadsTails::default();
        ht.rebuild(prob, seq);
        let mut moves = Vec::with_capacity(n * n.saturating_sub(1) / 2);
        let mut row = Vec::with_capacity(prob.n_machines);
        for i in 0..n {
            for j in i + 1..n {
                let middle = once(seq[j])
                    .chain(seq[i + 1..j].iter().copied())
                    .chain(once(seq[i]));
                let makespan = ht.makespan_replacing(prob, i, j, middle, &mut row);
                moves.push(Self {
                    i,
                    j,
                    gain: f64::from(makespan) - base,
                });
            }
        }
        moves.into_iter()
    }

    fn random_neighbor(
        prob: &FlowShop,
        sol: &FlowShopSolution,
        rng: &mut SmallRng,
    ) -> Option<Self> {
        let (i, j) = random_distinct_pair(sol.sequence.len(), rng)?;
        Some(Self::new(prob, sol, i, j))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    fn instance() -> FlowShop {
        let mut rng = SmallRng::seed_from_u64(5);
        let rows = (0..4)
            .map(|_| (0..7).map(|_| rng.random_range(1..50)).collect())
            .collect();
        FlowShop::new("random", rows)
    }

    /// The batched prices of `iter` agree with building each move alone, and
    /// applying a move lands on the makespan its gain promised.
    #[test]
    fn iter_prices_match_single_moves() {
        let fs = instance();
        let sol = fs.solution_from_sequence(vec![3, 0, 6, 1, 5, 2, 4]);

        let inserts: Vec<_> = FlowShopInsertNeighbor::iter(&fs, &sol).collect();
        assert_eq!(inserts.len(), 7 * 6);
        for m in &inserts {
            let single = FlowShopInsertNeighbor::new(&fs, &sol, m.from, m.to);
            assert_eq!(m.gain, single.gain);
            let mut after = sol.clone();
            m.apply_to_solution(&fs, &mut after).unwrap();
            assert_eq!(
                f64::from(after.objective),
                f64::from(sol.objective) + m.gain
            );
        }

        let swaps: Vec<_> = FlowShopSwapNeighbor::iter(&fs, &sol).collect();
        assert_eq!(swaps.len(), 7 * 6 / 2);
        for m in &swaps {
            assert_eq!(m.gain, FlowShopSwapNeighbor::new(&fs, &sol, m.i, m.j).gain);
        }
    }

    #[test]
    fn random_insert_never_draws_the_identity() {
        let fs = instance();
        let sol = fs.solution_from_sequence((0..7).collect());
        let mut rng = SmallRng::seed_from_u64(1);
        for _ in 0..200 {
            let m = FlowShopInsertNeighbor::random_neighbor(&fs, &sol, &mut rng).unwrap();
            assert_ne!(m.from, m.to);
        }
    }
}
