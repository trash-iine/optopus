use rand::Rng;
use rand::rngs::SmallRng;
use rand::seq::IteratorRandom;

use super::problem::{BinPacking, BinPackingSolution};
use crate::building_blocks::search::{TabuKey, TabuMemory};
use crate::error::OptError;
use crate::search_state::{EnabledTabu, Evaluable, Evaluate, MoveToNeighbor};

/// Draws before a random move falls back to listing the neighborhood.
///
/// A drawn pair is uniform over all pairs and kept only if feasible, which is
/// uniform over the feasible ones. When most pairs are infeasible the draws
/// rarely land, and the fallback lists them instead, which is uniform too.
const RANDOM_DRAWS: usize = 64;

/// `Σ (load / capacity)²` of a packing, read back from its objective
/// `B − S / B`, so a move can be priced without a pass over the bins.
fn fill_of(sol: &BinPackingSolution) -> f64 {
    let b = sol.num_bins() as f64;
    b * (b - sol.objective)
}

/// The objective after bin loads change, `changes` listing each touched bin's
/// load before and after.
fn energy_after(prob: &BinPacking, sol: &BinPackingSolution, changes: &[(u64, u64)]) -> f64 {
    let c = prob.capacity as f64;
    let sq = |l: u64| (l as f64 / c).powi(2);
    let mut fill = fill_of(sol);
    let mut bins = sol.num_bins();
    for &(before, after) in changes {
        fill += sq(after) - sq(before);
        if after == 0 {
            bins -= 1;
        }
    }
    if bins == 0 {
        0.0
    } else {
        bins as f64 - fill / bins as f64
    }
}

/// Empties bin `gone` by giving its number to the last bin, so the bins stay
/// numbered `0..num_bins()`, then recomputes the objective.
fn settle(prob: &BinPacking, sol: &mut BinPackingSolution) {
    if let Some(gone) = sol.loads.iter().position(|&l| l == 0) {
        let last = sol.loads.len() - 1;
        sol.loads.swap(gone, last);
        sol.loads.pop();
        for b in sol.bin_of.iter_mut().filter(|b| **b == last) {
            *b = gone;
        }
    }
    sol.objective = prob.energy(sol.loads.iter().copied());
}

/// Moves one item into another bin that has room for it.
///
/// Emptying a bin is how the bin count falls, and the move does it whenever it
/// takes the last item out.
#[derive(Debug, Clone)]
pub struct BinPackingRelocateNeighbor {
    /// The item that moves.
    pub item: usize,
    /// The bin it moves to.
    pub to: usize,
    /// Change in the objective (negative = improvement).
    pub gain: f64,
}

impl BinPackingRelocateNeighbor {
    /// Builds the move of `item` into bin `to`, or `None` if it does not fit or
    /// is already there.
    pub fn new(
        prob: &BinPacking,
        sol: &BinPackingSolution,
        item: usize,
        to: usize,
    ) -> Option<Self> {
        let from = sol.bin_of[item];
        let size = prob.sizes[item];
        if to == from || sol.loads[to] + size > prob.capacity {
            return None;
        }
        let changes = [
            (sol.loads[from], sol.loads[from] - size),
            (sol.loads[to], sol.loads[to] + size),
        ];
        Some(Self {
            item,
            to,
            gain: energy_after(prob, sol, &changes) - sol.objective,
        })
    }
}

impl Evaluate for BinPackingRelocateNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain)
    }
}

impl EnabledTabu for BinPackingRelocateNeighbor {
    /// The move is tabu while the item it moves is.
    fn is_move_enabled(&self, tabu: &TabuMemory, iteration: u64) -> bool {
        tabu.is_enabled(TabuKey::DenseVar(self.item), iteration)
    }

    /// Moving an item forbids moving it again for a tenure the memory draws.
    fn add_to_tabu_map(&self, tabu: &mut TabuMemory, iteration: u64, rng: &mut SmallRng) {
        tabu.forbid(TabuKey::DenseVar(self.item), iteration, rng);
    }
}

impl MoveToNeighbor<BinPacking> for BinPackingRelocateNeighbor {
    /// Hands this move's [`EnabledTabu`] policy to the search state, which is
    /// what holds the tabu map.
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }

    /// Compares through the cached `gain`, relative to the solution the move
    /// was built from (`src`).
    fn move_to_be_better_than(
        &self,
        _prob: &BinPacking,
        src: &BinPackingSolution,
        other: &BinPackingSolution,
    ) -> bool {
        self.evaluate()
            .improves_over(src.evaluate(), other.evaluate())
    }

    fn apply_to_solution(
        &self,
        prob: &BinPacking,
        sol: &mut BinPackingSolution,
    ) -> Result<(), OptError> {
        let size = prob.sizes[self.item];
        let from = sol.bin_of[self.item];
        sol.loads[from] -= size;
        sol.loads[self.to] += size;
        sol.bin_of[self.item] = self.to;
        settle(prob, sol);
        Ok(())
    }

    fn iter(prob: &BinPacking, sol: &BinPackingSolution) -> impl Iterator<Item = Self> + Send {
        let bins = sol.num_bins();
        (0..prob.num_items())
            .flat_map(move |item| (0..bins).map(move |to| (item, to)))
            .filter_map(move |(item, to)| Self::new(prob, sol, item, to))
    }

    fn random_neighbor(
        prob: &BinPacking,
        sol: &BinPackingSolution,
        rng: &mut SmallRng,
    ) -> Option<Self> {
        if sol.num_bins() < 2 {
            return None;
        }
        for _ in 0..RANDOM_DRAWS {
            let item = rng.random_range(0..prob.num_items());
            let to = rng.random_range(0..sol.num_bins());
            if let Some(m) = Self::new(prob, sol, item, to) {
                return Some(m);
            }
        }
        Self::iter(prob, sol).choose(rng)
    }
}

/// Exchanges two items of different sizes in different bins, when both bins
/// still fit.
///
/// It keeps the bin count and moves load from one bin to the other, which is
/// how a nearly full bin gets the last bit of room filled.
#[derive(Debug, Clone)]
pub struct BinPackingSwapNeighbor {
    /// The first item, `i < j`.
    pub i: usize,
    /// The second item.
    pub j: usize,
    /// Change in the objective (negative = improvement).
    pub gain: f64,
}

impl BinPackingSwapNeighbor {
    /// Builds the exchange of items `i` and `j`, or `None` if they share a
    /// bin, have the same size, or one bin would overflow.
    pub fn new(prob: &BinPacking, sol: &BinPackingSolution, i: usize, j: usize) -> Option<Self> {
        let (i, j) = (i.min(j), i.max(j));
        let (a, b) = (sol.bin_of[i], sol.bin_of[j]);
        let (si, sj) = (prob.sizes[i], prob.sizes[j]);
        if a == b || si == sj {
            return None;
        }
        let (la, lb) = (sol.loads[a] - si + sj, sol.loads[b] - sj + si);
        if la > prob.capacity || lb > prob.capacity {
            return None;
        }
        let changes = [(sol.loads[a], la), (sol.loads[b], lb)];
        Some(Self {
            i,
            j,
            gain: energy_after(prob, sol, &changes) - sol.objective,
        })
    }
}

impl Evaluate for BinPackingSwapNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain)
    }
}

impl EnabledTabu for BinPackingSwapNeighbor {
    /// Keyed by the two items, so swapping them back is what is blocked.
    fn is_move_enabled(&self, tabu: &TabuMemory, iteration: u64) -> bool {
        tabu.is_enabled((self.i, self.j), iteration)
    }

    /// Applying it forbids that pair.
    fn add_to_tabu_map(&self, tabu: &mut TabuMemory, iteration: u64, rng: &mut SmallRng) {
        tabu.forbid((self.i, self.j), iteration, rng);
    }
}

impl MoveToNeighbor<BinPacking> for BinPackingSwapNeighbor {
    /// Hands this move's [`EnabledTabu`] policy to the search state, which is
    /// what holds the tabu map.
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }

    /// Compares through the cached `gain`, relative to the solution the move
    /// was built from (`src`).
    fn move_to_be_better_than(
        &self,
        _prob: &BinPacking,
        src: &BinPackingSolution,
        other: &BinPackingSolution,
    ) -> bool {
        self.evaluate()
            .improves_over(src.evaluate(), other.evaluate())
    }

    fn apply_to_solution(
        &self,
        prob: &BinPacking,
        sol: &mut BinPackingSolution,
    ) -> Result<(), OptError> {
        let (a, b) = (sol.bin_of[self.i], sol.bin_of[self.j]);
        let (si, sj) = (prob.sizes[self.i], prob.sizes[self.j]);
        sol.loads[a] = sol.loads[a] - si + sj;
        sol.loads[b] = sol.loads[b] - sj + si;
        sol.bin_of.swap(self.i, self.j);
        sol.objective = prob.energy(sol.loads.iter().copied());
        Ok(())
    }

    fn iter(prob: &BinPacking, sol: &BinPackingSolution) -> impl Iterator<Item = Self> + Send {
        let n = prob.num_items();
        (0..n)
            .flat_map(move |i| (i + 1..n).map(move |j| (i, j)))
            .filter_map(move |(i, j)| Self::new(prob, sol, i, j))
    }

    fn random_neighbor(
        prob: &BinPacking,
        sol: &BinPackingSolution,
        rng: &mut SmallRng,
    ) -> Option<Self> {
        let n = prob.num_items();
        if n < 2 || sol.num_bins() < 2 {
            return None;
        }
        for _ in 0..RANDOM_DRAWS {
            let (i, j) = crate::building_blocks::search::random_distinct_pair(n, rng)?;
            if let Some(m) = Self::new(prob, sol, i, j) {
                return Some(m);
            }
        }
        Self::iter(prob, sol).choose(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search_state::ProblemTrait;
    use rand::SeedableRng;

    fn instance() -> BinPacking {
        let mut rng = SmallRng::seed_from_u64(9);
        let sizes = (0..30).map(|_| rng.random_range(20..100)).collect();
        BinPacking::new("random", 150, sizes)
    }

    /// Every move lands on the objective its gain promised, keeps the bins
    /// within capacity and numbered compactly.
    #[test]
    fn moves_land_on_the_objective_they_promise() {
        let bp = instance();
        let sol = bp.new_solution(&mut SmallRng::seed_from_u64(1));
        let check = |after: &BinPackingSolution, gain: f64| {
            // Rebuilding renumbers the bins, so the loads are compared as sets.
            let rebuilt = bp.solution_from_bins(after.bin_of.clone());
            let sorted = |loads: &[u64]| {
                let mut l = loads.to_vec();
                l.sort_unstable();
                l
            };
            assert_eq!(sorted(&rebuilt.loads), sorted(&after.loads));
            assert!(after.bin_of.iter().all(|&b| b < after.num_bins()));
            assert!((after.objective - rebuilt.objective).abs() < 1e-9);
            assert!((after.objective - (sol.objective + gain)).abs() < 1e-9);
        };
        let relocates: Vec<_> = BinPackingRelocateNeighbor::iter(&bp, &sol).collect();
        assert!(!relocates.is_empty());
        for m in relocates {
            let mut after = sol.clone();
            m.apply_to_solution(&bp, &mut after).unwrap();
            check(&after, m.gain);
        }
        for m in BinPackingSwapNeighbor::iter(&bp, &sol) {
            let mut after = sol.clone();
            m.apply_to_solution(&bp, &mut after).unwrap();
            check(&after, m.gain);
        }
    }

    /// Emptying a bin by a relocation lowers the count and renumbers.
    #[test]
    fn emptying_a_bin_drops_it() {
        let bp = BinPacking::new("tiny", 10, vec![6, 3, 1]);
        let sol = bp.solution_from_bins(vec![0, 1, 2]);
        let m = BinPackingRelocateNeighbor::new(&bp, &sol, 2, 1).unwrap();
        let mut after = sol.clone();
        m.apply_to_solution(&bp, &mut after).unwrap();
        assert_eq!(after.num_bins(), 2);
        assert!(m.gain < 0.0);
        assert!(after.bin_of.iter().all(|&b| b < 2));
    }
}
