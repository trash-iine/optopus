use crate::trait_defs::Rankable;
use rand::Rng;
use rand::rngs::SmallRng;
use rand::seq::IteratorRandom;

use super::problem::{BinPacking, BinPackingSolution};
use crate::building_blocks::search::pair_scan;
use crate::error::OptError;
use crate::search_state::{EnabledTabu, Evaluate, MoveToNeighbor};

/// Draws before a random move falls back to listing the neighborhood.
///
/// A drawn pair is uniform over all pairs and kept only if feasible, which is
/// uniform over the feasible ones. When most pairs are infeasible the draws
/// rarely land, and the fallback counts them instead, which is uniform too.
const RANDOM_DRAWS: usize = 64;

/// The objective and fill term once the loads of the touched bins change,
/// `changes` listing each one's load before and after, in O(1). A bin whose
/// load was 0 before is a new one.
fn priced(prob: &BinPacking, sol: &BinPackingSolution, changes: &[(u64, u64)]) -> (f64, f64) {
    let mut fill = sol.fill;
    let mut bins = sol.num_bins();
    for &(before, after) in changes {
        fill += prob.sq(after) - prob.sq(before);
        if before == 0 {
            bins += 1;
        }
        if after == 0 {
            bins -= 1;
        }
    }
    (BinPacking::energy_of(bins, fill), fill)
}

/// Moves one item into another bin that has room for it, or into a new bin of
/// its own.
///
/// Emptying a bin is how the bin count falls, and the move does it whenever it
/// takes the last item out. The new bin, numbered `num_bins()`, is offered to
/// every item that shares its bin, so the neighborhood is never empty while
/// some bin holds two items, however tight the packing.
#[derive(Debug, Clone, Rankable, Evaluate, EnabledTabu)]
#[tabu(dense(item))]
pub struct BinPackingRelocateNeighbor {
    /// The item that moves.
    pub item: usize,
    /// The bin it moves to, `num_bins()` for a new one.
    pub to: usize,
    /// Change in the objective (negative = improvement).
    #[objective(minimize)]
    pub gain: f64,
}

impl BinPackingRelocateNeighbor {
    /// Builds the move of `item` into bin `to`, `num_bins()` opening a new one,
    /// or `None` if it does not fit, is already there, or would only move a
    /// bin's sole item to a new bin.
    pub fn new(
        prob: &BinPacking,
        sol: &BinPackingSolution,
        item: usize,
        to: usize,
    ) -> Option<Self> {
        let from = sol.bin_of[item];
        let size = prob.sizes[item];
        let to_load = match sol.loads.get(to) {
            Some(&load) => load,
            None if to == sol.num_bins() && sol.loads[from] > size => 0,
            None => return None,
        };
        if to == from || to_load + size > prob.capacity {
            return None;
        }
        let changes = [
            (sol.loads[from], sol.loads[from] - size),
            (to_load, to_load + size),
        ];
        Some(Self {
            item,
            to,
            gain: priced(prob, sol, &changes).0 - sol.objective,
        })
    }

    /// A move drawn uniformly from the feasible ones, or `None` if there are
    /// none, without listing them.
    ///
    /// With the bins sorted by load, the bins an item fits into are a prefix,
    /// so each item's count is one binary search, plus the new bin when the
    /// item shares its bin, and the draw costs O(n log B) rather than the
    /// O(n B) of the full neighborhood.
    fn uniform_feasible(
        prob: &BinPacking,
        sol: &BinPackingSolution,
        rng: &mut SmallRng,
    ) -> Option<Self> {
        let mut order: Vec<usize> = (0..sol.num_bins()).collect();
        order.sort_unstable_by_key(|&b| sol.loads[b]);
        let fits_in = |item: usize| {
            let Some(room) = prob.capacity.checked_sub(prob.sizes[item]) else {
                return 0;
            };
            let k = order.partition_point(|&b| sol.loads[b] <= room);
            let from_load = sol.loads[sol.bin_of[item]];
            k - usize::from(from_load <= room) + usize::from(from_load > prob.sizes[item])
        };
        let total: usize = (0..prob.num_items()).map(fits_in).sum();
        if total == 0 {
            return None;
        }
        let mut r = rng.random_range(0..total);
        for item in 0..prob.num_items() {
            let k = fits_in(item);
            if r >= k {
                r -= k;
                continue;
            }
            let from = sol.bin_of[item];
            // Past the existing bins it fits into, the last move is the new bin.
            let room = prob.capacity - prob.sizes[item];
            let to = order
                .iter()
                .copied()
                .take_while(|&b| sol.loads[b] <= room)
                .filter(|&b| b != from)
                .nth(r)
                .unwrap_or(sol.num_bins());
            return Self::new(prob, sol, item, to);
        }
        unreachable!("the draw is below the total count")
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
        let to_load = sol.loads.get(self.to).copied().unwrap_or(0);
        let changes = [
            (sol.loads[from], sol.loads[from] - size),
            (to_load, to_load + size),
        ];
        (sol.objective, sol.fill) = priced(prob, sol, &changes);
        if self.to == sol.num_bins() {
            sol.loads.push(0);
        }
        sol.loads[from] -= size;
        sol.loads[self.to] += size;
        sol.bin_of[self.item] = self.to;
        // Close an emptied bin by giving its number to the last one, so the
        // bins stay numbered `0..num_bins()`.
        if sol.loads[from] == 0 {
            let last = sol.loads.len() - 1;
            sol.loads.swap_remove(from);
            for b in sol.bin_of.iter_mut().filter(|b| **b == last) {
                *b = from;
            }
        }
        Ok(())
    }

    fn iter(prob: &BinPacking, sol: &BinPackingSolution) -> impl Iterator<Item = Self> + Send {
        pair_scan(
            prob.num_items(),
            move |_| ((), 0..sol.num_bins() + 1),
            move |_, item, to| Self::new(prob, sol, item, to),
        )
    }

    fn random_neighbor(
        prob: &BinPacking,
        sol: &BinPackingSolution,
        rng: &mut SmallRng,
    ) -> Option<Self> {
        if prob.num_items() == 0 {
            return None;
        }
        for _ in 0..RANDOM_DRAWS {
            let item = rng.random_range(0..prob.num_items());
            let to = rng.random_range(0..=sol.num_bins());
            if let Some(m) = Self::new(prob, sol, item, to) {
                return Some(m);
            }
        }
        Self::uniform_feasible(prob, sol, rng)
    }
}

/// Exchanges two items of different sizes in different bins, when both bins
/// still fit.
///
/// It keeps the bin count and moves load from one bin to the other, which is
/// how a nearly full bin gets the last bit of room filled.
#[derive(Debug, Clone, Rankable, Evaluate, EnabledTabu)]
#[tabu(pair(i, j))]
pub struct BinPackingSwapNeighbor {
    /// The first item, `i < j`.
    pub i: usize,
    /// The second item.
    pub j: usize,
    /// Change in the objective (negative = improvement).
    #[objective(minimize)]
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
            gain: priced(prob, sol, &changes).0 - sol.objective,
        })
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
        let (la, lb) = (sol.loads[a] - si + sj, sol.loads[b] - sj + si);
        (sol.objective, sol.fill) = priced(prob, sol, &[(sol.loads[a], la), (sol.loads[b], lb)]);
        (sol.loads[a], sol.loads[b]) = (la, lb);
        sol.bin_of.swap(self.i, self.j);
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
            let rebuilt = bp.solution_from_bins(after.bin_of.clone()).unwrap();
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
        let sol = bp.solution_from_bins(vec![0, 1, 2]).unwrap();
        let m = BinPackingRelocateNeighbor::new(&bp, &sol, 2, 1).unwrap();
        let mut after = sol.clone();
        m.apply_to_solution(&bp, &mut after).unwrap();
        assert_eq!(after.num_bins(), 2);
        assert!(m.gain < 0.0);
        assert!(after.bin_of.iter().all(|&b| b < 2));
    }

    /// The draw without listing reaches exactly the moves the listing has,
    /// each about equally often.
    #[test]
    fn uniform_draw_covers_the_listed_moves_evenly() {
        let bp = instance();
        let sol = bp.new_solution(&mut SmallRng::seed_from_u64(3));
        let listed: std::collections::HashSet<_> = BinPackingRelocateNeighbor::iter(&bp, &sol)
            .map(|m| (m.item, m.to))
            .collect();
        let mut rng = SmallRng::seed_from_u64(4);
        let draws = 200 * listed.len();
        let mut counts = std::collections::HashMap::new();
        for _ in 0..draws {
            let m = BinPackingRelocateNeighbor::uniform_feasible(&bp, &sol, &mut rng).unwrap();
            *counts.entry((m.item, m.to)).or_insert(0usize) += 1;
        }
        assert_eq!(
            counts
                .keys()
                .copied()
                .collect::<std::collections::HashSet<_>>(),
            listed
        );
        assert!(counts.values().all(|&c| (100..300).contains(&c)));

        // Two full bins leave only the new bin, which every item is offered.
        let tight = BinPacking::new("full", 10, vec![6, 4, 5, 5]);
        let full = tight.solution_from_bins(vec![0, 0, 1, 1]).unwrap();
        let opening: Vec<_> = BinPackingRelocateNeighbor::iter(&tight, &full).collect();
        assert_eq!(opening.len(), 4);
        assert!(opening.iter().all(|m| m.to == 2 && m.gain > 0.0));
        let m = BinPackingRelocateNeighbor::uniform_feasible(&tight, &full, &mut rng).unwrap();
        assert_eq!(m.to, 2);

        // Items alone in their bins with no room elsewhere have no move at all.
        let apart = BinPacking::new("apart", 10, vec![6, 5]);
        let alone = apart.solution_from_bins(vec![0, 1]).unwrap();
        assert!(
            BinPackingRelocateNeighbor::iter(&apart, &alone)
                .next()
                .is_none()
        );
        assert!(BinPackingRelocateNeighbor::uniform_feasible(&apart, &alone, &mut rng).is_none());
    }

    /// Opening a bin adds it at the end and prices the extra bin.
    #[test]
    fn opening_a_bin_adds_it_last() {
        let bp = BinPacking::new("tiny", 10, vec![6, 4]);
        let sol = bp.solution_from_bins(vec![0, 0]).unwrap();
        let m = BinPackingRelocateNeighbor::new(&bp, &sol, 1, 1).unwrap();
        let mut after = sol.clone();
        m.apply_to_solution(&bp, &mut after).unwrap();
        assert_eq!(after.loads, vec![6, 4]);
        assert_eq!(after.bin_of, vec![0, 1]);
        let rebuilt = bp.solution_from_bins(after.bin_of.clone()).unwrap();
        assert!((after.objective - rebuilt.objective).abs() < 1e-9);
        assert!((after.objective - (sol.objective + m.gain)).abs() < 1e-9);
        assert!(BinPackingRelocateNeighbor::new(&bp, &after, 1, 2).is_none());
    }
}
