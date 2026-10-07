//! Bin packing's answer to [`Ruinable`], with items as the elements and bins
//! as the containers.
//!
//! This is the case the trait was written around, several containers that
//! compete for capacity and that come and go. Every open bin is a bucket, and
//! one more empty bin is always offered after them, so a repair never runs out
//! of somewhere to put an item. An item that does not fit a bin costs infinity
//! there, which the empty bin never does, so a repair never overfills. Regret
//! means something here, the gap between an item's best bin and its next best.

use super::problem::{BinPacking, BinPackingSolution};
use crate::trait_defs::Ruinable;

/// Marks an item that is not in any bin.
const NOT_PLACED: usize = usize::MAX;

/// A packing mid-ruin, with some items out of every bin.
///
/// No bin in it is empty. Taking the last item out of a bin closes the bin and
/// renumbers the rest, so the open bins are always `0..bins.len()`.
#[derive(Clone)]
pub struct BinPackingPartial {
    /// The items in each open bin.
    bins: Vec<Vec<usize>>,
    /// The load of each open bin.
    loads: Vec<u64>,
    /// The bin of each item, or [`NOT_PLACED`].
    bin_of: Vec<usize>,
    /// `Σ (load / capacity)²` over the open bins.
    fill: f64,
}

impl BinPackingPartial {
    fn energy(&self) -> f64 {
        BinPacking::energy_of(self.loads.len(), self.fill)
    }
}

impl Ruinable for BinPacking {
    type Element = usize;
    type Partial = BinPackingPartial;

    fn to_partial(&self, sol: &BinPackingSolution) -> BinPackingPartial {
        BinPackingPartial {
            bins: sol.bins(),
            loads: sol.loads.clone(),
            bin_of: sol.bin_of.clone(),
            fill: sol.loads.iter().map(|&l| self.sq(l)).sum(),
        }
    }

    fn finish(&self, partial: &BinPackingPartial) -> BinPackingSolution {
        self.solution_from_bins(partial.bin_of.clone())
    }

    fn elements(&self, partial: &BinPackingPartial, out: &mut Vec<usize>) {
        out.clear();
        out.extend(partial.bins.iter().flatten().copied());
    }

    fn num_elements(&self, partial: &BinPackingPartial) -> usize {
        partial.bins.iter().map(Vec::len).sum()
    }

    /// Takes the items out, closes the bins that empties, and renumbers the
    /// rest. The fill is summed again rather than adjusted.
    fn remove_all(&self, partial: &mut BinPackingPartial, set: &[usize]) {
        for &item in set {
            let b = partial.bin_of[item];
            let slot = partial.bins[b]
                .iter()
                .position(|&i| i == item)
                .expect("a placed item is in its bin");
            partial.bins[b].swap_remove(slot);
            partial.loads[b] -= self.sizes[item];
            partial.bin_of[item] = NOT_PLACED;
        }
        let mut b = 0;
        while b < partial.bins.len() {
            if partial.bins[b].is_empty() {
                partial.bins.swap_remove(b);
                partial.loads.swap_remove(b);
                for &item in partial.bins.get(b).into_iter().flatten() {
                    partial.bin_of[item] = b;
                }
            } else {
                b += 1;
            }
        }
        partial.fill = partial.loads.iter().map(|&l| self.sq(l)).sum();
    }

    fn removal_gain(&self, partial: &BinPackingPartial, element: usize) -> f64 {
        let b = partial.bin_of[element];
        let (before, after) = (partial.loads[b], partial.loads[b] - self.sizes[element]);
        let fill = partial.fill - self.sq(before) + self.sq(after);
        let bins = partial.loads.len() - usize::from(after == 0);
        partial.energy() - BinPacking::energy_of(bins, fill)
    }

    /// Items of similar size, smaller is closer.
    fn relatedness(&self, a: usize, b: usize) -> f64 {
        self.sizes[a].abs_diff(self.sizes[b]) as f64
    }

    /// Every open bin, and one empty bin after them.
    fn num_buckets(&self, partial: &BinPackingPartial) -> usize {
        partial.bins.len() + 1
    }

    /// A bin is a set, so it offers one place.
    fn num_places(&self, _partial: &BinPackingPartial, _bucket: usize) -> usize {
        1
    }

    fn insertion_cost(
        &self,
        partial: &BinPackingPartial,
        bucket: usize,
        _place: usize,
        element: usize,
    ) -> f64 {
        let size = self.sizes[element];
        let opened = bucket == partial.bins.len();
        let before = if opened { 0 } else { partial.loads[bucket] };
        if before + size > self.capacity {
            return f64::INFINITY;
        }
        let fill = partial.fill - self.sq(before) + self.sq(before + size);
        let bins = partial.loads.len() + usize::from(opened);
        BinPacking::energy_of(bins, fill) - partial.energy()
    }

    fn insert(
        &self,
        partial: &mut BinPackingPartial,
        bucket: usize,
        _place: usize,
        element: usize,
    ) {
        if bucket == partial.bins.len() {
            partial.bins.push(Vec::new());
            partial.loads.push(0);
        }
        let before = partial.loads[bucket];
        let after = before + self.sizes[element];
        partial.bins[bucket].push(element);
        partial.loads[bucket] = after;
        partial.bin_of[element] = bucket;
        partial.fill += self.sq(after) - self.sq(before);
    }

    fn partial_energy(&self, partial: &BinPackingPartial) -> f64 {
        partial.energy()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search_state::{Evaluate, ProblemTrait};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn instance() -> BinPacking {
        let mut rng = SmallRng::seed_from_u64(2);
        let sizes = (0..25).map(|_| rng.random_range(20..100)).collect();
        BinPacking::new("random", 150, sizes)
    }

    /// The prices agree with the objective of the packing they lead to.
    #[test]
    fn ruin_prices_match_the_objective() {
        let bp = instance();
        let sol = bp.new_solution(&mut SmallRng::seed_from_u64(3));
        let mut partial = bp.to_partial(&sol);
        assert!((bp.partial_energy(&partial) - sol.evaluate().minimized()).abs() < 1e-9);

        let taken = [0, 5, 9];
        let expected_gain = bp.removal_gain(&partial, 5);
        let alone = {
            let mut p = bp.to_partial(&sol);
            bp.remove_all(&mut p, &[5]);
            bp.partial_energy(&p)
        };
        assert!((bp.partial_energy(&partial) - alone - expected_gain).abs() < 1e-9);

        bp.remove_all(&mut partial, &taken);
        for &item in &taken {
            for bucket in 0..bp.num_buckets(&partial) {
                let cost = bp.insertion_cost(&partial, bucket, 0, item);
                if cost.is_infinite() {
                    continue;
                }
                let mut p = partial.clone();
                bp.insert(&mut p, bucket, 0, item);
                let moved = bp.partial_energy(&p) - bp.partial_energy(&partial);
                assert!((moved - cost).abs() < 1e-9);
            }
            let open = bp.num_buckets(&partial) - 1;
            bp.insert(&mut partial, open, 0, item);
        }
        let done = bp.finish(&partial);
        assert!((done.objective - bp.partial_energy(&partial)).abs() < 1e-9);
    }

    /// A packing ruined to nothing and recreated greedily is a valid packing,
    /// which is what a construction from an empty partial relies on.
    #[test]
    fn recreates_from_an_empty_partial() {
        use crate::building_blocks::search::greedy_insertion;

        let bp = instance();
        let sol = bp.new_solution(&mut SmallRng::seed_from_u64(4));
        let mut partial = bp.to_partial(&sol);
        let all: Vec<usize> = (0..bp.num_items()).collect();
        bp.remove_all(&mut partial, &all);
        assert_eq!(bp.num_buckets(&partial), 1);
        greedy_insertion(&bp, &mut partial, all, &mut SmallRng::seed_from_u64(5));
        let done = bp.finish(&partial);
        assert!(done.loads.iter().all(|&l| l <= bp.capacity));
        assert!(done.num_bins() as u64 >= bp.lower_bound());
    }
}
