use rand::rngs::SmallRng;

use super::problem::{BinPacking, BinPackingSolution};
use crate::search_state::Crossover;

/// A grouping crossover in the spirit of Falkenauer's.
///
/// Bins, not items, are what a packing passes on. The child takes whole bins
/// from the two parents alternately, fullest first, skipping any bin that holds
/// an item already placed, and packs the items left over by First Fit in
/// decreasing size. The full bins of both parents survive, which item-wise
/// crossovers break apart.
pub struct BinPackingGroupCrossover;

impl Crossover<BinPacking> for BinPackingGroupCrossover {
    fn crossover(
        &mut self,
        prob: &BinPacking,
        sol1: &BinPackingSolution,
        sol2: &BinPackingSolution,
        _rng: &mut SmallRng,
    ) -> Result<BinPackingSolution, crate::error::OptError> {
        let bins_by_fill = |sol: &BinPackingSolution| -> Vec<Vec<usize>> {
            let mut bins = vec![Vec::new(); sol.num_bins()];
            for (item, &b) in sol.bin_of.iter().enumerate() {
                bins[b].push(item);
            }
            let mut order: Vec<usize> = (0..bins.len()).collect();
            order.sort_by_key(|&b| std::cmp::Reverse(sol.loads[b]));
            order
                .into_iter()
                .map(|b| std::mem::take(&mut bins[b]))
                .collect()
        };
        let (first, second) = (bins_by_fill(sol1), bins_by_fill(sol2));

        const UNPLACED: usize = usize::MAX;
        let mut bin_of = vec![UNPLACED; prob.num_items()];
        let mut loads: Vec<u64> = Vec::new();
        let take = |bin: &[usize], bin_of: &mut [usize], loads: &mut Vec<u64>| {
            if bin.iter().all(|&item| bin_of[item] == UNPLACED) {
                for &item in bin {
                    bin_of[item] = loads.len();
                }
                loads.push(bin.iter().map(|&item| prob.sizes[item]).sum());
            }
        };
        for k in 0..first.len().max(second.len()) {
            if let Some(bin) = first.get(k) {
                take(bin, &mut bin_of, &mut loads);
            }
            if let Some(bin) = second.get(k) {
                take(bin, &mut bin_of, &mut loads);
            }
        }

        let mut rest: Vec<usize> = (0..prob.num_items())
            .filter(|&item| bin_of[item] == UNPLACED)
            .collect();
        rest.sort_by_key(|&item| std::cmp::Reverse(prob.sizes[item]));
        for item in rest {
            let size = prob.sizes[item];
            let bin = match loads.iter().position(|&l| l + size <= prob.capacity) {
                Some(b) => b,
                None => {
                    loads.push(0);
                    loads.len() - 1
                }
            };
            loads[bin] += size;
            bin_of[item] = bin;
        }
        Ok(prob.solution_from_bins(bin_of))
    }
}
