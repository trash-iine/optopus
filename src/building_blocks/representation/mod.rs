//! Helpers shared by the problems whose solutions are laid out the same way.
//!
//! One module per kind of decision variable. Binary vectors get crossover,
//! variable fixing and the paired moves, bounded integers get the ranges that
//! integer problems and branch and bound are stated in, and permutations get
//! order crossover. Hamming distance, which every representation is compared
//! by, sits beside them here.

mod binary;
mod integer;
mod permutation;

pub(crate) use binary::fixed_by;
pub use binary::{
    BinaryFixing, BinaryRelaxation, apply_swap_as_two_flips, binary_node, binary_ranges,
    binary_solution_with, binary_value, differing_pairs, lift_binary_solution,
    lift_compact_binary_solution, random_differing_pair, uniform_binary_crossover, variable_slots,
};
pub use integer::{DomainRestriction, IntVar, IntVars};
pub use permutation::order_crossover;

/// Hamming distance between two sequences of equal length: the number of
/// positions whose values differ.
///
/// The per-problem [`Distance`](crate::trait_defs::Distance) impls all
/// delegate to this function, the binary problems over their assignments and
/// the sequencing problems (TSP, job shop) over their permutations.
pub fn hamming_distance<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamming_distance_counts_disagreements() {
        assert_eq!(hamming_distance::<bool>(&[], &[]), 0);
        assert_eq!(
            hamming_distance(&[true, false, true], &[true, false, true]),
            0
        );
        assert_eq!(
            hamming_distance(&[true, false, true], &[false, false, false]),
            2
        );
    }
}
