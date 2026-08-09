//! Permutation crossover operators shared by sequencing problems.

use rand::Rng;
use rand::rngs::SmallRng;

/// Order crossover (OX) over two permutations of the same elements.
///
/// A random contiguous segment of `parent1` is copied to the child at its own
/// positions; the remaining slots are filled with the elements `parent1` did not
/// contribute, in the relative order they appear in `parent2`, starting just
/// after the segment and wrapping around. The result is a permutation of the
/// same elements whenever both parents are.
///
/// Used by the giant-tour encodings: [`crate::problem::VrpOrderCrossover`] and
/// Hybrid Genetic Search.
///
/// # Panics
/// Panics if the parents differ in length.
pub fn order_crossover(parent1: &[usize], parent2: &[usize], rng: &mut SmallRng) -> Vec<usize> {
    assert_eq!(
        parent1.len(),
        parent2.len(),
        "order crossover needs parents of equal length"
    );
    let n = parent1.len();
    if n == 0 {
        return Vec::new();
    }

    let a = rng.random_range(0..n);
    let b = rng.random_range(0..n);
    let (start, end) = if a <= b { (a, b) } else { (b, a) };

    let width = parent1.iter().copied().max().unwrap_or(0) + 1;
    let mut in_segment = vec![false; width];
    let mut child = vec![usize::MAX; n];
    for (slot, &element) in child[start..=end].iter_mut().zip(&parent1[start..=end]) {
        *slot = element;
        in_segment[element] = true;
    }

    let mut slot = (end + 1) % n;
    let mut source = (end + 1) % n;
    let mut filled = end - start + 1;
    while filled < n {
        let element = parent2[source];
        if !in_segment[element] {
            child[slot] = element;
            slot = (slot + 1) % n;
            filled += 1;
        }
        source = (source + 1) % n;
    }
    child
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn is_permutation_of(child: &[usize], parent: &[usize]) -> bool {
        let mut a = child.to_vec();
        let mut b = parent.to_vec();
        a.sort_unstable();
        b.sort_unstable();
        a == b
    }

    #[test]
    fn offspring_is_a_permutation() {
        let p1: Vec<usize> = (0..12).collect();
        let p2: Vec<usize> = (0..12).rev().collect();
        let mut rng = SmallRng::seed_from_u64(11);
        for _ in 0..100 {
            let child = order_crossover(&p1, &p2, &mut rng);
            assert!(is_permutation_of(&child, &p1));
        }
    }

    #[test]
    fn identical_parents_reproduce_themselves() {
        let p: Vec<usize> = vec![3, 1, 4, 0, 5, 9, 2, 6];
        let mut rng = SmallRng::seed_from_u64(5);
        for _ in 0..20 {
            assert_eq!(order_crossover(&p, &p, &mut rng), p);
        }
    }

    #[test]
    fn empty_parents_yield_an_empty_child() {
        let mut rng = SmallRng::seed_from_u64(0);
        assert!(order_crossover(&[], &[], &mut rng).is_empty());
    }

    #[test]
    #[should_panic(expected = "equal length")]
    fn mismatched_parents_panic() {
        let mut rng = SmallRng::seed_from_u64(0);
        order_crossover(&[0, 1], &[0], &mut rng);
    }
}
