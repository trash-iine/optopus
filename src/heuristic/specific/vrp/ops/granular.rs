//! Granular candidate lists: which customer pairs a move is allowed to touch.

use crate::problem::Vrp;

/// For each customer, its `granularity` nearest customers in ascending distance.
///
/// Index `0` (the depot) is present but empty so the list can be indexed by
/// customer id directly. Costs O(n²) once per instance; callers cache it.
pub(crate) fn build_neighbor_lists(prob: &Vrp, granularity: usize) -> Vec<Vec<usize>> {
    let n = prob.get_n();
    let mut lists = vec![Vec::new(); n + 1];
    let mut buffer: Vec<(f64, usize)> = Vec::with_capacity(n);
    for (c, list) in lists.iter_mut().enumerate().skip(1) {
        buffer.clear();
        buffer.extend(
            (1..=n)
                .filter(|&o| o != c)
                .map(|o| (prob.distance(c, o), o)),
        );
        let keep = granularity.min(buffer.len());
        if keep < buffer.len() {
            buffer.select_nth_unstable_by(keep, |a, b| a.0.total_cmp(&b.0));
            buffer.truncate(keep);
        }
        buffer.sort_by(|a, b| a.0.total_cmp(&b.0));
        *list = buffer.iter().map(|&(_, o)| o).collect();
    }
    lists
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Customers strung out along a line: the nearest partners of each are its
    /// immediate neighbors, in a known order.
    #[test]
    fn lists_hold_the_nearest_customers_in_order() {
        let prob = Vrp::new(
            "line",
            vec![(0.0, 5.0), (1.0, 0.0), (2.0, 0.0), (4.0, 0.0), (8.0, 0.0)],
            vec![0, 1, 1, 1, 1],
            10,
            1,
        );
        let lists = build_neighbor_lists(&prob, 2);
        assert!(lists[0].is_empty(), "the depot has no candidate partners");
        assert_eq!(lists[1], vec![2, 3]);
        assert_eq!(lists[3], vec![2, 1]);
        // A granularity above n - 1 keeps every other customer.
        assert_eq!(build_neighbor_lists(&prob, 99)[1], vec![2, 3, 4]);
    }
}
