//! The scan over two nested index ranges the pair moves list their
//! neighborhoods with.

use std::ops::Range;

/// The moves `build(row, i, j)` gives, for `i` in `0..n` and `j` in the range
/// `inner(i)` returns, in that order, skipping the pairs where it gives
/// `None`. `inner(i)` also returns what every pair of row `i` shares, built
/// once per row and handed to `build` as `row`.
///
/// It lists what `(0..n).flat_map(|i| inner(i).map(move |j| (i, j)))`
/// followed by a `filter_map` over `build` lists, as one iterator whose
/// `next` and `fold` run as plain loops over the two indices.
pub fn pair_scan<T, C, R, B>(n: usize, inner: R, build: B) -> PairScan<C, R, B>
where
    R: Fn(usize) -> (C, Range<usize>),
    B: Fn(&C, usize, usize) -> Option<T>,
{
    PairScan {
        n,
        i: 0,
        j: 0,
        row: None,
        inner,
        build,
    }
}

/// The iterator [`pair_scan`] returns. While row `i` is under way, `row`
/// holds what it shares and where its range ends, and `j` is its next pair.
pub struct PairScan<C, R, B> {
    n: usize,
    i: usize,
    j: usize,
    row: Option<(C, usize)>,
    inner: R,
    build: B,
}

impl<T, C, R, B> Iterator for PairScan<C, R, B>
where
    R: Fn(usize) -> (C, Range<usize>),
    B: Fn(&C, usize, usize) -> Option<T>,
{
    type Item = T;

    // The rows are walked in locals, from ranges `inner` has just built, and
    // the position is written back only when a move is found. A search that
    // rejects most pairs, `filter` looking for its first match, spends the
    // scan here, and a position kept in `self` is stored and the row's
    // operands reloaded on every pair. The rest of the current row is walked
    // apart from the rows after it, so those run from their start the way
    // `fold` runs them (decisions/0020).
    #[inline]
    fn next(&mut self) -> Option<T> {
        let mut i = self.i;
        if let Some((shared, end)) = &self.row {
            for j in self.j..*end {
                if let Some(m) = (self.build)(shared, i, j) {
                    self.j = j + 1;
                    return Some(m);
                }
            }
            self.row = None;
            i += 1;
        }
        while i < self.n {
            let (shared, Range { start, end }) = (self.inner)(i);
            for j in start..end {
                if let Some(m) = (self.build)(&shared, i, j) {
                    (self.i, self.j, self.row) = (i, j + 1, Some((shared, end)));
                    return Some(m);
                }
            }
            i += 1;
        }
        self.i = i;
        None
    }

    fn fold<A, F: FnMut(A, T) -> A>(self, init: A, mut f: F) -> A {
        let mut acc = init;
        let mut i = self.i;
        if let Some((shared, end)) = self.row {
            for j in self.j..end {
                if let Some(m) = (self.build)(&shared, i, j) {
                    acc = f(acc, m);
                }
            }
            i += 1;
        }
        for i in i..self.n {
            let (shared, range) = (self.inner)(i);
            for j in range {
                if let Some(m) = (self.build)(&shared, i, j) {
                    acc = f(acc, m);
                }
            }
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scan lists what the nested adapters list, whether it is read by
    /// `next`, by `fold`, or by `next` up to any point and `fold` after,
    /// with empty inner ranges and skipped pairs among them.
    #[test]
    fn pair_scan_lists_what_flat_map_lists() {
        for n in 0..7 {
            let range = |i: usize| i + 1..n.max(i + 1).min(i + 4);
            let inner = |i: usize| (10 * i, range(i));
            let build = |&shared: &usize, i: usize, j: usize| {
                (!(i + j).is_multiple_of(3)).then_some((shared, i, j))
            };
            let expected: Vec<_> = (0..n)
                .flat_map(|i| range(i).map(move |j| (i, j)))
                .filter_map(|(i, j)| build(&(10 * i), i, j))
                .collect();
            let by_next: Vec<_> = {
                let mut it = pair_scan(n, inner, build);
                std::iter::from_fn(|| it.next()).collect()
            };
            assert_eq!(by_next, expected, "n = {n}");

            for k in 0..=expected.len() {
                let mut it = pair_scan(n, inner, build);
                let head: Vec<_> = it.by_ref().take(k).collect();
                let all = it.fold(head, |mut acc, m| {
                    acc.push(m);
                    acc
                });
                assert_eq!(all, expected, "n = {n}, switched after {k}");
            }
        }
    }
}
