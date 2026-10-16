//! The scan over two nested index ranges the pair moves list their
//! neighborhoods with.

use std::ops::Range;

/// The moves `build(i, j)` gives, for `i` in `0..n` and `j` in `inner(i)`, in
/// that order, skipping the pairs where it gives `None`.
///
/// It lists what `(0..n).flat_map(|i| inner(i).map(move |j| (i, j)))`
/// followed by `filter_map(|(i, j)| build(i, j))` lists, as one iterator
/// whose `next` and `fold` run as plain loops over the two indices.
pub fn pair_scan<T, R, B>(n: usize, inner: R, build: B) -> PairScan<R, B>
where
    R: Fn(usize) -> Range<usize>,
    B: Fn(usize, usize) -> Option<T>,
{
    PairScan {
        n,
        i: 0,
        j: 0,
        inner,
        build,
    }
}

/// The iterator [`pair_scan`] returns, at row `i` with the `j`s of
/// `inner(i)` below `j` already listed.
pub struct PairScan<R, B> {
    n: usize,
    i: usize,
    j: usize,
    inner: R,
    build: B,
}

impl<T, R, B> Iterator for PairScan<R, B>
where
    R: Fn(usize) -> Range<usize>,
    B: Fn(usize, usize) -> Option<T>,
{
    type Item = T;

    // The rows are walked in locals, from ranges `inner` has just built, and
    // the position is written back only when a move is found. A search that
    // rejects most pairs, `filter` looking for its first match, spends the
    // scan here, and a position kept in `self` is stored and the row's
    // operands reloaded on every pair. The rest of the current row is walked
    // apart from the rows after it, so those run from their start the way
    // `fold` runs them (decisions/0032).
    #[inline]
    fn next(&mut self) -> Option<T> {
        let mut i = self.i;
        if i >= self.n {
            return None;
        }
        let row = (self.inner)(i);
        for j in row.start.max(self.j)..row.end {
            if let Some(m) = (self.build)(i, j) {
                self.j = j + 1;
                return Some(m);
            }
        }
        i += 1;
        while i < self.n {
            for j in (self.inner)(i) {
                if let Some(m) = (self.build)(i, j) {
                    (self.i, self.j) = (i, j + 1);
                    return Some(m);
                }
            }
            i += 1;
        }
        (self.i, self.j) = (i, 0);
        None
    }

    fn fold<A, F: FnMut(A, T) -> A>(self, init: A, mut f: F) -> A {
        let mut acc = init;
        if self.i < self.n {
            let row = (self.inner)(self.i);
            for j in row.start.max(self.j)..row.end {
                if let Some(m) = (self.build)(self.i, j) {
                    acc = f(acc, m);
                }
            }
        }
        for i in self.i + 1..self.n {
            for j in (self.inner)(i) {
                if let Some(m) = (self.build)(i, j) {
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
            let inner = |i: usize| i + 1..n.max(i + 1).min(i + 4);
            let build = |i: usize, j: usize| (!(i + j).is_multiple_of(3)).then_some((i, j));
            let expected: Vec<_> = (0..n)
                .flat_map(|i| inner(i).map(move |j| (i, j)))
                .filter_map(|(i, j)| build(i, j))
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
