//! Epoch-stamped index marker: a set of indices that clears in O(1).
//!
//! The graph operators that walk a neighborhood — a BFS over a cluster, an
//! independent-set selection — need "have I already seen this vertex?" many
//! times per call and a fresh set on every call. Clearing a `Vec<bool>` or a
//! `HashSet` costs O(n) per call, which dominates when the walk itself touches
//! only a handful of vertices.
//!
//! Stamping each entry with a generation counter instead makes the clear a
//! single increment: an index counts as marked only while its stamp equals the
//! current epoch, so bumping the epoch invalidates every stamp at once. The
//! only O(n) work is the wrap-around every 2^32 generations.

/// A set of indices in `0..capacity`, cleared in O(1) by
/// [`next_epoch`](Self::next_epoch).
///
/// ```
/// use optopus::common::EpochMarks;
///
/// let mut marks = EpochMarks::new();
/// marks.ensure_capacity(4);
///
/// marks.next_epoch();              // start a fresh generation
/// assert!(!marks.is_marked(2));
/// marks.mark(2);
/// assert!(marks.is_marked(2));
///
/// marks.next_epoch();              // O(1) clear
/// assert!(!marks.is_marked(2));
/// ```
#[derive(Debug, Clone)]
pub struct EpochMarks {
    /// Per-index generation stamp: index `i` is marked iff `stamps[i] == epoch`.
    stamps: Vec<u32>,
    /// The current generation. Never 0 — that value is reserved for "this entry
    /// has never been marked", which is what [`ensure_capacity`](Self::ensure_capacity)
    /// fills new entries with. Starting at 1 rather than 0 is what makes a
    /// freshly constructed marker read as empty instead of full.
    epoch: u32,
}

impl Default for EpochMarks {
    fn default() -> Self {
        Self::new()
    }
}

impl EpochMarks {
    /// Creates an empty marker. Call [`ensure_capacity`](Self::ensure_capacity)
    /// before marking.
    pub fn new() -> Self {
        Self {
            stamps: Vec::new(),
            epoch: 1,
        }
    }

    /// Grows the marker to hold indices `0..n`. Never shrinks.
    ///
    /// New entries are stamped 0, which no live epoch ever equals, so they
    /// start unmarked.
    pub fn ensure_capacity(&mut self, n: usize) {
        if self.stamps.len() < n {
            self.stamps.resize(n, 0);
        }
    }

    /// Starts a fresh generation, clearing every mark. O(1) except on the
    /// wrap-around every 2^32 generations, where the stamps are refilled.
    pub fn next_epoch(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamps.fill(0);
            self.epoch = 1;
        }
    }

    /// Returns whether `i` is marked in the current generation. Out-of-range
    /// indices read as unmarked.
    #[inline]
    pub fn is_marked(&self, i: usize) -> bool {
        self.stamps.get(i) == Some(&self.epoch)
    }

    /// Marks `i` for the current generation. Out-of-range indices are ignored.
    #[inline]
    pub fn mark(&mut self, i: usize) {
        if let Some(stamp) = self.stamps.get_mut(i) {
            *stamp = self.epoch;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marks_are_scoped_to_the_current_epoch() {
        let mut marks = EpochMarks::new();
        marks.ensure_capacity(4);
        marks.next_epoch();

        assert!(!marks.is_marked(0));
        marks.mark(0);
        marks.mark(2);
        assert!(marks.is_marked(0));
        assert!(marks.is_marked(2));
        assert!(!marks.is_marked(1));

        marks.next_epoch();
        for i in 0..4 {
            assert!(!marks.is_marked(i), "{i} survived the epoch bump");
        }
    }

    /// A fresh marker must read as empty. It would not if the initial epoch
    /// were 0, since that is also the stamp `ensure_capacity` fills in — every
    /// index would come back marked before a single `mark` call.
    #[test]
    fn nothing_is_marked_before_the_first_epoch() {
        let mut marks = EpochMarks::new();
        marks.ensure_capacity(3);
        for i in 0..3 {
            assert!(!marks.is_marked(i));
        }
        assert!(!EpochMarks::default().is_marked(0));
    }

    /// Growing must not resurrect marks: a new entry starts unmarked even
    /// though the live epoch has already been used.
    #[test]
    fn growing_leaves_new_entries_unmarked() {
        let mut marks = EpochMarks::new();
        marks.ensure_capacity(2);
        marks.next_epoch();
        marks.mark(1);

        marks.ensure_capacity(5);
        assert!(marks.is_marked(1), "existing marks must survive a grow");
        for i in 2..5 {
            assert!(!marks.is_marked(i), "new entry {i} must start unmarked");
        }
    }

    /// Out-of-range access is a no-op rather than a panic, mirroring how the
    /// callers size the marker from the graph and then index it by vertex.
    #[test]
    fn out_of_range_indices_are_inert() {
        let mut marks = EpochMarks::new();
        marks.ensure_capacity(2);
        marks.next_epoch();
        marks.mark(9);
        assert!(!marks.is_marked(9));
    }

    /// The wrap-around path: `epoch` must skip 0 and every stale stamp must be
    /// cleared, or an index marked 2^32 generations ago would read as marked.
    /// Driving this through an operator would take four billion perturbations.
    #[test]
    fn wrap_around_clears_stale_stamps() {
        let mut marks = EpochMarks::new();
        marks.ensure_capacity(2);

        // Land on the last epoch before the wrap and mark index 0 there.
        marks.epoch = u32::MAX;
        marks.mark(0);
        assert!(marks.is_marked(0));

        marks.next_epoch();
        assert_eq!(marks.epoch, 1, "epoch must skip 0 on wrap-around");
        assert!(!marks.is_marked(0), "stale stamp survived the wrap-around");

        // And the marker is still usable afterwards.
        marks.mark(1);
        assert!(marks.is_marked(1));
    }
}
