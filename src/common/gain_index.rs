//! Incrementally maintained index of the variables a predicate currently holds for.

/// **Advanced.** Unordered set of the variables whose cached gain currently
/// satisfies a caller-supplied predicate, with O(1) insert/remove via an
/// inverse position index.
///
/// The predicate itself lives at the call site: callers report the old and new
/// status through [`update`](Self::update) and this type only maintains the
/// membership bookkeeping.
///
/// Standard heuristics do not need this index; it exists for problem-specific
/// algorithms that iterate only over a marked subset of the variables, and it
/// costs one O(1) membership update per gain change once enabled.
///
/// MaxCut's `zero_gain` is its only consumer: the set of `gain == 0` vertices,
/// which feeds
/// [`PopulationAnnealingForMaxCut`](crate::heuristic::PopulationAnnealingForMaxCut)'s
/// non-local cluster moves. There were two *improving*-move indexes as well —
/// MaxCut's `positive_gain` and QUBO's `negative_gain` — and both were removed
/// once nothing read them: `positive_gain`'s last reader was Breakout Local
/// Search's own descent, replaced by [`LocalSearch`](crate::heuristic::LocalSearch)
/// at a measured 1.15x (see `docs/heuristics/breakout_local_search.md`), and
/// QUBO's never had one. Neither could be read from outside the crate anyway.
///
/// Dropping `positive_gain` shrank `MaxCutSolution` from 168 to 112 bytes and
/// made BLS **8% slower**, which is worth recording because it is *not* a cost
/// of the deletion. Padding the struct back by any amount from 8 bytes upward
/// restored the speed exactly (a cliff at one size, not a gradient); the
/// padding changes no allocation, so the heap layout of the hot arrays is
/// identical; and `MaxCutFlipNeighbor::apply_to_solution` is the same 396 bytes
/// of instructions either way, differing only in being placed 4 bytes further
/// along. Building with `-C llvm-args=-align-all-functions=6` erased the
/// difference entirely. It is hot-function code alignment, and it can flip on
/// any unrelated edit — so it is never a reason to keep code.
#[derive(Debug, Clone, Default)]
pub struct GainIndex {
    enabled: bool,
    /// Unordered list of member variables.
    members: Vec<usize>,
    /// `pos[v]` = position of `v` in `members`, or `-1` if absent.
    pos: Vec<i32>,
}

impl GainIndex {
    /// Returns `true` once [`enable`](Self::enable) has been called.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// The current members, in unspecified order.
    pub fn as_slice(&self) -> &[usize] {
        &self.members
    }

    /// Returns `true` if variable `v` is currently a member.
    pub fn contains(&self, v: usize) -> bool {
        self.pos.get(v).is_some_and(|&p| p >= 0)
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Builds the index from the current `gains`, marking it enabled.
    /// If already enabled, this is a no-op. O(n).
    pub fn enable<T>(&mut self, gains: &[T], member: impl Fn(&T) -> bool) {
        if self.enabled {
            return;
        }
        self.enabled = true;
        self.members.clear();
        self.pos = vec![-1i32; gains.len()];
        for (v, g) in gains.iter().enumerate() {
            if member(g) {
                self.pos[v] = self.members.len() as i32;
                self.members.push(v);
            }
        }
    }

    /// Records that variable `v`'s membership changes from `was_member` to
    /// `is_member`. No-op when the index is not enabled or the status is
    /// unchanged. O(1).
    #[inline]
    pub fn update(&mut self, v: usize, was_member: bool, is_member: bool) {
        if !self.enabled || was_member == is_member {
            return;
        }
        if is_member {
            self.pos[v] = self.members.len() as i32;
            self.members.push(v);
        } else {
            let pos = self.pos[v] as usize;
            let last = *self.members.last().expect("members non-empty");
            self.members.swap_remove(pos);
            if last != v {
                self.pos[last] = pos as i32;
            }
            self.pos[v] = -1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enable_collects_matching_variables() {
        let mut idx = GainIndex::default();
        assert!(!idx.is_enabled());
        idx.enable(&[1, -2, 3, 0, -5], |&g| g < 0);
        assert!(idx.is_enabled());
        let mut members = idx.as_slice().to_vec();
        members.sort_unstable();
        assert_eq!(members, vec![1, 4]);
        assert!(idx.contains(1) && idx.contains(4));
        assert!(!idx.contains(0) && !idx.contains(2));
    }

    #[test]
    fn enable_twice_is_noop() {
        let mut idx = GainIndex::default();
        idx.enable(&[-1], |&g: &i32| g < 0);
        idx.enable(&[1], |&g: &i32| g < 0); // must not rebuild
        assert_eq!(idx.as_slice(), &[0]);
    }

    #[test]
    fn update_maintains_membership_and_inverse_index() {
        let mut idx = GainIndex::default();
        idx.enable(&[-1, 2, -3], |&g| g < 0); // members: {0, 2}
        idx.update(1, false, true); // 1 joins
        idx.update(0, true, false); // 0 leaves (swap_remove moves another member)
        idx.update(2, true, true); // unchanged status: no-op
        let mut members = idx.as_slice().to_vec();
        members.sort_unstable();
        assert_eq!(members, vec![1, 2]);
        // inverse index stays consistent after swap_remove
        for (p, &v) in idx.as_slice().iter().enumerate() {
            assert!(idx.contains(v));
            assert_eq!(idx.as_slice()[p], v);
        }
    }

    #[test]
    fn update_before_enable_is_noop() {
        let mut idx = GainIndex::default();
        idx.update(0, false, true);
        assert!(idx.is_empty());
    }
}
