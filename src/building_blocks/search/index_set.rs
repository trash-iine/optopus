//! A set of indices drawn from uniformly.

/// A set of indices below a fixed bound, with membership, insertion and
/// removal in O(1) and its members laid out as a slice.
///
/// A weighted search keeps the constraints a solution violates in one, since
/// it both updates the set on every move and draws a member uniformly.
/// The order of [`as_slice`](Self::as_slice) follows the insertions and
/// removals, so two sets built by the same calls list their members alike.
///
/// ```
/// use optopus::building_blocks::search::IndexSet;
///
/// let mut set = IndexSet::new(5);
/// set.insert(3);
/// set.insert(1);
/// set.insert(3);
/// assert_eq!(set.as_slice(), &[3, 1]);
/// set.remove(3);
/// assert!(set.contains(1) && !set.contains(3));
/// ```
#[derive(Debug, Clone)]
pub struct IndexSet {
    members: Vec<usize>,
    /// `position[i]` is where `i` sits in `members`, or [`ABSENT`].
    position: Vec<usize>,
}

const ABSENT: usize = usize::MAX;

impl IndexSet {
    /// The empty set over the indices `0..bound`.
    pub fn new(bound: usize) -> Self {
        Self {
            members: Vec::new(),
            position: vec![ABSENT; bound],
        }
    }

    /// Whether `i` is a member.
    #[inline]
    pub fn contains(&self, i: usize) -> bool {
        self.position[i] != ABSENT
    }

    /// Adds `i`, which may already be a member.
    #[inline]
    pub fn insert(&mut self, i: usize) {
        if self.position[i] == ABSENT {
            self.position[i] = self.members.len();
            self.members.push(i);
        }
    }

    /// Removes `i`, which need not be a member.
    #[inline]
    pub fn remove(&mut self, i: usize) {
        let at = self.position[i];
        if at == ABSENT {
            return;
        }
        self.members.swap_remove(at);
        if let Some(&moved) = self.members.get(at) {
            self.position[moved] = at;
        }
        self.position[i] = ABSENT;
    }

    /// The members.
    #[inline]
    pub fn as_slice(&self) -> &[usize] {
        &self.members
    }

    /// How many members there are.
    #[inline]
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Whether there is no member.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follows_a_plain_set_through_random_updates() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(3);
        let mut set = IndexSet::new(20);
        let mut plain = [false; 20];
        for _ in 0..2_000 {
            let i = rng.random_range(0..20);
            if rng.random_bool(0.5) {
                set.insert(i);
                plain[i] = true;
            } else {
                set.remove(i);
                plain[i] = false;
            }
            let mut got = set.as_slice().to_vec();
            got.sort_unstable();
            let expected: Vec<usize> = (0..20).filter(|&i| plain[i]).collect();
            assert_eq!(got, expected);
            assert_eq!(set.len(), expected.len());
            assert!((0..20).all(|i| set.contains(i) == plain[i]));
        }
    }
}
