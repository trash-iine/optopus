//! The tabu memory a search carries, and the keys moves forbid in it.
//!
//! Every move that supports a tabu list forbids the same kind of thing: an
//! index, a pair, or a triple of indices, until some future iteration. This
//! module holds that one store; which keys a move reads and writes is the
//! move's own [`EnabledTabu`](crate::trait_defs::EnabledTabu) impl, and nothing
//! here decides it.

use rand::Rng;
use rand::rngs::SmallRng;
use std::collections::HashMap;

/// What a move forbids: the thing that has to stay put for a while after the
/// move is applied.
///
/// The three shapes are what the library's moves actually key on — a variable
/// or vertex index, an (endpoint, endpoint) or (customer, route) pair, a
/// (route, position, position) triple. They live in separate key spaces, so a
/// `Var(3)` and a `Pair(3, 0)` never collide, while two move types that key on
/// the same shape do share prohibitions — which is exactly what Breakout Local
/// Search's flips and swaps rely on.
///
/// The `From` impls are what let a move write `tabu.forbid(self.i, ..)` instead
/// of spelling the variant out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TabuKey {
    /// A dense index in `0..n`: a variable, a vertex, a position.
    Var(usize),
    /// A pair of indices, e.g. the two endpoints of an edge.
    Pair(usize, usize),
    /// A triple of indices, e.g. a route and two positions in it.
    Triple(usize, usize, usize),
}

impl From<usize> for TabuKey {
    fn from(i: usize) -> Self {
        TabuKey::Var(i)
    }
}

impl From<(usize, usize)> for TabuKey {
    fn from((a, b): (usize, usize)) -> Self {
        TabuKey::Pair(a, b)
    }
}

impl From<(usize, usize, usize)> for TabuKey {
    fn from((a, b, c): (usize, usize, usize)) -> Self {
        TabuKey::Triple(a, b, c)
    }
}

/// The tabu prohibitions a [`SearchState`](crate::search_state::SearchState)
/// remembers, together with the tenure range every write to them draws from.
///
/// The two always travel together — a map without the tenure cannot record a
/// move, and a tenure without the map has nothing to record into — and both
/// belong to the *search*, not to the heuristic driving it: the state is what
/// applies a move, so the state is what records it
/// ([`apply`](crate::search_state::SearchState::apply) does it inline). A
/// heuristic only sets the tenure it wants.
///
/// It is deliberately not a tabu policy. The policy lives in each move's
/// [`EnabledTabu`](crate::trait_defs::EnabledTabu) impl — which keys it reads,
/// which it writes, and whether they have to agree; this type owns only the
/// storage and the tenure, so it forbids exactly what a
/// [`TabuSearch`](crate::heuristic::TabuSearch) over the same neighborhood
/// would.
///
/// Each entry is the first iteration at which the key is free again, so `0`
/// (what a fresh or cleared entry holds) means "always allowed". Storing the
/// boundary rather than the last forbidden iteration keeps the test a plain
/// `iteration >= until`, with no off-by-one to get wrong at each call site.
///
/// ```
/// use optopus::common::TabuMemory;
/// use rand::SeedableRng;
///
/// let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
/// let mut tabu = TabuMemory::default();
/// tabu.set_tenure((5, 5));
///
/// assert!(tabu.is_enabled(2, 0));
/// tabu.forbid(2, 0, &mut rng);        // forbidden for the next 5 iterations
/// assert!(!tabu.is_enabled(2, 5));
/// assert!(tabu.is_enabled(2, 6));
///
/// // A different key space: (2, 0) is not the same thing as 2.
/// assert!(tabu.is_enabled((2, 0), 5));
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TabuMemory {
    /// `dense[i]` = first iteration at which `Var(i)` is free again.
    dense: Vec<u64>,
    /// The same, for the keys that are not a dense index.
    /// The compound keys.
    ///
    /// **This field was briefly boxed, and the reason is worth keeping.** A
    /// neighborhood scan walks three heap buffers in lockstep, once per
    /// candidate: the graph's vertex list, the solution's gains, and `dense`.
    /// Where the allocator puts those three decides whether they collide in the
    /// cache, and anything that perturbs the allocation sequence moves them.
    /// Under `lto = "thin"` storing this map inline drew an arrangement that
    /// cost `TabuSearch` **+47% on G32 and +35% on G1**, at identical work —
    /// bit-identical objective, move counts and per-run seeds.
    ///
    /// The map itself was never the cause. Leaving it inline and giving `dense`
    /// 4096 slots of slack also fixed it (+0.9%), and boxing the map while
    /// giving `dense` 512 slots of slack put the slowdown back (+7.6%).
    ///
    /// What actually removes it is `lto = "fat"`, now set in `Cargo.toml`:
    /// the same inline arrangement measures −2.0% / +0.2% under it. So the box
    /// is gone and the profile carries the fix.
    ///
    /// Ruled out along the way, each by measurement, so they are not re-tried:
    /// the struct's size (padding `SearchState` back to the same bytes costs
    /// nothing), the cache lines its fields occupy (`offset_of!` says four
    /// either way), inlining of `is_enabled` (no symbol survives in either
    /// binary), the `TabuKey` construction, the shape of the closure that reads
    /// the state, and code alignment (24 unrelated functions inserted ahead of
    /// it change nothing). Instruction counts in `run_once` are unchanged; the
    /// cost was stalls.
    sparse: HashMap<TabuKey, u64>,
    tenure: (u64, u64),
}

impl TabuMemory {
    /// The tenure range every [`forbid`](Self::forbid) draws from. `(0, 0)` —
    /// the default — records each move but frees it again on the next
    /// iteration, i.e. remembers without forbidding.
    pub fn tenure(&self) -> (u64, u64) {
        self.tenure
    }

    /// Sets the tenure range.
    ///
    /// # Panics
    ///
    /// Panics if `tenure.0 > tenure.1`, i.e. the range is empty — sampling from
    /// it would panic later, at a point far from the caller that set it.
    pub fn set_tenure(&mut self, tenure: (u64, u64)) {
        assert_valid_tenure(tenure);
        self.tenure = tenure;
    }

    /// Frees every key, keeping the dense buffer.
    pub fn clear(&mut self) {
        self.dense.fill(0);
        self.sparse.clear();
    }

    /// Grows the dense space to hold `Var(0..n)`. Never shrinks.
    ///
    /// Pure pre-allocation: [`forbid`](Self::forbid) grows on demand anyway,
    /// and does so without changing what it draws.
    pub fn reserve_vars(&mut self, n: usize) {
        if self.dense.len() < n {
            self.dense.resize(n, 0);
        }
    }

    /// Replaces these prohibitions and tenure with `other`'s, translating every
    /// boundary from `other`'s iteration counter into `now`.
    ///
    /// The translation is the whole point. Boundaries are absolute iterations,
    /// and a sub-run restarts its counter, so copying them as they stand would
    /// forbid a key for the parent's *elapsed* iterations rather than for the
    /// tenure it has left — a key blocked for 5 more steps at parent iteration
    /// 100 would come out blocked for 105 of the child's. What crosses is the
    /// remaining time; anything already expired arrives free.
    pub fn inherit(&mut self, other: &Self, other_iteration: u64, now: u64) {
        let remaining = |until: u64| now + (until - other_iteration);
        self.dense.clear();
        self.dense.extend(other.dense.iter().map(|&until| {
            if until > other_iteration {
                remaining(until)
            } else {
                0
            }
        }));
        self.sparse = other
            .sparse
            .iter()
            .filter(|&(_, &until)| until > other_iteration)
            .map(|(&key, &until)| (key, remaining(until)))
            .collect();
        self.tenure = other.tenure;
    }

    /// Returns whether `key` is free at `iteration`. A key nothing has recorded
    /// is free.
    #[inline]
    pub fn is_enabled(&self, key: impl Into<TabuKey>, iteration: u64) -> bool {
        match key.into() {
            TabuKey::Var(i) => self.dense.get(i).is_none_or(|&until| iteration >= until),
            key => self
                .sparse
                .get(&key)
                .is_none_or(|&until| iteration >= until),
        }
    }

    /// Forbids `key` for a tenure drawn from [`tenure`](Self::tenure), counted
    /// from `iteration`.
    ///
    /// Any boundary already recorded is overwritten — the most recent move
    /// wins, which is what lets a short tenure release a key early.
    #[inline]
    pub fn forbid(&mut self, key: impl Into<TabuKey>, iteration: u64, rng: &mut SmallRng) {
        let tabu_duration = rng.random_range(self.tenure.0..=self.tenure.1);
        // A move applied at `iteration` with tenure `d` stays blocked through
        // `iteration + d`, so the first iteration that frees it is one past.
        let until = iteration + tabu_duration + 1;
        match key.into() {
            TabuKey::Var(i) => {
                if i >= self.dense.len() {
                    self.dense.resize(i + 1, 0);
                }
                self.dense[i] = until;
            }
            key => {
                self.sparse.insert(key, until);
            }
        }
    }
}

/// Rejects an empty tenure range.
///
/// The range is sampled from on every record, far from whoever configured it,
/// so a heuristic that takes a tenure checks it in its constructor and this is
/// the one wording that check has.
///
/// # Panics
///
/// Panics if `tenure.0 > tenure.1`.
pub(crate) fn assert_valid_tenure(tenure: (u64, u64)) {
    if tenure.0 > tenure.1 {
        panic!(
            "Invalid tabu tenure range: left side should be smaller than or equal to the right side ({} <= {})",
            tenure.0, tenure.1
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn rng() -> SmallRng {
        SmallRng::seed_from_u64(7)
    }

    fn memory(tenure: (u64, u64)) -> TabuMemory {
        let mut memory = TabuMemory::default();
        memory.set_tenure(tenure);
        memory
    }

    /// A tenure of `d` drawn at `iteration` must forbid exactly the next `d`
    /// iterations — blocked through `iteration + d`, free at `iteration + d + 1`
    /// — and it must do so identically in both key spaces.
    #[test]
    fn a_tenure_of_d_blocks_exactly_d_iterations() {
        for key in [
            TabuKey::Var(0),
            TabuKey::Pair(1, 2),
            TabuKey::Triple(1, 2, 3),
        ] {
            let mut tabu = memory((3, 3));
            tabu.forbid(key, 100, &mut rng());

            for it in 100..=103 {
                assert!(!tabu.is_enabled(key, it), "{key:?} must be blocked at {it}");
            }
            assert!(tabu.is_enabled(key, 104), "{key:?} must be free at 104");
        }
    }

    /// The three shapes are separate spaces: recording one must say nothing
    /// about the others, however similar the indices look.
    #[test]
    fn the_key_spaces_do_not_collide() {
        let mut tabu = memory((50, 50));
        tabu.forbid(3usize, 0, &mut rng());

        assert!(!tabu.is_enabled(3usize, 10));
        assert!(tabu.is_enabled((3, 0), 10), "a pair is not a var");
        assert!(tabu.is_enabled((3, 0, 0), 10), "a triple is not a pair");
    }

    /// Unseen keys are movable, and `clear` returns every key to that state.
    #[test]
    fn unseen_and_cleared_keys_are_movable() {
        let mut tabu = memory((50, 50));
        assert!(
            tabu.is_enabled(999usize, 0),
            "an unseen key must be movable"
        );

        tabu.reserve_vars(4);
        tabu.forbid(1usize, 0, &mut rng());
        tabu.forbid((1, 2), 0, &mut rng());
        assert!(!tabu.is_enabled(1usize, 10));
        assert!(!tabu.is_enabled((1, 2), 10));

        tabu.clear();
        assert!(tabu.is_enabled(1usize, 10));
        assert!(tabu.is_enabled((1, 2), 10));
        assert_eq!(tabu.tenure(), (50, 50), "clear must not touch the tenure");
    }

    /// `forbid` must grow the dense space rather than silently dropping the
    /// entry, so a default-constructed memory still works.
    #[test]
    fn forbid_grows_the_dense_space() {
        let mut tabu = memory((2, 2));
        tabu.forbid(5usize, 0, &mut rng());
        assert!(!tabu.is_enabled(5usize, 1));
        assert!(
            tabu.is_enabled(4usize, 1),
            "growth must not block its neighbors"
        );
    }

    /// Inheriting carries the *remaining* prohibition, not the boundary: a key
    /// blocked for 3 more iterations at the parent's 100 must be blocked for 3
    /// more at the child's 0, not for 103.
    #[test]
    fn inheriting_carries_the_remaining_tenure_into_the_new_frame() {
        let mut parent = memory((3, 3));
        parent.forbid(1usize, 100, &mut rng());
        parent.forbid((4, 5), 100, &mut rng());
        parent.forbid(2usize, 0, &mut rng()); // long expired by iteration 100

        let mut child = TabuMemory::default();
        child.inherit(&parent, 100, 0);

        assert_eq!(child.tenure(), (3, 3), "the tenure crosses too");
        for it in 0..=3 {
            assert!(!child.is_enabled(1usize, it), "still blocked at {it}");
            assert!(!child.is_enabled((4, 5), it), "still blocked at {it}");
        }
        assert!(child.is_enabled(1usize, 4), "and free at 4");
        assert!(child.is_enabled((4, 5), 4), "and free at 4");
        assert!(
            child.is_enabled(2usize, 0),
            "a prohibition the parent had already outlived must arrive free"
        );
    }

    #[test]
    #[should_panic(expected = "Invalid tabu tenure range")]
    fn an_empty_tenure_range_panics() {
        TabuMemory::default().set_tenure((9, 2));
    }

    /// What the moves make of the store: two move types that key on the same
    /// shape share prohibitions, and two that do not are independent. Breakout
    /// Local Search depends on the first — a weak swap must not undo what the
    /// descent's flips wrote.
    mod sharing {
        use super::*;
        use crate::problem::{
            JobShopRelocateNeighbor, JobShopSwapNeighbor, MaxCut, MaxCutFlipNeighbor,
            MaxCutSolution, MaxCutSwapNeighbor,
        };
        use crate::trait_defs::EnabledTabu;

        #[test]
        fn max_cut_flip_and_swap_share_prohibitions() {
            let mc = MaxCut::from_edges([(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]);
            let sol = MaxCutSolution::new_from_assignment(&mc, vec![false, true, false, true]);
            let mut tabu = memory((9, 9));

            MaxCutSwapNeighbor::new(&mc, &sol, 1, 2).add_to_tabu_map(&mut tabu, 0, &mut rng());

            for v in [1usize, 2] {
                assert!(
                    !MaxCutFlipNeighbor::new(&mc, &sol, v).is_move_enabled(&tabu, 3),
                    "vertex {v} was moved by the swap and must be tabu for the flip too"
                );
            }
        }

        #[test]
        fn job_shop_swap_and_relocate_are_independent() {
            let swap = JobShopSwapNeighbor { i: 0, gain: 0.0 };
            let relocate = JobShopRelocateNeighbor {
                from: 0,
                to: 1,
                gain: 0.0,
            };
            let mut tabu = memory((9, 9));

            swap.add_to_tabu_map(&mut tabu, 0, &mut rng());
            assert!(!EnabledTabu::is_move_enabled(&swap, &tabu, 3));
            assert!(
                EnabledTabu::is_move_enabled(&relocate, &tabu, 3),
                "a swap keys on a position, a relocate on a pair"
            );

            relocate.add_to_tabu_map(&mut tabu, 0, &mut rng());
            assert!(!EnabledTabu::is_move_enabled(&relocate, &tabu, 3));
            assert!(
                !EnabledTabu::is_move_enabled(&swap, &tabu, 3),
                "and the swap's own key must survive"
            );
        }
    }
}
