//! The swap neighborhood every binary problem builds from its flip move.

use rand::Rng;

use crate::error::OptError;
use crate::trait_defs::{BinaryProblem, MoveToNeighbor};

/// Applies a swap move as two sequential flips (`i` then `j`).
///
/// The second flip reads the gain cached after the first, so the interaction
/// term between the two variables is accounted for. The per-problem
/// `*SwapNeighbor::apply_to_solution` impls delegate to this function.
pub fn apply_swap_as_two_flips<P: BinaryProblem>(
    prob: &P,
    sol: &mut P::Solution,
    i: usize,
    j: usize,
) -> Result<(), OptError> {
    P::flip_move(sol, i).apply_to_solution(prob, sol)?;
    P::flip_move(sol, j).apply_to_solution(prob, sol)?;
    Ok(())
}

/// Every pair `(i, j)` with `j < i` whose values differ in `sol`, in the
/// order the swap neighborhoods scan them. The per-problem
/// `*SwapNeighbor::iter` impls delegate to this function.
pub fn differing_pairs<P: BinaryProblem + Sync>(
    prob: &P,
    sol: &P::Solution,
) -> impl Iterator<Item = (usize, usize)> + Send
where
    P::Solution: Sync,
{
    prob.variable_indices().flat_map(move |i| {
        prob.variable_indices()
            .filter(move |&j| j < i && P::variable(sol, i) != P::variable(sol, j))
            .map(move |j| (i, j))
    })
}

/// Every pair [`differing_pairs`] yields, in the same order, with the value
/// `row(i)` lists for `j`, or `W::default()` when it lists none.
///
/// The swap moves price a pair by the interaction between its two variables,
/// which a lookup in `i`'s row finds by binary search. Here one pass over
/// `row(i)` fills a dense scratch before the pairs of `i` are listed and
/// another clears it after, so a pair costs O(1) and an `i` its row length on
/// top. The per-problem `*SwapNeighbor::iter` impls delegate to this function.
pub fn differing_pairs_by_row<P, W, I>(
    prob: &P,
    sol: &P::Solution,
    row: impl Fn(usize) -> I + Send,
) -> impl Iterator<Item = (usize, usize, W)> + Send
where
    P: BinaryProblem + Sync,
    P::Solution: Sync,
    W: Copy + Default + Send,
    I: Iterator<Item = (usize, W)>,
{
    RowPairs::<P, _, _, W, _>::new(
        sol,
        row,
        move || prob.variable_indices(),
        crate::trait_defs::variable_count(prob),
    )
}

/// The scan [`differing_pairs_by_row`] returns. `current` is the `i` whose
/// row is in `dense`, its value, and the `j`s still to list for it.
///
/// `fold`, which is how `max_by` and `filter` drive a scan, finishes the
/// current `i` and then runs the rest as two plain loops, which is what the
/// tour's 2-opt scan gained by (decisions/0031).
struct RowPairs<'a, P: BinaryProblem, R, K, W, O> {
    sol: &'a P::Solution,
    row: R,
    indices: K,
    dense: Vec<W>,
    outer: O,
    current: Option<(usize, bool, O)>,
}

impl<'a, P, R, K, W, I, O> RowPairs<'a, P, R, K, W, O>
where
    P: BinaryProblem,
    R: Fn(usize) -> I,
    K: Fn() -> O,
    W: Copy + Default,
    I: Iterator<Item = (usize, W)>,
    O: Iterator<Item = usize>,
{
    fn new(sol: &'a P::Solution, row: R, indices: K, len: usize) -> Self {
        let outer = indices();
        Self {
            sol,
            row,
            indices,
            dense: vec![W::default(); len],
            outer,
            current: None,
        }
    }

    fn fill(&mut self, i: usize) {
        for (j, w) in (self.row)(i) {
            if let Some(slot) = self.dense.get_mut(j) {
                *slot = w;
            }
        }
    }

    fn clear(&mut self, i: usize) {
        for (j, _) in (self.row)(i) {
            if let Some(slot) = self.dense.get_mut(j) {
                *slot = W::default();
            }
        }
    }
}

impl<P, R, K, W, I, O> Iterator for RowPairs<'_, P, R, K, W, O>
where
    P: BinaryProblem,
    R: Fn(usize) -> I,
    K: Fn() -> O,
    W: Copy + Default,
    I: Iterator<Item = (usize, W)>,
    O: Iterator<Item = usize>,
{
    type Item = (usize, usize, W);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((i, vi, js)) = &mut self.current {
                let (i, vi) = (*i, *vi);
                for j in js.by_ref() {
                    if j < i && vi != P::variable(self.sol, j) {
                        return Some((i, j, self.dense[j]));
                    }
                }
                self.clear(i);
                self.current = None;
            }
            let i = self.outer.next()?;
            self.fill(i);
            self.current = Some((i, P::variable(self.sol, i), (self.indices)()));
        }
    }

    fn fold<B, F: FnMut(B, Self::Item) -> B>(mut self, init: B, mut f: F) -> B {
        let mut acc = init;
        if let Some((i, vi, js)) = self.current.take() {
            for j in js {
                if j < i && vi != P::variable(self.sol, j) {
                    acc = f(acc, (i, j, self.dense[j]));
                }
            }
            self.clear(i);
        }
        for i in self.outer.by_ref() {
            for (j, w) in (self.row)(i) {
                if let Some(slot) = self.dense.get_mut(j) {
                    *slot = w;
                }
            }
            let vi = P::variable(self.sol, i);
            for j in (self.indices)() {
                if j < i && vi != P::variable(self.sol, j) {
                    acc = f(acc, (i, j, self.dense[j]));
                }
            }
            for (j, _) in (self.row)(i) {
                if let Some(slot) = self.dense.get_mut(j) {
                    *slot = W::default();
                }
            }
        }
        acc
    }
}

/// Draws of two variables before [`random_differing_pair`] falls back to its
/// counting pass. A draw lands with the probability that two variables
/// differ, about one half on a balanced assignment.
const PAIR_DRAWS: usize = 16;

/// A uniformly random pair `(i, j)` with `j < i` whose values differ in
/// `sol`, or `None` when every variable holds the same value.
///
/// Two variables are drawn with [`BinaryProblem::random_variable`] and kept
/// if they differ, which is uniform over the differing pairs and O(1) per
/// draw on the built-in problems. When sixteen draws all miss, the
/// values are nearly all equal, and one O(n) pass counts each side and draws
/// one variable from each, uniform too. Either way the distribution is that of
/// sampling [`differing_pairs`] uniformly. The per-problem
/// `*SwapNeighbor::random_neighbor` impls delegate to this function.
pub fn random_differing_pair<P: BinaryProblem>(
    prob: &P,
    sol: &P::Solution,
    rng: &mut impl Rng,
) -> Option<(usize, usize)> {
    for _ in 0..PAIR_DRAWS {
        let a = prob.random_variable(rng)?;
        let b = prob.random_variable(rng)?;
        if P::variable(sol, a) != P::variable(sol, b) {
            return Some((a.max(b), a.min(b)));
        }
    }
    let (mut n_false, mut n_true) = (0, 0);
    for i in prob.variable_indices() {
        if P::variable(sol, i) {
            n_true += 1;
        } else {
            n_false += 1;
        }
    }
    if n_false == 0 || n_true == 0 {
        return None;
    }
    let k_false = rng.random_range(0..n_false);
    let k_true = rng.random_range(0..n_true);
    let a = prob
        .variable_indices()
        .filter(|&i| !P::variable(sol, i))
        .nth(k_false)?;
    let b = prob
        .variable_indices()
        .filter(|&i| P::variable(sol, i))
        .nth(k_true)?;
    Some((a.max(b), a.min(b)))
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::building_blocks::instance::Graph;
    use crate::problem::MaxCut;

    /// Both the drawing and the counting path reach exactly the differing
    /// pairs, each about equally often.
    #[test]
    fn random_pairs_are_uniform_over_the_differing_pairs() {
        let mc = MaxCut::new(Graph::from_edges((0..12).map(|i| (i, (i + 1) % 12, 1.0))));
        let mut rng = SmallRng::seed_from_u64(5);
        // Balanced, where the draws land, and one-sided, where they mostly
        // miss and the counting pass answers.
        for ones in [6, 1] {
            let values: Vec<bool> = (0..12).map(|i| i < ones).collect();
            let sol = mc.solution_from_assignment(&values).unwrap();
            let listed: HashSet<_> = differing_pairs(&mc, &sol).collect();
            let mut counts = HashMap::new();
            for _ in 0..400 * listed.len() {
                let pair = random_differing_pair(&mc, &sol, &mut rng).unwrap();
                *counts.entry(pair).or_insert(0usize) += 1;
            }
            assert_eq!(counts.keys().copied().collect::<HashSet<_>>(), listed);
            assert!(
                counts.values().all(|&c| (250..550).contains(&c)),
                "{counts:?}"
            );
        }
        let same = mc.solution_from_assignment(&[false; 12]).unwrap();
        assert!(random_differing_pair(&mc, &same, &mut rng).is_none());
    }

    /// The swap scans that read the interaction from a dense row price every
    /// pair as the moves' own constructors do, in the order `differing_pairs`
    /// lists them, on MaxCut, QUBO and Vertex Cover.
    #[test]
    fn row_scans_price_pairs_as_the_constructors_do() {
        use crate::problem::{
            MaxCutSolution, MaxCutSwapNeighbor, Qubo, QuboSolution, QuboSwapNeighbor, VertexCover,
            VertexCoverSwapNeighbor,
        };
        use crate::search_state::{MoveToNeighbor, ProblemTrait};
        use rand::Rng;

        let mut rng = SmallRng::seed_from_u64(9);
        let edges: Vec<(usize, usize, f32)> = (0..40)
            .map(|_| {
                (
                    rng.random_range(0..15),
                    rng.random_range(0..15),
                    rng.random_range(-3.0..3.0),
                )
            })
            .filter(|&(a, b, _)| a != b)
            .collect();
        let graph = Graph::from_edges(edges.iter().copied());
        let mc = MaxCut::new(graph.clone());
        let qubo = Qubo::from_entries(edges.iter().map(|&(a, b, w)| (a, b, w as i32)));
        let vc = VertexCover::new(graph);
        for _ in 0..5 {
            let sol: MaxCutSolution = mc.new_solution(&mut rng);
            let scanned: Vec<_> = MaxCutSwapNeighbor::iter(&mc, &sol)
                .map(|m| (m.i, m.j, m.gain))
                .collect();
            let built: Vec<_> = differing_pairs(&mc, &sol)
                .map(|(i, j)| (i, j, MaxCutSwapNeighbor::new(&mc, &sol, i, j).gain))
                .collect();
            assert_eq!(scanned, built);
            // `collect` reads by `next`, `fold` by its own loops, and a scan
            // may switch from the first to the second at any pair.
            for k in 0..=built.len() {
                let mut it = MaxCutSwapNeighbor::iter(&mc, &sol);
                let head: Vec<_> = it.by_ref().take(k).map(|m| (m.i, m.j, m.gain)).collect();
                let all = it.fold(head, |mut acc, m| {
                    acc.push((m.i, m.j, m.gain));
                    acc
                });
                assert_eq!(all, built, "switched after {k}");
            }

            let sol: QuboSolution = qubo.new_solution(&mut rng);
            let scanned: Vec<_> = QuboSwapNeighbor::iter(&qubo, &sol)
                .map(|m| (m.i, m.j, m.gain))
                .collect();
            let built: Vec<_> = differing_pairs(&qubo, &sol)
                .map(|(i, j)| (i, j, QuboSwapNeighbor::new(&qubo, &sol, i, j).gain))
                .collect();
            assert_eq!(scanned, built);

            let sol = vc.new_solution(&mut rng);
            let scanned: Vec<_> = VertexCoverSwapNeighbor::iter(&vc, &sol)
                .map(|m| (m.i, m.j, m.gain))
                .collect();
            let built: Vec<_> = differing_pairs(&vc, &sol)
                .map(|(i, j)| (i, j, VertexCoverSwapNeighbor::new(&vc, &sol, i, j).gain))
                .collect();
            assert_eq!(scanned, built);
        }
    }
}
