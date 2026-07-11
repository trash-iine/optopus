//! Neighborhood move types for the [`GraphColoring`] problem.

use super::problem::{GraphColoring, GraphColoringSolution};
use crate::{
    common::TabuMemory,
    error::OptError,
    search_state::{EnabledTabu, Evaluable, Evaluate, MoveToNeighbor},
};
use rand::Rng;
use rand::rngs::SmallRng;

/// A recolor move: assign vertex `v` a new color.
///
/// `gain` holds the change in (penalty-augmented) objective after the move
/// (negative = improvement, since Graph Coloring is minimization). Applying it
/// refreshes the `gamma` row of every neighbor of `v`, i.e. O(degree(v)) work.
#[derive(Debug, Clone, Copy)]
pub struct GraphColoringRecolorNeighbor {
    /// The vertex to recolor.
    pub v: usize,
    /// The target color.
    pub new_color: usize,
    /// Change in objective after the recolor.
    pub gain: i64,
}

impl GraphColoringRecolorNeighbor {
    /// Builds the recolor of `v` to `new_color`, computing its gain from the
    /// solution's caches. Every construction site goes through here.
    pub fn new(
        prob: &GraphColoring,
        sol: &GraphColoringSolution,
        v: usize,
        new_color: usize,
    ) -> Self {
        Self {
            v,
            new_color,
            gain: prob.recolor_gain(sol, v, new_color),
        }
    }
}

impl Evaluate for GraphColoringRecolorNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain as f64)
    }
}

impl EnabledTabu for GraphColoringRecolorNeighbor {
    /// The move is tabu while vertex `v` is still blocked at the current iteration.
    fn is_move_enabled(&self, tabu: &TabuMemory, iteration: u64) -> bool {
        tabu.is_enabled(self.v, iteration)
    }

    /// Applying the move forbids vertex `v` for a tenure the memory draws.
    fn add_to_tabu_map(&self, tabu: &mut TabuMemory, iteration: u64, rng: &mut SmallRng) {
        tabu.forbid(self.v, iteration, rng);
    }
}

impl MoveToNeighbor<GraphColoring> for GraphColoringRecolorNeighbor {
    /// Hands this move's [`EnabledTabu`] policy to the search state, which is
    /// what holds the tabu map.
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }

    fn apply_to_solution(
        &self,
        prob: &GraphColoring,
        sol: &mut GraphColoringSolution,
    ) -> Result<(), OptError> {
        prob.recolor(sol, self.v, self.new_color);
        Ok(())
    }

    fn iter(
        prob: &GraphColoring,
        sol: &GraphColoringSolution,
    ) -> impl Iterator<Item = Self> + Send {
        let n = prob.graph.len();
        let k = prob.k;
        (0..n).flat_map(move |v| {
            let cur = sol.colors[v];
            (0..k)
                .filter(move |&c| c != cur)
                .map(move |c| Self::new(prob, sol, v, c))
        })
    }

    fn move_to_be_better_than(
        &self,
        _: &GraphColoring,
        src: &GraphColoringSolution,
        other: &GraphColoringSolution,
    ) -> bool {
        self.evaluate()
            .improves_over(src.evaluate(), other.evaluate())
    }

    /// O(1): a uniformly random vertex and a uniformly random color other
    /// than its current one.
    fn random_neighbor(
        prob: &GraphColoring,
        sol: &GraphColoringSolution,
        rng: &mut SmallRng,
    ) -> Option<Self> {
        let n = prob.graph.len();
        if n == 0 || prob.k < 2 {
            return None;
        }
        let v = rng.random_range(0..n);
        let cur = sol.colors[v];
        // Uniform color in `0..k` excluding the current one.
        let mut c = rng.random_range(0..prob.k - 1);
        if c >= cur {
            c += 1;
        }
        Some(Self::new(prob, sol, v, c))
    }
}

/// A swap move: exchange the colors of two differently colored vertices.
///
/// The number of colors used is invariant under a swap, so this move only
/// repairs (or introduces) conflicts. Only pairs with `colors[i] != colors[j]`
/// and `j < i` are enumerated. Applying the move performs two recolors in
/// sequence (`apply_to_iteration` returns `iter + 2`), each doing O(degree)
/// work.
#[derive(Debug, Clone, Copy)]
pub struct GraphColoringSwapNeighbor {
    pub i: usize,
    pub j: usize,
    /// Change in objective after the swap.
    pub gain: i64,
}

impl GraphColoringSwapNeighbor {
    /// Builds the swap of `i` and `j`, computing its gain from the solution's
    /// caches. The gain corrects for `(i, j)` being an edge, so every
    /// construction site goes through here and the correction cannot be
    /// forgotten at one of them.
    pub fn new(prob: &GraphColoring, sol: &GraphColoringSolution, i: usize, j: usize) -> Self {
        Self {
            i,
            j,
            gain: prob.swap_gain(sol, i, j),
        }
    }
}

impl Evaluate for GraphColoringSwapNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain as f64)
    }
}

impl EnabledTabu for GraphColoringSwapNeighbor {
    /// A swap is tabu unless both vertices it recolors are free.
    fn is_move_enabled(&self, tabu: &TabuMemory, iteration: u64) -> bool {
        tabu.is_enabled(self.i, iteration) && tabu.is_enabled(self.j, iteration)
    }

    /// Applying the swap forbids both vertices, each for its own drawn tenure.
    fn add_to_tabu_map(&self, tabu: &mut TabuMemory, iteration: u64, rng: &mut SmallRng) {
        tabu.forbid(self.i, iteration, rng);
        tabu.forbid(self.j, iteration, rng);
    }
}

impl MoveToNeighbor<GraphColoring> for GraphColoringSwapNeighbor {
    /// Hands this move's [`EnabledTabu`] policy to the search state, which is
    /// what holds the tabu map.
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }

    fn apply_to_iteration(&self, iter: u64) -> u64 {
        iter + 2
    }

    fn apply_to_solution(
        &self,
        prob: &GraphColoring,
        sol: &mut GraphColoringSolution,
    ) -> Result<(), OptError> {
        let ci = sol.colors[self.i];
        let cj = sol.colors[self.j];
        prob.recolor(sol, self.i, cj);
        prob.recolor(sol, self.j, ci);
        Ok(())
    }

    fn iter(
        prob: &GraphColoring,
        sol: &GraphColoringSolution,
    ) -> impl Iterator<Item = Self> + Send {
        let n = prob.graph.len();
        (0..n).flat_map(move |i| {
            (0..i)
                .filter(move |&j| sol.colors[j] != sol.colors[i])
                .map(move |j| Self::new(prob, sol, i, j))
        })
    }

    fn move_to_be_better_than(
        &self,
        _: &GraphColoring,
        src: &GraphColoringSolution,
        other: &GraphColoringSolution,
    ) -> bool {
        self.evaluate()
            .improves_over(src.evaluate(), other.evaluate())
    }

    /// Expected O(1): rejection-samples a pair of differently colored
    /// vertices, giving up after a bounded number of draws so a solution
    /// colored all one color, where the neighborhood is empty, returns
    /// `None` instead of looping.
    fn random_neighbor(
        prob: &GraphColoring,
        sol: &GraphColoringSolution,
        rng: &mut SmallRng,
    ) -> Option<Self> {
        let n = prob.graph.len();
        if n < 2 {
            return None;
        }
        for _ in 0..16 {
            let a = rng.random_range(0..n);
            let b = rng.random_range(0..n);
            if a != b && sol.colors[a] != sol.colors[b] {
                let (i, j) = (a.max(b), a.min(b));
                return Some(Self::new(prob, sol, i, j));
            }
        }
        None
    }
}
