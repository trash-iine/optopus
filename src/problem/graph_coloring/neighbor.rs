use super::problem::{GraphColoring, GraphColoringSolution};
use crate::{
    error::OptError,
    search_state::{EnabledTabu, Evaluable, Evaluate, MoveToNeighbor, Rankable},
};
use rand::Rng;

/// A recolor move: assign vertex `v` a new color.
#[derive(Debug, Clone, Copy)]
pub struct GraphColoringRecolorNeighbor {
    /// The vertex to recolor.
    pub v: usize,
    /// The target color.
    pub new_color: usize,
    /// Change in objective after the recolor.
    pub gain: i64,
}

impl Rankable for GraphColoringRecolorNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain < other.gain
    }
}

impl Evaluate for GraphColoringRecolorNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain as f64)
    }
}

impl EnabledTabu for GraphColoringRecolorNeighbor {
    /// Vec indexed by vertex ID; value = expiry iteration (recoloring `v` is
    /// forbidden until then).
    type TabuMap = Vec<u64>;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        tabu_map
            .get(self.v)
            .is_none_or(|&tenure| iteration > tenure)
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
        rng: &mut rand::rngs::SmallRng,
    ) {
        let d = rng.random_range(tabu_tenure.0..=tabu_tenure.1);
        if self.v >= tabu_map.len() {
            tabu_map.resize(self.v + 1, 0);
        }
        tabu_map[self.v] = iteration + d;
    }
}

impl MoveToNeighbor<GraphColoring> for GraphColoringRecolorNeighbor {
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
                .map(move |c| GraphColoringRecolorNeighbor {
                    v,
                    new_color: c,
                    gain: prob.recolor_gain(sol, v, c),
                })
        })
    }

    fn move_to_be_better_than(
        &self,
        _prob: &GraphColoring,
        src: &GraphColoringSolution,
        other: &GraphColoringSolution,
    ) -> bool {
        src.objective + self.gain < other.objective
    }

    fn random_neighbor(
        prob: &GraphColoring,
        sol: &GraphColoringSolution,
        rng: &mut rand::rngs::SmallRng,
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
        Some(Self {
            v,
            new_color: c,
            gain: prob.recolor_gain(sol, v, c),
        })
    }
}

/// A swap move: exchange the colors of two differently-colored vertices.
///
/// The number of colors used is invariant under a swap, so this move only
/// repairs (or introduces) conflicts.
#[derive(Debug, Clone, Copy)]
pub struct GraphColoringSwapNeighbor {
    pub i: usize,
    pub j: usize,
    /// Change in objective after the swap.
    pub gain: i64,
}

impl Rankable for GraphColoringSwapNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain < other.gain
    }
}

impl Evaluate for GraphColoringSwapNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain as f64)
    }
}

impl EnabledTabu for GraphColoringSwapNeighbor {
    type TabuMap = Vec<u64>;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        let enabled_i = tabu_map.get(self.i).is_none_or(|&t| iteration > t);
        let enabled_j = tabu_map.get(self.j).is_none_or(|&t| iteration > t);
        enabled_i && enabled_j
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
        rng: &mut rand::rngs::SmallRng,
    ) {
        let max_v = self.i.max(self.j);
        if max_v >= tabu_map.len() {
            tabu_map.resize(max_v + 1, 0);
        }
        let d = rng.random_range(tabu_tenure.0..=tabu_tenure.1);
        tabu_map[self.i] = iteration + d;
        let d = rng.random_range(tabu_tenure.0..=tabu_tenure.1);
        tabu_map[self.j] = iteration + d;
    }
}

impl MoveToNeighbor<GraphColoring> for GraphColoringSwapNeighbor {
    fn apply_to_iteration(&self, iter: u64) -> u64 {
        iter + 2 // counts as two recolors
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
                .map(move |j| GraphColoringSwapNeighbor {
                    i,
                    j,
                    gain: prob.swap_gain(sol, i, j),
                })
        })
    }

    fn move_to_be_better_than(
        &self,
        _prob: &GraphColoring,
        src: &GraphColoringSolution,
        other: &GraphColoringSolution,
    ) -> bool {
        src.objective + self.gain < other.objective
    }

    fn random_neighbor(
        prob: &GraphColoring,
        sol: &GraphColoringSolution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Option<Self> {
        let n = prob.graph.len();
        if n < 2 {
            return None;
        }
        // Sample a differing-color pair with a bounded number of attempts.
        for _ in 0..16 {
            let a = rng.random_range(0..n);
            let b = rng.random_range(0..n);
            if a != b && sol.colors[a] != sol.colors[b] {
                let (i, j) = (a.max(b), a.min(b));
                return Some(Self {
                    i,
                    j,
                    gain: prob.swap_gain(sol, i, j),
                });
            }
        }
        None
    }
}
