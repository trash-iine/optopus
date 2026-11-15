//! Uniform crossover, path relinking, and the lifts back from a crossover's
//! sub-problem.

use rand::Rng;

use crate::error::OptError;
use crate::trait_defs::{BinaryProblem, MoveToNeighbor, Rankable};

/// Uniform crossover over binary variables.
///
/// Clones `sol1` and, for each variable where the parents disagree, flips it to
/// `sol2`'s value with 50% probability. Because the flips go through the
/// problem's flip move, the offspring's cached `gain` and objective stay
/// consistent.
///
/// The per-problem `*UniformCrossover` operators all delegate to this function.
pub fn uniform_binary_crossover<P: BinaryProblem>(
    prob: &P,
    sol1: &P::Solution,
    sol2: &P::Solution,
    rng: &mut rand::rngs::SmallRng,
) -> Result<P::Solution, OptError> {
    let mut sol = sol1.clone();
    for i in prob.variable_indices() {
        if P::variable(&sol, i) != P::variable(sol2, i) && rng.random::<bool>() {
            prob.flip_move(&sol, i).apply_to_solution(prob, &mut sol)?;
        }
    }
    Ok(sol)
}

/// The order a relinking path flips its variables in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathOrder {
    /// The flip that leaves the best solution, at every step.
    Greedy,
    /// A flip drawn uniformly, at every step.
    Random,
}

/// Path relinking as a crossover over binary variables.
///
/// The offspring is a solution on a path from the first parent toward the
/// second. The path flips, one at a time, the variables the parents disagree
/// on, so every solution on it keeps what the parents share. Of the solutions
/// that are at least `distance_scale` of the way from both parents, the best
/// is returned. A solution next to a parent would fall back into that parent
/// under the improvement a genetic search runs on its offspring, which is what
/// the distance is for.
///
/// It implements [`Crossover`](crate::trait_defs::Crossover) for MaxCut, QUBO,
/// MaxSAT and vertex cover. With [`PathOrder::Greedy`] a path over `d`
/// disagreeing variables prices O(d²) flips, where the uniform crossover
/// applies `d / 2`.
///
/// ```
/// use optopus::building_blocks::representation::PathRelinkingCrossover;
/// use optopus::prelude::*;
///
/// let mc = MaxCut::new(Graph::from_edges([
///     (0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0),
/// ]));
/// let mut ga = GeneticAlgorithm::new(
///     StopCondition::iterations(200),
///     6,
///     PathRelinkingCrossover::greedy(),
///     Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
///         StopCondition::failed_updates(20),
///     )),
///     ParentSelection::Tournament,
/// );
/// let mut state = SearchState::new_with_seed(&mc, 1);
/// ga.run(&mut state).unwrap();
/// assert_eq!(state.best_solution.objective, 4.0);
/// ```
///
/// # References
///
/// - Wang, Y., Lü, Z., Glover, F. and Hao, J.-K. "Path relinking for
///   unconstrained binary quadratic programming." *European Journal of
///   Operational Research*, 223(3), 595-604, 2012.
#[derive(Debug, Clone, Copy)]
pub struct PathRelinkingCrossover {
    order: PathOrder,
    distance_scale: f64,
}

impl PathRelinkingCrossover {
    /// The share of the path an offspring keeps from each parent, unless
    /// [`with_distance_scale`](Self::with_distance_scale) says otherwise.
    pub const DISTANCE_SCALE: f64 = 1.0 / 3.0;

    /// A relinking that takes the best flip at every step.
    pub fn greedy() -> Self {
        Self {
            order: PathOrder::Greedy,
            distance_scale: Self::DISTANCE_SCALE,
        }
    }

    /// A relinking that draws its flips uniformly.
    pub fn random() -> Self {
        Self {
            order: PathOrder::Random,
            distance_scale: Self::DISTANCE_SCALE,
        }
    }

    /// Builder-style: how far, as a share of the path, the offspring stays
    /// from both parents. Past one half no solution qualifies, and the middle
    /// of the path is returned.
    ///
    /// # Panics
    ///
    /// Panics if `distance_scale` is outside `(0, 1]`.
    pub fn with_distance_scale(mut self, distance_scale: f64) -> Self {
        assert!(
            distance_scale > 0.0 && distance_scale <= 1.0,
            "distance_scale must be in (0, 1]"
        );
        self.distance_scale = distance_scale;
        self
    }

    /// The offspring on the path from `start` that flips the variables of
    /// `flips`, each at most once.
    ///
    /// `flips` is the set of variables on which `start` differs from the
    /// solution the path leads to. A caller with two parents passes
    /// [`differing_variables`], and a problem where two assignments are the
    /// same solution passes the set that leads to the nearer of them.
    ///
    /// With fewer than two variables to flip there is no solution between the
    /// parents, and `start` is returned.
    pub fn relink<P>(
        &self,
        prob: &P,
        start: &P::Solution,
        mut flips: Vec<usize>,
        rng: &mut rand::rngs::SmallRng,
    ) -> Result<P::Solution, OptError>
    where
        P: BinaryProblem,
        P::Solution: Rankable,
        P::Flip: Rankable,
    {
        let distance = flips.len();
        let mut sol = start.clone();
        if distance < 2 {
            return Ok(sol);
        }
        let nearest =
            ((self.distance_scale * distance as f64).ceil() as usize).clamp(1, distance / 2);
        let farthest = distance - nearest;
        let mut best: Option<P::Solution> = None;
        for step in 1..=farthest {
            let pick = match self.order {
                PathOrder::Random => rng.random_range(0..flips.len()),
                PathOrder::Greedy => {
                    let mut pick = 0;
                    let mut best_flip = prob.flip_move(&sol, flips[0]);
                    for (at, &i) in flips.iter().enumerate().skip(1) {
                        let flip = prob.flip_move(&sol, i);
                        if flip.is_better_than(&best_flip) {
                            best_flip = flip;
                            pick = at;
                        }
                    }
                    pick
                }
            };
            let i = flips.swap_remove(pick);
            prob.flip_move(&sol, i).apply_to_solution(prob, &mut sol)?;
            if step >= nearest {
                match &mut best {
                    Some(kept) if sol.is_better_than(kept) => kept.clone_from(&sol),
                    Some(_) => {}
                    None => best = Some(sol.clone()),
                }
            }
        }
        Ok(best.expect("the path has a solution between its two ends"))
    }
}

/// The variables on which `a` and `b` take different values, in index order.
pub fn differing_variables<P: BinaryProblem>(
    prob: &P,
    a: &P::Solution,
    b: &P::Solution,
) -> Vec<usize> {
    prob.variable_indices()
        .filter(|&i| P::variable(a, i) != P::variable(b, i))
        .collect()
}

/// Shared body for `SubProblemExtractable::lift_solution` when the sub-problem
/// keeps the parent's variable indices (MaxCut, VertexCover).
///
/// Clones `sol1`, then for every index in `sub_indices` where the parents
/// disagree (the sub-problem's free variables), flips the clone wherever it
/// differs from `sub_sol`. Fixed variables (parents agree) and free variables
/// absent from `sub_indices` inherit `sol1`. Because the flips go through the
/// problem's flip move, the lifted solution's cached `gain` and objective stay
/// consistent.
pub fn lift_binary_solution<P: BinaryProblem>(
    prob: &P,
    sol1: &P::Solution,
    sol2: &P::Solution,
    sub_sol: &P::Solution,
    sub_indices: impl Iterator<Item = usize>,
) -> P::Solution {
    let mut sol = sol1.clone();
    for i in sub_indices {
        if P::variable(sol1, i) == P::variable(sol2, i) {
            continue; // fixed variable — both parents agree
        }
        if P::variable(&sol, i) != P::variable(sub_sol, i) {
            prob.flip_move(&sol, i)
                .apply_to_solution(prob, &mut sol)
                .expect("flip on a valid variable index cannot fail");
        }
    }
    sol
}

/// Shared body for `SubProblemExtractable::lift_solution` when the sub-problem
/// re-indexes the free variables compactly (MaxSAT): the k-th
/// disagreeing variable of the parents corresponds to variable `k` of `sub_sol`.
pub fn lift_compact_binary_solution<P: BinaryProblem>(
    prob: &P,
    sol1: &P::Solution,
    sol2: &P::Solution,
    sub_sol: &P::Solution,
    n_vars: usize,
) -> P::Solution {
    let mut sol = sol1.clone();
    let mut sub_idx = 0;
    for i in 0..n_vars {
        if P::variable(sol1, i) == P::variable(sol2, i) {
            continue; // fixed variable — not part of the sub-problem
        }
        if P::variable(&sol, i) != P::variable(sub_sol, sub_idx) {
            prob.flip_move(&sol, i)
                .apply_to_solution(prob, &mut sol)
                .expect("flip on a valid variable index cannot fail");
        }
        sub_idx += 1;
    }
    sol
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::building_blocks::instance::Graph;
    use crate::problem::{MaxCut, MaxCutSolution, VertexCover};
    use crate::trait_defs::{Crossover, ProblemTrait};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn random_cut(n: usize, seed: u64) -> (MaxCut, SmallRng) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mc = MaxCut::new(Graph::erdos_renyi(n, 0.2, &mut rng));
        (mc, rng)
    }

    fn hamming(a: &MaxCutSolution, b: &MaxCutSolution) -> usize {
        a.x.iter().zip(&b.x).filter(|(p, q)| p != q).count()
    }

    /// The offspring keeps what the parents share, stays the asked distance
    /// from both, and carries caches a rebuild agrees with.
    #[test]
    fn the_offspring_lies_between_the_parents() {
        let (mc, mut rng) = random_cut(60, 1);
        for relinking in [
            PathRelinkingCrossover::greedy(),
            PathRelinkingCrossover::random(),
        ] {
            for _ in 0..20 {
                let a = mc.new_solution(&mut rng);
                let b = mc.new_solution(&mut rng);
                let flips = differing_variables(&mc, &a, &b);
                let d = flips.len();
                let child = relinking.relink(&mc, &a, flips, &mut rng).unwrap();
                for i in 0..60 {
                    if a.x[i] == b.x[i] {
                        assert_eq!(child.x[i], a.x[i]);
                    }
                }
                let from_a = hamming(&child, &a);
                assert_eq!(from_a + hamming(&child, &b), d);
                let nearest = (d as f64 / 3.0).ceil() as usize;
                assert!(from_a >= nearest && d - from_a >= nearest);
                let rebuilt = MaxCutSolution::new_from_assignment(&mc, child.x.clone());
                assert_eq!(child.objective, rebuilt.objective);
                assert_eq!(child.gain, rebuilt.gain);
            }
        }
    }

    /// The greedy path is at least as good at its first admitted step as any
    /// single flip order could make it, checked on the case of one admitted
    /// step: two flips, where the offspring is the better of the two.
    #[test]
    fn the_greedy_path_takes_the_better_flip() {
        let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (1, 2, 5.0), (2, 3, 1.0)]));
        let a = MaxCutSolution::new_from_assignment(&mc, vec![false, false, false, false]);
        let b = MaxCutSolution::new_from_assignment(&mc, vec![true, false, true, false]);
        let mut rng = SmallRng::seed_from_u64(0);
        let child = PathRelinkingCrossover::greedy()
            .relink(&mc, &a, differing_variables(&mc, &a, &b), &mut rng)
            .unwrap();
        // Flipping 2 cuts 5 + 1, flipping 0 cuts 1.
        assert_eq!(child.x, vec![false, false, true, false]);
        assert_eq!(child.objective, 6.0);
    }

    /// Parents one flip apart have nothing between them.
    #[test]
    fn parents_one_flip_apart_give_back_the_first() {
        let (mc, mut rng) = random_cut(10, 3);
        let a = mc.new_solution(&mut rng);
        let child = PathRelinkingCrossover::greedy()
            .relink(&mc, &a, vec![4], &mut rng)
            .unwrap();
        assert_eq!(child.x, a.x);
    }

    /// A cut and its complement are one solution, so relinking toward a
    /// parent's complement must walk the short way.
    #[test]
    fn max_cut_relinks_toward_the_nearer_labelling() {
        let (mc, mut rng) = random_cut(40, 5);
        let a = mc.new_solution(&mut rng);
        // The complement of `a` with three vertices moved.
        let mut x: Vec<bool> = a.x.iter().map(|v| !v).collect();
        for i in [3, 17, 29] {
            x[i] = !x[i];
        }
        let b = MaxCutSolution::new_from_assignment(&mc, x);
        let child = PathRelinkingCrossover::greedy()
            .crossover(&mc, &a, &b, &mut rng)
            .unwrap();
        assert!(hamming(&child, &a) <= 2, "the path is three flips long");
        assert!(hamming(&child, &a) >= 1);
    }

    #[test]
    fn it_runs_on_vertex_cover_and_replays_under_a_seed() {
        let mut rng = SmallRng::seed_from_u64(9);
        let vc = VertexCover::new(Graph::erdos_renyi(30, 0.2, &mut rng));
        let a = vc.new_solution(&mut rng);
        let b = vc.new_solution(&mut rng);
        let run = |seed| {
            let mut rng = SmallRng::seed_from_u64(seed);
            PathRelinkingCrossover::random()
                .crossover(&vc, &a, &b, &mut rng)
                .unwrap()
                .x
        };
        assert_eq!(run(4), run(4));
    }

    #[test]
    #[should_panic(expected = "distance_scale must be in (0, 1]")]
    fn a_zero_distance_scale_is_rejected() {
        let _ = PathRelinkingCrossover::greedy().with_distance_scale(0.0);
    }

    #[test]
    fn lift_binary_solution_fixes_agreeing_and_lifts_disagreeing_vars() {
        let mc = MaxCut::from_edges([(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]);
        // Variables 0 and 3 agree between the parents (fixed); 1 and 2 disagree (free).
        let sol1 = MaxCutSolution::new_from_assignment(&mc, vec![true, false, true, false]);
        let sol2 = MaxCutSolution::new_from_assignment(&mc, vec![true, true, false, false]);
        // The sub-solution proposes values for the free variables.
        let sub_sol = MaxCutSolution::new_from_assignment(&mc, vec![true, true, true, false]);

        let lifted = lift_binary_solution(&mc, &sol1, &sol2, &sub_sol, 0..sub_sol.x.len());

        // Fixed variables keep sol1's values; free variables take sub_sol's.
        assert_eq!(lifted.x, vec![true, true, true, false]);
        // The lifted solution's cached objective must match a from-scratch build.
        let rebuilt = MaxCutSolution::new_from_assignment(&mc, lifted.x.clone());
        assert_eq!(lifted.objective, rebuilt.objective);
        assert_eq!(lifted.gain, rebuilt.gain);
    }
}
