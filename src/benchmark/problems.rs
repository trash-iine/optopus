//! Per-problem registration for the benchmark.
//!
//! This file is the single place to touch when adding a new problem: one
//! line in the `problems!` table, which makes the [`ProblemKind`] variant and
//! the dispatch to the type, and one impl each of [`BenchmarkProblem`],
//! [`BenchmarkSolution`] and [`ConfigurableProblem`], whose neighbors are one
//! `neighbors!` table and whose crossovers are one `crossovers!` table.

use serde::{Deserialize, Serialize};

use super::config::{HeuristicConfig, NeighborKind};
use super::factory::{
    ConfigurableProblem, NeighborVisitor, ProvingSearch, build_heuristic, check_tenure,
    invalid_neighbor, unsupported,
};
use crate::building_blocks::representation::BinaryRelaxation;
use crate::error::OptError;
use crate::heuristic::{
    AdaptiveLargeNeighborhoodSearch, BranchAndBound, GreedyRandomizedConstruction, Heuristic,
    HybridGeneticSearchForVrp, LinKernighanHelsgaunForTsp, StopCondition, SubProblemBasedCrossover,
    WalkSatForSat, alns_for_flow_shop, alns_for_tsp, alns_for_vrp, bls_for_max_cut,
    iterated_greedy_for_flow_shop,
};
use crate::problem::{
    BinPacking, BinPackingGroupCrossover, BinPackingRelocateNeighbor, BinPackingSolution,
    BinPackingSwapNeighbor, EigenvalueRelaxation, FlowShop, FlowShopInsertNeighbor,
    FlowShopOrderCrossover, FlowShopSolution, FlowShopSwapNeighbor, GraphColoring,
    GraphColoringRecolorNeighbor, GraphColoringSolution, GraphColoringSwapNeighbor,
    GraphColoringUniformCrossover, JobShopPpxCrossover, JobShopRelocateNeighbor, JobShopScheduling,
    JobShopSolution, JobShopSwapNeighbor, MaxCutFlipNeighbor, MaxCutSolution, MaxCutSwapNeighbor,
    MaxCutUniformCrossover, Qap, QapOrderCrossover, QapSolution, QapSwapNeighbor, QuboFlipNeighbor,
    QuboSwapNeighbor, QuboUniformCrossover, SatUniformCrossover, TspOrderCrossover, VertexCover,
    VertexCoverFlipNeighbor, VertexCoverSolution, VertexCoverSwapNeighbor,
    VertexCoverUniformCrossover,
    max_cut::MaxCut,
    qubo::{Qubo, QuboSolution},
    sat::{Sat, SatFlipNeighbor, SatSolution, SatSwapNeighbor},
    tsp::{Tsp, TspRelocateNeighbor, TspSolution, TspTwoOptNeighbor},
    vrp::{
        Vrp, VrpOrderCrossover, VrpRelocateNeighbor, VrpSolution, VrpSwapNeighbor,
        VrpTwoOptNeighbor,
    },
};
use crate::search_state::{Crossover, Distance, Evaluate, ProblemTrait};
use crate::trait_defs::{BinaryProblem, BranchSpace, Relaxation, Ruinable};

// ---------------------------------------------------------------------------
// The problem table, the one list of the problems a config can name
// ---------------------------------------------------------------------------

/// What the problem table says about a problem.
pub(crate) trait Registered {
    /// The [`ProblemKind`] variant's name, as a config and the error messages
    /// spell the problem.
    const NAME: &'static str;
    /// Whether lower objectives are better. It is the direction the
    /// solution's `Evaluate` reports, which a test checks for every row, and
    /// the report reads it where it has no solution to ask.
    const MINIMIZE: bool;

    /// Reads an instance file, with the problem type's own `load_file`.
    fn load_instance(path: &str) -> Result<Self, OptError>
    where
        Self: Sized;
}

macro_rules! minimizes {
    (minimize) => {
        true
    };
    (maximize) => {
        false
    };
}

/// Writes [`ProblemKind`] with its `ALL`, `minimize` and `valid_neighbors`,
/// [`with_problem`], and every [`Registered`] impl from one list of
/// `Variant => ProblemType, direction;`. Every problem type reads its
/// instances with an inherent `load_file(path)`.
macro_rules! problems {
    ($($kind:ident => $ty:ty, $direction:ident;)+) => {
        /// Problem type discriminant used in config files.
        ///
        /// `FormulaProblem` and `IntegerProblem` are intentionally absent. They are
        /// library-only, built in code from an [`Expr`](crate::problem::Expr) or a
        /// closure, and have no instance file format for the benchmark runner to load.
        #[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
        pub enum ProblemKind {
            $($kind,)+
        }

        impl ProblemKind {
            /// Every kind, in declaration order, for code that sweeps the problems.
            pub const ALL: &[ProblemKind] = &[$(ProblemKind::$kind,)+];

            /// Whether this problem is minimized (drives report statistics).
            pub fn minimize(&self) -> bool {
                match self {
                    $(ProblemKind::$kind => <$ty as Registered>::MINIMIZE,)+
                }
            }

            /// The neighborhood kinds supported by this problem.
            pub fn valid_neighbors(&self) -> &'static [NeighborKind] {
                match self {
                    $(ProblemKind::$kind => <$ty as ConfigurableProblem>::VALID_NEIGHBORS,)+
                }
            }
        }

        /// Maps the runtime [`ProblemKind`] to the concrete problem type.
        pub(crate) fn with_problem<V: ProblemVisitor>(kind: &ProblemKind, visitor: V) -> V::Output {
            match kind {
                $(ProblemKind::$kind => visitor.visit::<$ty>(),)+
            }
        }

        $(
            impl Registered for $ty {
                const NAME: &'static str = stringify!($kind);
                const MINIMIZE: bool = minimizes!($direction);

                fn load_instance(path: &str) -> Result<Self, OptError> {
                    <$ty>::load_file(path)
                }
            }
        )+
    };
}

problems! {
    MaxCut => MaxCut, maximize;
    Qubo => Qubo, minimize;
    Sat => Sat, maximize;
    Tsp => Tsp, minimize;
    VertexCover => VertexCover, minimize;
    JobShop => JobShopScheduling, minimize;
    FlowShop => FlowShop, minimize;
    Qap => Qap, minimize;
    BinPacking => BinPacking, minimize;
    Vrp => Vrp, minimize;
    GraphColoring => GraphColoring, minimize;
}

/// Writes `VALID_NEIGHBORS` and `with_neighbor` inside a
/// [`ConfigurableProblem`] impl from one list of `NeighborKind => move type`,
/// so the kinds a config may name and the kinds that dispatch are one list.
macro_rules! neighbors {
    ($($kind:ident => $mv:ty),+ $(,)?) => {
        const VALID_NEIGHBORS: &'static [NeighborKind] = &[$(NeighborKind::$kind),+];

        fn with_neighbor<V: NeighborVisitor<Self>>(
            kind: &NeighborKind,
            visitor: V,
        ) -> Result<V::Output, OptError> {
            match kind {
                $(NeighborKind::$kind => Ok(visitor.visit::<$mv>()),)+
                #[allow(unreachable_patterns)]
                other => Err(invalid_neighbor::<Self>(other)),
            }
        }
    };
}

/// Writes `build_crossover` inside a [`ConfigurableProblem`] impl from one
/// list of `Name => operator`, the name a config's `crossover_kind` gives.
/// The first row is the default when the config names none.
macro_rules! crossovers {
    ($($name:ident => $op:expr),+ $(,)?) => {
        fn build_crossover(kind: Option<&str>) -> Result<Box<dyn Crossover<Self>>, OptError> {
            const NAMES: &[&str] = &[$(stringify!($name)),+];
            match kind.unwrap_or(NAMES[0]) {
                $(stringify!($name) => Ok(Box::new($op)),)+
                other => Err(unknown_crossover::<Self>(other, NAMES)),
            }
        }
    };
}

/// Writes `build_branch_and_bound` inside the [`ConfigurableProblem`] impl
/// of a binary problem whose one bound is [`BinaryRelaxation`].
macro_rules! binary_branch_and_bound {
    () => {
        fn build_branch_and_bound(
            config: &HeuristicConfig,
            cond: StopCondition,
        ) -> Result<Box<dyn ProvingSearch<Self>>, OptError> {
            bnb_relaxation(config, Self::NAME, false)?;
            branch_and_bound(config, cond, BinaryRelaxation)
        }
    };
}

/// The error for a `crossover_kind` the problem has no row for.
fn unknown_crossover<P: Registered>(kind: &str, names: &[&str]) -> OptError {
    OptError::Config(format!(
        "Unknown crossover_kind '{kind}' for {} (expected '{}')",
        P::NAME,
        names.join("' or '")
    ))
}

// ---------------------------------------------------------------------------
// BenchmarkProblem / BenchmarkSolution traits
// ---------------------------------------------------------------------------

/// Problem types whose solutions a report writes and a config gives back.
pub trait BenchmarkProblem: ProblemTrait + Sized {
    /// The solution `indices` stands for, the inverse of
    /// [`BenchmarkSolution::encode_as_indices`], so a report's `solution` can
    /// start a run or be scored again. Indices that do not make a solution of
    /// this instance are an [`OptError::Config`] saying why.
    fn decode_indices(&self, indices: &[usize]) -> Result<Self::Solution, OptError>;
}

/// The error for indices that do not make a solution of the instance.
fn misfit(err: OptError) -> OptError {
    let detail = match err {
        OptError::InvalidState(m) | OptError::Config(m) => m,
        other => other.to_string(),
    };
    OptError::Config(format!("the solution does not fit the instance: {detail}"))
}

/// The binary assignment that sets exactly the variables in `indices`.
fn binary_from_indices<P: BinaryProblem>(
    prob: &P,
    indices: &[usize],
) -> Result<P::Solution, OptError> {
    let n = prob.variable_indices().max().map_or(0, |i| i + 1);
    let mut x = vec![false; n];
    for &i in indices {
        if i >= n {
            return Err(misfit(OptError::InvalidState(format!(
                "index {i} is outside the {n} variables"
            ))));
        }
        if std::mem::replace(&mut x[i], true) {
            return Err(misfit(OptError::InvalidState(format!(
                "index {i} is given twice"
            ))));
        }
    }
    prob.solution_from_assignment(&x).map_err(misfit)
}

/// Solution types that expose generic metrics needed by the benchmark runner.
pub trait BenchmarkSolution: Clone + Evaluate {
    /// The objective a report gives, by default the number the solution's
    /// `Evaluate` reports.
    fn best_objective_f64(&self) -> f64 {
        self.evaluate().raw()
    }
    fn encode_as_indices(&self) -> Vec<usize>;
}

/// The positions of the `true` variables, a binary solution's encoding.
fn true_indices(x: &[bool]) -> Vec<usize> {
    x.iter()
        .enumerate()
        .filter(|&(_, &v)| v)
        .map(|(i, _)| i)
        .collect()
}

impl BenchmarkProblem for MaxCut {
    fn decode_indices(&self, indices: &[usize]) -> Result<Self::Solution, OptError> {
        binary_from_indices(self, indices)
    }
}

impl BenchmarkSolution for MaxCutSolution {
    fn encode_as_indices(&self) -> Vec<usize> {
        true_indices(&self.x)
    }
}

impl BenchmarkProblem for Qubo {
    fn decode_indices(&self, indices: &[usize]) -> Result<Self::Solution, OptError> {
        binary_from_indices(self, indices)
    }
}

impl BenchmarkSolution for QuboSolution {
    fn encode_as_indices(&self) -> Vec<usize> {
        true_indices(&self.x)
    }
}

impl BenchmarkProblem for Sat {
    fn decode_indices(&self, indices: &[usize]) -> Result<Self::Solution, OptError> {
        binary_from_indices(self, indices)
    }
}

impl BenchmarkSolution for SatSolution {
    fn encode_as_indices(&self) -> Vec<usize> {
        true_indices(&self.x)
    }
}

impl BenchmarkProblem for Tsp {
    fn decode_indices(&self, indices: &[usize]) -> Result<Self::Solution, OptError> {
        let tour = indices.to_vec();
        if let Some(&city) = tour.iter().find(|&&c| c >= self.get_n()) {
            return Err(misfit(OptError::InvalidState(format!(
                "city {city} is outside the {} cities",
                self.get_n()
            ))));
        }
        let objective = self.calculate_tour_length(&tour).map_err(misfit)?;
        Ok(TspSolution { tour, objective })
    }
}

impl BenchmarkSolution for TspSolution {
    fn encode_as_indices(&self) -> Vec<usize> {
        self.tour.clone()
    }
}

impl BenchmarkProblem for Vrp {
    fn decode_indices(&self, indices: &[usize]) -> Result<Self::Solution, OptError> {
        // `encode_as_indices` writes `0, r0…, 0, r1…, 0`, every slot when the
        // fleet has more than one vehicle type and the busy ones otherwise,
        // where any slot serves as well as another.
        if indices.first() != Some(&0) || indices.last() != Some(&0) {
            return Err(misfit(OptError::InvalidState(
                "the routes start and end at the depot 0".into(),
            )));
        }
        // Every 0 after the first closes a route, so `0, 0` is one idle slot.
        let mut routes = Vec::new();
        let mut route = Vec::new();
        for &c in &indices[1..] {
            if c == 0 {
                routes.push(std::mem::take(&mut route));
            } else {
                route.push(c);
            }
        }
        let slots = self.num_slots();
        if routes.len() != slots {
            if self.vehicle_types().len() > 1 {
                return Err(misfit(OptError::InvalidState(format!(
                    "a fleet of several vehicle types needs one route per slot, {slots}, \
                     but {} were given",
                    routes.len()
                ))));
            }
            routes.retain(|r| !r.is_empty());
            if routes.len() > slots {
                return Err(misfit(OptError::InvalidState(format!(
                    "{} routes for {slots} vehicles",
                    routes.len()
                ))));
            }
            routes.resize(slots, Vec::new());
        }
        self.solution_from_routes(routes).map_err(misfit)
    }
}

impl BenchmarkSolution for VrpSolution {
    fn encode_as_indices(&self) -> Vec<usize> {
        // Flatten with the depot (0) as the separator, `0, r0…, 0, r1…, 0`.
        // An idle slot leaves no trace when every vehicle is of one type. In a
        // fleet of several types it stays as an empty route, `0, 0`, so the
        // encoding still says which type ran each route.
        let several_types = self.used_count.len() > 1;
        let mut out = vec![0];
        for route in self
            .routes
            .iter()
            .filter(|route| several_types || !route.is_empty())
        {
            out.extend_from_slice(route);
            out.push(0);
        }
        out
    }
}

impl BenchmarkProblem for VertexCover {
    fn decode_indices(&self, indices: &[usize]) -> Result<Self::Solution, OptError> {
        binary_from_indices(self, indices)
    }
}

impl BenchmarkSolution for VertexCoverSolution {
    fn encode_as_indices(&self) -> Vec<usize> {
        true_indices(&self.x)
    }
}

impl BenchmarkProblem for JobShopScheduling {
    fn decode_indices(&self, indices: &[usize]) -> Result<Self::Solution, OptError> {
        let operations = indices.to_vec();
        let (objective, completion_times) = self.decode(&operations).map_err(misfit)?;
        Ok(JobShopSolution {
            operations,
            objective,
            completion_times,
        })
    }
}

impl BenchmarkSolution for JobShopSolution {
    fn encode_as_indices(&self) -> Vec<usize> {
        self.operations.clone()
    }
}

impl BenchmarkProblem for FlowShop {
    fn decode_indices(&self, indices: &[usize]) -> Result<Self::Solution, OptError> {
        self.solution_from_sequence(indices.to_vec())
            .map_err(misfit)
    }
}

impl BenchmarkSolution for FlowShopSolution {
    fn encode_as_indices(&self) -> Vec<usize> {
        self.sequence.clone()
    }
}

impl BenchmarkProblem for GraphColoring {
    fn decode_indices(&self, indices: &[usize]) -> Result<Self::Solution, OptError> {
        self.solution_from_colors(indices.to_vec()).map_err(misfit)
    }
}

impl BenchmarkSolution for GraphColoringSolution {
    fn encode_as_indices(&self) -> Vec<usize> {
        // The color assigned to each vertex (0-indexed color ids).
        self.colors.clone()
    }
}

impl ConfigurableProblem for MaxCut {
    neighbors! {
        Flip => MaxCutFlipNeighbor,
        Swap => MaxCutSwapNeighbor,
    }

    fn build_special_heuristic(
        config: &HeuristicConfig,
        cond: StopCondition,
    ) -> Result<Box<dyn Heuristic<Self>>, OptError> {
        match config {
            HeuristicConfig::BreakoutLocalSearch {
                tabu_tenure,
                t,
                l0,
                p0,
                q,
                ..
            } => {
                check_tenure(*tabu_tenure)?;
                // `q` and `1 - q` are the shares of the two weak perturbations.
                if !(0.0..=1.0).contains(q) {
                    return Err(OptError::Config(format!("'q' must be in [0, 1], got {q}")));
                }
                Ok(Box::new(bls_for_max_cut(
                    cond,
                    *tabu_tenure,
                    *t,
                    *l0,
                    *p0,
                    *q,
                )))
            }
            _ => Err(unsupported::<Self>(config)),
        }
    }

    fn build_branch_and_bound(
        config: &HeuristicConfig,
        cond: StopCondition,
    ) -> Result<Box<dyn ProvingSearch<Self>>, OptError> {
        match bnb_relaxation(config, Self::NAME, true)? {
            None => branch_and_bound(config, cond, BinaryRelaxation),
            Some(eigen) => branch_and_bound(config, cond, eigen),
        }
    }

    crossovers! {
        Uniform => MaxCutUniformCrossover,
        // Solves the sub-MaxCut formed by the variables the two parents
        // disagree on (TSHEA/MOH-style memetic recombination). The
        // sub-problem shrinks as the population converges, so a bounded
        // BLS is enough to solve it well.
        SubProblem => SubProblemBasedCrossover {
            sub_heuristic: Box::new(bls_for_max_cut(
                StopCondition::iterations(50_000).with_failed_updates(10_000),
                (6, 100),
                1_000,
                20,
                0.8,
                0.5,
            )),
        },
    }
}

impl ConfigurableProblem for Qubo {
    neighbors! {
        Flip => QuboFlipNeighbor,
        Swap => QuboSwapNeighbor,
    }

    binary_branch_and_bound!();

    crossovers! {
        Uniform => QuboUniformCrossover,
    }
}

impl ConfigurableProblem for Sat {
    neighbors! {
        Flip => SatFlipNeighbor,
        Swap => SatSwapNeighbor,
    }

    fn build_special_heuristic(
        config: &HeuristicConfig,
        cond: StopCondition,
    ) -> Result<Box<dyn Heuristic<Self>>, OptError> {
        match config {
            HeuristicConfig::WalkSat {
                noise,
                adaptive_noise,
                ..
            } => {
                let noise = noise.unwrap_or(0.3);
                if !(0.0..=1.0).contains(&noise) {
                    return Err(OptError::Config(
                        "'noise' must be within [0, 1]".to_string(),
                    ));
                }
                Ok(Box::new(WalkSatForSat::new(
                    cond,
                    noise,
                    adaptive_noise.unwrap_or(false),
                )))
            }
            _ => Err(unsupported::<Self>(config)),
        }
    }

    binary_branch_and_bound!();

    crossovers! {
        Uniform => SatUniformCrossover,
    }
}

impl ConfigurableProblem for Tsp {
    neighbors! {
        TwoOpt => TspTwoOptNeighbor,
        Relocate => TspRelocateNeighbor,
    }

    fn build_special_heuristic(
        config: &HeuristicConfig,
        cond: StopCondition,
    ) -> Result<Box<dyn Heuristic<Self>>, OptError> {
        match config {
            HeuristicConfig::LinKernighanHelsgaun {
                num_neighbors,
                max_depth,
                ..
            } => Ok(Box::new(LinKernighanHelsgaunForTsp::new(
                cond,
                num_neighbors.unwrap_or(5),
                max_depth.unwrap_or(5),
            ))),
            HeuristicConfig::GreedyRandomizedConstruction { .. }
            | HeuristicConfig::AdaptiveLargeNeighborhoodSearch { .. } => {
                ruin_and_recreate(config, cond, alns_for_tsp)
            }
            _ => Err(unsupported::<Self>(config)),
        }
    }

    crossovers! {
        Order => TspOrderCrossover,
    }
}

impl ConfigurableProblem for Vrp {
    neighbors! {
        Relocate => VrpRelocateNeighbor,
        Swap => VrpSwapNeighbor,
        TwoOpt => VrpTwoOptNeighbor,
    }

    fn build_special_heuristic(
        config: &HeuristicConfig,
        cond: StopCondition,
    ) -> Result<Box<dyn Heuristic<Self>>, OptError> {
        match config {
            HeuristicConfig::GreedyRandomizedConstruction { .. }
            | HeuristicConfig::AdaptiveLargeNeighborhoodSearch { .. } => {
                ruin_and_recreate(config, cond, alns_for_vrp)
            }
            HeuristicConfig::HybridGeneticSearch {
                min_population_size,
                generation_size,
                granularity,
                target_feasible,
                restart_generations,
                ..
            } => {
                let min_population_size = min_population_size.unwrap_or(25);
                let generation_size = generation_size.unwrap_or(40);
                let granularity = granularity.unwrap_or(20);
                let target_feasible = target_feasible.unwrap_or(0.2);
                // The constructor asserts the same, so a config fails here
                // rather than panicking in a run.
                if min_population_size < 4 {
                    return Err(OptError::Config(format!(
                        "'min_population_size' must be at least 4, got {min_population_size}"
                    )));
                }
                for (name, value) in [
                    ("generation_size", generation_size),
                    ("granularity", granularity),
                ] {
                    if value == 0 {
                        return Err(OptError::Config(format!("'{name}' must be at least 1")));
                    }
                }
                if !(target_feasible > 0.0 && target_feasible < 1.0) {
                    return Err(OptError::Config(format!(
                        "'target_feasible' must be in (0, 1), got {target_feasible}"
                    )));
                }
                Ok(Box::new(HybridGeneticSearchForVrp::new(
                    cond,
                    min_population_size,
                    generation_size,
                    granularity,
                    target_feasible,
                    Some(restart_generations.unwrap_or(20_000)),
                )))
            }
            _ => Err(unsupported::<Self>(config)),
        }
    }

    crossovers! {
        Order => VrpOrderCrossover,
    }
}

impl ConfigurableProblem for VertexCover {
    neighbors! {
        Flip => VertexCoverFlipNeighbor,
        Swap => VertexCoverSwapNeighbor,
    }

    binary_branch_and_bound!();

    crossovers! {
        Uniform => VertexCoverUniformCrossover,
    }
}

impl ConfigurableProblem for JobShopScheduling {
    neighbors! {
        Swap => JobShopSwapNeighbor,
        Relocate => JobShopRelocateNeighbor,
    }

    crossovers! {
        Ppx => JobShopPpxCrossover,
    }
}

impl BenchmarkProblem for Qap {
    fn decode_indices(&self, indices: &[usize]) -> Result<Self::Solution, OptError> {
        self.solution_from_assignment(indices.to_vec())
            .map_err(misfit)
    }
}

impl BenchmarkSolution for QapSolution {
    fn encode_as_indices(&self) -> Vec<usize> {
        self.assignment.clone()
    }
}

impl ConfigurableProblem for Qap {
    neighbors! {
        Swap => QapSwapNeighbor,
    }

    crossovers! {
        Order => QapOrderCrossover,
    }
}

/// The ALNS defaults, checked here so a bad config fails at startup rather
/// than panicking in the constructor.
fn alns_params(
    removal_fraction: Option<f64>,
    cooling_rate: Option<f64>,
) -> Result<(f64, f64), OptError> {
    let removal_fraction = removal_fraction.unwrap_or(0.15);
    let cooling_rate = cooling_rate.unwrap_or(0.9995);
    for (name, value) in [
        ("removal_fraction", removal_fraction),
        ("cooling_rate", cooling_rate),
    ] {
        if !(value > 0.0 && value <= 1.0) {
            return Err(OptError::Config(format!(
                "'{name}' must be in (0, 1], got {value}"
            )));
        }
    }
    Ok((removal_fraction, cooling_rate))
}

/// The GRASP construction or the ALNS of a `GreedyRandomizedConstruction` or
/// `AdaptiveLargeNeighborhoodSearch` config, on a `Ruinable` problem. `alns`
/// makes the problem's ALNS from the stop condition, the removal fraction
/// and the cooling rate, which are checked here so a bad config fails at
/// startup rather than panicking in a run.
fn ruin_and_recreate<P>(
    config: &HeuristicConfig,
    cond: StopCondition,
    alns: fn(StopCondition, f64, f64) -> AdaptiveLargeNeighborhoodSearch<P>,
) -> Result<Box<dyn Heuristic<P>>, OptError>
where
    P: Ruinable + 'static,
    P::Solution: Evaluate,
{
    match config {
        HeuristicConfig::GreedyRandomizedConstruction { alpha, .. } => {
            let alpha = alpha.unwrap_or(0.2);
            if !(0.0..=1.0).contains(&alpha) {
                return Err(OptError::Config(format!(
                    "'alpha' must be in [0, 1], got {alpha}"
                )));
            }
            Ok(Box::new(GreedyRandomizedConstruction::<P>::new(
                cond, alpha,
            )))
        }
        HeuristicConfig::AdaptiveLargeNeighborhoodSearch {
            removal_fraction,
            cooling_rate,
            ..
        } => {
            let (removal_fraction, cooling_rate) = alns_params(*removal_fraction, *cooling_rate)?;
            Ok(Box::new(alns(cond, removal_fraction, cooling_rate)))
        }
        _ => unreachable!("ruin_and_recreate is only called on GRASP and ALNS configs"),
    }
}

/// A `BranchAndBound` config's search, `steps[0]` searching each node and
/// `relaxation` pruning them.
fn branch_and_bound<P, R>(
    config: &HeuristicConfig,
    cond: StopCondition,
    relaxation: R,
) -> Result<Box<dyn ProvingSearch<P>>, OptError>
where
    P: ConfigurableProblem + BranchSpace,
    P::Solution: BenchmarkSolution + Distance + Evaluate,
    R: Relaxation<P> + 'static,
{
    let steps = config.steps();
    if steps.len() != 1 {
        return Err(OptError::Config(format!(
            "BranchAndBound requires exactly 1 step (the search of each node), but got {}",
            steps.len()
        )));
    }
    let node_search = build_heuristic::<P>(&steps[0])?;
    Ok(Box::new(BranchAndBound::new(cond, node_search, relaxation)))
}

/// The bound a `BranchAndBound` config names on `problem`: `None` for
/// "Binary", the default, and the eigenvalue bound with its limits for
/// "Eigenvalue", which only a problem with `has_eigenvalue` takes. The limits
/// are refused alongside any other bound, so a config cannot set them to no
/// effect.
fn bnb_relaxation(
    config: &HeuristicConfig,
    problem: &str,
    has_eigenvalue: bool,
) -> Result<Option<EigenvalueRelaxation>, OptError> {
    let HeuristicConfig::BranchAndBound {
        relaxation,
        eigen_max_iterations,
        eigen_lanczos_steps,
        eigen_certify_limit,
        ..
    } = config
    else {
        unreachable!("bnb_relaxation is only called on a BranchAndBound config");
    };
    let limits = [
        ("eigen_max_iterations", *eigen_max_iterations),
        ("eigen_lanczos_steps", *eigen_lanczos_steps),
        ("eigen_certify_limit", *eigen_certify_limit),
    ];
    match relaxation.as_deref().unwrap_or("Binary") {
        "Binary" => {
            if let Some((name, _)) = limits.iter().find(|(_, v)| v.is_some()) {
                return Err(OptError::Config(format!(
                    "'{name}' applies only to relaxation = 'Eigenvalue'"
                )));
            }
            Ok(None)
        }
        "Eigenvalue" if has_eigenvalue => {
            if let Some((name, _)) = limits.iter().find(|(_, v)| *v == Some(0)) {
                return Err(OptError::Config(format!("'{name}' must be at least 1")));
            }
            let mut eigen = EigenvalueRelaxation::new();
            if let Some(n) = *eigen_max_iterations {
                eigen = eigen.with_max_iterations(n);
            }
            if let Some(n) = *eigen_lanczos_steps {
                eigen = eigen.with_lanczos_steps(n);
            }
            if let Some(n) = *eigen_certify_limit {
                eigen = eigen.with_certify_limit(n);
            }
            Ok(Some(eigen))
        }
        other => Err(OptError::Config(format!(
            "Unknown relaxation '{other}' for {problem} (expected {})",
            if has_eigenvalue {
                "'Binary' or 'Eigenvalue'"
            } else {
                "'Binary'"
            }
        ))),
    }
}

impl BenchmarkProblem for BinPacking {
    fn decode_indices(&self, indices: &[usize]) -> Result<Self::Solution, OptError> {
        self.solution_from_bins(indices.to_vec()).map_err(misfit)
    }
}

impl BenchmarkSolution for BinPackingSolution {
    fn best_objective_f64(&self) -> f64 {
        self.num_bins() as f64
    }
    fn encode_as_indices(&self) -> Vec<usize> {
        self.bin_of.clone()
    }
}

impl ConfigurableProblem for BinPacking {
    neighbors! {
        Relocate => BinPackingRelocateNeighbor,
        Swap => BinPackingSwapNeighbor,
    }

    fn build_special_heuristic(
        config: &HeuristicConfig,
        cond: StopCondition,
    ) -> Result<Box<dyn Heuristic<Self>>, OptError> {
        match config {
            HeuristicConfig::GreedyRandomizedConstruction { .. }
            | HeuristicConfig::AdaptiveLargeNeighborhoodSearch { .. } => ruin_and_recreate(
                config,
                cond,
                AdaptiveLargeNeighborhoodSearch::<BinPacking>::new,
            ),
            _ => Err(unsupported::<Self>(config)),
        }
    }

    crossovers! {
        Group => BinPackingGroupCrossover,
    }
}

impl ConfigurableProblem for FlowShop {
    neighbors! {
        Swap => FlowShopSwapNeighbor,
        Relocate => FlowShopInsertNeighbor,
    }

    fn build_special_heuristic(
        config: &HeuristicConfig,
        cond: StopCondition,
    ) -> Result<Box<dyn Heuristic<Self>>, OptError> {
        match config {
            HeuristicConfig::IteratedGreedy {
                removal_count,
                temperature_factor,
                ..
            } => {
                let removal_count = removal_count.unwrap_or(4);
                let temperature_factor = temperature_factor.unwrap_or(0.4);
                if removal_count == 0 {
                    return Err(OptError::Config(
                        "'removal_count' must be at least 1".to_string(),
                    ));
                }
                if !(temperature_factor.is_finite() && temperature_factor > 0.0) {
                    return Err(OptError::Config(format!(
                        "'temperature_factor' must be positive and finite, got {temperature_factor}"
                    )));
                }
                Ok(Box::new(iterated_greedy_for_flow_shop(
                    cond,
                    removal_count,
                    temperature_factor,
                )))
            }
            HeuristicConfig::GreedyRandomizedConstruction { .. }
            | HeuristicConfig::AdaptiveLargeNeighborhoodSearch { .. } => {
                ruin_and_recreate(config, cond, alns_for_flow_shop)
            }
            _ => Err(unsupported::<Self>(config)),
        }
    }

    crossovers! {
        Order => FlowShopOrderCrossover,
    }
}

impl ConfigurableProblem for GraphColoring {
    neighbors! {
        // `Flip` is the single-vertex recolor.
        Flip => GraphColoringRecolorNeighbor,
        Swap => GraphColoringSwapNeighbor,
    }

    crossovers! {
        Uniform => GraphColoringUniformCrossover,
    }
}

/// Callback invoked with the concrete problem type for a [`ProblemKind`].
pub(crate) trait ProblemVisitor {
    type Output;
    fn visit<P>(self) -> Self::Output
    where
        P: ConfigurableProblem,
        P::Solution: BenchmarkSolution + Distance + Evaluate;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trait_defs::Evaluable;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    /// A small instance of `kind`. The match names every kind, so a new row of
    /// the problem table does not compile here until it has one.
    fn instance(kind: &ProblemKind) -> String {
        let written = |name: &str, text: &str| {
            let path = std::env::temp_dir().join(format!(
                "optopus_registered_{name}_{}.txt",
                std::process::id()
            ));
            std::fs::write(&path, text).unwrap();
            path.display().to_string()
        };
        match kind {
            ProblemKind::MaxCut => "data/instances/max_cut/sample.txt".into(),
            ProblemKind::Qubo => "data/instances/qubo/sample.txt".into(),
            ProblemKind::Sat => "data/instances/sat/sample.cnf".into(),
            ProblemKind::Tsp => "data/instances/tsp/test_data.txt".into(),
            ProblemKind::VertexCover => written("vertex_cover", "4 3\n1 2\n2 3\n3 4\n"),
            ProblemKind::JobShop => "data/instances/jssp/ft06.txt".into(),
            ProblemKind::FlowShop => "data/instances/flow_shop/taillard/ta001.txt".into(),
            ProblemKind::Qap => written("qap", "3\n0 1 2\n1 0 1\n2 1 0\n0 3 1\n3 0 2\n1 2 0\n"),
            ProblemKind::BinPacking => "data/instances/bin_packing/falkenauer/u120_00.txt".into(),
            ProblemKind::Vrp => "data/instances/vrp/demo16.vrp".into(),
            ProblemKind::GraphColoring => "data/instances/graph_coloring/example.txt".into(),
        }
    }

    /// The table's direction and the one a solution of the problem evaluates in.
    struct Directions<'a> {
        path: &'a str,
    }

    impl ProblemVisitor for Directions<'_> {
        type Output = (bool, bool);
        fn visit<P>(self) -> (bool, bool)
        where
            P: ConfigurableProblem,
            P::Solution: BenchmarkSolution + Distance + Evaluate,
        {
            let prob = P::load_instance(self.path).unwrap_or_else(|e| panic!("{}: {e}", self.path));
            let solution = prob.new_solution(&mut SmallRng::seed_from_u64(0));
            (
                P::MINIMIZE,
                matches!(solution.evaluate(), Evaluable::Minimize(_)),
            )
        }
    }

    /// The report summarises a problem in the direction its row states, and
    /// the heuristics search in the one its solution's `Evaluate` reports, so
    /// every row is checked against a solution of its problem.
    #[test]
    fn every_row_states_the_direction_its_solution_evaluates_in() {
        let temp = std::env::temp_dir();
        for kind in ProblemKind::ALL {
            let path = instance(kind);
            let (table, evaluated) = with_problem(kind, Directions { path: &path });
            assert_eq!(table, evaluated, "{kind:?} minimizes = {evaluated}");
            if std::path::Path::new(&path).starts_with(&temp) {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
}
