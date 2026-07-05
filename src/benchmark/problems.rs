//! Per-problem registration for the benchmark.
//!
//! This file is the single place to touch when adding a new problem:
//! implement [`BenchmarkProblem`] + [`BenchmarkSolution`] +
//! [`ConfigurableProblem`], and add the [`with_problem`] arm for the new
//! [`ProblemKind`] variant.

use super::config::{HeuristicConfig, NeighborKind, ProblemKind};
use super::factory::{ConfigurableProblem, NeighborVisitor, invalid_neighbor};
use crate::error::OptError;
use crate::heuristic::{
    BreakoutLocalSearchForMaxCut, Heuristic, LinKernighanHelsgaunForTsp, StopCondition,
};
use crate::problem::{
    JobShopPpxCrossover, JobShopRelocateNeighbor, JobShopScheduling, JobShopSolution,
    JobShopSwapNeighbor, MaxCutFlipNeighbor, MaxCutSolution, MaxCutSwapNeighbor,
    MaxCutUniformCrossover, QuboFlipNeighbor, QuboSwapNeighbor, QuboUniformCrossover,
    SatUniformCrossover, TspOrderCrossover, VertexCover, VertexCoverFlipNeighbor,
    VertexCoverSolution, VertexCoverSwapNeighbor, VertexCoverUniformCrossover,
    max_cut::MaxCut,
    qubo::{Qubo, QuboSolution},
    sat::{Sat, SatFlipNeighbor, SatSolution, SatSwapNeighbor},
    tsp_2d::{TspRelocateNeighbor, TspSolution, TspTwoOptNeighbor, TspWithCoordinates},
};
use crate::search_state::{Crossover, Distance, ProblemTrait};

// ---------------------------------------------------------------------------
// BenchmarkProblem / BenchmarkSolution traits
// ---------------------------------------------------------------------------

/// Problem types that can load an instance from a file path.
pub trait BenchmarkProblem: ProblemTrait + Sized {
    fn load_instance(path: &str) -> Result<Self, OptError>;
}

/// Solution types that expose generic metrics needed by the benchmark runner.
pub trait BenchmarkSolution: Clone {
    fn best_objective_f64(&self) -> f64;
    fn encode_as_indices(&self) -> Vec<usize>;
}

impl BenchmarkProblem for MaxCut {
    fn load_instance(path: &str) -> Result<Self, OptError> {
        MaxCut::load_file(path)
    }
}

impl BenchmarkSolution for MaxCutSolution {
    fn best_objective_f64(&self) -> f64 {
        self.objective as f64
    }
    fn encode_as_indices(&self) -> Vec<usize> {
        self.x
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v)
            .map(|(i, _)| i)
            .collect()
    }
}

impl BenchmarkProblem for Qubo {
    fn load_instance(path: &str) -> Result<Self, OptError> {
        Qubo::load_file(path)
    }
}

impl BenchmarkSolution for QuboSolution {
    fn best_objective_f64(&self) -> f64 {
        self.objective as f64
    }
    fn encode_as_indices(&self) -> Vec<usize> {
        self.x
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v)
            .map(|(i, _)| i)
            .collect()
    }
}

impl BenchmarkProblem for Sat {
    fn load_instance(path: &str) -> Result<Self, OptError> {
        Sat::load_file(path)
    }
}

impl BenchmarkSolution for SatSolution {
    fn best_objective_f64(&self) -> f64 {
        self.n_satisfied as f64
    }
    fn encode_as_indices(&self) -> Vec<usize> {
        self.x
            .iter()
            .enumerate()
            .filter(|&(_, v)| *v)
            .map(|(i, _)| i)
            .collect()
    }
}

impl BenchmarkProblem for TspWithCoordinates {
    fn load_instance(path: &str) -> Result<Self, OptError> {
        TspWithCoordinates::load_file(path)
    }
}

impl BenchmarkSolution for TspSolution {
    fn best_objective_f64(&self) -> f64 {
        self.objective
    }
    fn encode_as_indices(&self) -> Vec<usize> {
        self.tour.clone()
    }
}

impl BenchmarkProblem for VertexCover {
    fn load_instance(path: &str) -> Result<Self, OptError> {
        VertexCover::load_file(path)
    }
}

impl BenchmarkSolution for VertexCoverSolution {
    fn best_objective_f64(&self) -> f64 {
        // Use penalty-augmented objective so infeasible solutions are correctly penalized.
        self.objective as f64
    }
    fn encode_as_indices(&self) -> Vec<usize> {
        self.x
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v)
            .map(|(i, _)| i)
            .collect()
    }
}

impl BenchmarkProblem for JobShopScheduling {
    fn load_instance(path: &str) -> Result<Self, OptError> {
        JobShopScheduling::load_file(path)
    }
}

impl BenchmarkSolution for JobShopSolution {
    fn best_objective_f64(&self) -> f64 {
        self.objective as f64
    }
    fn encode_as_indices(&self) -> Vec<usize> {
        self.operations.clone()
    }
}

impl ConfigurableProblem for MaxCut {
    const NAME: &'static str = "MaxCut";
    const MINIMIZE: bool = false;
    const VALID_NEIGHBORS: &'static [NeighborKind] = &[NeighborKind::Flip, NeighborKind::Swap];

    fn with_neighbor<V: NeighborVisitor<Self>>(
        kind: &NeighborKind,
        visitor: V,
    ) -> Result<V::Output, OptError> {
        match kind {
            NeighborKind::Flip => Ok(visitor.visit::<MaxCutFlipNeighbor>()),
            NeighborKind::Swap => Ok(visitor.visit::<MaxCutSwapNeighbor>()),
            other => Err(invalid_neighbor::<Self>(other)),
        }
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
            } => Ok(Box::new(BreakoutLocalSearchForMaxCut::new(
                cond,
                *tabu_tenure,
                *t,
                *l0,
                *p0,
                *q,
            ))),
            _ => Err(OptError::Config(format!(
                "heuristic '{}' is not supported for MaxCut",
                config.kind_name()
            ))),
        }
    }

    fn build_crossover(kind: Option<&str>) -> Result<Box<dyn Crossover<Self>>, OptError> {
        match kind.unwrap_or("Uniform") {
            "Uniform" => Ok(Box::new(MaxCutUniformCrossover)),
            other => Err(OptError::Config(format!(
                "Unknown crossover_kind '{other}' for MaxCut (expected 'Uniform')"
            ))),
        }
    }
}

impl ConfigurableProblem for Qubo {
    const NAME: &'static str = "Qubo";
    const MINIMIZE: bool = true;
    const VALID_NEIGHBORS: &'static [NeighborKind] = &[NeighborKind::Flip, NeighborKind::Swap];

    fn with_neighbor<V: NeighborVisitor<Self>>(
        kind: &NeighborKind,
        visitor: V,
    ) -> Result<V::Output, OptError> {
        match kind {
            NeighborKind::Flip => Ok(visitor.visit::<QuboFlipNeighbor>()),
            NeighborKind::Swap => Ok(visitor.visit::<QuboSwapNeighbor>()),
            other => Err(invalid_neighbor::<Self>(other)),
        }
    }

    fn build_crossover(kind: Option<&str>) -> Result<Box<dyn Crossover<Self>>, OptError> {
        match kind.unwrap_or("Uniform") {
            "Uniform" => Ok(Box::new(QuboUniformCrossover)),
            other => Err(OptError::Config(format!(
                "Unknown crossover_kind '{other}' for Qubo (expected 'Uniform')"
            ))),
        }
    }
}

impl ConfigurableProblem for Sat {
    const NAME: &'static str = "Sat";
    const MINIMIZE: bool = false;
    const VALID_NEIGHBORS: &'static [NeighborKind] = &[NeighborKind::Flip, NeighborKind::Swap];

    fn with_neighbor<V: NeighborVisitor<Self>>(
        kind: &NeighborKind,
        visitor: V,
    ) -> Result<V::Output, OptError> {
        match kind {
            NeighborKind::Flip => Ok(visitor.visit::<SatFlipNeighbor>()),
            NeighborKind::Swap => Ok(visitor.visit::<SatSwapNeighbor>()),
            other => Err(invalid_neighbor::<Self>(other)),
        }
    }

    fn build_crossover(kind: Option<&str>) -> Result<Box<dyn Crossover<Self>>, OptError> {
        match kind.unwrap_or("Uniform") {
            "Uniform" => Ok(Box::new(SatUniformCrossover)),
            other => Err(OptError::Config(format!(
                "Unknown crossover_kind '{other}' for Sat (expected 'Uniform')"
            ))),
        }
    }
}

impl ConfigurableProblem for TspWithCoordinates {
    const NAME: &'static str = "Tsp";
    const MINIMIZE: bool = true;
    const VALID_NEIGHBORS: &'static [NeighborKind] =
        &[NeighborKind::TwoOpt, NeighborKind::Relocate];

    fn with_neighbor<V: NeighborVisitor<Self>>(
        kind: &NeighborKind,
        visitor: V,
    ) -> Result<V::Output, OptError> {
        match kind {
            NeighborKind::TwoOpt => Ok(visitor.visit::<TspTwoOptNeighbor>()),
            NeighborKind::Relocate => Ok(visitor.visit::<TspRelocateNeighbor>()),
            other => Err(invalid_neighbor::<Self>(other)),
        }
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
            _ => Err(OptError::Config(format!(
                "heuristic '{}' is not supported for Tsp",
                config.kind_name()
            ))),
        }
    }

    fn build_crossover(kind: Option<&str>) -> Result<Box<dyn Crossover<Self>>, OptError> {
        match kind.unwrap_or("Order") {
            "Order" => Ok(Box::new(TspOrderCrossover)),
            other => Err(OptError::Config(format!(
                "Unknown crossover_kind '{other}' for Tsp (expected 'Order')"
            ))),
        }
    }
}

impl ConfigurableProblem for VertexCover {
    const NAME: &'static str = "VertexCover";
    const MINIMIZE: bool = true;
    const VALID_NEIGHBORS: &'static [NeighborKind] = &[NeighborKind::Flip, NeighborKind::Swap];

    fn with_neighbor<V: NeighborVisitor<Self>>(
        kind: &NeighborKind,
        visitor: V,
    ) -> Result<V::Output, OptError> {
        match kind {
            NeighborKind::Flip => Ok(visitor.visit::<VertexCoverFlipNeighbor>()),
            NeighborKind::Swap => Ok(visitor.visit::<VertexCoverSwapNeighbor>()),
            other => Err(invalid_neighbor::<Self>(other)),
        }
    }

    fn build_crossover(kind: Option<&str>) -> Result<Box<dyn Crossover<Self>>, OptError> {
        match kind.unwrap_or("Uniform") {
            "Uniform" => Ok(Box::new(VertexCoverUniformCrossover)),
            other => Err(OptError::Config(format!(
                "Unknown crossover_kind '{other}' for VertexCover (expected 'Uniform')"
            ))),
        }
    }
}

impl ConfigurableProblem for JobShopScheduling {
    const NAME: &'static str = "JobShop";
    const MINIMIZE: bool = true;
    const VALID_NEIGHBORS: &'static [NeighborKind] = &[NeighborKind::Swap, NeighborKind::Relocate];

    fn with_neighbor<V: NeighborVisitor<Self>>(
        kind: &NeighborKind,
        visitor: V,
    ) -> Result<V::Output, OptError> {
        match kind {
            NeighborKind::Swap => Ok(visitor.visit::<JobShopSwapNeighbor>()),
            NeighborKind::Relocate => Ok(visitor.visit::<JobShopRelocateNeighbor>()),
            other => Err(invalid_neighbor::<Self>(other)),
        }
    }

    fn build_crossover(kind: Option<&str>) -> Result<Box<dyn Crossover<Self>>, OptError> {
        match kind.unwrap_or("Ppx") {
            "Ppx" => Ok(Box::new(JobShopPpxCrossover)),
            other => Err(OptError::Config(format!(
                "Unknown crossover_kind '{other}' for JobShop (expected 'Ppx')"
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Runtime problem dispatch -- the single place mapping ProblemKind to types
// ---------------------------------------------------------------------------

/// Callback invoked with the concrete problem type for a [`ProblemKind`].
pub(crate) trait ProblemVisitor {
    type Output;
    fn visit<P>(self) -> Self::Output
    where
        P: ConfigurableProblem,
        P::Solution: BenchmarkSolution + Distance;
}

/// Maps the runtime [`ProblemKind`] to the concrete problem type.
pub(crate) fn with_problem<V: ProblemVisitor>(kind: &ProblemKind, visitor: V) -> V::Output {
    match kind {
        ProblemKind::MaxCut => visitor.visit::<MaxCut>(),
        ProblemKind::Qubo => visitor.visit::<Qubo>(),
        ProblemKind::Sat => visitor.visit::<Sat>(),
        ProblemKind::Tsp => visitor.visit::<TspWithCoordinates>(),
        ProblemKind::VertexCover => visitor.visit::<VertexCover>(),
        ProblemKind::JobShop => visitor.visit::<JobShopScheduling>(),
    }
}

struct MinimizeVisitor;
impl ProblemVisitor for MinimizeVisitor {
    type Output = bool;
    fn visit<P>(self) -> bool
    where
        P: ConfigurableProblem,
        P::Solution: BenchmarkSolution + Distance,
    {
        P::MINIMIZE
    }
}

struct ValidNeighborsVisitor;
impl ProblemVisitor for ValidNeighborsVisitor {
    type Output = &'static [NeighborKind];
    fn visit<P>(self) -> &'static [NeighborKind]
    where
        P: ConfigurableProblem,
        P::Solution: BenchmarkSolution + Distance,
    {
        P::VALID_NEIGHBORS
    }
}

impl ProblemKind {
    /// Whether this problem is minimized (drives report statistics).
    pub fn minimize(&self) -> bool {
        with_problem(self, MinimizeVisitor)
    }

    /// The neighborhood kinds supported by this problem.
    pub fn valid_neighbors(&self) -> &'static [NeighborKind] {
        with_problem(self, ValidNeighborsVisitor)
    }
}
