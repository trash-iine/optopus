pub trait ProblemTrait {
    type Solution: Clone;
    type Objective: Clone;
    fn new_solution(&self, rng: &mut impl rand::Rng) -> Self::Solution;
    fn calculate_objective(&self, sol: &Self::Solution) -> Self::Objective;
    fn is_first_objective_better_than_second(
        &self,
        first: &Self::Objective,
        second: &Self::Objective,
    ) -> bool;
}

pub trait EnumerateMoveToNeighbor<MoveToNeighbor> {
    fn apply_to_iteration(&mut self, neighbor: &MoveToNeighbor);
    fn apply_to_solution(&mut self, neighbor: &MoveToNeighbor);
    fn apply_to_objective(&mut self, neighbor: &MoveToNeighbor);
    fn iter_on_move_to_neighbor(&self) -> impl Iterator<Item = MoveToNeighbor> + Send;
    fn is_move_to_be_better_than_currernt(&self, neighbor: &MoveToNeighbor) -> bool;
    fn is_move_to_be_better_than_best(&self, neighbor: &MoveToNeighbor) -> bool;
    fn is_first_move_better_than_second(
        &self,
        first: &MoveToNeighbor,
        second: &MoveToNeighbor,
    ) -> bool;
}

pub trait Evaluable<T> {
    fn evaluate(&self) -> T;
}

pub trait EnabledTabu {
    type TabuMap;
    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool;
    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
    );
}
