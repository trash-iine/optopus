pub trait Rankable {
    fn is_better_than(&self, other: &Self) -> bool;
}

pub fn filter_best<R: Rankable, T: Iterator<Item = R>>(iter: T) -> Vec<R> {
    let mut best_list = vec![];
    for r in iter {
        if best_list.is_empty() {
            best_list = vec![r];
        } else {
            let sample = &best_list[0];
            if r.is_better_than(sample) {
                best_list = vec![r];
            } else if !sample.is_better_than(&r) {
                best_list.push(r);
            }
        }
    }

    return best_list;
}

pub trait ProblemTrait {
    type Solution: Clone + Rankable;
    fn new_solution(&self, rng: &mut impl rand::Rng) -> Self::Solution;
}

pub trait MoveToNeigbor<Problem>
where
    Problem: ProblemTrait,
    Problem::Solution: Rankable,
{
    fn apply_to_iteration(&self, iter: u64) -> u64 {
        iter + 1
    }
    fn apply_to_solution(&self, prob: &Problem, sol: &mut Problem::Solution);
    fn iter(prob: &Problem, sol: &Problem::Solution) -> impl Iterator<Item = Self> + Send;
    fn move_to_be_better_than(
        &self,
        prob: &Problem,
        src: &Problem::Solution,
        other: &Problem::Solution,
    ) -> bool {
        let mut cloned = src.clone();
        self.apply_to_solution(prob, &mut cloned);
        cloned.is_better_than(other)
    }
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
