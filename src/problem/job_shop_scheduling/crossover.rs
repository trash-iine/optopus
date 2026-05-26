use rand::Rng;

use crate::search_state::Crossover;

use super::problem::{JobShopScheduling, JobShopSolution};

/// Precedence-Preserving Crossover (PPX) for permutation-with-repetition encodings.
///
/// At each position, randomly choose a parent. Append the next unconsumed
/// operation from that parent's sequence to the child, and consume the
/// matching operation (same job index, in order) from both parents. Because
/// each parent already encodes a precedence-feasible operation order, the
/// child does too.
pub struct JobShopPpxCrossover;

impl Crossover<JobShopScheduling> for JobShopPpxCrossover {
    fn crossover(
        &mut self,
        prob: &JobShopScheduling,
        sol1: &JobShopSolution,
        sol2: &JobShopSolution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Result<JobShopSolution, crate::error::OptError> {
        let n = sol1.operations.len();
        let mut a = sol1.operations.clone();
        let mut b = sol2.operations.clone();
        let mut child = Vec::with_capacity(n);

        for _ in 0..n {
            let job = if rng.random_bool(0.5) { a[0] } else { b[0] };
            child.push(job);
            // Remove the leftmost occurrence of `job` from each parent so
            // the head always exposes the next-eligible operation, preserving
            // each parent's relative ordering of remaining operations.
            let pa = a
                .iter()
                .position(|&x| x == job)
                .expect("parent must contain job");
            a.remove(pa);
            let pb = b
                .iter()
                .position(|&x| x == job)
                .expect("parent must contain job");
            b.remove(pb);
        }

        let (objective, completion_times) = prob.decode(&child)?;
        Ok(JobShopSolution {
            operations: child,
            objective,
            completion_times,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search_state::ProblemTrait;
    use rand::SeedableRng;

    fn make_inst() -> JobShopScheduling {
        JobShopScheduling::new(
            "tiny".to_string(),
            3,
            vec![
                vec![(0, 3), (1, 2), (2, 4)],
                vec![(1, 1), (0, 5), (2, 3)],
                vec![(0, 2), (2, 1), (1, 4)],
            ],
        )
    }

    fn make_sol(inst: &JobShopScheduling, ops: Vec<usize>) -> JobShopSolution {
        let (objective, completion_times) = inst.decode(&ops).unwrap();
        JobShopSolution {
            operations: ops,
            objective,
            completion_times,
        }
    }

    #[test]
    fn test_ppx_child_is_valid_permutation() {
        let inst = make_inst();
        let a = make_sol(&inst, vec![0, 1, 2, 0, 1, 2, 0, 1, 2]);
        let b = make_sol(&inst, vec![2, 1, 0, 2, 0, 1, 1, 2, 0]);
        let mut cx = JobShopPpxCrossover;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        for _ in 0..20 {
            let child = cx.crossover(&inst, &a, &b, &mut rng).unwrap();
            assert_eq!(child.operations.len(), 9);
            let mut counts = vec![0usize; inst.n_jobs];
            for &j in &child.operations {
                counts[j] += 1;
            }
            assert_eq!(counts, vec![inst.n_machines; inst.n_jobs]);
        }
    }

    #[test]
    fn test_ppx_identical_parents_yields_same_sequence() {
        let inst = make_inst();
        let a = make_sol(&inst, vec![0, 1, 2, 0, 1, 2, 0, 1, 2]);
        let mut cx = JobShopPpxCrossover;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let child = cx.crossover(&inst, &a, &a, &mut rng).unwrap();
        assert_eq!(child.operations, a.operations);
    }

    #[test]
    fn test_ppx_random_parents_random_inst() {
        let inst = make_inst();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let a = inst.new_solution(&mut rng);
        let b = inst.new_solution(&mut rng);
        let mut cx = JobShopPpxCrossover;
        let child = cx.crossover(&inst, &a, &b, &mut rng).unwrap();
        assert_eq!(child.operations.len(), inst.n_jobs * inst.n_machines);
    }
}
