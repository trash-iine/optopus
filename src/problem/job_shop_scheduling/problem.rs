use std::fs::File;
use std::io::{BufRead, BufReader};

use rand::seq::SliceRandom;

use crate::error::OptError;
use crate::search_state::{Distance, ProblemTrait, Rankable};

/// A solution to the Job Shop Scheduling problem.
///
/// `operations` is a permutation-with-repetition of length `n_jobs * n_machines`
/// where each job index appears exactly `n_machines` times. The k-th occurrence
/// (0-indexed) of job `j` represents operation `O(j, k)`.
///
/// `objective` is the makespan (Cmax) obtained by left-shift semi-active
/// scheduling, and `completion_times` are the per-position completion times
/// in the same order as `operations`.
#[derive(Debug, Clone)]
pub struct JobShopSolution {
    pub operations: Vec<usize>,
    pub objective: u32,
    pub completion_times: Vec<u32>,
}

impl Rankable for JobShopSolution {
    fn is_better_than(&self, other: &Self) -> bool {
        self.objective < other.objective
    }
}

impl Distance for JobShopSolution {
    /// Position-based dissimilarity: number of positions where the job index differs.
    fn distance(&self, other: &Self) -> usize {
        self.operations
            .iter()
            .zip(other.operations.iter())
            .filter(|(a, b)| a != b)
            .count()
    }
}

/// A Job Shop Scheduling instance.
///
/// `jobs[j]` is the ordered list of `(machine, duration)` pairs that job `j`
/// must execute. Operations within a job must be processed in this order
/// (precedence constraint), and a machine can process only one operation at a time.
#[derive(Debug, Clone)]
pub struct JobShopScheduling {
    pub name: String,
    pub n_jobs: usize,
    pub n_machines: usize,
    pub jobs: Vec<Vec<(usize, u32)>>,
}

impl JobShopScheduling {
    /// Creates a new instance from `(machine, duration)` sequences.
    pub fn new(name: String, n_machines: usize, jobs: Vec<Vec<(usize, u32)>>) -> Self {
        let n_jobs = jobs.len();
        Self {
            name,
            n_jobs,
            n_machines,
            jobs,
        }
    }

    /// Loads an instance from a Taillard / OR-Library standard format file.
    ///
    /// Format:
    /// ```text
    /// n_jobs n_machines
    /// machine duration machine duration ...   (n_machines pairs per job, one job per line)
    /// ...
    /// ```
    /// Machine indices are 0-indexed. Empty lines and `#`-prefixed comment lines are ignored.
    pub fn load_file(path: impl AsRef<std::path::Path>) -> Result<Self, OptError> {
        let path = path.as_ref();
        let path_display = path.display().to_string();
        let err = |line: usize, detail: String| OptError::FileLoad {
            path: path_display.clone(),
            line,
            detail,
        };

        let file = File::open(path).map_err(|e| err(0, format!("failed to open file: {e}")))?;
        let reader = BufReader::new(file);

        let mut tokens: Vec<(usize, String)> = Vec::new();
        for (idx, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| err(idx + 1, format!("failed to read line: {e}")))?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            for tok in trimmed.split_whitespace() {
                tokens.push((idx + 1, tok.to_string()));
            }
        }

        let mut iter = tokens.into_iter();
        let parse_usize = |entry: Option<(usize, String)>, what: &str| -> Result<usize, OptError> {
            let (line_num, tok) =
                entry.ok_or_else(|| err(0, format!("unexpected end of file, expected {what}")))?;
            tok.parse::<usize>()
                .map_err(|e| err(line_num, format!("failed to parse {what} '{tok}': {e}")))
        };
        let parse_u32 = |entry: Option<(usize, String)>, what: &str| -> Result<u32, OptError> {
            let (line_num, tok) =
                entry.ok_or_else(|| err(0, format!("unexpected end of file, expected {what}")))?;
            tok.parse::<u32>()
                .map_err(|e| err(line_num, format!("failed to parse {what} '{tok}': {e}")))
        };

        let n_jobs = parse_usize(iter.next(), "n_jobs")?;
        let n_machines = parse_usize(iter.next(), "n_machines")?;

        let mut jobs = Vec::with_capacity(n_jobs);
        for j in 0..n_jobs {
            let mut ops = Vec::with_capacity(n_machines);
            for k in 0..n_machines {
                let machine =
                    parse_usize(iter.next(), &format!("machine for job {j} operation {k}"))?;
                let duration =
                    parse_u32(iter.next(), &format!("duration for job {j} operation {k}"))?;
                if machine >= n_machines {
                    return Err(err(
                        0,
                        format!(
                            "machine index {machine} out of range (n_machines = {n_machines}) for job {j} operation {k}"
                        ),
                    ));
                }
                ops.push((machine, duration));
            }
            jobs.push(ops);
        }

        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("jssp")
            .to_string();

        Ok(Self::new(name, n_machines, jobs))
    }

    /// Decodes an operation sequence into a left-shift semi-active schedule.
    ///
    /// Returns `(makespan, completion_times)` where `completion_times[i]` is
    /// the finish time of `operations[i]`.
    pub fn decode(&self, operations: &[usize]) -> Result<(u32, Vec<u32>), OptError> {
        let expected_len = self.n_jobs * self.n_machines;
        if operations.len() != expected_len {
            return Err(OptError::InvalidState(format!(
                "operations length {} does not match n_jobs * n_machines = {}",
                operations.len(),
                expected_len
            )));
        }

        let mut machine_release = vec![0u32; self.n_machines];
        let mut job_release = vec![0u32; self.n_jobs];
        let mut job_op_idx = vec![0usize; self.n_jobs];
        let mut completion_times = Vec::with_capacity(operations.len());
        let mut makespan = 0u32;

        for (i, &j) in operations.iter().enumerate() {
            if j >= self.n_jobs {
                return Err(OptError::InvalidState(format!(
                    "operations[{i}] = {j} is out of range (n_jobs = {})",
                    self.n_jobs
                )));
            }
            let k = job_op_idx[j];
            if k >= self.n_machines {
                return Err(OptError::InvalidState(format!(
                    "job {j} appears more than {} times in operations",
                    self.n_machines
                )));
            }
            let (machine, duration) = self.jobs[j][k];
            let start = machine_release[machine].max(job_release[j]);
            let finish = start + duration;
            machine_release[machine] = finish;
            job_release[j] = finish;
            job_op_idx[j] = k + 1;
            completion_times.push(finish);
            if finish > makespan {
                makespan = finish;
            }
        }

        for (j, &count) in job_op_idx.iter().enumerate() {
            if count != self.n_machines {
                return Err(OptError::InvalidState(format!(
                    "job {j} appears {count} times in operations, expected {}",
                    self.n_machines
                )));
            }
        }

        Ok((makespan, completion_times))
    }

    /// Computes the makespan of an operation sequence without allocating a
    /// per-operation `completion_times` vector.
    ///
    /// Functionally equivalent to `decode(operations).map(|(m, _)| m)` but
    /// avoids one Vec allocation — useful in neighbor evaluation loops
    /// (e.g. `MoveToNeighbor::move_to_be_better_than`) where only the final
    /// objective is needed.
    pub(crate) fn compute_makespan(&self, operations: &[usize]) -> Result<u32, OptError> {
        let expected_len = self.n_jobs * self.n_machines;
        if operations.len() != expected_len {
            return Err(OptError::InvalidState(format!(
                "operations length {} does not match n_jobs * n_machines = {}",
                operations.len(),
                expected_len
            )));
        }

        let mut machine_release = vec![0u32; self.n_machines];
        let mut job_release = vec![0u32; self.n_jobs];
        let mut job_op_idx = vec![0usize; self.n_jobs];
        let mut makespan = 0u32;

        for (i, &j) in operations.iter().enumerate() {
            if j >= self.n_jobs {
                return Err(OptError::InvalidState(format!(
                    "operations[{i}] = {j} is out of range (n_jobs = {})",
                    self.n_jobs
                )));
            }
            let k = job_op_idx[j];
            if k >= self.n_machines {
                return Err(OptError::InvalidState(format!(
                    "job {j} appears more than {} times in operations",
                    self.n_machines
                )));
            }
            let (machine, duration) = self.jobs[j][k];
            let start = machine_release[machine].max(job_release[j]);
            let finish = start + duration;
            machine_release[machine] = finish;
            job_release[j] = finish;
            job_op_idx[j] = k + 1;
            if finish > makespan {
                makespan = finish;
            }
        }

        for (j, &count) in job_op_idx.iter().enumerate() {
            if count != self.n_machines {
                return Err(OptError::InvalidState(format!(
                    "job {j} appears {count} times in operations, expected {}",
                    self.n_machines
                )));
            }
        }

        Ok(makespan)
    }
}

impl ProblemTrait for JobShopScheduling {
    type Solution = JobShopSolution;

    fn new_solution(&self, rng: &mut impl rand::Rng) -> JobShopSolution {
        let mut operations = Vec::with_capacity(self.n_jobs * self.n_machines);
        for j in 0..self.n_jobs {
            for _ in 0..self.n_machines {
                operations.push(j);
            }
        }
        operations.shuffle(rng);
        let (objective, completion_times) = self
            .decode(&operations)
            .expect("operation sequence generated by new_solution must be valid");
        JobShopSolution {
            operations,
            objective,
            completion_times,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_tmp(contents: &str) -> std::path::PathBuf {
        use std::io::Write;
        let mut path = std::env::temp_dir();
        let unique = format!(
            "optopus_jssp_{}_{}.txt",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        path.push(unique);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        path
    }

    /// Tiny 2-jobs × 2-machines fixture.
    /// job 0: M0(2) → M1(3)
    /// job 1: M1(1) → M0(4)
    fn make_small_instance() -> JobShopScheduling {
        JobShopScheduling::new(
            "tiny".to_string(),
            2,
            vec![vec![(0, 2), (1, 3)], vec![(1, 1), (0, 4)]],
        )
    }

    #[test]
    fn test_decode_optimal_sequence() {
        let inst = make_small_instance();
        // Sequence [1, 0, 0, 1]:
        //   pos 0: job1 op0 on M1, [0..1] -> finish 1
        //   pos 1: job0 op0 on M0, [0..2] -> finish 2
        //   pos 2: job0 op1 on M1, max(2, 1)=2..5 -> finish 5
        //   pos 3: job1 op1 on M0, max(2, 1)=2..6 -> finish 6
        // makespan = 6
        let (makespan, completions) = inst.decode(&[1, 0, 0, 1]).unwrap();
        assert_eq!(makespan, 6);
        assert_eq!(completions, vec![1, 2, 5, 6]);
    }

    #[test]
    fn test_decode_invalid_length() {
        let inst = make_small_instance();
        assert!(inst.decode(&[0, 1, 0]).is_err());
    }

    #[test]
    fn test_decode_invalid_count() {
        let inst = make_small_instance();
        // job 0 appears 3 times, job 1 only once → out-of-range error on 3rd job-0 op.
        assert!(inst.decode(&[0, 0, 0, 1]).is_err());
    }

    #[test]
    fn test_load_file_basic() {
        let contents = "\
2 2
0 2 1 3
1 1 0 4
";
        let path = write_tmp(contents);
        let inst = JobShopScheduling::load_file(path.to_str().unwrap()).unwrap();
        assert_eq!(inst.n_jobs, 2);
        assert_eq!(inst.n_machines, 2);
        assert_eq!(inst.jobs[0], vec![(0, 2), (1, 3)]);
        assert_eq!(inst.jobs[1], vec![(1, 1), (0, 4)]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_file_with_comments_and_blank_lines() {
        let contents = "\
# tiny instance
2 2

0 2 1 3
# comment in the middle
1 1 0 4
";
        let path = write_tmp(contents);
        let inst = JobShopScheduling::load_file(path.to_str().unwrap()).unwrap();
        assert_eq!(inst.n_jobs, 2);
        assert_eq!(inst.jobs[1], vec![(1, 1), (0, 4)]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_file_machine_out_of_range() {
        let contents = "\
2 2
0 2 5 3
1 1 0 4
";
        let path = write_tmp(contents);
        let result = JobShopScheduling::load_file(path.to_str().unwrap());
        assert!(result.is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_new_solution_is_valid() {
        let inst = make_small_instance();
        let mut rng = rand::rng();
        let sol = inst.new_solution(&mut rng);
        assert_eq!(sol.operations.len(), 4);
        let mut counts = vec![0usize; inst.n_jobs];
        for &j in &sol.operations {
            counts[j] += 1;
        }
        assert_eq!(counts, vec![2, 2]);
    }

    #[test]
    fn test_load_ft06() {
        let inst = JobShopScheduling::load_file("data/jssp/ft06.txt").unwrap();
        assert_eq!(inst.n_jobs, 6);
        assert_eq!(inst.n_machines, 6);
        // Sanity: the canonical optimal makespan for ft06 is 55.
        let mut rng = rand::rng();
        let sol = inst.new_solution(&mut rng);
        assert!(sol.objective >= 55);
    }
}
