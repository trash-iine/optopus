use rand::seq::SliceRandom;

use super::taillard::HeadsTails;
use crate::error::OptError;
use crate::search_state::{Distance, Evaluable, Evaluate, ProblemTrait};

/// A solution to the permutation flow shop, the order the jobs run in.
///
/// Every machine processes the jobs in this one order.
#[derive(Debug, Clone)]
pub struct FlowShopSolution {
    /// A permutation of `0..n_jobs`, the job run first coming first.
    pub sequence: Vec<usize>,
    /// The makespan of `sequence`, when the last job leaves the last machine.
    pub objective: u32,
}

impl Evaluate for FlowShopSolution {
    /// The flow shop minimizes the makespan.
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(f64::from(self.objective))
    }
}

impl Distance for FlowShopSolution {
    /// The number of positions holding different jobs.
    fn distance(&self, other: &Self) -> usize {
        crate::building_blocks::representation::hamming_distance(&self.sequence, &other.sequence)
    }
}

/// A permutation flow shop instance, `Fm | prmu | Cmax`.
///
/// `n_jobs` jobs pass through `n_machines` machines in the same machine order,
/// and every machine takes the jobs in one common order, the solution. A
/// machine runs one job at a time and a job is on one machine at a time. The
/// makespan, the time the last job leaves the last machine, is minimized.
#[derive(Debug, Clone)]
pub struct FlowShop {
    /// Instance name, from the file stem or given by the caller.
    pub name: String,
    /// Number of jobs.
    pub n_jobs: usize,
    /// Number of machines.
    pub n_machines: usize,
    /// Processing times, machine-major, `processing[i * n_jobs + j]` for job
    /// `j` on machine `i`.
    processing: Vec<u32>,
}

impl FlowShop {
    /// Creates an instance from one row of processing times per machine, the
    /// layout Taillard's instances are published in.
    ///
    /// # Panics
    ///
    /// Panics if the rows differ in length.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// // Two machines, three jobs.
    /// let fs = FlowShop::new("tiny", vec![vec![3, 2, 4], vec![2, 5, 1]]);
    /// assert_eq!(fs.makespan(&[0, 1, 2]), 11);
    /// ```
    pub fn new(name: impl Into<String>, rows: Vec<Vec<u32>>) -> Self {
        let n_machines = rows.len();
        let n_jobs = rows.first().map_or(0, Vec::len);
        assert!(
            rows.iter().all(|row| row.len() == n_jobs),
            "every machine needs a processing time for every job"
        );
        Self {
            name: name.into(),
            n_jobs,
            n_machines,
            processing: rows.into_iter().flatten().collect(),
        }
    }

    /// Loads an instance in Taillard's layout.
    ///
    /// ```text
    /// n_jobs n_machines
    /// p(0,0) p(0,1) ... p(0,n-1)     one line per machine
    /// ...
    /// ```
    ///
    /// Blank lines are skipped. Taillard's own files carry a seed and bounds
    /// in the header and are converted to this layout by
    /// `data/instances/scripts/gen_taillard_pfsp.py`.
    pub fn load_file(path: impl AsRef<std::path::Path>) -> Result<Self, OptError> {
        use crate::building_blocks::instance::InstanceLines;

        let path = path.as_ref();
        let mut lines = InstanceLines::open(path)?;
        let header = lines
            .next_data_line()?
            .ok_or_else(|| lines.err("file is empty, expected header 'n_jobs n_machines'"))?;
        let mut tokens = header.split_whitespace();
        let n_jobs: usize = lines.parse_next(&mut tokens, "n_jobs")?;
        let n_machines: usize = lines.parse_next(&mut tokens, "n_machines")?;

        let mut rows = Vec::with_capacity(n_machines);
        for i in 0..n_machines {
            let line = lines.next_data_line()?.ok_or_else(|| {
                lines.err(format!(
                    "unexpected end of file, expected the row of machine {i}"
                ))
            })?;
            let mut tokens = line.split_whitespace();
            let row = (0..n_jobs)
                .map(|j| lines.parse_next(&mut tokens, &format!("time of job {j} on machine {i}")))
                .collect::<Result<Vec<u32>, _>>()?;
            if tokens.next().is_some() {
                return Err(lines.err(format!("machine {i} has more than {n_jobs} times")));
            }
            rows.push(row);
        }

        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("flow_shop");
        Ok(Self::new(name, rows))
    }

    /// The processing time of `job` on `machine`.
    #[inline]
    pub fn processing_time(&self, machine: usize, job: usize) -> u32 {
        self.processing[machine * self.n_jobs + job]
    }

    /// The sum of every processing time, the scale Ruiz and Stützle set their
    /// acceptance temperature by.
    pub fn total_processing_time(&self) -> u64 {
        self.processing.iter().map(|&p| u64::from(p)).sum()
    }

    /// The makespan of a sequence of distinct jobs, in O(nm).
    ///
    /// A partial sequence is fine and is scheduled as given.
    pub fn makespan(&self, sequence: &[usize]) -> u32 {
        let mut done = vec![0u32; self.n_machines];
        for &job in sequence {
            let mut prev_machine = 0;
            for (i, cell) in done.iter_mut().enumerate() {
                *cell = (*cell).max(prev_machine) + self.processing_time(i, job);
                prev_machine = *cell;
            }
        }
        done.last().copied().unwrap_or(0)
    }

    /// Wraps a sequence as a solution, computing its makespan.
    ///
    /// # Panics
    ///
    /// Panics if `sequence` is not a permutation of `0..n_jobs`.
    pub fn solution_from_sequence(&self, sequence: Vec<usize>) -> FlowShopSolution {
        let mut seen = vec![false; self.n_jobs];
        for &job in &sequence {
            assert!(
                job < self.n_jobs && !std::mem::replace(&mut seen[job], true),
                "a flow shop sequence is a permutation of the jobs"
            );
        }
        assert_eq!(sequence.len(), self.n_jobs, "every job has to be scheduled");
        let objective = self.makespan(&sequence);
        FlowShopSolution {
            sequence,
            objective,
        }
    }

    /// The NEH heuristic of Nawaz, Enscore and Ham (1983).
    ///
    /// Jobs are taken by decreasing total processing time, ties by index, and
    /// each is inserted where the partial sequence's makespan is smallest, the
    /// earliest such place on a tie. With Taillard's heads and tails every
    /// insertion scan is O(nm), so the whole construction is O(n²m).
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// let fs = FlowShop::new("tiny", vec![vec![3, 2, 4], vec![2, 5, 1]]);
    /// let neh = fs.neh();
    /// assert!(neh.objective <= fs.makespan(&[0, 1, 2]));
    /// ```
    pub fn neh(&self) -> FlowShopSolution {
        let mut order: Vec<usize> = (0..self.n_jobs).collect();
        let total = |j: usize| -> u32 {
            (0..self.n_machines)
                .map(|i| self.processing_time(i, j))
                .sum()
        };
        order.sort_by_key(|&j| std::cmp::Reverse(total(j)));

        let mut sequence = Vec::with_capacity(self.n_jobs);
        let mut ht = HeadsTails::default();
        for job in order {
            ht.rebuild(self, &sequence);
            let place = (0..=sequence.len())
                .min_by_key(|&place| ht.makespan_with(self, place, job))
                .unwrap_or(0);
            sequence.insert(place, job);
        }
        self.solution_from_sequence(sequence)
    }
}

impl ProblemTrait for FlowShop {
    type Solution = FlowShopSolution;

    /// A uniformly random order of the jobs.
    fn new_solution(&self, rng: &mut impl rand::Rng) -> FlowShopSolution {
        let mut sequence: Vec<usize> = (0..self.n_jobs).collect();
        sequence.shuffle(rng);
        self.solution_from_sequence(sequence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two machines, three jobs, worked by hand.
    ///
    /// ```text
    /// order 0, 1, 2
    /// machine 0 finishes jobs at 3, 5, 9
    /// machine 1 finishes jobs at 5, 10, 11
    /// ```
    #[test]
    fn makespan_matches_a_hand_schedule() {
        let fs = FlowShop::new("tiny", vec![vec![3, 2, 4], vec![2, 5, 1]]);
        assert_eq!(fs.makespan(&[0, 1, 2]), 11);
        // Johnson's rule for two machines puts job 1 (2 < 5) first, then
        // job 0 (3 > 2), then job 2 (4 > 1), which is optimal.
        assert_eq!(fs.makespan(&[1, 0, 2]), 10);
        assert_eq!(fs.makespan(&[]), 0);
    }

    #[test]
    fn neh_returns_a_priced_permutation() {
        let fs = FlowShop::new(
            "small",
            vec![
                vec![5, 9, 8, 10, 1],
                vec![9, 3, 10, 1, 8],
                vec![9, 4, 5, 8, 6],
            ],
        );
        let neh = fs.neh();
        let mut sorted = neh.sequence.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..5).collect::<Vec<_>>());
        assert_eq!(neh.objective, fs.makespan(&neh.sequence));
    }

    /// NEH on Taillard's first instance gives 1286, the value the flow shop
    /// literature reports for it, which also checks that the generated file is
    /// Taillard's.
    #[test]
    fn neh_on_ta001_matches_the_literature() {
        let fs = FlowShop::load_file("data/instances/flow_shop/taillard/ta001.txt").unwrap();
        assert_eq!((fs.n_jobs, fs.n_machines), (20, 5));
        assert_eq!(fs.neh().objective, 1286);
    }

    #[test]
    fn load_file_reads_taillard_layout() {
        let dir = std::env::temp_dir().join("optopus_flow_shop_load");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.txt");
        std::fs::write(&path, "3 2\n3 2 4\n\n2 5 1\n").unwrap();
        let fs = FlowShop::load_file(&path).unwrap();
        assert_eq!((fs.n_jobs, fs.n_machines), (3, 2));
        assert_eq!(fs.name, "tiny");
        assert_eq!(fs.processing_time(1, 1), 5);

        std::fs::write(&path, "3 2\n3 2 4 7\n2 5 1\n").unwrap();
        assert!(FlowShop::load_file(&path).is_err());
        std::fs::write(&path, "3 2\n3 2 4\n").unwrap();
        assert!(FlowShop::load_file(&path).is_err());
    }
}
