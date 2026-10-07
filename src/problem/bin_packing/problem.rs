use rand::seq::SliceRandom;

use crate::error::OptError;
use crate::search_state::{Distance, Evaluable, Evaluate, ProblemTrait};

/// A packing, every item in a bin and no bin over capacity.
///
/// Bins are numbered `0..num_bins()` with no empty one among them, so the
/// number of bins in use is the length of `loads`.
#[derive(Debug, Clone)]
pub struct BinPackingSolution {
    /// `bin_of[i]` is the bin item `i` is in.
    pub bin_of: Vec<usize>,
    /// `loads[b]` is the total size packed into bin `b`.
    pub loads: Vec<u64>,
    /// `num_bins − mean((load / capacity)²)`, see [`BinPacking::energy`].
    pub objective: f64,
    /// `Σ (load / capacity)²`, kept beside the loads so a move prices the
    /// objective in O(1).
    pub(super) fill: f64,
}

impl BinPackingSolution {
    /// The number of bins in use.
    pub fn num_bins(&self) -> usize {
        self.loads.len()
    }

    /// The items in each bin.
    pub fn bins(&self) -> Vec<Vec<usize>> {
        let mut bins = vec![Vec::new(); self.num_bins()];
        for (item, &b) in self.bin_of.iter().enumerate() {
            bins[b].push(item);
        }
        bins
    }
}

/// Renumbers bins in order of their first item, so two maps that differ only
/// in how their bins are numbered come out equal.
fn relabel(bin_of: &[usize]) -> Vec<usize> {
    let width = bin_of.iter().copied().max().map_or(0, |b| b + 1);
    let mut label = vec![usize::MAX; width];
    let mut next = 0;
    bin_of
        .iter()
        .map(|&b| {
            if label[b] == usize::MAX {
                label[b] = next;
                next += 1;
            }
            label[b]
        })
        .collect()
}

impl Evaluate for BinPackingSolution {
    /// Bin packing minimizes the bins used, with fuller bins breaking ties.
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.objective)
    }
}

impl Distance for BinPackingSolution {
    /// Hamming distance after relabelling the bins of each packing in order
    /// of their first item, so two packings that differ only in how their bins
    /// are numbered are at distance zero.
    fn distance(&self, other: &Self) -> usize {
        crate::building_blocks::representation::hamming_distance(
            &relabel(&self.bin_of),
            &relabel(&other.bin_of),
        )
    }
}

/// A one-dimensional bin packing instance.
///
/// Items of the given sizes go into bins of one capacity, none over it, and
/// the number of bins is minimized. That objective is flat, every packing with
/// the same count ties, so the search minimizes
///
/// ```text
/// num_bins − (1 / num_bins) Σ_b (load_b / capacity)²
/// ```
///
/// instead, Falkenauer's fill term subtracted. The term lies in `(0, 1]`, so
/// fewer bins always wins, and among packings with the same count it prefers
/// the ones whose bins are fuller, which are the ones a bin can be emptied
/// from.
#[derive(Debug, Clone)]
pub struct BinPacking {
    /// Instance name, from the file stem or given by the caller.
    pub name: String,
    /// The capacity of every bin.
    pub capacity: u64,
    /// The size of every item.
    pub sizes: Vec<u64>,
}

impl BinPacking {
    /// Creates an instance.
    ///
    /// # Panics
    ///
    /// Panics if the capacity is zero or an item is empty or larger than a
    /// bin, since no packing would exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// let bp = BinPacking::new("tiny", 10, vec![6, 4, 5, 5, 3]);
    /// assert_eq!(bp.lower_bound(), 3);
    /// ```
    pub fn new(name: impl Into<String>, capacity: u64, sizes: Vec<u64>) -> Self {
        assert!(capacity > 0, "the capacity must be positive");
        assert!(
            sizes.iter().all(|&s| s > 0 && s <= capacity),
            "every item has to be non-empty and fit in a bin"
        );
        Self {
            name: name.into(),
            capacity,
            sizes,
        }
    }

    /// Loads an instance, `n capacity` followed by the `n` item sizes, all
    /// whitespace separated.
    pub fn load_file(path: impl AsRef<std::path::Path>) -> Result<Self, OptError> {
        use crate::building_blocks::instance::InstanceLines;

        let path = path.as_ref();
        let mut tokens = InstanceLines::open(path)?.into_tokens()?;
        let n: usize = tokens.next("n")?;
        let capacity: u64 = tokens.next("capacity")?;
        let sizes = (0..n)
            .map(|i| tokens.next(&format!("size of item {i}")))
            .collect::<Result<Vec<u64>, _>>()?;
        tokens.expect_end()?;
        if capacity == 0 || sizes.iter().any(|&s| s == 0 || s > capacity) {
            return Err(tokens.err("every item has to be non-empty and fit in a bin"));
        }

        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("bin_packing");
        Ok(Self::new(name, capacity, sizes))
    }

    /// The number of items.
    pub fn num_items(&self) -> usize {
        self.sizes.len()
    }

    /// `⌈Σ size / capacity⌉`, the bins the items need even if they could be
    /// cut.
    pub fn lower_bound(&self) -> u64 {
        self.sizes.iter().sum::<u64>().div_ceil(self.capacity)
    }

    /// The objective of a packing with these bin loads, see [`BinPacking`].
    pub fn energy(&self, loads: impl IntoIterator<Item = u64>) -> f64 {
        let (bins, fill) = loads
            .into_iter()
            .filter(|&l| l > 0)
            .fold((0, 0.0), |(n, f), l| (n + 1, f + self.sq(l)));
        Self::energy_of(bins, fill)
    }

    /// `(load / capacity)²`, one bin's share of the fill term.
    pub(super) fn sq(&self, load: u64) -> f64 {
        (load as f64 / self.capacity as f64).powi(2)
    }

    /// The objective of `bins` open bins whose fill terms sum to `fill`. The
    /// one place the objective is written.
    pub(super) fn energy_of(bins: usize, fill: f64) -> f64 {
        if bins == 0 {
            0.0
        } else {
            bins as f64 - fill / bins as f64
        }
    }

    /// Wraps an item to bin map as a solution, numbering the bins compactly
    /// in order of their first item and computing the loads and objective.
    ///
    /// # Panics
    ///
    /// Panics if `bin_of` does not cover every item or a bin is over capacity.
    pub fn solution_from_bins(&self, bin_of: Vec<usize>) -> BinPackingSolution {
        assert_eq!(bin_of.len(), self.num_items(), "every item needs a bin");
        let bin_of = relabel(&bin_of);
        let mut loads = vec![0; bin_of.iter().copied().max().map_or(0, |b| b + 1)];
        for (&b, &size) in bin_of.iter().zip(&self.sizes) {
            loads[b] += size;
        }
        assert!(
            loads.iter().all(|&l| l <= self.capacity),
            "a bin is over capacity"
        );
        let fill = loads.iter().map(|&l| self.sq(l)).sum();
        BinPackingSolution {
            bin_of,
            objective: Self::energy_of(loads.len(), fill),
            loads,
            fill,
        }
    }

    /// First Fit over the items in the given order, each into the first bin
    /// with room.
    pub fn first_fit(&self, order: impl IntoIterator<Item = usize>) -> BinPackingSolution {
        let mut bin_of = vec![0; self.num_items()];
        self.first_fit_into(&mut Vec::new(), &mut bin_of, order);
        self.solution_from_bins(bin_of)
    }

    /// First Fit of the items in `order` into the bins `loads` already
    /// describes, opening new ones as needed.
    pub(super) fn first_fit_into(
        &self,
        loads: &mut Vec<u64>,
        bin_of: &mut [usize],
        order: impl IntoIterator<Item = usize>,
    ) {
        for item in order {
            let size = self.sizes[item];
            let bin = match loads.iter().position(|&l| l + size <= self.capacity) {
                Some(b) => b,
                None => {
                    loads.push(0);
                    loads.len() - 1
                }
            };
            loads[bin] += size;
            bin_of[item] = bin;
        }
    }
}

impl ProblemTrait for BinPacking {
    type Solution = BinPackingSolution;

    /// First Fit over a uniformly random order of the items.
    fn new_solution(&self, rng: &mut impl rand::Rng) -> BinPackingSolution {
        let mut order: Vec<usize> = (0..self.num_items()).collect();
        order.shuffle(rng);
        self.first_fit(order)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_prefers_fewer_bins_then_fuller_ones() {
        let bp = BinPacking::new("tiny", 10, vec![6, 4, 5, 5, 3]);
        // Three bins, two of them full.
        let full = bp.solution_from_bins(vec![0, 0, 1, 1, 2]);
        assert_eq!(full.num_bins(), 3);
        // Three bins, none full.
        let loose = bp.solution_from_bins(vec![0, 1, 1, 2, 0]);
        // Four bins.
        let four = bp.solution_from_bins(vec![0, 1, 2, 2, 3]);
        assert!(full.objective < loose.objective);
        assert!(loose.objective < four.objective);
        assert_eq!(
            full.distance(&bp.solution_from_bins(vec![2, 2, 0, 0, 1])),
            0
        );
    }

    #[test]
    fn first_fit_never_overfills() {
        let bp = BinPacking::new("tiny", 10, vec![6, 4, 5, 5, 3, 7, 2]);
        let sol = bp.first_fit(0..7);
        assert!(sol.loads.iter().all(|&l| l <= 10));
        assert_eq!(sol.loads.iter().sum::<u64>(), 32);
        assert!(sol.num_bins() as u64 >= bp.lower_bound());
    }

    #[test]
    fn load_file_reads_the_split_layout() {
        let bp = BinPacking::load_file("data/instances/bin_packing/falkenauer/t60_00.txt").unwrap();
        assert_eq!((bp.num_items(), bp.capacity), (60, 1000));
        assert_eq!(bp.lower_bound(), 20);

        let dir = std::env::temp_dir().join("optopus_bin_packing_load");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.txt");
        std::fs::write(&path, "2 10\n4 11\n").unwrap();
        assert!(BinPacking::load_file(&path).is_err());
        std::fs::write(&path, "2 10\n4 5 6\n").unwrap();
        assert!(BinPacking::load_file(&path).is_err());
    }
}
