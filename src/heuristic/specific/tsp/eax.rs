//! A genetic algorithm for the symmetric TSP built on edge assembly
//! crossover.

use rand::Rng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

use crate::building_blocks::search::MIN_IMPROVEMENT;
use crate::error::OptError;
use crate::heuristic::{Heuristic, StopCondition};
use crate::problem::tsp::{AnchoredTourDescent, NeighborLists, Tsp, TspSolution};
use crate::search_state::SearchState;
use crate::trait_defs::{LocalRepair, Rankable, Ruinable};

/// A slot of an adjacency or a position that holds nothing.
const NONE: u32 = u32::MAX;

/// Which offspring of a pair replaces the first parent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EaxSelection {
    /// The shortest offspring.
    Greedy,
    /// The offspring that shortens the tour the most per unit of edge
    /// entropy the population loses by taking it.
    #[default]
    Entropy,
}

/// The cities of the cycle through `start` in `adj`, from `start` on.
fn around(adj: &[[u32; 2]], start: u32) -> impl Iterator<Item = u32> + '_ {
    let mut prev = NONE;
    let mut cur = Some(start);
    std::iter::from_fn(move || {
        let here = cur?;
        let [x, y] = adj[here as usize];
        let next = if x != prev { x } else { y };
        prev = here;
        cur = (next != start).then_some(next);
        Some(here)
    })
}

/// A tour as the two neighbors of every city, with its length.
struct Member {
    adj: Vec<[u32; 2]>,
    length: f64,
}

impl Member {
    fn from_tour(tour: &[usize], length: f64) -> Self {
        let n = tour.len();
        let mut adj = vec![[NONE; 2]; n];
        for (k, &city) in tour.iter().enumerate() {
            adj[city] = [tour[(k + n - 1) % n] as u32, tour[(k + 1) % n] as u32];
        }
        Self { adj, length }
    }

    /// The cities in tour order from city zero.
    fn tour(&self) -> Vec<usize> {
        around(&self.adj, 0).map(|c| c as usize).collect()
    }
}

/// How many members of the population hold each edge.
///
/// An edge is filed under its smaller end. A city has a handful of distinct
/// neighbors across a population, so a list per city is searched faster than
/// a table is hashed, and takes memory in proportion to the edges in use.
#[derive(Default)]
struct EdgeCounts {
    by_city: Vec<Vec<(u32, u32)>>,
}

impl EdgeCounts {
    fn reset(&mut self, n: usize) {
        self.by_city.clear();
        self.by_city.resize(n, Vec::new());
    }

    /// Adds one holder of the edge between `u` and `v`, or takes one away,
    /// and returns how many held it before.
    fn change(&mut self, u: u32, v: u32, added: bool) -> u32 {
        let (lo, hi) = (u.min(v), u.max(v));
        let list = &mut self.by_city[lo as usize];
        let Some(at) = list.iter().position(|e| e.0 == hi) else {
            assert!(added, "a removed edge is one the population holds");
            list.push((hi, 1));
            return 0;
        };
        let held = list[at].1;
        if added {
            list[at].1 += 1;
        } else if held == 1 {
            list.swap_remove(at);
        } else {
            list[at].1 -= 1;
        }
        held
    }

    fn add_member(&mut self, member: &Member) {
        for (v, nbs) in member.adj.iter().enumerate() {
            for &w in nbs {
                if (v as u32) < w {
                    self.change(v as u32, w, true);
                }
            }
        }
    }
}

/// One change of an edge between a parent and an offspring.
#[derive(Clone, Copy)]
struct EdgeChange {
    added: bool,
    u: u32,
    v: u32,
}

/// One write to an adjacency slot, with what it replaced.
#[derive(Clone, Copy)]
struct SlotWrite {
    city: u32,
    slot: u8,
    old: u32,
    new: u32,
}

/// The working state of one mating, the first parent with one offspring's
/// changes applied on top.
#[derive(Default)]
struct Mating {
    /// The first parent's adjacency, edited per offspring and undone.
    work: Vec<[u32; 2]>,
    /// The first parent's cities in tour order, and each city's index there.
    order: Vec<u32>,
    inv: Vec<u32>,

    // AB-cycle construction. Type 0 is an edge of the first parent, type 1
    // of the second.
    remaining: [Vec<[u32; 2]>; 2],
    count: [Vec<u8>; 2],
    /// Where on the walk a city was left by an edge of each type.
    left_at: [Vec<u32>; 2],
    walk: Vec<u32>,
    open: Vec<u32>,
    /// The cycles end to end. In each, the edge from an even index to the
    /// next is the first parent's and the edge after it the second's.
    cycles: Vec<u32>,
    spans: Vec<(u32, u32)>,
    gains: Vec<f64>,
    /// The nearest cities of each city a join looks among, fetched once per
    /// mating.
    near: NeighborLists,

    // One offspring.
    writes: Vec<SlotWrite>,
    changes: Vec<EdgeChange>,
    cuts: Vec<u32>,
    parent: Vec<u32>,
    size: Vec<u32>,
    roots: Vec<u32>,
    in_unit: Vec<bool>,
    unit: Vec<u32>,

    // The best offspring so far.
    best_writes: Vec<SlotWrite>,
    best_changes: Vec<EdgeChange>,
}

impl Mating {
    fn ensure(&mut self, n: usize) {
        if self.inv.len() == n {
            return;
        }
        self.inv = vec![0; n];
        for t in 0..2 {
            self.remaining[t] = vec![[NONE; 2]; n];
            self.count[t] = vec![0; n];
            self.left_at[t] = vec![NONE; n];
        }
        self.in_unit = vec![false; n];
    }

    /// Reads the two parents and splits the edges only one of them has into
    /// AB-cycles, closed walks that alternate between the parents' edges.
    fn set_parents(&mut self, prob: &Tsp, a: &Member, b: &Member, rng: &mut SmallRng) {
        let n = a.adj.len();
        self.ensure(n);
        self.work.clear();
        self.work.extend_from_slice(&a.adj);

        self.order.clear();
        for (k, c) in around(&a.adj, 0).enumerate() {
            self.order.push(c);
            self.inv[c as usize] = k as u32;
        }
        self.near = prob.nearest_neighbors(Self::FAR.min(n - 1));

        self.open.clear();
        for v in 0..n {
            let (ea, eb) = (a.adj[v], b.adj[v]);
            let mut k = 0;
            for &w in &ea {
                if !eb.contains(&w) {
                    self.remaining[0][v][k] = w;
                    k += 1;
                }
            }
            self.count[0][v] = k as u8;
            let mut k = 0;
            for &w in &eb {
                if !ea.contains(&w) {
                    self.remaining[1][v][k] = w;
                    k += 1;
                }
            }
            self.count[1][v] = k as u8;
            if k > 0 {
                self.open.push(v as u32);
            }
        }

        self.cycles.clear();
        self.spans.clear();
        self.gains.clear();
        while let Some(start) = self.draw_open(rng) {
            self.walk_from(prob, start, rng);
        }
    }

    /// A city that still has an edge outside every cycle, drawn uniformly.
    fn draw_open(&mut self, rng: &mut SmallRng) -> Option<u32> {
        while !self.open.is_empty() {
            let at = rng.random_range(0..self.open.len());
            let v = self.open[at];
            if self.count[0][v as usize] > 0 {
                return Some(v);
            }
            self.open.swap_remove(at);
        }
        None
    }

    fn take_edge(&mut self, ty: usize, u: u32, v: u32) {
        for (x, y) in [(u, v), (v, u)] {
            let slots = &mut self.remaining[ty][x as usize];
            let k = &mut self.count[ty][x as usize];
            if slots[0] == y {
                slots[0] = slots[1];
            }
            *k -= 1;
        }
    }

    /// Walks from `start`, an edge of one parent and then of the other,
    /// and cuts a cycle out of the walk each time it closes one, until the
    /// walk is back at its start with nothing left on it.
    fn walk_from(&mut self, prob: &Tsp, start: u32, rng: &mut SmallRng) {
        self.walk.clear();
        self.walk.push(start);
        let mut cur = start;
        let mut ty = 0usize;
        loop {
            let pick = if self.count[ty][cur as usize] == 2 {
                rng.random_range(0..2)
            } else {
                0
            };
            let next = self.remaining[ty][cur as usize][pick];
            self.take_edge(ty, cur, next);
            self.left_at[ty][cur as usize] = (self.walk.len() - 1) as u32;
            self.walk.push(next);

            // The walk closes a cycle at `next` when it once left `next` by
            // an edge of the other parent, so the two alternate there too.
            let other = 1 - ty;
            let since = self.left_at[other][next as usize];
            cur = next;
            ty = other;
            if since == NONE {
                continue;
            }
            let since = since as usize;
            let last = self.walk.len() - 1;
            let first = self.cycles.len();
            // Edge `k` of the walk has type `k % 2`, and a stored cycle
            // starts with an edge of the first parent.
            if since.is_multiple_of(2) {
                self.cycles.extend_from_slice(&self.walk[since..last]);
            } else {
                self.cycles.extend_from_slice(&self.walk[since + 1..last]);
                self.cycles.push(self.walk[since]);
            }
            let cycle = &self.cycles[first..];
            let len = cycle.len();
            let mut gain = 0.0;
            for j in (0..len).step_by(2) {
                let (p, q, r) = (cycle[j], cycle[j + 1], cycle[(j + 2) % len]);
                gain += prob.distance(p as usize, q as usize);
                gain -= prob.distance(q as usize, r as usize);
            }
            self.spans.push((first as u32, self.cycles.len() as u32));
            self.gains.push(gain);

            for k in since..last {
                let v = self.walk[k] as usize;
                self.left_at[k % 2][v] = NONE;
            }
            self.walk.truncate(since + 1);
            if since == 0 {
                return;
            }
        }
    }

    fn write(&mut self, city: u32, old: u32, new: u32) {
        let slots = &mut self.work[city as usize];
        let slot = if slots[0] == old { 0 } else { 1 };
        debug_assert_eq!(slots[slot], old);
        slots[slot] = new;
        self.writes.push(SlotWrite {
            city,
            slot: slot as u8,
            old,
            new,
        });
    }

    fn undo(&mut self) {
        for w in self.writes.iter().rev() {
            self.work[w.city as usize][w.slot as usize] = w.old;
        }
    }

    fn find(&mut self, mut s: u32) -> u32 {
        while self.parent[s as usize] != s {
            let up = self.parent[s as usize];
            self.parent[s as usize] = self.parent[up as usize];
            s = up;
        }
        s
    }

    /// The segment of the first parent's order that holds position `pos`,
    /// once the order is cut after each position of `cuts`.
    fn segment_of(&self, pos: u32) -> u32 {
        (self.cuts.partition_point(|&c| c < pos) % self.cuts.len()) as u32
    }

    /// Builds the offspring that takes AB-cycle `which` from the second
    /// parent, on `work`, and returns how much shorter than the first parent
    /// it is. The cycle leaves sub-tours, which are joined smallest first.
    fn build(&mut self, prob: &Tsp, which: usize, rng: &mut SmallRng) -> f64 {
        let n = self.work.len() as u32;
        let (first, end) = self.spans[which];
        let (first, len) = (first as usize, (end - first) as usize);
        self.writes.clear();
        self.changes.clear();
        self.cuts.clear();

        for idx in 0..len {
            let v = self.cycles[first + idx];
            let before = self.cycles[first + (idx + len - 1) % len];
            let after = self.cycles[first + (idx + 1) % len];
            let (from_a, from_b) = if idx % 2 == 0 {
                (after, before)
            } else {
                (before, after)
            };
            self.write(v, from_a, from_b);
            if idx % 2 == 0 {
                self.changes.push(EdgeChange {
                    added: false,
                    u: v,
                    v: after,
                });
                // The removed edge joins two neighbors in the order, and
                // the cut falls after the earlier of them.
                let (p, q) = (self.inv[v as usize], self.inv[after as usize]);
                self.cuts.push(if (p + 1) % n == q { p } else { q });
            } else {
                self.changes.push(EdgeChange {
                    added: true,
                    u: v,
                    v: after,
                });
            }
        }

        // Each stretch of the order between two cuts is still a path, and
        // the added edges say which stretches now hang together.
        self.cuts.sort_unstable();
        let m = self.cuts.len();
        self.parent.clear();
        self.parent.extend(0..m as u32);
        self.size.clear();
        self.size.push(self.cuts[0] + n - self.cuts[m - 1]);
        for i in 1..m {
            self.size.push(self.cuts[i] - self.cuts[i - 1]);
        }
        for idx in (1..len).step_by(2) {
            let u = self.cycles[first + idx];
            let v = self.cycles[first + (idx + 1) % len];
            let ru = self.find(self.segment_of(self.inv[u as usize]));
            let rv = self.find(self.segment_of(self.inv[v as usize]));
            if ru != rv {
                self.parent[ru as usize] = rv;
                self.size[rv as usize] += self.size[ru as usize];
            }
        }
        self.roots.clear();
        for s in 0..m as u32 {
            if self.parent[s as usize] == s {
                self.roots.push(s);
            }
        }

        let mut gain = self.gains[which];
        while self.roots.len() > 1 {
            let at = (0..self.roots.len())
                .min_by_key(|&i| self.size[self.roots[i] as usize])
                .expect("there is more than one sub-tour");
            let root = self.roots.swap_remove(at);
            gain += self.join(prob, root, rng);
        }
        gain
    }

    /// Joins the sub-tour of segment `root` to another by the cheapest
    /// exchange of one of its edges against an edge near it, and returns
    /// what the exchange shortens.
    fn join(&mut self, prob: &Tsp, root: u32, rng: &mut SmallRng) -> f64 {
        let n = self.work.len();
        let m = self.cuts.len();
        let first_pos = if root == 0 {
            (self.cuts[m - 1] + 1) % n as u32
        } else {
            self.cuts[root as usize - 1] + 1
        };
        self.unit.clear();
        for c in around(&self.work, self.order[first_pos as usize]) {
            self.unit.push(c);
            self.in_unit[c as usize] = true;
        }

        // (a, b) leaves the sub-tour, (c, d) leaves another, and (a, c)
        // and (b, d) come in.
        let mut best: Option<(f64, [u32; 4])> = None;
        let dist = |i: u32, j: u32| prob.distance(i as usize, j as usize);
        for reach in [Self::NEAR, Self::FAR] {
            for &a in &self.unit {
                let bs = self.work[a as usize];
                let ab = bs.map(|b| dist(a, b));
                for &c in self.near[a as usize].iter().take(reach) {
                    if self.in_unit[c] {
                        continue;
                    }
                    let c = c as u32;
                    let ds = self.work[c as usize];
                    let ac = dist(a, c);
                    let cd = ds.map(|d| dist(c, d));
                    let ad = ds.map(|d| dist(a, d));
                    let bc = bs.map(|b| dist(b, c));
                    for (i, b) in bs.into_iter().enumerate() {
                        for (j, d) in ds.into_iter().enumerate() {
                            let out = ab[i] + cd[j];
                            // Straight, a to c and b to d, then crossed, a
                            // to d and b to c.
                            for (g, x, y) in
                                [(out - ac - dist(b, d), c, d), (out - ad[j] - bc[i], d, c)]
                            {
                                if best.is_none_or(|(held, _)| g > held) {
                                    best = Some((g, [a, b, x, y]));
                                }
                            }
                        }
                    }
                }
            }
            if best.is_some() {
                break;
            }
        }
        let (gain, [a, b, c, d]) = best.unwrap_or_else(|| {
            // Nothing outside the sub-tour is near it, so any edge will do.
            let a = self.unit[rng.random_range(0..self.unit.len())];
            let b = self.work[a as usize][0];
            let c = (0..n as u32)
                .find(|&v| !self.in_unit[v as usize])
                .expect("another sub-tour exists");
            let d = self.work[c as usize][0];
            let gain = prob.distance(a as usize, b as usize)
                + prob.distance(c as usize, d as usize)
                - prob.distance(a as usize, c as usize)
                - prob.distance(b as usize, d as usize);
            (gain, [a, b, c, d])
        });

        self.write(a, b, c);
        self.write(b, a, d);
        self.write(c, d, a);
        self.write(d, c, b);
        for (added, u, v) in [(false, a, b), (false, c, d), (true, a, c), (true, b, d)] {
            self.changes.push(EdgeChange { added, u, v });
        }

        let other = self.find(self.segment_of(self.inv[c as usize]));
        self.parent[root as usize] = other;
        self.size[other as usize] += self.size[root as usize];
        for &v in &self.unit {
            self.in_unit[v as usize] = false;
        }
        gain
    }

    /// How close a join looks first, and how far it looks when that finds
    /// nothing.
    const NEAR: usize = 10;
    const FAR: usize = 50;

    /// The edge entropy the population loses when the first parent becomes
    /// the offspring on `work`. `counts` is edited along the way and put
    /// back, so an edge that the offspring both gains and loses cancels.
    fn entropy_loss(&self, counts: &mut EdgeCounts, population: usize) -> f64 {
        let term = |held: u32| {
            if held == 0 {
                0.0
            } else {
                let share = f64::from(held) / population as f64;
                share * share.ln()
            }
        };
        let mut delta = 0.0;
        for c in &self.changes {
            let held = counts.change(c.u, c.v, c.added);
            let now = if c.added { held + 1 } else { held - 1 };
            delta += term(held) - term(now);
        }
        for c in self.changes.iter().rev() {
            counts.change(c.u, c.v, !c.added);
        }
        -delta
    }
}

/// A genetic algorithm for the symmetric TSP built on edge assembly
/// crossover.
///
/// The population starts as random tours, each taken to a local optimum.
/// Every generation pairs each member with the next in a random order of the
/// population. A pair makes up to `offspring_per_pair` offspring, and the one
/// the selection prefers replaces the first parent of the pair, if it is
/// shorter than that parent. A member is therefore only ever replaced by a
/// tour made mostly of its own edges, which is what keeps the population
/// apart for long.
///
/// # The crossover
///
/// The edges that only one of the two parents has split into AB-cycles,
/// closed walks that alternate between an edge of the first parent and an
/// edge of the second. An offspring is the first parent with the edges of
/// one AB-cycle exchanged, its own taken out and the second parent's put in.
/// Every city keeps two edges, so the result is a set of sub-tours. They are
/// joined smallest first, each by removing one of its edges and one edge of
/// another sub-tour near it and adding the two edges that connect them, at
/// the least cost.
///
/// An offspring differs from its first parent by one cycle and a few joins,
/// so a mating reads both parents once and builds each offspring in time
/// that grows with the cycle and the sub-tours it joins. That is less than
/// the tour unless the cycle splits it into large parts.
///
/// # The selection
///
/// [`EaxSelection::Greedy`] takes the shortest offspring.
/// [`EaxSelection::Entropy`], the default, divides what an offspring
/// shortens by the edge entropy the population would lose by taking it,
/// where the entropy is `−Σ p ln p` over the share `p` of members that hold
/// each edge. An offspring that shortens the tour by bringing in edges the
/// population has few of is preferred to one that copies edges most members
/// already have.
///
/// # When it stops
///
/// Besides its stop condition, a run ends by itself in two cases. One is a
/// population whose tours all have the same length. The other is
/// stagnation. Once the best tour has stood for `1500 / offspring_per_pair`
/// generations, a tenth of the generations run so far becomes the limit, and
/// the run ends when the best tour has stood that long.
///
/// One `run_once` is one mating, or the construction of one member while the
/// population is being built.
///
/// # What is not here
///
/// This is the first stage of the published algorithm. That algorithm goes
/// on to a second stage when the first stagnates, which exchanges several
/// neighboring AB-cycles at once and searches for a block that meets the
/// rest of the tour at few cities. The second stage is not implemented, so
/// a run ends where the published one would change stages. The distances
/// have to be symmetric.
///
/// # Example
///
/// ```
/// use optopus::prelude::*;
///
/// let coords = (0..40)
///     .map(|i| {
///         let theta = std::f64::consts::TAU * i as f64 / 40.0;
///         (theta.cos() * 10.0, theta.sin() * 10.0)
///     })
///     .collect();
/// let tsp = Tsp::new("ring".into(), coords);
/// let mut state = SearchState::new_with_seed(&tsp, 1);
/// EdgeAssemblyCrossoverForTsp::new(StopCondition::iterations(2_000), 20, 10)
///     .run(&mut state)?;
/// assert!(state.best_solution.objective < 62.9);
/// # Ok::<(), optopus::error::OptError>(())
/// ```
///
/// # References
///
/// - Nagata, Y. and Kobayashi, S. "A Powerful Genetic Algorithm Using Edge
///   Assembly Crossover for the Traveling Salesman Problem." *INFORMS Journal
///   on Computing*, 25(2), 346-363, 2013.
pub struct EdgeAssemblyCrossover {
    stop_condition: StopCondition,
    population_size: usize,
    offspring_per_pair: usize,
    selection: EaxSelection,
    members: Vec<Member>,
    counts: EdgeCounts,
    /// The order the members are mated in this generation, and how far into
    /// it the run is.
    mating_order: Vec<usize>,
    next_pair: usize,
    generation: u64,
    /// The generation that last found a better tour, and the limit on the
    /// generations since then once one is set.
    improved_in: u64,
    stagnation_limit: Option<u64>,
    finished: bool,
    mating: Mating,
    /// The AB-cycles of the current pair, the first ones the offspring.
    picks: Vec<u32>,
    descent: AnchoredTourDescent,
}

impl EdgeAssemblyCrossover {
    /// The population size of the published algorithm.
    pub const POPULATION_SIZE: usize = 300;
    /// The offspring a pair makes in the published algorithm.
    pub const OFFSPRING_PER_PAIR: usize = 30;
    /// The generations without a better tour after which a limit on them is
    /// set, times the offspring per pair.
    const STAGNATION_OFFSPRING: usize = 1500;

    /// A search with the entropy selection.
    ///
    /// # Panics
    ///
    /// Panics if `population_size` is below two or `offspring_per_pair` is
    /// zero.
    pub fn new(
        stop_condition: StopCondition,
        population_size: usize,
        offspring_per_pair: usize,
    ) -> Self {
        assert!(population_size >= 2, "population_size must be at least 2");
        assert!(
            offspring_per_pair >= 1,
            "offspring_per_pair must be at least 1"
        );
        Self {
            stop_condition,
            population_size,
            offspring_per_pair,
            selection: EaxSelection::default(),
            members: Vec::new(),
            counts: EdgeCounts::default(),
            mating_order: Vec::new(),
            next_pair: 0,
            generation: 0,
            improved_in: 0,
            stagnation_limit: None,
            finished: false,
            mating: Mating::default(),
            picks: Vec::new(),
            // A member has to be a local optimum, so the descent runs until
            // a pass changes nothing.
            descent: AnchoredTourDescent::new().with_max_passes(usize::MAX),
        }
    }

    /// Builder-style: which offspring of a pair replaces the first parent.
    pub fn with_selection(mut self, selection: EaxSelection) -> Self {
        self.selection = selection;
        self
    }

    /// The generations run so far.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Adds one member, the incumbent for the first and a random tour for
    /// the rest, each taken to a local optimum.
    fn grow(&mut self, state: &mut SearchState<'_, Tsp>) {
        let prob: &Tsp = state.instance;
        let start = if self.members.is_empty() {
            state.solution.clone()
        } else {
            crate::trait_defs::ProblemTrait::new_solution(prob, &mut state.rng)
        };
        let mut partial = prob.to_partial(&start);
        let cities: Vec<usize> = (0..prob.get_n()).collect();
        self.descent
            .repair_around(prob, &mut partial, &cities, &mut state.rng);
        let sol = prob.finish(&partial);
        let member = Member::from_tour(&sol.tour, sol.objective);
        if self.selection == EaxSelection::Entropy {
            self.counts.add_member(&member);
        }
        self.members.push(member);
        if self.members.len() == 1 || sol.is_better_than(&state.solution) {
            state.solution = sol;
        }
        state.iteration += 1;
        state.n_accepted += 1;
        state.update_best();
    }

    /// Closes a generation: draws the next mating order and applies the two
    /// rules that end a run.
    fn turn_generation(&mut self, rng: &mut SmallRng) {
        // Only the first call comes before any mating.
        if self.next_pair > 0 {
            self.generation += 1;
            let stagnant = self.generation - self.improved_in;
            let patience = (Self::STAGNATION_OFFSPRING / self.offspring_per_pair) as u64;
            match self.stagnation_limit {
                None if stagnant == patience.max(1) => {
                    self.stagnation_limit = Some(self.generation / 10);
                }
                Some(limit) if limit <= stagnant => self.finished = true,
                _ => {}
            }
            let (lo, hi) = self
                .members
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), m| {
                    (lo.min(m.length), hi.max(m.length))
                });
            if hi - lo < 1e-3 {
                self.finished = true;
            }
        }
        self.mating_order.clear();
        self.mating_order.extend(0..self.population_size);
        self.mating_order.shuffle(rng);
        self.next_pair = 0;
    }

    /// Mates the next pair. Returns whether the first parent was replaced.
    fn mate(&mut self, state: &mut SearchState<'_, Tsp>) -> bool {
        let prob: &Tsp = state.instance;
        let ia = self.mating_order[self.next_pair];
        let ib = self.mating_order[(self.next_pair + 1) % self.population_size];
        self.next_pair += 1;

        let mating = &mut self.mating;
        mating.set_parents(prob, &self.members[ia], &self.members[ib], &mut state.rng);
        let cycles = mating.spans.len();
        self.picks.clear();
        self.picks.extend(0..cycles as u32);
        let take = self.offspring_per_pair.min(cycles);
        let (chosen, _) = self.picks.partial_shuffle(&mut state.rng, take);

        let mut best_score = 0.0;
        let mut best_gain = 0.0;
        for &which in chosen.iter() {
            let gain = mating.build(prob, which as usize, &mut state.rng);
            let mut kept = false;
            if gain > MIN_IMPROVEMENT {
                let score = match self.selection {
                    EaxSelection::Greedy => gain,
                    EaxSelection::Entropy => {
                        let loss = mating.entropy_loss(&mut self.counts, self.population_size);
                        gain / loss.max(1e-8)
                    }
                };
                if score > best_score {
                    best_score = score;
                    best_gain = gain;
                    kept = true;
                }
            }
            mating.undo();
            if kept {
                // The next offspring starts from empty edit lists, so the two
                // pairs trade places instead of being copied.
                std::mem::swap(&mut mating.best_writes, &mut mating.writes);
                std::mem::swap(&mut mating.best_changes, &mut mating.changes);
            }
        }
        if best_score <= 0.0 {
            return false;
        }

        let member = &mut self.members[ia];
        for w in &mating.best_writes {
            member.adj[w.city as usize][w.slot as usize] = w.new;
        }
        member.length -= best_gain;
        if self.selection == EaxSelection::Entropy {
            for c in &mating.best_changes {
                self.counts.change(c.u, c.v, c.added);
            }
        }

        if member.length < state.best_solution.objective - MIN_IMPROVEMENT {
            let tour = member.tour();
            // Measured afresh, so the length a long line of offspring
            // carried along does not drift from the tour.
            let objective = prob
                .calculate_tour_length(&tour)
                .expect("an offspring visits every city once");
            member.length = objective;
            if objective < state.best_solution.objective - MIN_IMPROVEMENT {
                state.solution = TspSolution { tour, objective };
                state.update_best();
                self.improved_in = self.generation + 1;
            }
        } else if member.length < state.solution.objective - MIN_IMPROVEMENT {
            // Not a new best, but the best member, which the solution
            // follows so that a step composed after this one starts there.
            let tour = member.tour();
            let objective = prob
                .calculate_tour_length(&tour)
                .expect("an offspring visits every city once");
            state.solution = TspSolution { tour, objective };
        }
        true
    }
}

impl Heuristic<Tsp> for EdgeAssemblyCrossover {
    fn clear(&mut self) {
        self.members.clear();
        self.mating_order.clear();
        self.next_pair = 0;
        self.generation = 0;
        self.improved_in = 0;
        self.stagnation_limit = None;
        self.finished = false;
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn is_done<'a>(&self, state: &SearchState<'a, Tsp>) -> bool {
        self.finished || self.stop_condition.is_done(state)
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, Tsp>) -> Result<(), OptError> {
        let prob: &Tsp = state.instance;
        let n = prob.get_n();
        if n < 5 {
            // Every tour of four cities or fewer is a local optimum of the
            // descent or one exchange from it, and there is nothing to mate.
            self.finished = true;
            return Ok(());
        }
        if self.members.is_empty() {
            // Every distance formula over coordinates is symmetric. A store
            // that keeps only the nearest neighbours answers `false` because
            // it has no matrix to check, so it is asked for its coordinates.
            if !prob.is_symmetric() && prob.coordinates().is_none() {
                return Err(OptError::InvalidState(
                    "EdgeAssemblyCrossover needs symmetric distances".into(),
                ));
            }
            if self.selection == EaxSelection::Entropy {
                self.counts.reset(n);
            }
        }
        if self.members.len() < self.population_size {
            self.grow(state);
            return Ok(());
        }
        if self.next_pair >= self.mating_order.len() {
            self.turn_generation(&mut state.rng);
            if self.finished {
                return Ok(());
            }
        }
        if self.mate(state) {
            state.iteration += 1;
            state.n_accepted += 1;
        } else {
            state.progress_iteration();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trait_defs::ProblemTrait;
    use rand::SeedableRng;

    fn scattered(n: usize, seed: u64) -> Tsp {
        let mut rng = SmallRng::seed_from_u64(seed);
        let coords = (0..n)
            .map(|_| (rng.random_range(0.0..100.0), rng.random_range(0.0..100.0)))
            .collect();
        Tsp::new("scattered".into(), coords)
    }

    fn member(tsp: &Tsp, rng: &mut SmallRng) -> Member {
        let sol = tsp.new_solution(rng);
        Member::from_tour(&sol.tour, sol.objective)
    }

    fn edges(adj: &[[u32; 2]]) -> std::collections::BTreeSet<(u32, u32)> {
        adj.iter()
            .enumerate()
            .flat_map(|(v, nbs)| {
                nbs.iter()
                    .map(move |&w| ((v as u32).min(w), (v as u32).max(w)))
            })
            .collect()
    }

    /// The AB-cycles use each edge that one parent alone has exactly once,
    /// and each alternates between the two parents.
    #[test]
    fn ab_cycles_partition_the_edges_the_parents_disagree_on() {
        let tsp = scattered(60, 1);
        let mut rng = SmallRng::seed_from_u64(2);
        let mut mating = Mating::default();
        for _ in 0..20 {
            let (a, b) = (member(&tsp, &mut rng), member(&tsp, &mut rng));
            mating.set_parents(&tsp, &a, &b, &mut rng);
            let (ea, eb) = (edges(&a.adj), edges(&b.adj));
            let mut seen_a = std::collections::BTreeSet::new();
            let mut seen_b = std::collections::BTreeSet::new();
            for &(first, end) in &mating.spans {
                let cycle = &mating.cycles[first as usize..end as usize];
                assert!(cycle.len() >= 4 && cycle.len() % 2 == 0);
                for j in 0..cycle.len() {
                    let (u, v) = (cycle[j], cycle[(j + 1) % cycle.len()]);
                    let e = (u.min(v), u.max(v));
                    if j % 2 == 0 {
                        assert!(ea.contains(&e) && !eb.contains(&e));
                        assert!(seen_a.insert(e), "an edge is in one cycle only");
                    } else {
                        assert!(eb.contains(&e) && !ea.contains(&e));
                        assert!(seen_b.insert(e), "an edge is in one cycle only");
                    }
                }
            }
            assert_eq!(seen_a.len(), ea.difference(&eb).count());
            assert_eq!(seen_b.len(), eb.difference(&ea).count());
        }
    }

    /// Every offspring is one tour over all the cities, its length is the
    /// parent's less the reported gain, and undoing it gives the parent back.
    #[test]
    fn an_offspring_is_a_tour_and_its_gain_is_its_length() {
        let tsp = scattered(80, 3);
        let mut rng = SmallRng::seed_from_u64(4);
        let mut mating = Mating::default();
        for _ in 0..10 {
            let (a, b) = (member(&tsp, &mut rng), member(&tsp, &mut rng));
            mating.set_parents(&tsp, &a, &b, &mut rng);
            for which in 0..mating.spans.len() {
                let gain = mating.build(&tsp, which, &mut rng);
                let child = Member {
                    adj: mating.work.clone(),
                    length: 0.0,
                };
                let tour = child.tour();
                assert_eq!(tour.len(), 80, "the sub-tours were all joined");
                let length = tsp.calculate_tour_length(&tour).unwrap();
                assert!((a.length - gain - length).abs() < 1e-6);
                mating.undo();
                assert_eq!(mating.work, a.adj);
            }
        }
    }

    /// The loss the selection divides by is the change in the entropy of the
    /// counts, and the counts are left as they were.
    #[test]
    fn the_entropy_loss_is_the_change_in_entropy() {
        let tsp = scattered(50, 5);
        let mut rng = SmallRng::seed_from_u64(6);
        let members: Vec<Member> = (0..6).map(|_| member(&tsp, &mut rng)).collect();
        let mut counts = EdgeCounts::default();
        counts.reset(50);
        for m in &members {
            counts.add_member(m);
        }
        let entropy = |counts: &EdgeCounts| -> f64 {
            counts
                .by_city
                .iter()
                .flatten()
                .map(|&(_, held)| {
                    let p = f64::from(held) / 6.0;
                    -p * p.ln()
                })
                .sum()
        };
        let before = entropy(&counts);

        let mut mating = Mating::default();
        mating.set_parents(&tsp, &members[0], &members[1], &mut rng);
        let _ = mating.build(&tsp, 0, &mut rng);
        let loss = mating.entropy_loss(&mut counts, 6);
        assert!(
            (entropy(&counts) - before).abs() < 1e-12,
            "the counts are put back"
        );

        let child = Member {
            adj: mating.work.clone(),
            length: 0.0,
        };
        let mut after = EdgeCounts::default();
        after.reset(50);
        after.add_member(&child);
        for m in &members[1..] {
            after.add_member(m);
        }
        assert!((before - entropy(&after) - loss).abs() < 1e-9);
    }

    #[test]
    fn it_improves_on_its_first_members_and_replays_under_a_seed() {
        let tsp = scattered(100, 7);
        for selection in [EaxSelection::Greedy, EaxSelection::Entropy] {
            let run = || {
                let mut state = SearchState::new_with_seed(&tsp, 9);
                let mut search =
                    EdgeAssemblyCrossover::new(StopCondition::iterations(3_000), 20, 10)
                        .with_selection(selection);
                search.run(&mut state).unwrap();
                let first = search
                    .members
                    .iter()
                    .map(|m| m.length)
                    .fold(f64::INFINITY, f64::min);
                let best = &state.best_solution;
                assert_eq!(
                    tsp.calculate_tour_length(&best.tour).unwrap(),
                    best.objective
                );
                assert!(best.objective <= first + 1e-9);
                best.objective
            };
            assert_eq!(run(), run());
        }
    }

    /// A population that has become one tour ends the run by itself.
    /// The solution a run leaves is the best tour of the population, not the
    /// last member built, so a step composed after it starts there.
    #[test]
    fn the_solution_left_is_the_best_member() {
        let tsp = scattered(60, 21);
        let mut state = SearchState::new_with_seed(&tsp, 2);
        let mut search = EdgeAssemblyCrossover::new(StopCondition::iterations(200), 12, 8);
        search.run(&mut state).unwrap();
        let shortest = search
            .members
            .iter()
            .map(|m| m.length)
            .fold(f64::INFINITY, f64::min);
        let sol = &state.solution;
        assert!(
            sol.objective <= shortest + 1e-6,
            "{} against {shortest}",
            sol.objective
        );
        assert_eq!(tsp.calculate_tour_length(&sol.tour).unwrap(), sol.objective);
    }

    #[test]
    fn a_converged_population_ends_the_run() {
        let tsp = scattered(30, 11);
        let mut state = SearchState::new_with_seed(&tsp, 1);
        let mut search = EdgeAssemblyCrossover::new(StopCondition::new(None, None, None), 6, 10)
            .with_selection(EaxSelection::Greedy);
        search.run(&mut state).unwrap();
        assert!(search.finished);
        assert!(search.generation() >= 1);
    }

    #[test]
    fn asymmetric_distances_are_an_error() {
        let matrix = (0..6)
            .map(|i| {
                (0..6)
                    .map(|j| if i == j { 0.0 } else { (i * 7 + j) as f64 })
                    .collect()
            })
            .collect();
        let tsp = Tsp::from_distance_matrix("asym".into(), matrix).unwrap();
        let mut state = SearchState::new_with_seed(&tsp, 1);
        let err = EdgeAssemblyCrossover::new(StopCondition::iterations(10), 4, 4)
            .run(&mut state)
            .unwrap_err();
        assert!(err.to_string().contains("symmetric"), "{err}");
    }

    /// An instance that stores only nearest neighbours has no matrix to
    /// call symmetric, and its distances are symmetric all the same.
    #[test]
    fn an_instance_of_nearest_neighbours_runs() {
        let mut rng = SmallRng::seed_from_u64(13);
        let coords: Vec<(f64, f64)> = (0..60)
            .map(|_| (rng.random_range(0.0..100.0), rng.random_range(0.0..100.0)))
            .collect();
        let tsp = Tsp::with_nearest_neighbors(
            "sparse".into(),
            coords,
            crate::problem::tsp::EdgeWeightType::Euc2d,
            8,
        );
        let mut state = SearchState::new_with_seed(&tsp, 1);
        EdgeAssemblyCrossover::new(StopCondition::iterations(500), 10, 10)
            .run(&mut state)
            .unwrap();
        let best = &state.best_solution;
        assert_eq!(
            tsp.calculate_tour_length(&best.tour).unwrap(),
            best.objective
        );
    }

    #[test]
    #[should_panic(expected = "population_size must be at least 2")]
    fn a_population_of_one_is_rejected() {
        let _ = EdgeAssemblyCrossover::new(StopCondition::iterations(1), 1, 30);
    }
}
