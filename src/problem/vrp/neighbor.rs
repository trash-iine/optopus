use rand::Rng;
use rand::rngs::SmallRng;

use super::ops::pricing::{SlotEdit, apply, price};
use super::ops::{before, insertion_cost, node_at, removal_gain, reversal_delta};
use super::problem::{Vrp, VrpSolution};
use crate::common::{TabuMemory, permutation::random_distinct_pair};
use crate::error::OptError;
use crate::trait_defs::{EnabledTabu, Evaluable, Evaluate, MoveToNeighbor};

// ---------------------------------------------------------------------------
// Relocate (inter-slot shift)
// ---------------------------------------------------------------------------

/// Moves one customer from `(from_r, from_i)` to position `to_i` of a *different*
/// slot `to_r`.
///
/// This is the only move that can change which vehicles are used — a source slot
/// can fall empty and a destination slot can wake up — so it is the only one
/// carrying `fixed_cost`, `used_count` and `min_count_shortfall` deltas. Because
/// slots of different vehicle types differ in capacity, speed, cost and
/// route-time limit, relocating across a type boundary is also how the search
/// reassigns work between vehicle types.
#[derive(Debug, Clone)]
pub struct VrpRelocateNeighbor {
    /// Source slot index.
    pub from_r: usize,
    /// Position of the customer within the source route.
    pub from_i: usize,
    /// Destination slot index (`!= from_r`).
    pub to_r: usize,
    /// Insertion position within the destination route (`0..=len`).
    pub to_i: usize,
    /// The relocated customer (cached for `apply` / tabu keying).
    pub customer: usize,
    /// Change in objective (negative = improvement).
    pub gain: f64,
}

/// The customer lifted out of its route, and what that costs: everything a
/// relocation's source side contributes, which is the same for every
/// destination it could be offered.
struct Lifted {
    from_r: usize,
    customer: usize,
    /// Distance the source route loses, already negated for the edit.
    distance_delta: f64,
    demand: i64,
    service: f64,
}

impl Lifted {
    #[inline]
    fn new(prob: &Vrp, sol: &VrpSolution, from_r: usize, from_i: usize) -> Self {
        let customer = sol.routes[from_r][from_i];
        Self {
            from_r,
            customer,
            distance_delta: -removal_gain(prob, &sol.routes[from_r], from_i, 1),
            demand: prob.demands[customer],
            service: prob.service_times[customer],
        }
    }

    #[inline]
    fn edits(&self, prob: &Vrp, sol: &VrpSolution, to_r: usize, to_i: usize) -> [SlotEdit; 2] {
        let c = self.customer;
        let detour = insertion_cost(prob, &sol.routes[to_r], to_i, c, c);
        [
            SlotEdit::remove(
                self.from_r,
                self.distance_delta,
                self.demand,
                self.service,
                1,
            ),
            SlotEdit::insert(to_r, detour, self.demand, self.service, 1),
        ]
    }
}

impl VrpRelocateNeighbor {
    /// The source's and the destination's share of the edit, against `sol`
    /// as it stands before the move. Priced by [`new`](Self::new) to rank the
    /// move and again by [`apply_to_solution`](MoveToNeighbor::apply_to_solution)
    /// to write the caches, the second time being one price per applied move
    /// against thousands per scan, so the move itself carries only its gain.
    fn edits(
        prob: &Vrp,
        sol: &VrpSolution,
        from_r: usize,
        from_i: usize,
        to_r: usize,
        to_i: usize,
    ) -> [SlotEdit; 2] {
        Lifted::new(prob, sol, from_r, from_i).edits(prob, sol, to_r, to_i)
    }

    /// Builds the relocation of `routes[from_r][from_i]` to position `to_i` of
    /// slot `to_r`, computing every cached delta.
    ///
    /// Every construction site goes through here. The cached deltas are what
    /// [`apply_to_solution`](MoveToNeighbor::apply_to_solution) trusts to update
    /// the solution's caches without recomputing the routes, so a hand-filled
    /// move would silently desynchronize all of them.
    ///
    /// # Panics
    ///
    /// Panics if `to_r == from_r` (the neighborhood is inter-slot only: the gain
    /// formula treats source and destination as independent, and the
    /// remove-then-insert in `apply_to_solution` would shift the indices of a
    /// single route), if the slot indices are out of range, if
    /// `from_i >= routes[from_r].len()`, or if `to_i > routes[to_r].len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// let prob = Vrp::with_fleet(
    ///     "demo",
    ///     vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (-1.0, 1.0)],
    ///     vec![0, 1, 1, 1],
    ///     vec![0.0, 0.0, 0.0, 0.0],
    ///     vec![VehicleType::new("truck", 3, 1.0, 2)],
    ///     ObjectiveMode::TotalTime,
    ///     1.0,
    ///     false,
    /// );
    /// let sol = prob.solution_from_routes(vec![vec![1, 2], vec![3]]);
    ///
    /// let m = VrpRelocateNeighbor::new(&prob, &sol, 0, 1, 1, 0);
    /// assert_eq!(m.customer, 2);
    ///
    /// let mut after = sol.clone();
    /// m.apply_to_solution(&prob, &mut after).unwrap();
    /// assert_eq!(after.routes, vec![vec![1], vec![2, 3]]);
    /// assert!((after.objective - (sol.objective + m.gain)).abs() < 1e-9);
    /// ```
    pub fn new(
        prob: &Vrp,
        sol: &VrpSolution,
        from_r: usize,
        from_i: usize,
        to_r: usize,
        to_i: usize,
    ) -> Self {
        assert_ne!(from_r, to_r, "relocate is an inter-slot move");
        let lifted = Lifted::new(prob, sol, from_r, from_i);
        Self::with_lifted(
            prob,
            sol,
            &lifted,
            from_i,
            to_r,
            to_i,
            prob.penalty_weight(),
        )
    }

    /// The move that puts an already-lifted customer at `(to_r, to_i)`. The
    /// scan builds the lifted half once per source position and the penalty
    /// once per neighborhood, both of which `new` pays per candidate.
    #[inline]
    fn with_lifted(
        prob: &Vrp,
        sol: &VrpSolution,
        lifted: &Lifted,
        from_i: usize,
        to_r: usize,
        to_i: usize,
        penalty: f64,
    ) -> Self {
        let edits = lifted.edits(prob, sol, to_r, to_i);
        Self {
            from_r: lifted.from_r,
            from_i,
            to_r,
            to_i,
            customer: lifted.customer,
            gain: price(prob, sol, penalty, &edits).gain,
        }
    }
}

impl Evaluate for VrpRelocateNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain)
    }
}

impl EnabledTabu for VrpRelocateNeighbor {
    /// Keyed by `(customer, slot)`: may this customer enter its destination
    /// slot?
    fn is_move_enabled(&self, tabu: &TabuMemory, iteration: u64) -> bool {
        tabu.is_enabled((self.customer, self.to_r), iteration)
    }

    /// Forbids returning the moved customer to the slot it just left, so the
    /// key it writes is deliberately not the key
    /// [`is_move_enabled`](Self::is_move_enabled) reads.
    fn add_to_tabu_map(&self, tabu: &mut TabuMemory, iteration: u64, rng: &mut SmallRng) {
        tabu.forbid((self.customer, self.from_r), iteration, rng);
    }
}

impl MoveToNeighbor<Vrp> for VrpRelocateNeighbor {
    /// Hands this move's [`EnabledTabu`] policy to the search state, which is
    /// what holds the tabu map.
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }

    fn apply_to_solution(&self, prob: &Vrp, sol: &mut VrpSolution) -> Result<(), OptError> {
        let edits = Self::edits(prob, sol, self.from_r, self.from_i, self.to_r, self.to_i);
        let price = price(prob, sol, prob.penalty_weight(), &edits);
        sol.routes[self.from_r].remove(self.from_i);
        sol.routes[self.to_r].insert(self.to_i, self.customer);
        apply(prob, sol, &edits, &price);
        Ok(())
    }

    /// Every customer into every position of every other slot, where an
    /// idle slot counts once per vehicle type: the other idle slots of the
    /// type would offer the same route under another label.
    fn iter(prob: &Vrp, sol: &VrpSolution) -> impl Iterator<Item = Self> + Send {
        let penalty = prob.penalty_weight();
        // Every place a customer could go, built once for the neighborhood:
        // the lifted source half is then the only thing each candidate needs
        // on top of its destination.
        let places: std::sync::Arc<[(usize, usize)]> = prob
            .destination_slots(&sol.routes)
            .flat_map(|to_r| (0..=sol.routes[to_r].len()).map(move |to_i| (to_r, to_i)))
            .collect();
        (0..sol.routes.len()).flat_map(move |from_r| {
            let places = places.clone();
            (0..sol.routes[from_r].len()).flat_map(move |from_i| {
                let places = places.clone();
                let lifted = Lifted::new(prob, sol, from_r, from_i);
                (0..places.len())
                    .map(move |k| places[k])
                    .filter(move |&(to_r, _)| to_r != from_r)
                    .map(move |(to_r, to_i)| {
                        Self::with_lifted(prob, sol, &lifted, from_i, to_r, to_i, penalty)
                    })
            })
        })
    }

    fn move_to_be_better_than(&self, _: &Vrp, src: &VrpSolution, other: &VrpSolution) -> bool {
        self.evaluate()
            .improves_over(src.evaluate(), other.evaluate())
    }

    fn random_neighbor(prob: &Vrp, sol: &VrpSolution, rng: &mut SmallRng) -> Option<Self> {
        for _ in 0..64 {
            let (from_r, to_r) = random_distinct_pair(sol.routes.len(), rng)?;
            let len_from = sol.routes[from_r].len();
            if len_from == 0 || !prob.is_destination_slot(&sol.routes, to_r) {
                continue;
            }
            let from_i = rng.random_range(0..len_from);
            let to_i = rng.random_range(0..=sol.routes[to_r].len());
            return Some(Self::new(prob, sol, from_r, from_i, to_r, to_i));
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Swap (inter-slot customer exchange)
// ---------------------------------------------------------------------------

/// Exchanges the customer at `(r1, i1)` with the one at `(r2, i2)` in a
/// *different* slot (`r1 != r2`).
///
/// Both routes keep their length, so occupancy — and with it `used_count`, the
/// fixed cost and the `min_count` shortfall — cannot change.
#[derive(Debug, Clone)]
pub struct VrpSwapNeighbor {
    pub r1: usize,
    pub i1: usize,
    pub r2: usize,
    pub i2: usize,
    /// Customer originally at `(r1, i1)`.
    pub c1: usize,
    /// Customer originally at `(r2, i2)`.
    pub c2: usize,
    /// Change in objective (negative = improvement).
    pub gain: f64,
}

/// One side of a swap: the customer, where it sits, and what its route pays
/// for it. The same for every partner it could be exchanged with.
struct Swapped {
    slot: usize,
    customer: usize,
    prev: usize,
    next: usize,
    /// Distance the route pays to visit `customer` between `prev` and `next`.
    detour: f64,
    demand: i64,
    service: f64,
}

impl Swapped {
    #[inline]
    fn new(prob: &Vrp, sol: &VrpSolution, slot: usize, pos: usize) -> Self {
        let route = &sol.routes[slot];
        let customer = route[pos];
        let (prev, next) = (before(route, pos), node_at(route, pos + 1));
        Self {
            slot,
            customer,
            prev,
            next,
            detour: prob.distance(prev, customer) + prob.distance(customer, next),
            demand: prob.demands[customer],
            service: prob.service_times[customer],
        }
    }

    /// What this side's route pays once it holds `other`'s customer instead.
    #[inline]
    fn receiving(&self, prob: &Vrp, other: &Self) -> f64 {
        prob.distance(self.prev, other.customer) + prob.distance(other.customer, self.next)
            - self.detour
    }

    #[inline]
    fn edits(prob: &Vrp, a: &Self, b: &Self) -> [SlotEdit; 2] {
        SlotEdit::exchange(
            (a.slot, a.receiving(prob, b), (a.demand, a.service, 1)),
            (b.slot, b.receiving(prob, a), (b.demand, b.service, 1)),
        )
    }
}

impl VrpSwapNeighbor {
    /// The two slots' shares of the edit against the pre-move `sol`, priced
    /// once to rank and once to apply, as for `VrpRelocateNeighbor`.
    fn edits(
        prob: &Vrp,
        sol: &VrpSolution,
        r1: usize,
        i1: usize,
        r2: usize,
        i2: usize,
    ) -> [SlotEdit; 2] {
        Swapped::edits(
            prob,
            &Swapped::new(prob, sol, r1, i1),
            &Swapped::new(prob, sol, r2, i2),
        )
    }

    /// Builds the exchange of `routes[r1][i1]` and `routes[r2][i2]`, computing
    /// every cached delta.
    ///
    /// Every construction site goes through here, for the same reason as
    /// [`VrpRelocateNeighbor::new`].
    ///
    /// # Panics
    ///
    /// Panics if `r1 == r2` (the neighborhood is inter-slot only: the gain
    /// formula treats the two routes as independent), if the slot indices are
    /// out of range, or if `i1` / `i2` is out of range for its route.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// let prob = Vrp::with_fleet(
    ///     "demo",
    ///     vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (-1.0, 1.0)],
    ///     vec![0, 1, 1, 1],
    ///     vec![0.0, 0.0, 0.0, 0.0],
    ///     vec![VehicleType::new("truck", 3, 1.0, 2)],
    ///     ObjectiveMode::TotalTime,
    ///     1.0,
    ///     false,
    /// );
    /// let sol = prob.solution_from_routes(vec![vec![1, 2], vec![3]]);
    ///
    /// let m = VrpSwapNeighbor::new(&prob, &sol, 0, 1, 1, 0);
    /// assert_eq!((m.c1, m.c2), (2, 3));
    ///
    /// let mut after = sol.clone();
    /// m.apply_to_solution(&prob, &mut after).unwrap();
    /// assert_eq!(after.routes, vec![vec![1, 3], vec![2]]);
    /// assert!((after.objective - (sol.objective + m.gain)).abs() < 1e-9);
    /// ```
    pub fn new(prob: &Vrp, sol: &VrpSolution, r1: usize, i1: usize, r2: usize, i2: usize) -> Self {
        assert_ne!(r1, r2, "swap is an inter-slot move");
        let (a, b) = (
            Swapped::new(prob, sol, r1, i1),
            Swapped::new(prob, sol, r2, i2),
        );
        Self::with_sides(prob, sol, &a, i1, &b, i2, prob.penalty_weight())
    }

    /// The move exchanging two already-read sides. The scan reads each side
    /// once per position and the penalty once per neighborhood, both of
    /// which `new` pays per candidate.
    #[inline]
    fn with_sides(
        prob: &Vrp,
        sol: &VrpSolution,
        a: &Swapped,
        i1: usize,
        b: &Swapped,
        i2: usize,
        penalty: f64,
    ) -> Self {
        let edits = Swapped::edits(prob, a, b);
        Self {
            r1: a.slot,
            i1,
            r2: b.slot,
            i2,
            c1: a.customer,
            c2: b.customer,
            gain: price(prob, sol, penalty, &edits).gain,
        }
    }
}

impl Evaluate for VrpSwapNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain)
    }
}

impl EnabledTabu for VrpSwapNeighbor {
    /// Keyed by the unordered customer pair.
    fn is_move_enabled(&self, tabu: &TabuMemory, iteration: u64) -> bool {
        let key = (self.c1.min(self.c2), self.c1.max(self.c2));
        tabu.is_enabled(key, iteration)
    }

    /// Applying it forbids that customer pair, so an immediate re-swap is out.
    fn add_to_tabu_map(&self, tabu: &mut TabuMemory, iteration: u64, rng: &mut SmallRng) {
        let key = (self.c1.min(self.c2), self.c1.max(self.c2));
        tabu.forbid(key, iteration, rng);
    }
}

impl MoveToNeighbor<Vrp> for VrpSwapNeighbor {
    /// Hands this move's [`EnabledTabu`] policy to the search state, which is
    /// what holds the tabu map.
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }

    fn apply_to_solution(&self, prob: &Vrp, sol: &mut VrpSolution) -> Result<(), OptError> {
        let edits = Self::edits(prob, sol, self.r1, self.i1, self.r2, self.i2);
        let price = price(prob, sol, prob.penalty_weight(), &edits);
        sol.routes[self.r1][self.i1] = self.c2;
        sol.routes[self.r2][self.i2] = self.c1;
        apply(prob, sol, &edits, &price);
        Ok(())
    }

    fn iter(prob: &Vrp, sol: &VrpSolution) -> impl Iterator<Item = Self> + Send {
        let penalty = prob.penalty_weight();
        let v = sol.routes.len();
        (0..v).flat_map(move |r1| {
            ((r1 + 1)..v).flat_map(move |r2| {
                (0..sol.routes[r1].len()).flat_map(move |i1| {
                    let a = Swapped::new(prob, sol, r1, i1);
                    (0..sol.routes[r2].len()).map(move |i2| {
                        let b = Swapped::new(prob, sol, r2, i2);
                        Self::with_sides(prob, sol, &a, i1, &b, i2, penalty)
                    })
                })
            })
        })
    }

    fn move_to_be_better_than(&self, _: &Vrp, src: &VrpSolution, other: &VrpSolution) -> bool {
        self.evaluate()
            .improves_over(src.evaluate(), other.evaluate())
    }

    fn random_neighbor(prob: &Vrp, sol: &VrpSolution, rng: &mut SmallRng) -> Option<Self> {
        for _ in 0..64 {
            let (r1, r2) = random_distinct_pair(sol.routes.len(), rng)?;
            let (len1, len2) = (sol.routes[r1].len(), sol.routes[r2].len());
            if len1 == 0 || len2 == 0 {
                continue;
            }
            let i1 = rng.random_range(0..len1);
            let i2 = rng.random_range(0..len2);
            return Some(Self::new(prob, sol, r1, i1, r2, i2));
        }
        None
    }
}

// ---------------------------------------------------------------------------
// 2-opt (intra-route segment reversal)
// ---------------------------------------------------------------------------

/// Reverses the segment `routes[r][p..=q]` (`p < q`).
///
/// The route keeps its customers, so load, service time, overload, occupancy and
/// fixed cost are all invariant: the only primitive change is the travelled
/// distance, which propagates into the route's time (and from there the makespan
/// and the route-time excess) and into the distance-proportional cost.
#[derive(Debug, Clone)]
pub struct VrpTwoOptNeighbor {
    pub r: usize,
    pub p: usize,
    pub q: usize,
    /// Change in objective (negative = improvement).
    pub gain: f64,
}

impl VrpTwoOptNeighbor {
    /// The slot's share of the edit against the pre-move `sol`, priced once
    /// to rank and once to apply, as for `VrpRelocateNeighbor`.
    fn edit(prob: &Vrp, sol: &VrpSolution, r: usize, p: usize, q: usize) -> [SlotEdit; 1] {
        [SlotEdit::reorder(
            r,
            reversal_delta(prob, &sol.routes[r], p, q),
        )]
    }

    /// Builds the reversal of `routes[r][p..=q]`, computing every cached delta.
    ///
    /// Every construction site goes through here: the deltas are applied without
    /// recomputing the route.
    ///
    /// # Panics
    ///
    /// Panics if `p >= q` (a segment of fewer than two customers has nothing to
    /// reverse), if `r` is out of range, or if `q >= routes[r].len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// let prob = Vrp::with_fleet(
    ///     "demo",
    ///     vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
    ///     vec![0, 1, 1, 1],
    ///     vec![0.0, 0.0, 0.0, 0.0],
    ///     vec![VehicleType::new("truck", 3, 1.0, 1)],
    ///     ObjectiveMode::TotalTime,
    ///     1.0,
    ///     false,
    /// );
    /// // Detour: depot -> 2 -> 1 -> 3 -> depot.
    /// let sol = prob.solution_from_routes(vec![vec![2, 1, 3]]);
    ///
    /// let m = VrpTwoOptNeighbor::new(&prob, &sol, 0, 0, 1);
    /// assert!(m.gain < 0.0);
    ///
    /// let mut after = sol.clone();
    /// m.apply_to_solution(&prob, &mut after).unwrap();
    /// assert_eq!(after.routes, vec![vec![1, 2, 3]]);
    /// assert!((after.objective - (sol.objective + m.gain)).abs() < 1e-9);
    /// ```
    pub fn new(prob: &Vrp, sol: &VrpSolution, r: usize, p: usize, q: usize) -> Self {
        assert!(p < q, "2-opt needs a segment of at least two customers");
        Self::priced(prob, sol, r, p, q, prob.penalty_weight())
    }

    /// The move under an already-read penalty, which the scan reads once per
    /// neighborhood where `new` reads it per candidate.
    #[inline]
    fn priced(prob: &Vrp, sol: &VrpSolution, r: usize, p: usize, q: usize, penalty: f64) -> Self {
        let edit = Self::edit(prob, sol, r, p, q);
        Self {
            r,
            p,
            q,
            gain: price(prob, sol, penalty, &edit).gain,
        }
    }
}

impl Evaluate for VrpTwoOptNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.gain)
    }
}

impl EnabledTabu for VrpTwoOptNeighbor {
    /// Keyed by the `(slot, p, q)` triple.
    fn is_move_enabled(&self, tabu: &TabuMemory, iteration: u64) -> bool {
        tabu.is_enabled((self.r, self.p, self.q), iteration)
    }

    /// Applying it forbids that triple.
    fn add_to_tabu_map(&self, tabu: &mut TabuMemory, iteration: u64, rng: &mut SmallRng) {
        tabu.forbid((self.r, self.p, self.q), iteration, rng);
    }
}

impl MoveToNeighbor<Vrp> for VrpTwoOptNeighbor {
    /// Hands this move's [`EnabledTabu`] policy to the search state, which is
    /// what holds the tabu map.
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }

    fn apply_to_solution(&self, prob: &Vrp, sol: &mut VrpSolution) -> Result<(), OptError> {
        let edit = Self::edit(prob, sol, self.r, self.p, self.q);
        let price = price(prob, sol, prob.penalty_weight(), &edit);
        sol.routes[self.r][self.p..=self.q].reverse();
        apply(prob, sol, &edit, &price);
        Ok(())
    }

    fn iter(prob: &Vrp, sol: &VrpSolution) -> impl Iterator<Item = Self> + Send {
        let penalty = prob.penalty_weight();
        let v = sol.routes.len();
        (0..v).flat_map(move |r| {
            let len = sol.routes[r].len();
            (0..len).flat_map(move |p| {
                ((p + 1)..len).map(move |q| Self::priced(prob, sol, r, p, q, penalty))
            })
        })
    }

    fn move_to_be_better_than(&self, _: &Vrp, src: &VrpSolution, other: &VrpSolution) -> bool {
        self.evaluate()
            .improves_over(src.evaluate(), other.evaluate())
    }

    fn random_neighbor(prob: &Vrp, sol: &VrpSolution, rng: &mut SmallRng) -> Option<Self> {
        let v = sol.routes.len();
        let candidates: Vec<usize> = (0..v).filter(|&r| sol.routes[r].len() >= 2).collect();
        if candidates.is_empty() {
            return None;
        }
        let r = candidates[rng.random_range(0..candidates.len())];
        let (a, b) = random_distinct_pair(sol.routes[r].len(), rng)?;
        let (p, q) = (a.min(b), a.max(b));
        Some(Self::new(prob, sol, r, p, q))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::vrp::{ObjectiveMode, VehicleType};
    use rand::SeedableRng;

    /// Six customers in two mirrored triples, two truck slots (capacity 3,
    /// speed 1, route-time limit 8) and two van slots (capacity 2, speed 2,
    /// unconstrained).
    ///
    /// The mirror symmetry is deliberate: `[1, 2]` and `[3, 4]` on the two
    /// trucks give *exactly* equal route times, which is what the tied-makespan
    /// branch needs.
    fn prob(mode: ObjectiveMode) -> Vrp {
        Vrp::with_fleet(
            "t",
            vec![
                (0.0, 0.0),
                (1.0, 1.0),
                (2.0, 2.0),
                (-1.0, 1.0),
                (-2.0, 2.0),
                (1.0, -1.0),
                (2.0, -2.0),
            ],
            vec![0, 1, 1, 1, 1, 1, 1],
            vec![0.0, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25],
            vec![
                VehicleType::new("truck", 3, 1.0, 2)
                    .with_costs(4.0, 0.2)
                    .with_min_count(1)
                    .with_max_route_time(8.0),
                VehicleType::new("van", 2, 2.0, 2)
                    .with_costs(1.0, 0.5)
                    .with_min_count(1),
            ],
            mode,
            1.0,
            false,
        )
    }

    /// What a relocate prices, re-derived the way `apply_to_solution` does.
    fn priced(
        prob: &Vrp,
        sol: &VrpSolution,
        m: &VrpRelocateNeighbor,
    ) -> ([SlotEdit; 2], crate::problem::vrp::ops::pricing::EditPrice) {
        let edits = VrpRelocateNeighbor::edits(prob, sol, m.from_r, m.from_i, m.to_r, m.to_i);
        let price = price(prob, sol, prob.penalty_weight(), &edits);
        (edits, price)
    }

    fn both_modes() -> [Vrp; 2] {
        [
            prob(ObjectiveMode::TotalTime),
            prob(ObjectiveMode::Makespan),
        ]
    }

    /// Every cached field of `sol` must equal a from-scratch recomputation.
    fn assert_caches_consistent(prob: &Vrp, sol: &VrpSolution) {
        let want = prob.solution_from_routes(sol.routes.clone());
        let close = |a: f64, b: f64, what: &str| {
            assert!(
                (a - b).abs() < 1e-6,
                "{what} drift: {a} vs {b} (routes {:?})",
                sol.routes
            );
        };
        assert_eq!(sol.route_loads, want.route_loads, "route_loads drift");
        assert_eq!(sol.used_count, want.used_count, "used_count drift");
        assert_eq!(sol.overload, want.overload, "overload drift");
        assert_eq!(
            sol.min_count_shortfall, want.min_count_shortfall,
            "min_count_shortfall drift"
        );
        for s in 0..prob.num_slots() {
            close(
                sol.route_distance[s],
                want.route_distance[s],
                "route_distance",
            );
            close(sol.route_time[s], want.route_time[s], "route_time");
        }
        close(sol.total_time, want.total_time, "total_time");
        close(sol.makespan, want.makespan, "makespan");
        close(sol.time_excess, want.time_excess, "time_excess");
        close(sol.total_cost, want.total_cost, "total_cost");
        close(sol.objective, want.objective, "objective");
        prob.validate_routes(&sol.routes).unwrap();
    }

    fn start_routes() -> Vec<Vec<usize>> {
        vec![vec![1, 2], vec![3, 4], vec![5], vec![6]]
    }

    #[test]
    fn relocate_apply_matches_recompute_in_both_modes() {
        for prob in both_modes() {
            for start in [
                start_routes(),
                vec![vec![1, 2, 5], vec![3, 4, 6], vec![], vec![]],
            ] {
                let sol = prob.solution_from_routes(start);
                for nb in VrpRelocateNeighbor::iter(&prob, &sol) {
                    let mut s = sol.clone();
                    nb.apply_to_solution(&prob, &mut s).unwrap();
                    assert_caches_consistent(&prob, &s);
                    assert!(
                        (s.objective - (sol.objective + nb.gain)).abs() < 1e-6,
                        "gain mismatch"
                    );
                }
            }
        }
    }

    #[test]
    fn swap_apply_matches_recompute_in_both_modes() {
        for prob in both_modes() {
            let sol = prob.solution_from_routes(start_routes());
            for nb in VrpSwapNeighbor::iter(&prob, &sol) {
                let mut s = sol.clone();
                nb.apply_to_solution(&prob, &mut s).unwrap();
                assert_caches_consistent(&prob, &s);
                assert!((s.objective - (sol.objective + nb.gain)).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn two_opt_apply_matches_recompute_in_both_modes() {
        for prob in both_modes() {
            let sol = prob.solution_from_routes(vec![vec![1, 2, 5], vec![3, 4, 6], vec![], vec![]]);
            for nb in VrpTwoOptNeighbor::iter(&prob, &sol) {
                let mut s = sol.clone();
                nb.apply_to_solution(&prob, &mut s).unwrap();
                assert_caches_consistent(&prob, &s);
                assert!((s.objective - (sol.objective + nb.gain)).abs() < 1e-6);
            }
        }
    }

    /// Applying a whole chain of moves must not drift: this is what a real run
    /// does, and every step trusts the previous step's caches.
    #[test]
    fn a_long_chain_of_moves_does_not_drift() {
        for prob in both_modes() {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(17);
            let mut sol = prob.solution_from_routes(start_routes());
            for step in 0..200 {
                match step % 3 {
                    0 => {
                        if let Some(m) = VrpRelocateNeighbor::random_neighbor(&prob, &sol, &mut rng)
                        {
                            m.apply_to_solution(&prob, &mut sol).unwrap();
                        }
                    }
                    1 => {
                        if let Some(m) = VrpSwapNeighbor::random_neighbor(&prob, &sol, &mut rng) {
                            m.apply_to_solution(&prob, &mut sol).unwrap();
                        }
                    }
                    _ => {
                        if let Some(m) = VrpTwoOptNeighbor::random_neighbor(&prob, &sol, &mut rng) {
                            m.apply_to_solution(&prob, &mut sol).unwrap();
                        }
                    }
                }
                assert_caches_consistent(&prob, &sol);
            }
        }
    }

    // -----------------------------------------------------------------------
    // The three makespan branches
    // -----------------------------------------------------------------------

    /// Neither affected slot holds the makespan: the O(1) branch.
    #[test]
    fn makespan_survives_a_move_between_two_short_routes() {
        let prob = prob(ObjectiveMode::Makespan);
        let sol = prob.solution_from_routes(vec![vec![1, 2, 6], vec![3], vec![4], vec![5]]);
        let holder = sol.makespan;
        assert!(
            sol.route_time[0] > sol.route_time[1] + 1e-6
                && sol.route_time[0] > sol.route_time[2] + 1e-6,
            "slot 0 must be the unique maximum"
        );

        // Move customer 4 from slot 2 to slot 1 — neither is the holder.
        let m = VrpRelocateNeighbor::new(&prob, &sol, 2, 0, 1, 0);
        let mut after = sol.clone();
        m.apply_to_solution(&prob, &mut after).unwrap();
        assert!(
            (after.makespan - holder).abs() < 1e-9,
            "makespan must not move"
        );
        assert_caches_consistent(&prob, &after);
    }

    /// The holder shrinks: the rescan branch must find the new, lower maximum.
    #[test]
    fn makespan_drops_when_its_holder_gives_a_customer_away() {
        let prob = prob(ObjectiveMode::Makespan);
        let sol = prob.solution_from_routes(vec![vec![1, 2, 6], vec![3], vec![4], vec![5]]);
        let before = sol.makespan;

        // Take customer 6 off the longest route.
        let m = VrpRelocateNeighbor::new(&prob, &sol, 0, 2, 3, 1);
        let mut after = sol.clone();
        m.apply_to_solution(&prob, &mut after).unwrap();
        assert!(after.makespan < before - 1e-6, "makespan must drop");
        assert_caches_consistent(&prob, &after);
    }

    /// Two slots tied at the maximum: shrinking one leaves the makespan where it
    /// was, which only the rescan branch gets right.
    #[test]
    fn a_tied_makespan_is_held_up_by_the_other_slot() {
        let prob = prob(ObjectiveMode::Makespan);
        let sol = prob.solution_from_routes(start_routes());
        assert!(
            (sol.route_time[0] - sol.route_time[1]).abs() < 1e-12,
            "the fixture's two trucks must tie: {} vs {}",
            sol.route_time[0],
            sol.route_time[1]
        );
        let tie = sol.makespan;

        // Move customer 1 off slot 0 onto a van; slot 1 still holds the tie value.
        let m = VrpRelocateNeighbor::new(&prob, &sol, 0, 0, 2, 0);
        let mut after = sol.clone();
        m.apply_to_solution(&prob, &mut after).unwrap();
        assert!(after.route_time[0] < tie - 1e-6, "slot 0 must have shrunk");
        assert!((after.makespan - tie).abs() < 1e-9, "the tie must hold");
        assert_caches_consistent(&prob, &after);
    }

    // -----------------------------------------------------------------------
    // Occupancy transitions (relocate only)
    // -----------------------------------------------------------------------

    #[test]
    fn waking_an_idle_vehicle_charges_its_fixed_cost() {
        let prob = prob(ObjectiveMode::TotalTime);
        // Both vans idle: the van type's min_count of 1 is unmet.
        let sol = prob.solution_from_routes(vec![vec![1, 2, 5], vec![3, 4, 6], vec![], vec![]]);
        assert_eq!(sol.used_count, vec![2, 0]);
        assert_eq!(sol.min_count_shortfall, 1);

        let m = VrpRelocateNeighbor::new(&prob, &sol, 0, 2, 2, 0);
        let (edits, price) = priced(&prob, &sol, &m);
        assert_eq!(price.delta_shortfall, -1);
        // The van's fixed cost of 1.0 on top of the distance-proportional part.
        let variable = 0.2 * edits[0].delta_distance + 0.5 * edits[1].delta_distance;
        assert!((price.delta_cost - (1.0 + variable)).abs() < 1e-9);

        let mut after = sol.clone();
        m.apply_to_solution(&prob, &mut after).unwrap();
        assert_eq!(after.used_count, vec![2, 1]);
        assert_eq!(after.min_count_shortfall, 0);
        assert_caches_consistent(&prob, &after);
    }

    #[test]
    fn emptying_a_route_refunds_its_fixed_cost_and_may_create_a_shortfall() {
        let prob = prob(ObjectiveMode::TotalTime);
        // Slot 2 is the only van in use; emptying it re-opens the shortfall.
        let sol = prob.solution_from_routes(vec![vec![1, 2], vec![3, 4, 6], vec![5], vec![]]);
        assert_eq!(sol.used_count, vec![2, 1]);
        assert_eq!(sol.min_count_shortfall, 0);

        let m = VrpRelocateNeighbor::new(&prob, &sol, 2, 0, 0, 2);
        let (edits, price) = priced(&prob, &sol, &m);
        assert_eq!(price.delta_shortfall, 1);
        // The van's fixed cost of 1.0 is refunded.
        let variable = 0.5 * edits[0].delta_distance + 0.2 * edits[1].delta_distance;
        assert!((price.delta_cost - (variable - 1.0)).abs() < 1e-9);

        let mut after = sol.clone();
        m.apply_to_solution(&prob, &mut after).unwrap();
        assert_eq!(after.used_count, vec![2, 0]);
        assert_eq!(after.min_count_shortfall, 1);
        // The emptied slot is left exactly clean, not at rounding residue.
        assert_eq!(after.route_distance[2], 0.0);
        assert_eq!(after.route_time[2], 0.0);
        assert_caches_consistent(&prob, &after);
    }

    /// Moving the last customer of one slot into an empty slot of the *same*
    /// type keeps the used count where it was — the case a per-slot shortfall
    /// delta would double-count.
    #[test]
    fn a_transfer_within_one_type_leaves_the_used_count_alone() {
        let prob = prob(ObjectiveMode::TotalTime);
        let sol = prob.solution_from_routes(vec![vec![1, 2, 3, 4, 6], vec![], vec![5], vec![]]);
        assert_eq!(sol.used_count, vec![1, 1]);

        // Slot 2 -> slot 3: both vans, so the type's used count is unchanged.
        let m = VrpRelocateNeighbor::new(&prob, &sol, 2, 0, 3, 0);
        let (_, price) = priced(&prob, &sol, &m);
        assert_eq!(price.delta_shortfall, 0);
        // One van's fixed cost is charged and one refunded; the distance
        // moves with the customer, so the variable part cancels too.
        assert!(price.delta_cost.abs() < 1e-12);

        let mut after = sol.clone();
        m.apply_to_solution(&prob, &mut after).unwrap();
        assert_eq!(after.used_count, vec![1, 1]);
        assert_caches_consistent(&prob, &after);
    }

    /// A relocation across a type boundary is a vehicle-type change: the same
    /// customer is priced and timed by the destination type.
    #[test]
    fn relocating_across_types_reprices_the_customer() {
        let prob = prob(ObjectiveMode::TotalTime);
        let sol = prob.solution_from_routes(vec![vec![1, 2], vec![3, 4], vec![5], vec![6]]);
        // Truck (speed 1, rate 0.2) -> van (speed 2, rate 0.5).
        let m = VrpRelocateNeighbor::new(&prob, &sol, 0, 1, 2, 1);
        let mut after = sol.clone();
        m.apply_to_solution(&prob, &mut after).unwrap();
        assert_caches_consistent(&prob, &after);
        // The destination's time grew at half the distance it added.
        let (edits, _) = priced(&prob, &sol, &m);
        let to_time_delta = after.route_time[2] - sol.route_time[2];
        assert!((to_time_delta - (edits[1].delta_distance / 2.0 + 0.5)).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // Samplers and contracts
    // -----------------------------------------------------------------------

    #[test]
    fn random_neighbors_are_members_of_iter() {
        let prob = prob(ObjectiveMode::Makespan);
        let sol = prob.solution_from_routes(vec![vec![1, 2], vec![3, 4], vec![5], vec![6]]);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(3);

        let relocs: Vec<_> = VrpRelocateNeighbor::iter(&prob, &sol).collect();
        for _ in 0..60 {
            if let Some(m) = VrpRelocateNeighbor::random_neighbor(&prob, &sol, &mut rng) {
                assert!(relocs.iter().any(|r| r.from_r == m.from_r
                    && r.from_i == m.from_i
                    && r.to_r == m.to_r
                    && r.to_i == m.to_i));
            }
        }

        let swaps: Vec<_> = VrpSwapNeighbor::iter(&prob, &sol).collect();
        for _ in 0..60 {
            if let Some(m) = VrpSwapNeighbor::random_neighbor(&prob, &sol, &mut rng) {
                assert!(swaps.iter().any(|s| {
                    (s.r1 == m.r1 && s.i1 == m.i1 && s.r2 == m.r2 && s.i2 == m.i2)
                        || (s.r1 == m.r2 && s.i1 == m.i2 && s.r2 == m.r1 && s.i2 == m.i1)
                }));
            }
        }

        let two_opts: Vec<_> = VrpTwoOptNeighbor::iter(&prob, &sol).collect();
        for _ in 0..60 {
            if let Some(m) = VrpTwoOptNeighbor::random_neighbor(&prob, &sol, &mut rng) {
                assert!(m.p < m.q);
                assert!(
                    two_opts
                        .iter()
                        .any(|n| n.r == m.r && n.p == m.p && n.q == m.q)
                );
            }
        }
    }

    /// `None` from `random_neighbor` reaches SA / LAHC as "the neighborhood is
    /// empty" and aborts the run, so it must only happen when it really is.
    #[test]
    fn two_opt_random_neighbor_never_gives_up_on_a_non_empty_neighborhood() {
        let prob = prob(ObjectiveMode::TotalTime);
        let sol = prob.solution_from_routes(vec![vec![1, 2], vec![3, 4], vec![5, 6], vec![]]);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
        for _ in 0..300 {
            let m = VrpTwoOptNeighbor::random_neighbor(&prob, &sol, &mut rng)
                .expect("three routes have two customers each");
            assert!(m.p < m.q);
        }
        // One route of three among singletons is still a non-empty neighborhood.
        let singles = prob.solution_from_routes(vec![vec![1], vec![2], vec![3], vec![4, 5, 6]]);
        assert!(VrpTwoOptNeighbor::random_neighbor(&prob, &singles, &mut rng).is_some());
    }

    #[test]
    #[should_panic(expected = "inter-slot")]
    fn relocate_new_rejects_the_same_slot() {
        let prob = prob(ObjectiveMode::TotalTime);
        let sol = prob.solution_from_routes(start_routes());
        let _ = VrpRelocateNeighbor::new(&prob, &sol, 0, 0, 0, 2);
    }

    #[test]
    #[should_panic(expected = "inter-slot")]
    fn swap_new_rejects_the_same_slot() {
        let prob = prob(ObjectiveMode::TotalTime);
        let sol = prob.solution_from_routes(start_routes());
        let _ = VrpSwapNeighbor::new(&prob, &sol, 0, 0, 0, 1);
    }

    #[test]
    #[should_panic(expected = "at least two customers")]
    fn two_opt_new_rejects_a_degenerate_segment() {
        let prob = prob(ObjectiveMode::TotalTime);
        let sol = prob.solution_from_routes(start_routes());
        let _ = VrpTwoOptNeighbor::new(&prob, &sol, 0, 1, 1);
    }

    // -----------------------------------------------------------------------
    // Smoke test: a generic heuristic must work unchanged
    // -----------------------------------------------------------------------

    #[test]
    fn local_search_never_worsens_the_initial_solution() {
        use crate::heuristic::{Heuristic, LocalSearch, StopCondition};
        use crate::search_state::SearchState;

        for prob in both_modes() {
            let mut state = SearchState::new_with_seed(&prob, 5);
            let initial = state.solution.objective;
            let mut ls: LocalSearch<VrpRelocateNeighbor> =
                LocalSearch::new(StopCondition::iterations(5_000));
            ls.run(&mut state).unwrap();

            assert!(state.best_solution.objective <= initial + 1e-9);
            assert_caches_consistent(&prob, &state.best_solution);
        }
    }
}
