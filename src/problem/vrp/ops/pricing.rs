//! The one answer to "what does this route edit cost".
//!
//! A route edit is described per slot, as the change in travelled distance,
//! load, service time and customer count of each slot it touches
//! ([`SlotEdit`]). [`price`] turns that into the objective delta under a given
//! penalty and the aggregate deltas the caches move by, and [`apply`] moves
//! them. The generic moves, the descent and the ruin-and-recreate operators
//! all go through this pair, so there is exactly one place that knows how a
//! slot's vehicle type, the objective mode and the three soft constraints
//! enter a move's value.
//!
//! The distance part of an edit is the caller's, since only the caller knows
//! which edges it exchanges; everything downstream of a distance delta is
//! here.
//!
//! `price` runs once per candidate move, which on a routing problem is the
//! hot loop of every search, so it reads only what the gain needs: a term a
//! vehicle type does not charge (an infinite route-time limit, a zero cost,
//! an unchanged occupancy) costs one predictable branch, and the per-slot
//! values after the edit are derived again by `apply`, once per applied move.

use super::super::problem::{
    ObjectiveMode, Vrp, VrpSolution, overload_of, shortfall_of, time_excess_of,
};

/// Change in total overflow when `demand` is added to a route carrying
/// `load`; a negative `demand` is a removal.
#[inline]
fn excess_delta_insert(capacity: i64, load: i64, demand: i64) -> i64 {
    overload_of(load + demand, capacity) - overload_of(load, capacity)
}

/// The most slots one edit touches: an inter-route move has a source and a
/// destination, an intra-route move one slot. Everything downstream is sized
/// by the edit's own `N`, so this only bounds what a caller may ask for.
const MAX_EDITS: usize = 2;

/// Tolerance for "this slot is the one currently holding the makespan".
///
/// Judged conservatively: a slot within this distance *below* the current
/// makespan counts as holding it, which only ever costs an extra rescan.
const MAKESPAN_TIE_EPS: f64 = 1e-9;

/// One slot's share of a route edit.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SlotEdit {
    /// The slot whose route changes.
    pub slot: usize,
    /// Change in the slot's travelled distance, from the exchanged edges.
    pub delta_distance: f64,
    /// Total demand entering the slot minus the demand leaving it.
    pub delta_load: i64,
    /// Total service time entering the slot minus that leaving it.
    pub delta_service: f64,
    /// Customers entering the slot minus customers leaving it. Together with
    /// the route's current length this says whether the slot falls idle or
    /// is put to use, which is what moves the fixed cost, the used count and
    /// the minimum-count shortfall.
    pub delta_count: isize,
}

impl SlotEdit {
    /// An edit that keeps the slot's customers and only reorders them.
    pub(crate) fn reorder(slot: usize, delta_distance: f64) -> Self {
        Self {
            slot,
            delta_distance,
            delta_load: 0,
            delta_service: 0.0,
            delta_count: 0,
        }
    }

    /// An edit that adds customers of total `demand` and `service` to the slot.
    pub(crate) fn insert(
        slot: usize,
        delta_distance: f64,
        demand: i64,
        service: f64,
        count: usize,
    ) -> Self {
        Self {
            slot,
            delta_distance,
            delta_load: demand,
            delta_service: service,
            delta_count: count as isize,
        }
    }

    /// An edit that takes customers of total `demand` and `service` out of
    /// the slot.
    pub(crate) fn remove(
        slot: usize,
        delta_distance: f64,
        demand: i64,
        service: f64,
        count: usize,
    ) -> Self {
        Self {
            slot,
            delta_distance,
            delta_load: -demand,
            delta_service: -service,
            delta_count: -(count as isize),
        }
    }

    /// The pair of edits where two slots trade content: each gives up what
    /// the other receives, so the loads, service times and counts mirror.
    /// `(slot, delta_distance, content)` per side, `content` being the
    /// `(demand, service, count)` that slot gives up.
    pub(crate) fn exchange(
        (slot_a, delta_distance_a, given_a): (usize, f64, (i64, f64, usize)),
        (slot_b, delta_distance_b, given_b): (usize, f64, (i64, f64, usize)),
    ) -> [Self; 2] {
        let (demand_a, service_a, count_a) = given_a;
        let (demand_b, service_b, count_b) = given_b;
        [
            Self {
                slot: slot_a,
                delta_distance: delta_distance_a,
                delta_load: demand_b - demand_a,
                delta_service: service_b - service_a,
                delta_count: count_b as isize - count_a as isize,
            },
            Self {
                slot: slot_b,
                delta_distance: delta_distance_b,
                delta_load: demand_a - demand_b,
                delta_service: service_a - service_b,
                delta_count: count_a as isize - count_b as isize,
            },
        ]
    }

    /// Whether the slot, holding `len` customers now, falls idle or is put
    /// to use by this edit.
    #[inline]
    fn occupancy(&self, len: usize) -> (bool, bool) {
        if self.delta_count == 0 {
            return (false, false);
        }
        let new_len = len
            .checked_add_signed(self.delta_count)
            .expect("an edit cannot remove more customers than the route has");
        (len > 0 && new_len == 0, len == 0 && new_len > 0)
    }

    /// The slot's distance and time after the edit, given what it has now.
    /// Emptying a route is exactly "lose everything it had": an emptied slot
    /// is set to a clean `0.0` rather than the rounding residue of
    /// subtracting what it had.
    #[inline]
    fn after(&self, speed: f64, old_distance: f64, old_time: f64, empties: bool) -> (f64, f64) {
        if empties {
            (0.0, 0.0)
        } else {
            (
                old_distance + self.delta_distance,
                old_time + (self.delta_distance / speed + self.delta_service),
            )
        }
    }
}

/// What a priced edit does to the aggregates, computed once per candidate.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EditPrice {
    /// Change in the objective under the penalty the edit was priced with;
    /// negative is an improvement.
    pub gain: f64,
    pub delta_total_time: f64,
    pub delta_cost: f64,
    pub delta_overload: i64,
    pub delta_time_excess: f64,
    pub delta_shortfall: i64,
    /// The makespan after the edit, resolved here only when the objective
    /// reads it. Under `TotalTime` it is `None` and [`apply`] refreshes the
    /// cache instead, so the candidate scan never pays for a maximum nobody
    /// charges.
    pub makespan_after: Option<f64>,
}

/// Prices `edits` against `sol`, which must still hold the pre-edit routes
/// and caches. `penalty` weighs the three violations exactly as
/// [`Vrp::penalty_weight`] does in the objective.
///
/// `N` is the number of slots the edit touches, one or two, fixed at the
/// call site so the per-slot loop unrolls.
#[inline(always)]
pub(crate) fn price<const N: usize>(
    prob: &Vrp,
    sol: &VrpSolution,
    penalty: f64,
    edits: &[SlotEdit; N],
) -> EditPrice {
    const {
        assert!(N >= 1 && N <= MAX_EDITS, "an edit touches one or two slots");
    }
    debug_assert!(
        N < 2 || edits[0].slot != edits[N - 1].slot,
        "one slot, one edit"
    );

    let mut delta_total_time = 0.0;
    let mut delta_cost = 0.0;
    let mut delta_overload = 0;
    let mut delta_time_excess = 0.0;
    // Occupancy changes, one per edit; both edits may hit one vehicle type,
    // and the shortfall is then read once from the combined change.
    let mut occupancy = [(usize::MAX, 0i64); N];
    let mut occupancy_changed = false;
    let mut new_times = [0.0; N];

    for (k, e) in edits.iter().enumerate() {
        let vt = prob.slot_terms(e.slot);
        let t = vt.vehicle_type;
        let (empties, wakes) = e.occupancy(sol.routes[e.slot].len());
        let old_time = sol.route_time[e.slot];

        let (delta_distance, delta_time) = if empties {
            (-sol.route_distance[e.slot], -old_time)
        } else {
            (
                e.delta_distance,
                e.delta_distance / vt.speed + e.delta_service,
            )
        };
        delta_total_time += delta_time;
        delta_overload += excess_delta_insert(vt.capacity, sol.route_loads[e.slot], e.delta_load);

        if vt.max_route_time.is_finite() || prob.objective_mode() == ObjectiveMode::Makespan {
            let new_time = if empties { 0.0 } else { old_time + delta_time };
            new_times[k] = new_time;
            delta_time_excess += time_excess_of(new_time, vt.max_route_time)
                - time_excess_of(old_time, vt.max_route_time);
        }
        if vt.variable_cost_per_distance != 0.0 {
            delta_cost += vt.variable_cost_per_distance * delta_distance;
        }
        if wakes || empties {
            let change = i64::from(wakes) - i64::from(empties);
            delta_cost += vt.fixed_cost * change as f64;
            occupancy[k] = (t, change);
            occupancy_changed = true;
        }
    }

    let mut delta_shortfall = 0;
    if occupancy_changed {
        // Two edits may hit one vehicle type, whose shortfall then moves by
        // the combined change rather than by each edit's on its own.
        for k in 0..N {
            for j in (k + 1)..N {
                if occupancy[j].0 == occupancy[k].0 {
                    occupancy[k].1 += occupancy[j].1;
                    occupancy[j] = (usize::MAX, 0);
                }
            }
        }
        for &(t, change) in occupancy.iter().filter(|(_, c)| *c != 0) {
            let min_count = prob.vehicle_types()[t].min_count;
            let old = sol.used_count[t] as i64;
            delta_shortfall += shortfall_of(min_count, old + change) - shortfall_of(min_count, old);
        }
    }

    let (time_component_delta, makespan_after) = match prob.objective_mode() {
        ObjectiveMode::TotalTime => (delta_total_time, None),
        ObjectiveMode::Makespan => {
            let new = makespan_after(sol, edits, &new_times);
            (new - sol.makespan, Some(new))
        }
    };
    let gain = time_component_delta
        + prob.cost_weight() * delta_cost
        + penalty * (delta_overload as f64 + delta_time_excess + delta_shortfall as f64);

    EditPrice {
        gain,
        delta_total_time,
        delta_cost,
        delta_overload,
        delta_time_excess,
        delta_shortfall,
        makespan_after,
    }
}

/// The makespan `sol` would have if the edited slots took the route times in
/// `new_times`.
///
/// The makespan is a min-max aggregate, so it is not additive the way every
/// other cache is. If no edited slot currently holds the maximum, nothing can
/// have been removed from it and the new one is the old one raised by
/// whatever the edited slots now take, O(1). If an edited slot holds it (ties
/// included, judged with [`MAKESPAN_TIE_EPS`]), the maximum may have been
/// the only thing keeping the value up, and every slot is rescanned:
/// O(num_slots), the fleet size, not the customer count. Empty slots have
/// route time `0.0`, so flooring at `0.0` yields "the largest route time
/// over the non-empty slots".
fn makespan_after<const N: usize>(
    sol: &VrpSolution,
    edits: &[SlotEdit; N],
    new_times: &[f64; N],
) -> f64 {
    if holds_makespan(sol, edits) {
        rescan_makespan(sol, edits, new_times)
    } else {
        new_times.iter().fold(sol.makespan, |m, &t| m.max(t))
    }
}

/// Whether any edited slot is the one currently holding the makespan, ties
/// included.
#[inline]
fn holds_makespan<const N: usize>(sol: &VrpSolution, edits: &[SlotEdit; N]) -> bool {
    edits
        .iter()
        .any(|e| sol.route_time[e.slot] >= sol.makespan - MAKESPAN_TIE_EPS)
}

/// The largest route time over every slot, reading the edited ones from
/// `new_times`.
fn rescan_makespan<const N: usize>(
    sol: &VrpSolution,
    edits: &[SlotEdit; N],
    new_times: &[f64; N],
) -> f64 {
    let mut m = 0.0_f64;
    for (s, &old_time) in sol.route_time.iter().enumerate() {
        let t = edits
            .iter()
            .position(|e| e.slot == s)
            .map_or(old_time, |k| new_times[k]);
        m = m.max(t);
    }
    m
}

/// Moves every cache of `sol` by a price, the routes having already been
/// edited by the caller. `sol.objective` moves by `price.gain`, so it stays
/// the objective under whatever penalty the edit was priced with.
pub(crate) fn apply<const N: usize>(
    prob: &Vrp,
    sol: &mut VrpSolution,
    edits: &[SlotEdit; N],
    price: &EditPrice,
) {
    // Read before the per-slot caches are overwritten: under `TotalTime` the
    // makespan is refreshed here rather than resolved per candidate.
    let held_max = price.makespan_after.is_none() && holds_makespan(sol, edits);

    let mut new_times = [0.0; N];
    for (k, e) in edits.iter().enumerate() {
        let vt = prob.slot_terms(e.slot);
        let t = vt.vehicle_type;
        // The routes are post-edit, so the length the pricing saw is
        // recovered before asking the same occupancy question it asked.
        let was = sol.routes[e.slot].len().wrapping_add_signed(-e.delta_count);
        let (empties, wakes) = e.occupancy(was);
        if empties {
            sol.used_count[t] -= 1;
        } else if wakes {
            sol.used_count[t] += 1;
        }
        sol.route_loads[e.slot] += e.delta_load;
        let (distance, time) = e.after(
            vt.speed,
            sol.route_distance[e.slot],
            sol.route_time[e.slot],
            empties,
        );
        sol.route_distance[e.slot] = distance;
        sol.route_time[e.slot] = time;
        new_times[k] = time;
    }

    sol.total_time += price.delta_total_time;
    sol.overload += price.delta_overload;
    sol.time_excess += price.delta_time_excess;
    sol.min_count_shortfall += price.delta_shortfall;
    sol.total_cost += price.delta_cost;
    sol.objective += price.gain;

    sol.makespan = match price.makespan_after {
        Some(m) => m,
        None if held_max => sol.route_time.iter().copied().fold(0.0_f64, f64::max),
        None => new_times.iter().fold(sol.makespan, |m, &t| m.max(t)),
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The excess delta must agree with recomputing the overflow, which is
    /// what the applied edit's `overload` cache is later checked against.
    #[test]
    fn excess_delta_matches_a_recompute() {
        let capacity = 10;
        for load in [0, 5, 10, 14] {
            for demand in [-7, -1, 1, 4, 7] {
                assert_eq!(
                    excess_delta_insert(capacity, load, demand),
                    overload_of(load + demand, capacity) - overload_of(load, capacity)
                );
            }
        }
    }
}
