//! The two heuristics built on the shared route machinery run on a
//! heterogeneous fleet, under both objective modes.
//!
//! The generic moves are covered where they live, by the unit tests in
//! `problem/vrp/neighbor.rs`, which exhaust the neighborhood against a
//! from-scratch rebuild on the same kind of fixture. What only an
//! integration test reaches is ALNS and HGS, which price their edits through
//! `RouteState`, the granular descent and the giant-tour split rather than
//! through the move types, so a fleet term dropped anywhere in that stack
//! shows up here and nowhere else.
//!
//! The fixture is the committed `demo_fleet.toml`: two trucks and two vans of
//! different capacity, speed and costs, with a route-time limit on the trucks
//! and a minimum count of one truck.

use optopus::heuristic::{Heuristic, HybridGeneticSearchForVrp, StopCondition, alns_for_vrp};
use optopus::problem::{ObjectiveMode, Vrp};
use optopus::search_state::SearchState;

fn instance(mode: ObjectiveMode) -> Vrp {
    let vrp =
        Vrp::load_file("data/instances/vrp/demo_fleet.toml").expect("the committed fixture loads");
    assert!(
        vrp.vehicle_types().len() >= 2,
        "the fixture is heterogeneous"
    );
    vrp.with_objective_mode(mode)
}

/// The best solution is a partition over every slot, and the objective it
/// reports is the one the problem would price its routes at.
fn check(vrp: &Vrp, state: &SearchState<'_, Vrp>, what: &str) {
    let best = &state.best_solution;
    vrp.validate_routes(&best.routes)
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    let fresh = vrp.solution_from_routes(best.routes.clone());
    assert!(
        (best.objective - fresh.objective).abs() < 1e-6,
        "{what}: reported {} where its routes price at {}",
        best.objective,
        fresh.objective
    );
    assert!(
        best.objective <= state.initial_solution.objective + 1e-9,
        "{what}: worsened the start"
    );
}

fn run<H: Heuristic<Vrp>>(vrp: &Vrp, mut heuristic: H, seed: u64, what: &str) {
    let mut state = SearchState::new_with_seed(vrp, seed);
    heuristic.run(&mut state).unwrap();
    check(vrp, &state, what);
}

#[test]
fn the_route_machinery_heuristics_run_on_a_heterogeneous_fleet() {
    for mode in [ObjectiveMode::TotalTime, ObjectiveMode::Makespan] {
        let vrp = instance(mode);
        run(
            &vrp,
            alns_for_vrp(StopCondition::iterations(300), 0.3, 0.99),
            5,
            &format!("AdaptiveLargeNeighborhoodSearch under {mode:?}"),
        );
        run(
            &vrp,
            HybridGeneticSearchForVrp::new(StopCondition::iterations(60), 4, 6, 3, 0.2, None),
            6,
            &format!("HybridGeneticSearch under {mode:?}"),
        );
    }
}

/// The two modes rank the same routes differently, so a search under one is
/// not simply a search under the other with a different label.
#[test]
fn the_objective_mode_is_what_the_heuristics_optimize() {
    let total = instance(ObjectiveMode::TotalTime);
    let makespan = instance(ObjectiveMode::Makespan);
    let mut a = SearchState::new_with_seed(&total, 9);
    let mut b = SearchState::new_with_seed(&makespan, 9);
    alns_for_vrp(StopCondition::iterations(400), 0.3, 0.99)
        .run(&mut a)
        .unwrap();
    alns_for_vrp(StopCondition::iterations(400), 0.3, 0.99)
        .run(&mut b)
        .unwrap();
    // Re-price each best under the other mode: the makespan search must not
    // lose to the total-time search on the makespan, nor vice versa.
    let a_under_makespan = makespan.solution_from_routes(a.best_solution.routes.clone());
    let b_under_total = total.solution_from_routes(b.best_solution.routes.clone());
    assert!(b.best_solution.objective <= a_under_makespan.objective + 1e-9);
    assert!(a.best_solution.objective <= b_under_total.objective + 1e-9);
}
