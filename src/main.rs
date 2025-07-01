use optopus::{
    algorithm::{
        BangBangSimulatedAnnealing, Heuristic, LocalSearch, SimulatedAnnealing, StopCondition,
        TabuSearch,
    },
    problem::max_cut::{MaxCut, MaxCutFlipNeighbor},
    search_state::{EnumerateMoveToNeighbor, Evaluable, ProblemTrait, SearchState},
};

fn run_benchmark<Problem, MoveToNeighbor>(problem: Problem, stop_condition: StopCondition)
where
    Problem: ProblemTrait,
    Problem::Objective: std::fmt::Display,
    for<'a> SearchState<'a, Problem>: EnumerateMoveToNeighbor<MoveToNeighbor>,
    MoveToNeighbor: Clone + std::hash::Hash + std::cmp::Eq + Evaluable<f64>,
{
    let ls = LocalSearch::<MoveToNeighbor>::new(stop_condition.clone());
    let ts = TabuSearch::<MoveToNeighbor>::new(stop_condition.clone(), 60, 3);
    let sa = SimulatedAnnealing::new(stop_condition.clone(), 100.0, 0.999);
    let bbsa = BangBangSimulatedAnnealing::new(stop_condition.clone(), 100.0, 0.999, 1.0, 50.0);

    let hs: Vec<(&str, Box<dyn Heuristic<Problem>>)> = vec![
        ("Local Search", Box::new(ls)),
        ("Tabu Search", Box::new(ts)),
        ("Simulated Annealing", Box::new(sa)),
        ("Bang-Bang Simulated Annealing", Box::new(bbsa)),
    ];

    for heuristic in hs.iter() {
        let mut state = SearchState::new(&problem, rand::rng());
        tracing::info!("Start {}", heuristic.0);
        heuristic.1.run(&mut state);
        tracing::info!(
            "{} best: {} (iter: {}, time: {}[s])",
            heuristic.0,
            state.best_objective,
            state.best_iteration,
            (state.best_time - state.start_time).as_secs_f64()
        );
    }
}

fn main() {
    tracing_subscriber::fmt::init();

    /*
    let mut mc = MaxCut::new();
    mc.add_weight(0, 1, 1.0);
    mc.add_weight(0, 2, 1.0);
    mc.add_weight(0, 1, 2.0);

    for i0 in [false, true] {
        for i1 in [false, true] {
            for i2 in [false, true] {
                let mut sol = HashMap::new();
                sol.insert(0, i0);
                sol.insert(1, i1);
                sol.insert(2, i2);

                println!("{:?}: {}", sol, mc.calculate_cut_size(&sol))
            }
        }
    }
     */

    let mc = MaxCut::load_from_file("data/max_cut/G1").unwrap();
    // let sc = StopCondition::new(Some(10000), None, None);
    let sc = StopCondition::new(None, Some(std::time::Duration::from_secs(10)), None);
    run_benchmark::<_, MaxCutFlipNeighbor>(mc, sc);
}
