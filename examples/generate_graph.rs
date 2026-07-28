//! Random graph generation example.
//!
//! Generates one graph per structural model (Erdős–Rényi, Barabási–Albert,
//! Watts–Strogatz) from a fixed seed and writes each to a MaxCut-compatible
//! edge-list file. The same seed always yields the same graph, portably.
//!
//! Run with:
//! ```
//! cargo run --example generate_graph
//! ```

use optopus::prelude::*;

fn main() {
    // (min, max) inclusive integer weight range.
    let weight_range = (1, 10);
    let mut rng = seeded_rng(42);

    let graphs = [
        (
            "erdos_renyi",
            Graph::erdos_renyi(100, 0.05, &mut rng).with_random_weights(weight_range, &mut rng),
        ),
        (
            "barabasi_albert",
            Graph::barabasi_albert(100, 3, &mut rng).with_random_weights(weight_range, &mut rng),
        ),
        (
            "watts_strogatz",
            Graph::watts_strogatz(100, 6, 0.2, &mut rng)
                .with_random_weights(weight_range, &mut rng),
        ),
    ];

    let out_dir = std::env::temp_dir();
    for (name, graph) in graphs {
        let path = out_dir.join(format!("optopus_{name}.txt"));
        graph.write_to_file(&path).expect("write graph");

        println!(
            "{name:16} -> {} vertices with edges, {} edges  ({})",
            graph.num_vertices(),
            graph.num_edges(),
            path.display()
        );

        // The written file can be loaded straight back as a MaxCut instance.
        let mc = MaxCut::load_file(&path).expect("reload as MaxCut");
        assert_eq!(mc.graph.num_edges(), graph.num_edges());
    }
}
