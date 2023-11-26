use ndarray::Array;
use rand::RngCore;
use rayon::prelude::*;
use stone_model::{stats::FlightStats, util::Random, *};
use tqdm::Iter;

/// Grid search for dye parameters
fn main() {
    let setup = Setup {
        outbound_steps: 1500,
        inbound_steps: 1500,
        acceleration_out: 0.15,
        acceleration_in: 0.1,
        vary_speed: true,
        record_memory: false,
    };

    let beta_space = Array::linspace(0.0, 0.9, 20);
    let h_space = Array::logspace(10.0, -4.0, 0.0, 20);
    let w0_space = Array::logspace(10.0, -40.0, 0.0, 20);
    let samples = 30;

    let grid = itertools::iproduct!(h_space.iter(), w0_space.iter(), beta_space.iter())
        .tqdm()
        .par_bridge()
        .map(|params @ (&h, &w0, &beta)| {
            let random = Random::new(0.1, 0.0, None);
            let mut distances = Vec::with_capacity(samples);
            for _ in 0..samples {
                let mut setup = setup.clone();
                setup.outbound_steps = 1000 + (random.rng().next_u32() as usize) % 1000;
                let outbound = setup.generate_outbound(&random);
                let mut cx = create_weight_logistic_cx(&random, h, w0, beta);
                let result = run_homing_trial(&setup, &mut cx, outbound);
                let stats = FlightStats::analyze(&setup, &result);
                distances.push(stats.min_distance_to_home);
            }

            (params, distances.iter().sum::<f32>() / (samples as f32))
        });

    let grid = grid.collect::<Vec<_>>();
    let best = grid.iter().fold(
        ((&f32::NAN, &f32::NAN, &f32::NAN), f32::INFINITY),
        |a, b| if a.1 < b.1 { a } else { *b },
    );
    println!("{:?}", best);
}
