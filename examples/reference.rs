use stone_model::{util::Random, movement::reconstruct_path, *};

fn main() {
    let random = Random::new(0.1, 0.0);
    let setup = Setup {
        outbound_steps: 1500,
        inbound_steps: 1500,
        acceleration_out: 0.15,
        acceleration_in: 0.1,
        vary_speed: true,
    };

    let mut cx = create_reference_cx(&random);
    let result = run_homing_trial(&mut cx, &random, &setup);

    println!("{:?}", reconstruct_path(&result.physical_states));
}
