use stone_model::{movement::reconstruct_path, util::Random, *};

fn main() {
    let random = Random::new(0.1, 0.0, COMMON_SEED);
    let setup = Setup {
        outbound_steps: 1500,
        inbound_steps: 1500,
        acceleration_out: 0.15,
        acceleration_in: 0.1,
        vary_speed: true,
        record_memory: true,
    };

    let outbound = setup.generate_outbound(&random);
    let mut cx = create_reference_cx(&random);
    let result = run_homing_trial(&setup, &mut cx, outbound);

    println!("{:?}", reconstruct_path(&result.physical_states));
}
