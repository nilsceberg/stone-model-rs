use stone_model::{util::Random, *};

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
    let mut cx = create_weight_logistic_cx(&random, 7.8E-3, 6.2E-5, 0.3);
    let result = run_homing_trial(&setup, &mut cx, outbound);

    //println!("{:?}", reconstruct_path(&result.physical_states));
    println!("{:?}", result.memory_record.unwrap());
}
