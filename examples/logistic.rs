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

    //let (h, w0, beta) = (0.012742752, 6.1584935e-5, 0.2368421);
    //let (h, w0, beta) = (0.007847599, 6.1584935e-5, 0.42631575);
    //let (h, w0, beta) = (0.0060, 6.1584935e-5, 0.42631575);
    let (h, w0, beta) = (0.0098329304, 6.1584935e-6, 0.2631579);

    let outbound = setup.generate_outbound(&random);
    let mut cx = create_weight_logistic_cx(&random, h, w0, beta);
    let result = run_homing_trial(&setup, &mut cx, outbound);

    result.print();
}
