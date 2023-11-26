use stone_model::{model::network::Noise, movement::reconstruct_path, *};

fn main() {
    let noise = Noise::new(0.1, 0.0);
    let setup = Setup {
        outbound_steps: 1500,
        inbound_steps: 1500,
        acceleration_out: 0.15,
        acceleration_in: 0.1,
        vary_speed: true,
    };

    let mut cx = create_reference_cx(&noise);
    let result = run_homing_trial(&mut cx, &setup);

    println!("{:?}", reconstruct_path(&result.physical_states));
}
