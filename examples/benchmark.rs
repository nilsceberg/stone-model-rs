use stone_model::{util::Random, *};

fn main() {
    let duration = std::time::Duration::from_secs(10);
    let now = std::time::Instant::now();
    let then = now + duration;

    let random = Random::new(0.1, 0.0);
    let setup = Setup {
        outbound_steps: 1500,
        inbound_steps: 1500,
        acceleration_out: 0.15,
        acceleration_in: 0.1,
        vary_speed: true,
    };

    let mut times = 0;
    while std::time::Instant::now() < then {
        let mut cx = create_reference_cx(&random);
        run_homing_trial(&mut cx, &random, &setup);
        times += 1;
    }

    println!(
        "rust: simulated {} flights per second",
        (times as f32) / duration.as_secs_f32()
    );
}
