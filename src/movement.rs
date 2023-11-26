use nalgebra::{DVector, SVector, Vector2};
use rand::{distributions::Distribution, Rng};
use rand_distr::Normal;

pub const DEFAULT_ACCELERATION: f32 = 0.15;
pub const DEFAULT_DRAG: f32 = 0.15;

#[derive(Debug, Clone)]
pub struct PhysicalState {
    pub velocity: Vector2<f32>,
    pub heading: f32,
}

impl PhysicalState {
    pub fn next(&self, rotation: f32, acceleration: f32, drag: f32) -> PhysicalState {
        PhysicalState {
            velocity: (self.velocity
                + Vector2::new(self.heading.sin(), self.heading.cos()) * acceleration)
                * (1.0 - drag),
            heading: (self.heading + rotation).rem_euclid(std::f32::consts::TAU),
        }
    }
}

pub fn generate_outbound(rng: &mut impl Rng, steps: usize, acceleration: f32, vary_speed: bool) -> Vec<PhysicalState> {
    let mut states = Vec::with_capacity(steps);

    // Generate random motor outputs
    let rotations = generate_rotations(rng, steps);
    let accelerations = generate_accelerations(rng, steps, acceleration, vary_speed);

    // Step from initial physical state
    let mut state = PhysicalState {
        heading: 0.0,
        velocity: Vector2::zeros(),
    };

    for i in 0..steps {
        state = state.next(rotations[i], accelerations[i], DEFAULT_DRAG);
        states.push(state.clone());
    }

    states
}

pub fn generate_rotations(rng: &mut impl Rng, steps: usize) -> DVector<f32> {
    // Sample turns from a Normal(0, 0.1) distribution, which is
    // very close to the von Mises(0, 100) used in Stone et al. (2017).
    let turn_distribution = Normal::new(0.0, 0.1).unwrap();
    let turns = DVector::from_iterator(steps, turn_distribution.sample_iter(rng).take(steps));

    let filter = SVector::<f32, 2>::repeat(1.0);
    let turns = turns.convolve_same(filter / (filter.len() as f32));

    turns
}

pub fn generate_accelerations(
    rng: &mut impl Rng,
    steps: usize,
    acceleration: f32,
    vary_speed: bool,
) -> DVector<f32> {
    if vary_speed {
        let mut accelerations = DVector::zeros(steps);

        // Choose a new acceleration every INTERVAL steps and lerp
        const INTERVAL: usize = 50;
        let mut prev_key = 0.0;
        let mut next_key = rng.gen::<f32>() * acceleration;

        for i in 0..steps {
            let offset = i.rem_euclid(INTERVAL);
            if offset == 0 {
                prev_key = next_key;
                next_key = rng.gen::<f32>() * acceleration;
            }

            let factor = (offset as f32) / (INTERVAL as f32);
            accelerations[i] = prev_key + (next_key - prev_key) * factor;
        }

        accelerations
    }
    else {
        DVector::repeat(steps, acceleration)
    }
}

pub fn reconstruct_path(states: &[PhysicalState]) -> Vec<Vector2<f32>> {
    let mut position = Vector2::<f32>::zeros();
    let mut path = Vec::with_capacity(states.len());

    path.push(position);
    for state in states {
        position += state.velocity;
        path.push(position);
    }

    path
}
