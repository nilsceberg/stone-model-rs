use crate::{movement::reconstruct_path, FlightData, Setup};

#[derive(Default, Debug)]
pub struct FlightStats {
    pub min_distance_to_home: f32,
}

impl FlightStats {
    pub fn analyze(setup: &Setup, result: &FlightData) -> FlightStats {
        let mut stats = FlightStats::default();
        stats.min_distance_to_home = f32::INFINITY;

        let path = reconstruct_path(&result.physical_states);
        for position in &path[(setup.outbound_steps + 1)..] {
            let distance = position.magnitude();
            if distance < stats.min_distance_to_home {
                stats.min_distance_to_home = distance;
            }
        }

        stats
    }
}
