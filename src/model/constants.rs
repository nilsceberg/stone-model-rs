pub const DRAG: f64 = 0.15;
pub const MEAN_ACCELERATION: f64 = 0.15;

pub const N_COLUMNS: usize = 8;

// Anatomy
pub const N_TL2: usize = 16;
pub const N_CL1: usize = 16;
pub const N_TB1: usize = 8;
pub const N_TN1: usize = 2;
pub const N_TN2: usize = 2;
pub const N_CPU4: usize = 16;
pub const N_PONTINE: usize = 16;
pub const N_AMP: usize = 16;
pub const N_CPU1A: usize = 14;
pub const N_CPU1B: usize = 2;
pub const N_CPU1: usize = N_CPU1A + N_CPU1B;

// Tuned parameters:
pub const TL2_SLOPE_TUNED: f32 = 6.8;
pub const TL2_BIAS_TUNED: f32 = 3.0;

pub const CL1_SLOPE_TUNED: f32 = 3.0;
pub const CL1_BIAS_TUNED: f32 = -0.5;

pub const TB1_SLOPE_TUNED: f32 = 5.0;
pub const TB1_BIAS_TUNED: f32 = 0.0;

pub const CPU4_SLOPE_TUNED: f32 = 5.0;
pub const CPU4_BIAS_TUNED: f32 = 2.5;

pub const AMP_SLOPE_TUNED: f32 = 100.0;
pub const AMP_BIAS_TUNED: f32 = 0.0;

pub const CPU1_SLOPE_TUNED: f32 = 7.5;
pub const CPU1_BIAS_TUNED: f32 = -1.0;

pub const PONTINE_SLOPE_TUNED: f32 = 5.0;
pub const PONTINE_BIAS_TUNED: f32 = 2.5;

pub const PONTINE_SLOPE_TUNED_LINEAR: f32 = 1.0;
pub const PONTINE_BIAS_TUNED_LINEAR: f32 = 0.0;

pub const MOTOR_SLOPE_TUNED: f32 = 1.0;
pub const MOTOR_BIAS_TUNED: f32 = 3.0;

pub const CPU4_MEM_GAIN: f32 = 0.005 * 0.5;
pub const CPU4_MEM_FADE: f32 = 0.125;
