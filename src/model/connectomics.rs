use nalgebra::matrix;

use super::constants::{
    self, N_AMP, N_CL1, N_CPU1A, N_CPU1B, N_CPU4, N_PONTINE, N_TB1, N_TN1, N_TN2,
};
use super::network::WeightMatrix;

pub const W_CL1_TB1: WeightMatrix<N_TB1, N_CL1> = matrix![
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
];

pub const W_TN1_CPU4: WeightMatrix<N_CPU4, N_TN1> = matrix![
    1.0, 0.0;
    1.0, 0.0;
    1.0, 0.0;
    1.0, 0.0;
    1.0, 0.0;
    1.0, 0.0;
    1.0, 0.0;
    1.0, 0.0;
    0.0, 1.0;
    0.0, 1.0;
    0.0, 1.0;
    0.0, 1.0;
    0.0, 1.0;
    0.0, 1.0;
    0.0, 1.0;
    0.0, 1.0;
];

pub const W_TN2_CPU4: WeightMatrix<N_CPU4, N_TN2> = matrix![
    1.0, 0.0;
    1.0, 0.0;
    1.0, 0.0;
    1.0, 0.0;
    1.0, 0.0;
    1.0, 0.0;
    1.0, 0.0;
    1.0, 0.0;
    0.0, 1.0;
    0.0, 1.0;
    0.0, 1.0;
    0.0, 1.0;
    0.0, 1.0;
    0.0, 1.0;
    0.0, 1.0;
    0.0, 1.0;
];

pub const W_TB1_CPU1A: WeightMatrix<N_CPU1A, N_TB1> = matrix![
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
];

pub const W_TB1_CPU1B: WeightMatrix<N_CPU1B, N_TB1> = matrix![
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
];

pub const W_TB1_CPU4: WeightMatrix<N_CPU4, N_TB1> = matrix![
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
];

pub const W_CPU4_AMP: WeightMatrix<N_AMP, N_CPU4> = matrix![
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
];

pub const W_CPU4_PONTINE: WeightMatrix<N_PONTINE, N_CPU4> = matrix![
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
];

pub const W_PONTINE_AMP: WeightMatrix<N_AMP, N_PONTINE> = matrix![
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
];

pub const W_AMP_CPU1A: WeightMatrix<N_CPU1A, N_AMP> = matrix![
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
];

pub const W_AMP_CPU1B: WeightMatrix<N_CPU1B, N_AMP> = matrix![
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
];

pub const W_CPU1A_MOTOR: WeightMatrix<2, N_CPU1A> = matrix![
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
];

pub const W_CPU1B_MOTOR: WeightMatrix<2, N_CPU1B> = matrix![
    0.0, 1.0;
    1.0, 0.0;
];

pub fn generate_tb_tb_weights() -> WeightMatrix<N_TB1, N_TB1> {
    use ndarray::prelude::*;

    let mut w: Array2<f32> = Array::zeros((constants::N_TB1, constants::N_TB1));
    let x = Array::linspace(0.0, 2.0 * std::f32::consts::PI, constants::N_TB1 + 1);
    let x = x.slice(s![..-1]);
    let sinusoid = -(x.mapv(f32::cos) - 1.0) / 2.0;

    for i in 0..constants::N_TB1 {
        let rolled = ndarray::concatenate(
            Axis(0),
            &[
                sinusoid.slice(s![N_TB1 - i..]),
                sinusoid.slice(s![..N_TB1 - i]),
            ],
        )
        .unwrap();
        //let rolled =
        //    ndarray::concatenate(Axis(0), &[sinusoid.slice(s![i..]), sinusoid.slice(s![..i])])
        //        .unwrap();
        w.slice_mut(s![i, ..]).assign(&rolled);
    }

    WeightMatrix::<N_TB1, N_TB1>::from_vec(w.into_raw_vec())
}
