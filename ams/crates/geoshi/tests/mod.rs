use geoshi::diffuser::{Diffuser, DiffusionConfig, DiffusionMethod, AnisotropicDiffuser, GraphDiffuser};
use ndarray::Array2;
use xuid::e8_lattice::quantize_to_e8;

#[test]
fn e8_quantization_is_consistent() {
    let point: [f64; 8] = [0.1, -1.2, 3.3, -2.1, 0.4, 1.0, -0.5, 2.3];
    let q = quantize_to_e8(&point);
    // quantized result should consist of finite numbers and reasonable norms
    for &x in &q {
        assert!(x.is_finite());
        assert!(x.abs() <= 8.0);
    }
    // Requantize the same point and ensure determinism
    let q2 = quantize_to_e8(&point);
    assert_eq!(q, q2);
}

#[test]
fn diffuser_heat_smooths_values() {
    let config = DiffusionConfig { method: DiffusionMethod::Heat, iterations: 200, time_step: 0.1, ..Default::default() };
    let diffuser = Diffuser::new(config);
    let mut grid = Array2::from_shape_vec((3, 3), vec![0.0_f64; 9]).unwrap();
    grid[[1, 1]] = 10.0; // Center peak

    let out = diffuser.diffuse_grid(grid.view()).unwrap();
    // After diffusion center value should decrease while neighbors increase
    assert!(out[[1,1]] < 10.0);
    assert!(out[[1,0]] > 0.0 || out[[0,1]] > 0.0);
}

#[test]
fn anisotropic_preserves_edges() {
    let config = DiffusionConfig { method: DiffusionMethod::Heat, iterations: 5, time_step: 0.1, ..Default::default() };
    let diff = AnisotropicDiffuser::new(config);
    let grid = Array2::from_shape_vec((3, 3), vec![0.0_f64, 1.0, 0.0,
                                                      1.0, 10.0, 1.0,
                                                      0.0, 1.0, 0.0]).unwrap();
    let out = diff.diffuse(grid.view()).unwrap();
    // Edge values should remain higher than distant corners
    assert!(out[[0,1]] >= out[[0,0]]);
}

#[test]
fn graph_diffuser_balances_values() {
    // 3-node line graph: 0 <-> 1 <-> 2
    let adjacency = vec![vec![1], vec![0,2], vec![1]];
    let weights = vec![vec![0.0, 1.0, 0.0], vec![1.0, 0.0, 1.0], vec![0.0, 1.0, 0.0]];
    let initial = vec![10.0, 0.0, 0.0];
    let diffuser = GraphDiffuser::new(adjacency, weights);
    let out = diffuser.diffuse(&initial, 5).unwrap();
    // Values should have propagated from node 0 to node 1 and 2
    assert!(out[1] > 0.0);
    assert!(out[2] > 0.0);
    // Mass/energy should not increase (weights may lead to small numerical difference)
    let sum_in: f64 = initial.iter().sum();
    let sum_out: f64 = out.iter().sum();
    assert!(sum_out >= 0.0);
    assert!(sum_out <= sum_in + 1e-6);
}
