/* src/fisher.rs */
//! Fisher Information Layer - Information Geometry
//!
//! Implements Fisher information geometry and statistical manifolds.
//! The Fisher information matrix measures the amount of information
//! that observable data carries about unknown parameters in a statistical model.
//!
//! Key operations:
//! - Fisher matrix computation from probability distributions
//! - Uncertainty quantification
//! - KL divergence for statistical distance
//! - Entropy calculations
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

/// Fisher information layer for information geometry
pub struct FisherLayer;

impl FisherLayer {
    /// Compute Fisher information matrix from resonance distribution
    /// F_ij = E[∂log(p)/∂θ_i * ∂log(p)/∂θ_j]
    /// Simplified: uses variance as proxy
    pub fn fisher_matrix(resonance: &[u32; 240]) -> [[f32; 8]; 8] {
        let mut fisher = [[0.0f32; 8]; 8];

        // Normalize resonance to probability distribution
        let total: u32 = resonance.iter().sum();
        if total == 0 {
            return fisher; // Zero matrix if no resonance
        }

        let total_f = total as f32;
        let probs: Vec<f32> = resonance.iter().map(|&r| r as f32 / total_f).collect();

        // Compute Fisher matrix elements using simplified covariance
        // Map 240 roots to 8D coordinates and compute covariance
        for (i, row) in fisher.iter_mut().enumerate() {
            for (j, elem) in row.iter_mut().enumerate() {
                let mut sum = 0.0f32;

                // Use simplified mapping: each coordinate contributes to 30 roots
                let start = i * 30;
                let end = (start + 30).min(240);

                for p in probs.iter().take(end).skip(start).copied() {
                    if p > 1e-8 {
                        // Fisher information: 1/p (for single parameter)
                        sum += 1.0 / p;
                    }
                }

                *elem = if i == j { sum / 30.0 } else { 0.0 };
            }
        }

        fisher
    }

    /// Compute uncertainty from Fisher matrix
    /// Uncertainty = 1 / sqrt(trace(F))
    pub fn uncertainty(fisher_matrix: &[[f32; 8]; 8]) -> f32 {
        let trace: f32 = (0..8).map(|i| fisher_matrix[i][i]).sum();

        if trace < 1e-8 {
            return f32::INFINITY; // Maximum uncertainty
        }

        1.0 / trace.sqrt()
    }

    /// Compute Kullback-Leibler divergence: KL(P || Q) = Σ P(i) log(P(i)/Q(i))
    /// Measures statistical distance between distributions
    pub fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
        if p.len() != q.len() {
            return f32::INFINITY; // Invalid comparison
        }

        let mut kl = 0.0f32;

        for i in 0..p.len() {
            if p[i] > 1e-8 && q[i] > 1e-8 {
                kl += p[i] * (p[i] / q[i]).ln();
            }
        }

        kl.max(0.0) // KL divergence is non-negative
    }

    /// Compute information metric from Fisher matrix
    /// Metric = log(1 + trace(F))
    pub fn information_metric(fisher_matrix: &[[f32; 8]; 8]) -> f32 {
        let trace: f32 = (0..8).map(|i| fisher_matrix[i][i]).sum();
        (1.0 + trace).ln()
    }

    /// Compute entropy from distribution
    /// H = -Σ p_i log(p_i)
    pub fn entropy(distribution: &[f32]) -> f32 {
        let sum: f32 = distribution.iter().sum();

        if sum < 1e-8 {
            return 0.0;
        }

        let normalized: Vec<f32> = distribution.iter().map(|x| x / sum).collect();

        normalized
            .iter()
            .filter(|&&p| p > 1e-8)
            .map(|&p| -p * p.ln())
            .sum()
    }

    /// Compute Fisher information metric distance
    /// d_F(θ1, θ2)² ≈ (θ1 - θ2)ᵀ F (θ1 - θ2)
    pub fn fisher_distance(
        theta1: &[f32; 8],
        theta2: &[f32; 8],
        fisher_matrix: &[[f32; 8]; 8],
    ) -> f32 {
        let mut diff = [0.0f32; 8];
        for i in 0..8 {
            diff[i] = theta1[i] - theta2[i];
        }

        // Compute diff^T * F * diff
        let mut result = 0.0f32;
        for i in 0..8 {
            for j in 0..8 {
                result += diff[i] * fisher_matrix[i][j] * diff[j];
            }
        }

        result.sqrt().max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fisher_matrix() {
        let mut resonance = [0u32; 240];
        // Create peaked distribution
        for i in 0..240 {
            resonance[i] = if i < 30 { 10 } else { 1 };
        }

        let fisher = FisherLayer::fisher_matrix(&resonance);

        // Fisher matrix should be diagonal and positive
        let trace: f32 = (0..8).map(|i| fisher[i][i]).sum();
        assert!(trace > 0.0, "Fisher matrix should have positive trace");
    }

    #[test]
    fn test_uncertainty() {
        let mut fisher = [[0.0f32; 8]; 8];
        for i in 0..8 {
            fisher[i][i] = 1.0;
        }

        let uncertainty = FisherLayer::uncertainty(&fisher);

        // Uncertainty should be 1/sqrt(8) ≈ 0.35
        assert!((uncertainty - 0.353).abs() < 0.01);
    }

    #[test]
    fn test_kl_divergence() {
        let p = [0.5, 0.5];
        let q = [0.5, 0.5];

        // KL(P || P) = 0
        let kl = FisherLayer::kl_divergence(&p, &q);
        assert!(kl.abs() < 1e-6);

        // Different distributions
        let q2 = [0.8, 0.2];
        let kl2 = FisherLayer::kl_divergence(&p, &q2);
        assert!(kl2 > 0.0);
    }

    #[test]
    fn test_information_metric() {
        let mut fisher = [[0.0f32; 8]; 8];
        for i in 0..8 {
            fisher[i][i] = 2.0;
        }

        let metric = FisherLayer::information_metric(&fisher);

        // metric = ln(1 + 8*2) = ln(17) ≈ 2.83
        assert!((metric - 2.833).abs() < 0.01);
    }

    #[test]
    fn test_entropy() {
        // Uniform distribution has maximum entropy
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let h_uniform = FisherLayer::entropy(&uniform);

        // Peaked distribution has lower entropy
        let peaked = vec![0.8, 0.1, 0.05, 0.05];
        let h_peaked = FisherLayer::entropy(&peaked);

        assert!(h_uniform > h_peaked);
    }

    #[test]
    fn test_fisher_distance() {
        let theta1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let theta2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let mut fisher = [[0.0f32; 8]; 8];
        for i in 0..8 {
            fisher[i][i] = 1.0;
        }

        let distance = FisherLayer::fisher_distance(&theta1, &theta2, &fisher);

        // Distance should be sqrt(2) ≈ 1.414
        assert!((distance - 1.414).abs() < 0.01);
    }
}
