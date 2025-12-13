/* src/symplectic.rs */
//! Symplectic T*E8 Layer - Hamiltonian Dynamics
//!
//! Implements symplectic phase space over E8 for Hamiltonian dynamics.
//! Phase space has 16 dimensions: 8 positions (q) + 8 momenta (p).
//!
//! Key operations:
//! - Hamiltonian computation (H = T + V)
//! - Symplectic evolution (Velocity Verlet integrator)
//! - Möbius kicks and drifts
//! - Poisson brackets
//! - Phase space conservation
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

/// Symplectic phase space layer for Hamiltonian dynamics
pub struct SymplecticLayer {
    /// Symplectic 2-form ω (16×16 for 8 positions + 8 momenta)
    /// Standard form: ω[i][i+8] = 1, ω[i+8][i] = -1
    pub omega: [[f32; 16]; 16],
}

impl SymplecticLayer {
    /// Create new symplectic layer with standard symplectic form
    pub fn new() -> Self {
        let mut omega = [[0.0f32; 16]; 16];

        // Standard symplectic form: ω = Σ dp_i ∧ dq_i
        for i in 0..8 {
            omega[i][i + 8] = 1.0; // dq_i ∧ dp_i = 1
            omega[i + 8][i] = -1.0; // dp_i ∧ dq_i = -1
        }

        Self { omega }
    }

    /// Compute Hamiltonian: H = kinetic + potential
    /// H = ½ Σ p_i² + V(q)
    /// where V(q) is a simple harmonic potential
    pub fn hamiltonian(&self, _q: &[f32; 8], p: &[f32; 8]) -> f32 {
        // Kinetic energy: T = ½ Σ p_i² (use SIMD norm2)
        #[cfg(feature = "simd")]
        let kinetic = {
            use super::gf8::gf8_norm2_simd;
            gf8_norm2_simd(p) * 0.5
        };
        #[cfg(not(feature = "simd"))]
        let kinetic: f32 = p.iter().map(|&pi| pi * pi).sum::<f32>() * 0.5;

        // Potential energy: V = ½ k Σ q_i² (harmonic oscillator)
        let k = 0.1;
        #[cfg(feature = "simd")]
        let potential = {
            use super::gf8::gf8_norm2_simd;
            gf8_norm2_simd(_q) * 0.5 * k
        };
        #[cfg(not(feature = "simd"))]
        let potential: f32 = _q.iter().map(|&qi| qi * qi).sum::<f32>() * 0.5 * k;

        kinetic + potential
    }

    /// Compute force F = -∂V/∂q for harmonic potential
    fn compute_force(&self, q: &[f32; 8]) -> [f32; 8] {
        let k = 0.1;
        q.map(|qi| -k * qi)
    }

    /// Evolve system using Velocity Verlet integrator (symplectic)
    /// Preserves phase space volume and energy
    pub fn evolve(&self, q: &mut [f32; 8], p: &mut [f32; 8], dt: f32) {
        #[cfg(feature = "simd")]
        {
            use super::gf8::gf8_add_inplace_slice_simd;
            // Half-step momentum update: p(t+dt/2) = p(t) + (dt/2) * F(t)
            let force = self.compute_force(q);
            let scaled_force = force.map(|fi| 0.5 * dt * fi);
            gf8_add_inplace_slice_simd(p, &scaled_force);

            // Full-step position update: q(t+dt) = q(t) + dt * p(t+dt/2)
            let scaled_p = p.map(|pi| dt * pi);
            gf8_add_inplace_slice_simd(q, &scaled_p);

            // Recompute force at new position
            let force_new = self.compute_force(q);

            // Complete momentum update: p(t+dt) = p(t+dt/2) + (dt/2) * F(t+dt)
            let scaled_force_new = force_new.map(|fi| 0.5 * dt * fi);
            gf8_add_inplace_slice_simd(p, &scaled_force_new);
        }
        #[cfg(not(feature = "simd"))]
        {
            // Half-step momentum update: p(t+dt/2) = p(t) + (dt/2) * F(t)
            let force = self.compute_force(q);
            for i in 0..8 {
                p[i] += 0.5 * dt * force[i];
            }

            // Full-step position update: q(t+dt) = q(t) + dt * p(t+dt/2)
            for i in 0..8 {
                q[i] += dt * p[i];
            }

            // Recompute force at new position
            let force_new = self.compute_force(q);

            // Complete momentum update: p(t+dt) = p(t+dt/2) + (dt/2) * F(t+dt)
            for i in 0..8 {
                p[i] += 0.5 * dt * force_new[i];
            }
        }
    }

    /// Apply symplectic kick (instantaneous momentum change)
    pub fn kick(&self, p: &mut [f32; 8], force: &[f32; 8], dt: f32) {
        #[cfg(feature = "simd")]
        {
            use super::gf8::gf8_add_inplace_slice_simd;
            let scaled_force = force.map(|fi| fi * dt);
            gf8_add_inplace_slice_simd(p, &scaled_force);
        }
        #[cfg(not(feature = "simd"))]
        {
            for i in 0..8 {
                p[i] += force[i] * dt;
            }
        }
    }

    /// Apply symplectic drift (position update from momentum)
    pub fn drift(&self, q: &mut [f32; 8], p: &[f32; 8], dt: f32) {
        #[cfg(feature = "simd")]
        {
            use super::gf8::gf8_add_inplace_slice_simd;
            let scaled_p = p.map(|pi| pi * dt);
            gf8_add_inplace_slice_simd(q, &scaled_p);
        }
        #[cfg(not(feature = "simd"))]
        {
            for i in 0..8 {
                q[i] += p[i] * dt;
            }
        }
    }

    /// Convert position and momentum to phase space coordinates
    pub fn to_phase_space(&self, q: &[f32; 8], p: &[f32; 8]) -> [f32; 16] {
        let mut phase = [0.0f32; 16];
        phase[..8].copy_from_slice(q);
        phase[8..].copy_from_slice(p);
        phase
    }

    /// Extract position and momentum from phase space coordinates
    pub fn from_phase_space(&self, phase: &[f32; 16]) -> ([f32; 8], [f32; 8]) {
        let mut q = [0.0f32; 8];
        let mut p = [0.0f32; 8];
        q.copy_from_slice(&phase[..8]);
        p.copy_from_slice(&phase[8..]);
        (q, p)
    }

    /// Compute Poisson bracket {f, g} for coordinate functions
    /// {q_i, p_i} = 1, {p_i, q_i} = -1, others = 0
    pub fn poisson_bracket(&self, i: usize, j: usize) -> f32 {
        if (0..8).contains(&i) && (8..16).contains(&j) && i + 8 == j {
            1.0 // {q_i, p_i} = 1
        } else if (8..16).contains(&i) && (0..8).contains(&j) && i - 8 == j {
            -1.0 // {p_i, q_i} = -1
        } else {
            0.0 // All other brackets vanish
        }
    }

    /// Verify energy conservation (approximately)
    pub fn verify_energy_conservation(
        &self,
        before: &([f32; 8], [f32; 8]),
        after: &([f32; 8], [f32; 8]),
    ) -> bool {
        let h_before = self.hamiltonian(&before.0, &before.1);
        let h_after = self.hamiltonian(&after.0, &after.1);

        let relative_error = (h_before - h_after).abs() / h_before.abs().max(1e-8);
        relative_error < 0.1 // Allow 10% error
    }
}

impl Default for SymplecticLayer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symplectic_form() {
        let sym = SymplecticLayer::new();

        // Check antisymmetry
        for i in 0..16 {
            for j in 0..16 {
                assert_eq!(sym.omega[i][j], -sym.omega[j][i]);
            }
        }

        // Check structure
        for i in 0..8 {
            assert_eq!(sym.omega[i][i + 8], 1.0);
            assert_eq!(sym.omega[i + 8][i], -1.0);
        }
    }

    #[test]
    fn test_hamiltonian() {
        let sym = SymplecticLayer::new();

        let q = [0.0; 8];
        let p = [1.0; 8];

        let h = sym.hamiltonian(&q, &p);
        // H = ½ * 8 * 1.0 = 4.0 (kinetic only)
        assert!((h - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_phase_space_conversion() {
        let sym = SymplecticLayer::new();

        let q = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        let phase = sym.to_phase_space(&q, &p);
        let (q_back, p_back) = sym.from_phase_space(&phase);

        for i in 0..8 {
            assert!((q[i] - q_back[i]).abs() < 1e-6);
            assert!((p[i] - p_back[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_poisson_bracket() {
        let sym = SymplecticLayer::new();

        // {q_0, p_0} = 1
        assert_eq!(sym.poisson_bracket(0, 8), 1.0);

        // {p_0, q_0} = -1
        assert_eq!(sym.poisson_bracket(8, 0), -1.0);

        // {q_0, q_1} = 0
        assert_eq!(sym.poisson_bracket(0, 1), 0.0);

        // {p_0, p_1} = 0
        assert_eq!(sym.poisson_bracket(8, 9), 0.0);
    }

    #[test]
    fn test_symplectic_evolution() {
        let sym = SymplecticLayer::new();

        let mut q = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut p = [0.0; 8];

        let before = (q, p);

        // Evolve for small time step
        sym.evolve(&mut q, &mut p, 0.1);

        let after = (q, p);

        // Energy should be approximately conserved
        assert!(sym.verify_energy_conservation(&before, &after));
    }

    #[test]
    fn test_kick_and_drift() {
        let sym = SymplecticLayer::new();

        let mut q = [0.0; 8];
        let mut p = [1.0; 8];

        // Apply drift: q = q + p * dt
        sym.drift(&mut q, &p, 0.1);
        for i in 0..8 {
            assert!((q[i] - 0.1).abs() < 1e-6);
        }

        // Apply kick: p = p + F * dt
        let force = [1.0; 8];
        sym.kick(&mut p, &force, 0.1);
        for i in 0..8 {
            assert!((p[i] - 1.1).abs() < 1e-6);
        }
    }
}
