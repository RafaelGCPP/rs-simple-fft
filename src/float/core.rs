// src/float/core.rs

use num_complex::Complex32;
use core::f32::consts::PI;

#[cfg(feature = "std")]
use std::f32;
#[cfg(not(feature = "std"))]
use libm::Libm;

// --- Public Auxiliary Functions for the Module (pub(crate)) ---

/// Computes the rotation factors (Twiddle Factors) for an FFT of size N.
pub(crate) fn precompute_twiddles(twiddles: &mut [Complex32], n: usize) {
    // Note que geramos apenas N/2 fatores, pois é o necessário para Radix-2
    for j in 0..(n / 2) {
        let angle = -2.0 * PI * (j as f32) / (n as f32);
        let (sin, cos) = sin_cos(angle);
        twiddles[j] = Complex32::new(cos, sin);
    }
}

/// Fills the bit-reversal table.
pub(crate) fn precompute_bitrev(bitrev: &mut [usize], n: usize) {
    bitrev[0] = 0;
    let mut j = 0;
    for i in 1..n {
        let mut k = n >> 1;
        while j >= k {
            j -= k;
            k >>= 1;
        }
        j += k;
        bitrev[i] = j;
    }
}

/// Agnostic helper function for sin/cos
fn sin_cos(angle: f32) -> (f32, f32) {
    #[cfg(feature = "std")]
    return (angle.sin(), angle.cos());
    
    #[cfg(not(feature = "std"))]
    return (libm::sinf(angle), libm::cosf(angle));
}

/// This function is the direct equivalent of `radix_2_dit_fft` from your C code.
/// It is not pub(crate) for the end user, only for internal use by the real and complex modules.
pub(crate) fn radix_2_dit_fft_core<const INVERSE: bool>(
    buffer: &mut [Complex32], 
    twiddles: &[Complex32], 
    bitrev: &[usize],
    twiddle_stride: usize
) {
    let n = buffer.len();

    // 1. Bit-reverse
    for i in 1..(n - 1) {
        let j = bitrev[i];
        if i < j {
            buffer.swap(i, j);
        }
    }

    // 2. Butterfly
    let mut stride = 1;
    let mut tw_index = n >> 1;

    while stride < n {
        let jmax = n - stride;
        
        for j in (0..jmax).step_by(stride << 1) {
            for i in 0..stride {
                let mut w = twiddles[i * tw_index * twiddle_stride];

                // The compiler will completely remove this IF because INVERSE is a compile-time constant
                if INVERSE {
                    w = w.conj();
                }

                let index = j + i;
                let a = buffer[index];
                let b = buffer[index + stride];
                let t = b * w;

                let mut v1 = a + t;
                let mut v2 = a - t;

                // Stage normalization to avoid saturation (fixed-point behavior)
                // The compiler will optimize this for INVERSE = true/false
                if INVERSE {
                    v1 = v1.scale(0.5);
                    v2 = v2.scale(0.5);
                }

                buffer[index] = v1;
                buffer[index + stride] = v2;
            }
        }
        stride <<= 1;
        tw_index >>= 1;
    }
}

#[cfg(test)]
#[path = "core_tests.rs"]
mod tests;