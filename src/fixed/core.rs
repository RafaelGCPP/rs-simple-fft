// src/fixed/core.rs

use super::types::{ Fixed, ComplexFixed };
use core::f64::consts::PI;

/// Fractional bits for twiddle factors (high precision).
/// Using Q31 format for maximum precision in twiddle factors.
pub const TWIDDLE_FRAC: u32 = 31;

/// Computes the rotation factors (Twiddle Factors) for an FFT of size N.
/// Twiddle factors are stored in Q31 format for maximum precision.
pub(crate) fn precompute_twiddles(twiddles: &mut [ComplexFixed<TWIDDLE_FRAC>], n: usize) {
    // We generate only N/2 factors, as required for Radix-2
    for j in 0..(n / 2) {
        let angle = -2.0 * PI * (j as f64) / (n as f64);
        let (sin, cos) = (angle.sin(), angle.cos());
        twiddles[j] = ComplexFixed::new(
            Fixed::<TWIDDLE_FRAC>::from_f64(cos),
            Fixed::<TWIDDLE_FRAC>::from_f64(sin),
        );
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

/// Radix-2 Decimation-in-Time FFT core for fixed-point complex numbers.
/// 
/// This is the fixed-point equivalent of `radix_2_dit_fft_core` from the float module.
/// 
/// # Type Parameters
/// - `FRAC`: Fractional bits for the input/output buffer
/// - `INVERSE`: If true, performs inverse FFT with conjugate twiddles and scaling
/// 
/// # Arguments
/// - `buffer`: Input/output buffer of complex fixed-point numbers
/// - `twiddles`: Precomputed twiddle factors in Q31 format
/// - `bitrev`: Precomputed bit-reversal indices
/// - `twiddle_stride`: Stride for accessing twiddle factors (for smaller FFT sizes)
pub(crate) fn radix_2_dit_fft_core<const FRAC: u32, const INVERSE: bool>(
    buffer: &mut [ComplexFixed<FRAC>], 
    twiddles: &[ComplexFixed<TWIDDLE_FRAC>], 
    bitrev: &[usize],
    twiddle_stride: usize
) {
    let n = buffer.len();

    // 1. Bit-reverse permutation
    for i in 1..(n - 1) {
        let j = bitrev[i];
        if i < j {
            buffer.swap(i, j);
        }
    }

    // 2. Butterfly stages
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
                
                // Butterfly: t = b * w
                let t = b * w;

                let mut v1 = a + t;
                let mut v2 = a - t;

                // Stage normalization to avoid overflow (essential for fixed-point)
                // In inverse FFT, we scale by 0.5 at each stage instead of 1/N at the end
                if INVERSE {
                    v1 = v1.scale_half();
                    v2 = v2.scale_half();
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