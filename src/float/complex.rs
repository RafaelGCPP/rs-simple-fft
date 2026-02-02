use crate::common::{FftError, FftProcess}; // Adicione FftProcess aqui
use num_complex::Complex32; // Complex<f32>
use super::core::{radix_2_dit_fft_core, precompute_twiddles, precompute_bitrev};

// In no_std, we need to import math functions from somewhere.
// If the "std" feature is enabled, we use native f32::sin/cos.
// Otherwise, we use libm via the Float trait from num_traits.
// #[cfg(feature = "std")]
// use std::f32;
#[cfg(not(feature = "std"))]
use libm::Libm;

/// Structure that holds the precomputed tables (Twiddle factors and Bit Reverse).
/// This replaces passing 'twiddle' and 'bitrev' around in every function.
pub struct CplxFft<'a> {
    twiddles: &'a mut [Complex32],
    bitrev: &'a mut [usize],
    n: usize,
}

impl<'a> CplxFft<'a> {
    /// Initializes the tables (Port from `fft_init.c`)
    pub fn new(
        twiddles: &'a mut [Complex32], 
        bitrev: &'a mut [usize], 
        n: usize
    ) -> Result<Self, FftError> {
        if !n.is_power_of_two() {
            return Err(FftError::NotPowerOfTwo);
        }
        if twiddles.len() < n / 2 || bitrev.len() < n {
            return Err(FftError::BufferTooSmall);
        }

        let mut fft = Self { twiddles, bitrev, n };
        fft.precompute();
        Ok(fft)
    }

    /// Precomputes Twiddle Factors and Bit Reverse Table
    fn precompute(&mut self) {
        precompute_bitrev(self.bitrev, self.n);
        precompute_twiddles(self.twiddles, self.n);
    }

    /// Executes the FFT in-place (Port from `radix_2_dit_fft` in `fft_core.c`)
    pub fn process(&self, buffer: &mut [Complex32], inverse: bool) -> Result<(), FftError> {
        if buffer.len() != self.n {
            return Err(FftError::SizeMismatch);
        }

        if inverse {
            radix_2_dit_fft_core::<true>(buffer, self.twiddles, self.bitrev, 1);
        } else {
            radix_2_dit_fft_core::<false>(buffer, self.twiddles, self.bitrev, 1);
        }

        Ok(())
    }
}

// Implementação da trait FftProcess para CplxFft
impl<'a> FftProcess<Complex32> for CplxFft<'a> {
    fn process(&self, buffer: &mut [Complex32], inverse: bool) -> Result<(), FftError> {
        self.process(buffer, inverse)
    }
}

#[cfg(test)]
#[path = "complex_tests.rs"] 
mod tests;