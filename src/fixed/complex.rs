use super::core::{TWIDDLE_FRAC, precompute_bitrev, precompute_twiddles, radix_2_dit_fft_core};
use super::types::ComplexFixed;
use crate::common::{CplxFft, FftError, FftProcess};

impl<'a> CplxFft<'a, ComplexFixed<TWIDDLE_FRAC>> {
    /// Initializes the tables.
    pub fn new(
        twiddles: &'a mut [ComplexFixed<TWIDDLE_FRAC>],
        bitrev: &'a mut [usize],
        n: usize,
    ) -> Result<Self, FftError> {
        if !n.is_power_of_two() {
            return Err(FftError::NotPowerOfTwo);
        }
        if twiddles.len() < n / 2 {
            return Err(FftError::BufferTooSmall);
        }
        if bitrev.len() < n {
            return Err(FftError::BufferTooSmall);
        }

        let mut fft = Self {
            twiddles,
            bitrev,
            n,
        };
        fft.precompute();
        Ok(fft)
    }

    /// Precomputes Twiddle Factors and Bit Reverse Table
    fn precompute(&mut self) {
        precompute_bitrev(self.bitrev, self.n);
        precompute_twiddles(self.twiddles, self.n);
    }

    /// Executes the FFT in-place for a specific fixed-point format.
    pub fn process<const FRAC: u32>(
        &self,
        buffer: &mut [ComplexFixed<FRAC>],
        inverse: bool,
    ) -> Result<(), FftError> {
        if buffer.len() != self.n {
            return Err(FftError::SizeMismatch);
        }

        if inverse {
            radix_2_dit_fft_core::<FRAC, true>(buffer, self.twiddles, self.bitrev, 1);
        } else {
            radix_2_dit_fft_core::<FRAC, false>(buffer, self.twiddles, self.bitrev, 1);
        }

        Ok(())
    }
}

// Implement FftProcess for ANY fixed-point precision.
// This allows the same CplxFft instance to be reused for buffers with different Q-formats.
impl<'a, const FRAC: u32> FftProcess<ComplexFixed<FRAC>>
    for CplxFft<'a, ComplexFixed<TWIDDLE_FRAC>>
{
    fn process(&self, buffer: &mut [ComplexFixed<FRAC>], inverse: bool) -> Result<(), FftError> {
        self.process(buffer, inverse)
    }
}

#[cfg(test)]
#[path = "complex_tests.rs"]
mod tests;
