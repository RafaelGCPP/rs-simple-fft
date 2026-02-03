use super::core::{precompute_bitrev, precompute_twiddles, radix_2_dit_fft_core, TWIDDLE_FRAC};
use super::types::{Fixed, ComplexFixed};
use crate::common::{FftError, FftProcess, RealFft};
use core::slice;


impl<'a> RealFft<'a, ComplexFixed<TWIDDLE_FRAC>> {
    /// Initializes the Real FFT.
    /// Note that 'n' here is the number of REAL samples.
    pub fn new(
        twiddles: &'a mut [ComplexFixed<TWIDDLE_FRAC>],
        bitrev: &'a mut [usize],
        n: usize,
    ) -> Result<Self, FftError> {
        if !n.is_power_of_two() {
            return Err(FftError::NotPowerOfTwo);
        }
        
        // For an N-point RFFT, we need auxiliary tables
        // compatible with the underlying N/2-point Complex FFT,
        // BUT the post-processing needs finer twiddles (size N).
        // The original C code actually generates twiddles for N/2 * 2 = N.
        if twiddles.len() < n / 2 || bitrev.len() < n / 2 {
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

    fn precompute(&mut self) {
        // 1. Bitrev is generated for N/2 (size of the internal FFT)
        precompute_bitrev(self.bitrev, self.n / 2);

        // 2. Twiddles are generated for N (full circle, though size N/2)
        // This is what allows the post-processing to work
        precompute_twiddles(self.twiddles, self.n);
    }

    /// Executes the Real FFT Forward.
    /// The result is packed:
    /// - buffer[0].re = DC (Frequency 0)
    /// - buffer[0].im = Nyquist (Frequency N/2)
    /// - buffer[1..N/2] = Normal positive frequencies.
    fn rfft<const FRAC: u32>(&self, buffer: &mut [Fixed<FRAC>]) -> Result<(), FftError> {
        if buffer.len() != self.n {
            return Err(FftError::SizeMismatch);
        }

        // C TRICK: Reinterpret fixed array as ComplexFixed array
        // Safety: ComplexFixed is repr(C) of two Fixeds, and alignment is compatible.
        let cbuffer =
            unsafe { slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut ComplexFixed<FRAC>, self.n / 2) };

        // FFT of the complex sequence of N/2 points, interleaved from real input
        radix_2_dit_fft_core::<FRAC, false>(cbuffer, self.twiddles, self.bitrev, 2);

        // Unweaving
        let n_half = self.n / 2;
        let n_quarter = n_half / 2;

        // 0-indexed component (DC and Nyquist)
        {
            let val = cbuffer[0];
            // DC component = even.real + odd.imag = c[0].re + c[0].im
            // Nyquist component = even.real - odd.imag = c[0].re - c[0].im
            
            // In Fixed point, adding two N-bit numbers can overflow. 
            // Often FFT output formats assume growth.
            // However, this implementation keeps same storage type.
            // The caller must ensure headroom or accept wrap/saturation.
            // For safety equivalent to floating point 'addition', we just add.
            let dc = val.re + val.im;
            let nyquist = val.re - val.im;

            cbuffer[0] = ComplexFixed::new(dc, nyquist);
        }

        cbuffer[n_quarter] = cbuffer[n_quarter].conj();
        
        // Main unweaving loop
        for i in 1..n_quarter {
            let idx_a = i;
            let idx_b = n_half - i;

            let val_a = cbuffer[idx_a];
            let val_b = cbuffer[idx_b];
            let val_b_conj = val_b.conj();

            // even = (cdata[i] + conj(cdata[n/2-i])) / 2
            let even = (val_a + val_b_conj).scale_half();

            // odd = (cdata[i] - conj(cdata[n/2-i])) / 2
            let odd = (val_a - val_b_conj).scale_half();

            // Twiddle calculation
            let w = self.twiddles[i];

            // tmp1 = odd * w
            let tmp1 = odd * w;

            // tmp = I * tmp1 (re: -tmp1.im, im: tmp1.re)
            let tmp = ComplexFixed::new(
                Fixed::from_bits(tmp1.im.to_bits().wrapping_neg()), 
                tmp1.re
            );

            // cdata[i] = even - I * odd * w  => even - tmp
            cbuffer[idx_a] = even - tmp;

            // cdata[n/2-i] = conj(even) - I * conj(odd) * conj(w)
            //                conj(even) - I * conj(odd * w)
            //                conj(even + I * odd * w) => conj(even + tmp)
            let val_b_res = (even + tmp).conj();
            cbuffer[idx_b] = val_b_res;
        }

        Ok(())
    }

    fn irfft<const FRAC: u32>(&self, buffer: &mut [Fixed<FRAC>]) -> Result<(), FftError> {
        if buffer.len() != self.n {
            return Err(FftError::SizeMismatch);
        }

        let cbuffer =
            unsafe { slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut ComplexFixed<FRAC>, self.n / 2) };

        let n_half = self.n / 2;
        let n_quarter = n_half / 2;

        // 1. Reweaving

        cbuffer[0] = ComplexFixed::new(
            (cbuffer[0].re + cbuffer[0].im).scale_half(),
            (cbuffer[0].re - cbuffer[0].im).scale_half(),
        );
        cbuffer[n_quarter] = cbuffer[n_quarter].conj();

        for i in 1..n_quarter  {
            let idx_a = i;
            let idx_b = n_half - i;

            let val_a = cbuffer[idx_a];
            let val_b = cbuffer[idx_b];

            // even = (cdata[i] + conj(cdata[n/2-i])) / 2
            let even = (val_a + val_b.conj()).scale_half();

            // odd = (cdata[i] - conj(cdata[n/2-i])) / 2
            let odd = (val_a - val_b.conj()).scale_half();

            // w = conj(twd[i])
            let w = self.twiddles[i].conj();
            
            // NOTE CAREFULLY:
            // "w" here is Q31. "odd" is Q<FRAC>.
            // "odd * w" results in Q<FRAC>.
            // The multiplication logic inside ComplexFixed handles the mixed Q-format correctly
            // assuming the implementation of Mul<ComplexFixed<TWIDDLE_FRAC>> exists.
            
            let tmp1 = odd * w;
            
            // tmp = I * odd * w
            let tmp = ComplexFixed::new(
                Fixed::from_bits(tmp1.im.to_bits().wrapping_neg()), 
                tmp1.re
            );

            cbuffer[idx_a] = even + tmp;

            cbuffer[idx_b] = (even - tmp).conj();
        }

        // 2. Inverse FFT of the complex sequence of N/2 points
        // The core will handle 1/2 scaling per stage
        radix_2_dit_fft_core::<FRAC, true>(cbuffer, self.twiddles, self.bitrev, 2);

        Ok(())
    }

    pub fn process<const FRAC: u32>(&self, buffer: &mut [Fixed<FRAC>], inverse: bool) -> Result<(), FftError> {
        if inverse {
            self.irfft(buffer)
        } else {
            self.rfft(buffer)
        }
    }
}

// Implement trait for generic FRAC
impl<'a, const FRAC: u32> FftProcess<Fixed<FRAC>> for RealFft<'a,ComplexFixed<TWIDDLE_FRAC>> {
    fn process(&self, buffer: &mut [Fixed<FRAC>], inverse: bool) -> Result<(), FftError> {
        self.process(buffer, inverse)
    }
}

#[cfg(test)]
#[path = "real_tests.rs"]
mod tests;
