use super::core::{precompute_bitrev, precompute_twiddles, radix_2_dit_fft_core};
use crate::common::{FftError, FftProcess, RealFft};
use core::slice;
use num_complex::Complex32;

#[cfg(not(feature = "std"))]
use libm::Libm;
#[cfg(feature = "std")]
use std::f32;

// pub struct RealFft<'a> {
//     twiddles: &'a mut [Complex32],
//     bitrev: &'a mut [usize],
//     n: usize,
// }

impl<'a> RealFft<'a, Complex32> {
    /// Initializes the Real FFT.
    /// Note that 'n' here is the number of REAL samples.
    pub fn new(
        twiddles: &'a mut [Complex32],
        bitrev: &'a mut [usize],
        n: usize,
    ) -> Result<Self, FftError> {
        if !n.is_power_of_two() {
            return Err(FftError::NotPowerOfTwo);
        }
        // For an N-point RFFT, we need auxiliary tables
        // compatible with the underlying N/2-point Complex FFT,
        // BUT the post-processing needs finer twiddles (size N).
        // The original C code actually generates twiddles for N/2 * 2 = N (see fft_init.c).
        if twiddles.len() < n / 2 || bitrev.len() < n / 2 {
            return Err(FftError::BufferTooSmall);
        }

        let mut fft: RealFft<'a, num_complex::Complex<f32>> = Self {
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

        // 2. Twiddles are generated for N (full circle)
        // This is what allows the post-processing to work
        precompute_twiddles(self.twiddles, self.n);
    }

    /// Executes the Real FFT Forward.
    /// The result is packed:
    /// - buffer[0].re = DC (Frequency 0)
    /// - buffer[0].im = Nyquist (Frequency N/2)
    /// - buffer[1..N/2] = Normal positive frequencies.
    fn rfft(&self, buffer: &mut [f32]) -> Result<(), FftError> {
        if buffer.len() != self.n {
            return Err(FftError::SizeMismatch);
        }

        // C TRICK: Reinterpret float array as Complex array
        // Safety: Complex32 is repr(C) of two f32s, and alignment is compatible.
        let cbuffer =
            unsafe { slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut Complex32, self.n / 2) };

        // FFT of the complex sequence of N/2 points, interleaved from real input
        // This basically creates a complex FFT of the even and odd indexed samples
        // where the odd indexed samples are multiplied by j (the imaginary unit).

        radix_2_dit_fft_core::<false>(cbuffer, self.twiddles, self.bitrev, 2);

        // Unweaving
        let n_half = self.n / 2;
        let n_quarter = n_half / 2;

        // 0-indexed component (DC and Nyquist)
        {
            let val = cbuffer[0];

            // Because of how the complex FFT output is structured for real input,
            // c[0] will have the DC and Nyquist components intertwined.

            // The real part of c[0] is composed by the DC component of the
            // even indexed sequence and the nyquist component of the odd indexed
            // sequence.

            // The imaginary part of c[0] is composed by the DC component of the
            // odd indexed sequence and the nyquist component of the even indexed
            // sequence.

            // hence we can reconstruct them as follows:
            // DC component = even.real + odd.imag = c[0].re + c[0].im
            // Nyquist component = even.real - odd.imag = c[0].re - c[0].im

            cbuffer[0] = Complex32::new(val.re + val.im, val.re - val.im);
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
            let even = (val_a + val_b_conj).scale(0.5);

            // odd = (cdata[i] - conj(cdata[n/2-i])) / 2
            let odd = (val_a - val_b_conj).scale(0.5);

            // Twiddle calculation
            // C: w = twd[i]; (Note que twd aqui é a tabela completa de tamanho N/2)
            let w = self.twiddles[i];

            // tmp1 = odd * w
            let tmp1 = odd * w;

            // tmp = I * tmp1 (re: -tmp1.im, im: tmp1.re)
            let tmp = Complex32::new(-tmp1.im, tmp1.re);

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

    fn irfft(&self, buffer: &mut [f32]) -> Result<(), FftError> {
        if buffer.len() != self.n {
            return Err(FftError::SizeMismatch);
        }

        let cbuffer =
            unsafe { slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut Complex32, self.n / 2) };

        let n_half = self.n / 2;
        let n_quarter = n_half / 2;

        // 1. Reweaving

        cbuffer[0] = Complex32::new(
            (cbuffer[0].re + cbuffer[0].im) * 0.5,
            (cbuffer[0].re - cbuffer[0].im) * 0.5,
        );
        cbuffer[n_quarter] = cbuffer[n_quarter].conj();

        for i in 1..n_quarter {
            let idx_a = i;
            let idx_b = n_half - i;

            let val_a = cbuffer[idx_a];
            let val_b = cbuffer[idx_b];

            // even = (cdata[i] + conj(cdata[n/2-i])) / 2
            let even = (val_a + val_b.conj()).scale(0.5);

            // odd = (cdata[i] - conj(cdata[n/2-i])) / 2
            let odd = (val_a - val_b.conj()).scale(0.5);

            // w = conj(twd[i])
            let w = self.twiddles[i].conj();

            let tmp1 = odd * w;
            // tmp = I * odd * w
            let tmp = Complex32::new(-tmp1.im, tmp1.re);

            cbuffer[idx_a] = even + tmp;

            cbuffer[idx_b] = (even - tmp).conj();
        }

        // 2. Inverse FFT of the complex sequence of N/2 points
        radix_2_dit_fft_core::<true>(cbuffer, self.twiddles, self.bitrev, 2);

        Ok(())
    }

    pub fn process(&self, buffer: &mut [f32], inverse: bool) -> Result<(), FftError> {
        if buffer.len() != self.n {
            return Err(FftError::SizeMismatch);
        }

        if inverse {
            self.irfft(buffer)?;
        } else {
            self.rfft(buffer)?;
        }

        Ok(())
    }
}

// Implementação da trait FftProcess para RealFft
impl<'a> FftProcess<f32> for RealFft<'a, Complex32> {
    fn process(&self, buffer: &mut [f32], inverse: bool) -> Result<(), FftError> {
        self.process(buffer, inverse)
    }
}

#[cfg(test)]
#[path = "real_tests.rs"]
mod tests;
