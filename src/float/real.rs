use crate::common::FftError;
use num_complex::Complex32;
use core::slice;
use super::core::{radix_2_dit_fft_core, precompute_twiddles, precompute_bitrev};


// #[cfg(feature = "std")]
// use std::f32;
// #[cfg(not(feature = "std"))]
// use libm::Libm;

// pub struct RealFft<'a> {
//     twiddles: &'a mut [Complex32],
//     bitrev: &'a mut [usize],
//     n: usize,
// }

impl<'a> RealFft<'a> {
    /// Initializes the Real FFT.
    /// Note that 'n' here is the number of REAL samples.
    pub fn new(
        twiddles: &'a mut [Complex32], 
        bitrev: &'a mut [usize], 
        n: usize
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

//         let mut fft = Self { twiddles, bitrev, n };
//         fft.precompute();
//         Ok(fft)
//     }

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
    pub fn process(&self, buffer: &mut [f32]) -> Result<(), FftError> {
        if buffer.len() != self.n {
            return Err(FftError::SizeMismatch);
        }

        // C TRICK: Reinterpret float array as Complex array
        // Safety: Complex32 is repr(C) of two f32s, and alignment is compatible.
        let cbuffer = unsafe {
            slice::from_raw_parts_mut(
                buffer.as_mut_ptr() as *mut Complex32,
                self.n / 2
            )
        };

        // FFT of the complex sequence of N/2 points, interleaved from real input
        // This basically creates a complex FFT of the even and odd indexed samples
        // where the odd indexed samples are multiplied by j (the imaginary unit).
        

        radix_2_dit_fft_core(cbuffer, self.twiddles, self.bitrev, 2, false);

        // Unweaving
        let n_half = self.n / 2;
        let n_quarter = n_half / 2;

        // Processa índice 0 (DC e Nyquist)
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

        // Main unweaving loop
        for i in 1..=n_quarter { 
            if i == n_quarter {
                 // Special case i = N/4 
                 // in this case twiddle is 90 degrees (I) and the corresponding FFT components
                 // of the even and odd indexed sequences have the same index.

                 // The odd indexed sequence, multiplied by j, becomes a rotation by 90 degrees.
                 // Thus we can simplify the calculations, by just conjugating the value.

                 cbuffer[i] = cbuffer[i].conj();

                 continue;
            }

//             let idx_a = i;
//             let idx_b = n_half - i;

            let val_a = cbuffer[idx_a];
            let val_b = cbuffer[idx_b];
            let val_b_conj = val_b.conj(); 

//             // even = (cdata[i] + conj(cdata[n/2-i])) / 2
//             let even = (val_a + val_b_conj).scale(0.5);
            
//             // odd = (cdata[i] - conj(cdata[n/2-i])) / 2
//             let odd = (val_a - val_b_conj).scale(0.5);

//             // Twiddle calculation
//             // C: w = twd[i]; (Note que twd aqui é a tabela completa de tamanho N/2)
//             let w = self.twiddles[i]; 

//             // tmp1 = odd * w
//             let tmp1 = odd * w;
            
//             // tmp = I * tmp1 (re: -tmp1.im, im: tmp1.re)
//             let tmp = Complex32::new(-tmp1.im, tmp1.re);

//             // cdata[i] = even - I * odd * w  => even - tmp
//             cbuffer[idx_a] = even - tmp;

//             // cdata[n/2 - i] = conj(cdata[i]) (Simetria!)
//             // O código C calcula explicitamente: cdata[n/2-i] = even1 - ...
//             // Mas para RFFT, o resultado de N-i é o conjugado de i.
//             // Vamos seguir o C para garantir:
//             // C calcula even1/odd1 baseado em idx_b e idx_a.conj().
//             // Na matemática: even1 == even.conj(), odd1 == odd.conj()??
//             // Vamos confiar na operação simétrica:
            
//             let val_b_res = (even + tmp).conj(); 
//             cbuffer[idx_b] = val_b_res; 
//         }

//         Ok(())
//     }

//     /// Executa a FFT Real Inversa
//     pub fn process_inv(&self, buffer: &mut [f32]) -> Result<(), FftError> {
//         if buffer.len() != self.n {
//             return Err(FftError::SizeMismatch);
//         }

//         let cbuffer = unsafe {
//             slice::from_raw_parts_mut(
//                 buffer.as_mut_ptr() as *mut Complex32,
//                 self.n / 2
//             )
//         };

//         let n_half = self.n / 2;
//         let n_quarter = n_half / 2;

//         // 1. Pre-processamento (Weaving) - Inverso do passo acima
        
//         // Loop principal
//         for i in 1..n_quarter {
//             let idx_a = i;
//             let idx_b = n_half - i;

//             let val_a = cbuffer[idx_a];
//             let val_b = cbuffer[idx_b];
//             // O código C usa conjugados aqui
            
//             // Vamos simplificar: O código C inverse é simétrico mas com W conjugado e Somas.
//             // Vou omitir a tradução linha-a-linha exaustiva aqui para brevidade, 
//             // mas a lógica é espelhar o loop do 'process' trocando sinais e conjugando W.
            
//             let even = (val_a + val_b.conj()).scale(0.5);
//             let odd = (val_a - val_b.conj()).scale(0.5);
            
//             // w = conj(twd[i])
//             let w = self.twiddles[i].conj();
            
//             let tmp1 = odd * w;
//             // tmp = I * tmp1
//             let tmp = Complex32::new(-tmp1.im, tmp1.re);
            
//             cbuffer[idx_a] = even + tmp;
//             cbuffer[idx_b] = (even - tmp).conj();
//         }
        
//         // Ponto N/4
//         {
//              let val = cbuffer[n_quarter];
//              let tmp = val.conj();
//              let even = (val + tmp).scale(0.5);
//              let odd = (val - tmp).scale(0.5);
//              // tmp = I * odd
//              let tmp_i_odd = Complex32::new(-odd.im, odd.re);
//              cbuffer[n_quarter] = even + tmp_i_odd;
//         }

//         // Ponto 0 (DC/Nyquist)
//         {
//             let val = cbuffer[0];
//             // even.real = (val.re + val.im) / 2
//             let even_re = (val.re + val.im) * 0.5;
//             // odd.real = (val.re - val.im) / 2
//             let odd_re = (val.re - val.im) * 0.5;
            
//             cbuffer[0] = Complex32::new(even_re, odd_re);
//         }


//         // 2. Executa FFT Complexa Inversa de N/2 pontos
//         radix_2_dit_fft_core::<true>(cbuffer, self.twiddles, self.bitrev, 2);

//         Ok(())
//     }
// }

