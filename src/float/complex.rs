use crate::common::FftError; 
use num_complex::Complex32; // Complex<f32>
use core::f32::consts::PI;

// Em no_std, precisamos importar funções matemáticas de algum lugar.
// Se a feature "std" estiver ativa, usamos f32::sin/cos nativos.
// Se não, usamos a libm via trait Float do num_traits.
#[cfg(feature = "std")]
use std::f32;
#[cfg(not(feature = "std"))]
use libm::Libm;

/// Estrutura que segura as tabelas pré-computadas (Twiddle factors e Bit Reverse).
/// Isso substitui passar 'twiddle' e 'bitrev' soltos em toda função.
pub struct CplxFft<'a> {
    twiddles: &'a mut [Complex32],
    bitrev: &'a mut [usize],
    n: usize,
}

impl<'a> CplxFft<'a> {
    /// Inicializa as tabelas (Port do `fft_init.c`)
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

    /// Precomputa Twiddle Factors e Bit Reverse Table
    fn precompute(&mut self) {
        // 1. Bit Reverse Table (Port de `precompute_bitrev_table`)
        self.bitrev[0] = 0;
        let mut j = 0;
        for i in 1..self.n {
            let mut k = self.n >> 1;
            while j >= k {
                j -= k;
                k >>= 1;
            }
            j += k;
            self.bitrev[i] = j;
        }

        // 2. Twiddle Factors (Port de `precompute_twiddle_factors`)
        // Nota: Rust usa iteradores, mas manteremos o loop for clássico para fidelidade ao algoritmo C
        for j in 0..(self.n / 2) {
            let angle = -2.0 * PI * (j as f32) / (self.n as f32);
            // Aqui usamos a "magia" para funcionar em no_std ou std
            let (sin, cos) = sin_cos(angle);
            self.twiddles[j] = Complex32::new(cos, sin);
        }
    }

    /// Executa a FFT in-place (Port de `radix_2_dit_fft` em `fft_core.c`)
    pub fn process(&self, buffer: &mut [Complex32], inverse: bool) -> Result<(), FftError> {
        if buffer.len() != self.n {
            return Err(FftError::SizeMismatch);
        }

        // 1. Bit-reverse permutation
        for i in 1..(self.n - 1) {
            let j = self.bitrev[i];
            if i < j {
                buffer.swap(i, j);
            }
        }

        // 2. Butterfly operations
        let mut stride = 1;
        let mut tw_index = self.n >> 1;

        while stride < self.n {
            let jmax = self.n - stride;
            
            // Loop externo de blocos
            for j in (0..jmax).step_by(stride << 1) {
                // Loop interno (butterfly)
                for i in 0..stride {
                    let mut w = self.twiddles[i * tw_index];
                    
                    // Se for inversa, conjugamos o twiddle factor
                    if inverse {
                        w = w.conj();
                    }

                    let index = j + i;
                    let a = buffer[index];
                    let b = buffer[index + stride];

                    // Operação Butterfly:
                    // t = w * b
                    // buf[index] = a + t
                    // buf[index + stride] = a - t
                    let t = b * w;
                    buffer[index] = a + t;
                    buffer[index + stride] = a - t;
                }
            }
            stride <<= 1;
            tw_index >>= 1;
        }

        // 3. Normalização para FFT Inversa (Se necessário)
        // O código C original faz a divisão por 2 a cada estágio (cplx_half).
        // Aqui, para simplificar e melhorar precisão, costuma-se dividir tudo no final.
        // Mas se quiser seguir o C estritamente:
        if inverse {
            let factor = 1.0 / (self.n as f32);
            for x in buffer.iter_mut() {
                *x = x.scale(factor);
            }
        }

        Ok(())
    }
}

// Helper para calcular Seno e Cosseno de forma agnóstica (std ou no_std)
fn sin_cos(angle: f32) -> (f32, f32) {
    #[cfg(feature = "std")]
    return (angle.sin(), angle.cos());

    #[cfg(not(feature = "std"))]
    return (libm::sinf(angle), libm::cosf(angle));
}