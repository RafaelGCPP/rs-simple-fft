// src/common.rs

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum FftError {
    SizeMismatch,
    NotPowerOfTwo,
    BufferTooSmall,
    InvalidStride,
}

use core::fmt;

impl fmt::Display for FftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FftError::SizeMismatch => write!(f, "Data buffer size does not match FFT size"),
            FftError::NotPowerOfTwo => write!(f, "Size must be a power of 2"),
            FftError::BufferTooSmall => write!(f, "Auxiliary buffers are too small"),
            FftError::InvalidStride => write!(f, "Invalid stride configuration"),
        }
    }
}

pub trait FftProcess<T> {
    fn process(&self, buffer: &mut [T], inverse: bool) -> Result<(), FftError>;
}

#[cfg(feature = "std")]
impl std::error::Error for FftError {}

/// Generic RealFFT struct.
/// T represents the Complex Number type used for twiddle factors.
pub struct RealFft<'a, T> {
    pub twiddles: &'a mut [T],
    pub bitrev: &'a mut [usize],
    pub n: usize,
}

/// Generic CplxFft struct.
/// T represents the Complex Number type used for twiddle factors.
pub struct CplxFft<'a, T> {
    pub twiddles: &'a mut [T],
    pub bitrev: &'a mut [usize],
    pub n: usize,
}

/// Trait to handle generic Scalar operations for FFT packing/unpacking.
/// It bridges the gap between Real and Complex representations.
pub trait FftNum: Copy + PartialEq + core::fmt::Debug {
    type Complex: Copy + core::fmt::Debug;

    fn from_f64(v: f64) -> Self;
    fn zero() -> Self;
    fn val_to_complex(re: Self, im: Self) -> Self::Complex;
    fn complex_re(c: &Self::Complex) -> Self;
    fn complex_im(c: &Self::Complex) -> Self;
    fn negate(self) -> Self;
}

#[cfg(feature = "std")]
use std::f32;

#[cfg(not(feature = "std"))]
use core::f32;

// Implement for standard floats (using num_complex)
impl FftNum for f32 {
    type Complex = num_complex::Complex<f32>;

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn val_to_complex(re: Self, im: Self) -> Self::Complex {
        num_complex::Complex::new(re, im)
    }
    #[inline]
    fn complex_re(c: &Self::Complex) -> Self {
        c.re
    }
    #[inline]
    fn complex_im(c: &Self::Complex) -> Self {
        c.im
    }
    #[inline]
    fn negate(self) -> Self {
        -self
    }
}

/// Expands the packed Real FFT format into a full complex array of size N.
///
/// The output will be Hermitian symmetric: X[k] = conj(X[N-k]).
pub fn unpack_rfft_spectrum<T: FftNum>(packed: &[T], output: &mut [T::Complex]) {
    let n = packed.len();
    assert_eq!(output.len(), n, "Output buffer must be size N");
    assert_eq!(n % 2, 0, "Input size must be even");

    // 1. DC Component
    output[0] = T::val_to_complex(packed[0], T::zero());

    // 2. Nyquist Component (packed at index 1)
    output[n / 2] = T::val_to_complex(packed[1], T::zero());

    // 3. Positive Frequencies & Negative Frequencies (Conjugates)
    for k in 1..n / 2 {
        let re = packed[2 * k];
        let im = packed[2 * k + 1];

        // Positive freq k
        output[k] = T::val_to_complex(re, im);

        // Negative freq N-k = conj(k)
        // conj(a + bi) = a - bi
        output[n - k] = T::val_to_complex(re, T::negate(im));
    }
}

/// Packs a full complex spectrum of size N into the compact Real FFT format.
/// Only the DC, Nyquist, and positive frequencies are read from `full`.
pub fn pack_rfft_spectrum<T: FftNum>(full: &[T::Complex], output: &mut [T]) {
    let n = full.len();
    assert_eq!(output.len(), n, "Output buffer must be size N");

    // Output[0] = DC.Real
    output[0] = T::complex_re(&full[0]);
    // Output[1] = Nyquist.Real (Index N/2)
    output[1] = T::complex_re(&full[n / 2]);

    for k in 1..n / 2 {
        output[2 * k] = T::complex_re(&full[k]);
        output[2 * k + 1] = T::complex_im(&full[k]);
    }
}
