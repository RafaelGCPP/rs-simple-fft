// src/lib.rs
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(test)]
extern crate std;

pub mod common;
pub mod fixed;
pub mod float;

// Re-exporta o erro para ficar acessível globalmente
pub use common::CplxFft;
pub use common::FftError;
pub use common::FftProcess;
pub use common::RealFft;
pub use fixed::ComplexFixed;
pub use fixed::Fixed;
use num_complex::Complex32;

pub type ComplexQ23 = ComplexFixed<23>;
pub type ComplexQ16 = ComplexFixed<16>;

pub type CplxFFTQ23 = CplxFft<'static, ComplexQ23>;
pub type CplxFFTQ16 = CplxFft<'static, ComplexQ16>;
pub type RealFFTQ23 = RealFft<'static, Fixed<23>>;
pub type RealFFTQ16 = RealFft<'static, Fixed<16>>;

pub type CplxFFT32 = CplxFft<'static, Complex32>;
pub type RealFFT32 = RealFft<'static, f32>;
