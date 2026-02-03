// src/lib.rs
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(test)]
extern crate std;

pub mod common;
pub mod float;
pub mod fixed;

// Re-exporta o erro para ficar acessível globalmente
pub use common::FftError;
pub use common::FftProcess;
pub use float::CplxFft as FloatCplxFft;
pub use float::RealFft as FloatRealFft;
pub use fixed::CplxFft as FixedCplxFft;
pub use fixed::RealFft as FixedRealFft;
pub use fixed::Fixed;
pub use fixed::ComplexFixed;