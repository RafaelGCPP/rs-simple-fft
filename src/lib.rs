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
