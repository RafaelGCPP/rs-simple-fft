// src/lib.rs
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(test)]
extern crate std;

pub mod common;
pub mod float;
pub mod fixed;

// Re-exporta o erro para ficar acessível globalmente
pub use common::FftError;