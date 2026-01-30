#![no_std]

// Enables the standard library only for tests,
// so you can run 'cargo test' on your PC normally.
#[cfg(test)]
extern crate std;

pub mod common;
pub mod float;
pub mod fixed;

// The rest of the code will come later...