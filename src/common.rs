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