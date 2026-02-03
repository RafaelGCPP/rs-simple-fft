use super::super::core::TWIDDLE_FRAC;
use super::super::types::{ComplexFixed, Fixed};
use super::*;

fn to_f64<const FRAC: u32>(val: Fixed<FRAC>) -> f64 {
    val.to_bits() as f64 / (1u64 << FRAC) as f64
}

fn assert_fixed_close<const FRAC: u32>(val: Fixed<FRAC>, expected: f64, tolerance: f64) {
    let float_val = to_f64(val);
    assert!(
        (float_val - expected).abs() < tolerance,
        "Expected: {:.4}, Got: {:.4} (diff: {:.4})",
        expected,
        float_val,
        (float_val - expected).abs()
    );
}

#[test]
fn test_rfft_forward_impulse() {
    // RFFT of [1, 0, 0, 0] -> DC=1, Nyq=0, F1=0...
    const FRAC: u32 = 15;
    let n = 4;
    let mut buffer = [
        Fixed::<FRAC>::from_int(1),
        Fixed::<FRAC>::from_int(0),
        Fixed::<FRAC>::from_int(0),
        Fixed::<FRAC>::from_int(0),
    ];

    let mut twiddles =
        vec![ComplexFixed::<TWIDDLE_FRAC>::new(Fixed::from_bits(0), Fixed::from_bits(0)); n / 2];
    // NOTE: RealFft allocates bitrev tables of size N/2
    let mut bitrev = vec![0; n / 2];

    let fft = RealFft::<ComplexFixed<TWIDDLE_FRAC>>::new(&mut twiddles, &mut bitrev, n).unwrap();
    fft.process(&mut buffer, false).unwrap();

    // Output structure for N=4:
    // buffer[0]: DC (Re) + Nyquist (Im)
    // buffer[1]: Real part of freq 1
    // buffer[2]: Imag part of freq 1 (N/2 packed)
    // ... wait, DIT RFFT pack format is:
    // buffer[0] = DC
    // buffer[1] = Nyquist? No.
    // Let's check implementation.

    // Implementation says:
    // cbuffer[0] = Complex(DC, Nyquist)
    // So buffer[0] = DC, buffer[1] = Nyquist.
    // buffer[2] = Real(F1), buffer[3] = Imag(F1). ('cbuffer[1]')

    // Expected FFT of [1, 0, 0, 0] is [1, 1, 1, 1] (Hermitian redundant)
    // DC = 1, Nyquist = 1
    // F1 = 1 + 0j

    let dc = buffer[0];
    let nyq = buffer[1];
    let f1_re = buffer[2];
    let f1_im = buffer[3];

    assert_fixed_close(dc, 1.0, 0.01);
    assert_fixed_close(nyq, 1.0, 0.01);
    assert_fixed_close(f1_re, 1.0, 0.01);
    assert_fixed_close(f1_im, 0.0, 0.01);
}

#[test]
fn test_rfft_inverse_impulse() {
    // Inverse RFFT of Flat spectrum (impulse)
    // Input packed:
    // DC=1, Nyq=0
    // F1=0
    // Should result in constant or similar?
    // Let's do round trip.

    const FRAC: u32 = 15;
    let n = 8;

    // Simple cosine wave: freq 1.
    // cos(2pi * 1 * t / 8) -> [1, 0.707, 0, -0.707, -1, -0.707, 0, 0.707]
    // FFT should have spike at index 1.

    let mut input = [
        Fixed::<FRAC>::from_f64(1.0),
        Fixed::<FRAC>::from_f64(0.7071),
        Fixed::<FRAC>::from_f64(0.0),
        Fixed::<FRAC>::from_f64(-0.7071),
        Fixed::<FRAC>::from_f64(-1.0),
        Fixed::<FRAC>::from_f64(-0.7071),
        Fixed::<FRAC>::from_f64(0.0),
        Fixed::<FRAC>::from_f64(0.7071),
    ];

    // Keep a copy for check
    let original = input.clone();

    let mut twiddles =
        vec![ComplexFixed::<TWIDDLE_FRAC>::new(Fixed::from_bits(0), Fixed::from_bits(0)); n / 2];
    let mut bitrev = vec![0; n / 2];

    let fft = RealFft::<ComplexFixed<TWIDDLE_FRAC>>::new(&mut twiddles, &mut bitrev, n).unwrap();

    // Forward
    fft.process(&mut input, false).unwrap();

    // Inverse
    fft.process(&mut input, true).unwrap();

    // Check (Input should be recovered)
    for i in 0..n {
        assert_fixed_close(input[i], to_f64(original[i]), 0.1);
    }
}
