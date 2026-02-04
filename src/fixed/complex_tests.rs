use super::super::types::{ComplexFixed, Fixed};
use super::*;

// Access the TWIDDLE_FRAC constant from the core module
use super::super::core::TWIDDLE_FRAC;

fn to_f64<const FRAC: u32>(val: Fixed<FRAC>) -> f64 {
    val.to_bits() as f64 / (1u64 << FRAC) as f64
}

fn assert_complex_close<const FRAC: u32>(
    val: ComplexFixed<FRAC>,
    expected_re: f64,
    expected_im: f64,
    tolerance: f64,
) {
    let re = to_f64(val.re);
    let im = to_f64(val.im);

    let dist = ((re - expected_re).powi(2) + (im - expected_im).powi(2)).sqrt();

    assert!(
        dist < tolerance,
        "Error too large. Expected: ({:.4}, {:.4}), Got: ({:.4}, {:.4}), Dist: {:.4}",
        expected_re,
        expected_im,
        re,
        im,
        dist
    );
}

#[test]
fn test_fft_forward_q15() {
    const FRAC: u32 = 15;
    let n = 8;

    // Inputs from the floating point test
    let input_f64 = [
        (1.0, 2.0),
        (3.0, 4.0),
        (5.0, 6.0),
        (7.0, 8.0),
        (-8.0, -7.0),
        (-6.0, -5.0),
        (-4.0, -3.0),
        (-2.0, -1.0),
    ];

    // Expected outputs from floating point FFT (unscaled)
    // Note: Magnitudes are around 30.0, which fits easily in Q15 (max 65536)
    let expected_f64 = [
        (-4.0, 4.0),
        (30.72792, -12.72792),
        (-16.0, 0.0),
        (12.72792, 5.27208),
        (-8.0, -8.0),
        (5.27208, 12.72792),
        (0.0, -16.0),
        (-12.72792, 30.72792),
    ];

    let mut buffer: Vec<ComplexFixed<FRAC>> = input_f64
        .iter()
        .map(|&(re, im)| ComplexFixed::new(Fixed::from_f64(re), Fixed::from_f64(im)))
        .collect();

    let mut twiddles =
        vec![ComplexFixed::<TWIDDLE_FRAC>::new(Fixed::from_bits(0), Fixed::from_bits(0)); n / 2];
    let mut bitrev = vec![0; n];

    let fft =
        CplxFft::<'_, ComplexFixed<TWIDDLE_FRAC>>::new(&mut twiddles, &mut bitrev, n).unwrap();

    fft.process(&mut buffer, false).unwrap();

    // Tolerance check:
    // Fixed point noise comes from:
    // 1. Initial quantization (small)
    // 2. Twiddle factor quantization (Q31, very small)
    // 3. Multiplication truncation (Q15 * Q31 -> Q15) in each stage
    // For N=8, error shouldn't be too huge.
    // Let's suggest 0.1 as a safe starting margin.
    for (i, &val) in buffer.iter().enumerate() {
        assert_complex_close(val, expected_f64[i].0, expected_f64[i].1, 0.1);
    }
}

#[test]
fn test_fft_inverse_q15() {
    const FRAC: u32 = 15;
    let n = 8;

    // Input is the "Output" of the forward transform
    let input_f64 = [
        (-4.0, 4.0),
        (30.72792, -12.72792),
        (-16.0, 0.0),
        (12.72792, 5.27208),
        (-8.0, -8.0),
        (5.27208, 12.72792),
        (0.0, -16.0),
        (-12.72792, 30.72792),
    ];

    // Expected output is the original input
    // The fixed-point inverse FFT includes 1/N scaling distributed across stages
    let expected_f64 = [
        (1.0, 2.0),
        (3.0, 4.0),
        (5.0, 6.0),
        (7.0, 8.0),
        (-8.0, -7.0),
        (-6.0, -5.0),
        (-4.0, -3.0),
        (-2.0, -1.0),
    ];

    let mut buffer: Vec<ComplexFixed<FRAC>> = input_f64
        .iter()
        .map(|&(re, im)| ComplexFixed::new(Fixed::from_f64(re), Fixed::from_f64(im)))
        .collect();

    let mut twiddles =
        vec![ComplexFixed::<TWIDDLE_FRAC>::new(Fixed::from_bits(0), Fixed::from_bits(0)); n / 2];
    let mut bitrev = vec![0; n];

    let fft =
        CplxFft::<'_, ComplexFixed<TWIDDLE_FRAC>>::new(&mut twiddles, &mut bitrev, n).unwrap();

    // Run Inverse FFT
    fft.process(&mut buffer, true).unwrap();

    for (i, &val) in buffer.iter().enumerate() {
        assert_complex_close(val, expected_f64[i].0, expected_f64[i].1, 0.1);
    }
}
