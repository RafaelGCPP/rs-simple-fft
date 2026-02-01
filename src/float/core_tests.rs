use super::*;
use num_complex::Complex32;
use core::f32::consts::PI;

const EPSILON: f32 = 1e-4;

fn assert_feq(a: f32, b: f32) {
    assert!(
        (a - b).abs() < EPSILON, 
        "Float mismatch: {} vs {}", a, b
    );
}

fn assert_cplx_eq(a: Complex32, b: Complex32) {
    assert!(
        (a - b).l1_norm() < EPSILON, 
        "Complex mismatch: {} vs {}", a, b
    );
}

#[test]
fn test_precompute_bitrev_8() {
    let n = 8;
    let mut bitrev = vec![0; n];
    precompute_bitrev(&mut bitrev, n);

    // Expected bit reversal for N=8:
    // 0 (000) -> 0 (000)
    // 1 (001) -> 4 (100)
    // 2 (010) -> 2 (010)
    // 3 (011) -> 6 (110)
    // 4 (100) -> 1 (001)
    // 5 (101) -> 5 (101)
    // 6 (110) -> 3 (011)
    // 7 (111) -> 7 (111)
    let expected = vec![0, 4, 2, 6, 1, 5, 3, 7];
    assert_eq!(bitrev, expected);
}

#[test]
fn test_precompute_twiddles_8() {
    let n = 8;
    let mut twiddles = vec![Complex32::default(); n / 2];
    precompute_twiddles(&mut twiddles, n);

    // Twiddles are e^(-j * 2*pi * k / N) for k=0..N/2-1
    // k=0: exp(0) = 1
    // k=1: exp(-j * 2*pi * 1/8) = exp(-j * pi/4) = sqrt(2)/2 - j*sqrt(2)/2
    // k=2: exp(-j * 2*pi * 2/8) = exp(-j * pi/2) = -j
    // k=3: exp(-j * 2*pi * 3/8) = -sqrt(2)/2 - j*sqrt(2)/2

    assert_cplx_eq(twiddles[0], Complex32::new(1.0, 0.0));
    
    let sqrt2_2 = (2.0f32).sqrt() / 2.0;
    assert_cplx_eq(twiddles[1], Complex32::new(sqrt2_2, -sqrt2_2));
    assert_cplx_eq(twiddles[2], Complex32::new(0.0, -1.0));
    assert_cplx_eq(twiddles[3], Complex32::new(-sqrt2_2, -sqrt2_2));
}

#[test]
fn test_radix_2_dit_fft_core_basic() {
    // Simple DC signal check without the wrapper overhead
    let n = 4;
    let mut buffer = vec![
        Complex32::new(1.0, 0.0), 
        Complex32::new(1.0, 0.0), 
        Complex32::new(1.0, 0.0), 
        Complex32::new(1.0, 0.0)
    ];
    let mut twiddles = vec![Complex32::default(); n/2];
    let mut bitrev = vec![0; n];
    
    precompute_bitrev(&mut bitrev, n);
    precompute_twiddles(&mut twiddles, n);

    // Run Forward FFT
    radix_2_dit_fft_core::<false>(&mut buffer, &twiddles, &bitrev, 1);

    // Expected: [4, 0, 0, 0]
    assert_cplx_eq(buffer[0], Complex32::new(4.0, 0.0));
    assert_cplx_eq(buffer[1], Complex32::new(0.0, 0.0));
    assert_cplx_eq(buffer[2], Complex32::new(0.0, 0.0));
    assert_cplx_eq(buffer[3], Complex32::new(0.0, 0.0));
    
    // Run Inverse FFT
    radix_2_dit_fft_core::<true>(&mut buffer, &twiddles, &bitrev, 1);

    // Expected: [1, 1, 1, 1] 
    for sample in buffer {
        assert_cplx_eq(sample, Complex32::new(1.0, 0.0));
    }
}

#[test]
fn test_sin_cos() {
    let angle = PI / 4.0; // 45 degrees
    let (s, c) = sin_cos(angle);
    let sqrt2_2 = (2.0f32).sqrt() / 2.0;
    assert_feq(s, sqrt2_2);
    assert_feq(c, sqrt2_2);
}