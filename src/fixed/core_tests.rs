use super::*;
use super::super::types::{ComplexFixed, Fixed};

const FRAC: u32 = 16;
type C = ComplexFixed<FRAC>;
type F = Fixed<FRAC>;

#[test]
fn test_precompute_bitrev() {
    let n = 8;
    let mut bitrev = vec![0; n];
    precompute_bitrev(&mut bitrev, n);
    
    assert_eq!(bitrev[0], 0);
    assert_eq!(bitrev[1], 4);
    assert_eq!(bitrev[2], 2);
    assert_eq!(bitrev[3], 6);
    assert_eq!(bitrev[4], 1);
    assert_eq!(bitrev[5], 5);
    assert_eq!(bitrev[6], 3);
    assert_eq!(bitrev[7], 7);
}

#[test]
fn test_precompute_twiddles() {
    let n = 4;
    let mut twiddles = vec![ComplexFixed::<TWIDDLE_FRAC>::new(Fixed::from_bits(0), Fixed::from_bits(0)); n/2];
    precompute_twiddles(&mut twiddles, n);
    
    // N=4 -> N/2 = 2 twiddles
    // k=0 -> angle=0 -> cos=1, sin=0
    // k=1 -> angle=-pi/2 -> cos=0, sin=-1
    
    let t0 = twiddles[0];
    let t1 = twiddles[1];
    
    // Check magnitudes roughly
    // Q31: 1.0 might be saturated to i32::MAX or wrap. 
    // If it wraps to negative, that is bad.
    // If it saturates to MAX, that's fine.
    
    let one_q31 = Fixed::<TWIDDLE_FRAC>::from_f64(1.0).to_bits();
    let zero_q31 = Fixed::<TWIDDLE_FRAC>::from_f64(0.0).to_bits();
    let minus_one_q31 = Fixed::<TWIDDLE_FRAC>::from_f64(-1.0).to_bits();
    
    assert_eq!(t0.im.to_bits(), zero_q31);
    assert_eq!(t0.re.to_bits(), one_q31);
    
    assert_eq!(t1.re.to_bits(), zero_q31);
    assert_eq!(t1.im.to_bits(), minus_one_q31);
}

#[test]
fn test_fft_core_forward_impulse() {
    // Impulse at 0 -> DC out (flat)
    let n = 8;
    // Input: [1.0, 0, ... 0]
    let mut buffer = vec![C::new(F::from_int(0), F::from_int(0)); n];
    buffer[0] = C::new(F::from_int(1), F::from_int(0));
    
    let mut twiddles = vec![ComplexFixed::<TWIDDLE_FRAC>::new(Fixed::from_bits(0), Fixed::from_bits(0)); n/2];
    precompute_twiddles(&mut twiddles, n);
    
    let mut bitrev = vec![0; n];
    precompute_bitrev(&mut bitrev, n);
    
    // Forward FFT
    radix_2_dit_fft_core::<FRAC, false>(&mut buffer, &twiddles, &bitrev, 1);
    
    // Expected output: [1, 1, 1, 1, ..., 1]
    let one = F::from_int(1).to_bits();
    let zero = F::from_int(0).to_bits();
    
    for (i, val) in buffer.iter().enumerate() {
        assert_eq!(val.re.to_bits(), one, "Real part at index {}", i);
        assert_eq!(val.im.to_bits(), zero, "Imaginary part at index {}", i);
    }
}

#[test]
fn test_fft_core_inverse_flat() {
    // Input: [1, 1, ..., 1]
    // Inverse FFT should be [1, 0, ..., 0] (because of scaling 1/N internal to INVERSE routine)
    let n = 8;
    let mut buffer = vec![C::new(F::from_int(1), F::from_int(0)); n];
    
    let mut twiddles = vec![ComplexFixed::<TWIDDLE_FRAC>::new(Fixed::from_bits(0), Fixed::from_bits(0)); n/2];
    precompute_twiddles(&mut twiddles, n);
    
    let mut bitrev = vec![0; n];
    precompute_bitrev(&mut bitrev, n);
    
    // Inverse FFT
    radix_2_dit_fft_core::<FRAC, true>(&mut buffer, &twiddles, &bitrev, 1);
    
    // Expected output: [1, 0, ..., 0]
    let one = F::from_int(1).to_bits();
    let zero = F::from_int(0).to_bits();
    
    // Check index 0
    assert_eq!(buffer[0].re.to_bits(), one, "Real part at index 0");
    assert_eq!(buffer[0].im.to_bits(), zero, "Imag part at index 0");
    
    // Check others
    for i in 1..n {
        assert_eq!(buffer[i].re.to_bits(), zero, "Real part at index {}", i);
        assert_eq!(buffer[i].im.to_bits(), zero, "Imag part at index {}", i);
    }
}
