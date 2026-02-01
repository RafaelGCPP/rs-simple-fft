use super::CplxFft; // Use your library name (see notice below)
use num_complex::Complex32;

fn assert_complex_close(val: Complex32, expected: Complex32) {
    let tolerance = 1e-4;
    let diff = (val - expected).l1_norm();
    assert!(
        diff < tolerance,
        "Error. Expected: {}, Got: {}", expected, val
    );
}

#[test]
fn test_fft_roundtrip() {
    let n = 8;

    let input = [
        Complex32::new(1.0, 2.0), 
        Complex32::new(3.0, 4.0),
        Complex32::new(5.0, 6.0), 
        Complex32::new(7.0, 8.0),
        Complex32::new(-8.0, -7.0), 
        Complex32::new(-6.0, -5.0),
        Complex32::new(-4.0, -3.0), 
        Complex32::new(-2.0, -1.0),
    ];

    let expected_fft = [
        Complex32::new(-4.0, 4.0),
        Complex32::new(30.72792, -12.72792),
        Complex32::new(-16.0, 0.0),
        Complex32::new(12.72792, 5.27208),
        Complex32::new(-8.0, -8.0),
        Complex32::new(5.27208, 12.72792),
        Complex32::new(0.0, -16.0),
        Complex32::new(-12.72792, 30.72792),
    ];

    let mut buffer = input.to_vec(); // to_vec to allocate on the heap
    let mut twiddles = vec![Complex32::new(0., 0.); n / 2];
    let mut bitrev = vec![0; n];

    let fft = CplxFft::new(&mut twiddles, &mut bitrev, n).unwrap();

    // 1. Forward
    fft.process(&mut buffer, false).unwrap();
    
    // 2. Check FFT output
    for (i, &val) in buffer.iter().enumerate() {
        assert_complex_close(val, expected_fft[i]);
    }

    // 3. Inverse
    fft.process(&mut buffer, true).unwrap();

    // 4. Check
    for (i, &val) in buffer.iter().enumerate() {
        assert_complex_close(val, input[i]);
    }
}