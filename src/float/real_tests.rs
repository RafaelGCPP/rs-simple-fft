use super::RealFft;
use crate::common::{pack_rfft_spectrum, unpack_rfft_spectrum};
use num_complex::Complex32;

fn assert_float_close(val: f32, expected: f32) {
    let tolerance = 1e-4;
    let diff = (val - expected).abs();
    assert!(
        diff < tolerance,
        "Error. Expected: {}, Got: {}",
        expected,
        val
    );
}

#[test]
fn test_fft_forward() {
    let n = 16;
    let input: [f32; 16] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
    ];
    let expected_fft: [f32; 16] = [
        0.0,
        -8.0,
        9.0,
        -45.24605542913263,
        -8.0,
        19.31370849898476,
        9.0,
        -13.469451863989402,
        -8.0,
        8.0,
        9.0,
        -6.013607741273688,
        -8.0,
        3.313708498984761,
        9.0,
        -1.7902113064169214,
    ];

    let mut buffer = input.to_vec();
    let mut twiddles = vec![Complex32::new(0., 0.); n];
    let mut bitrev = vec![0; n / 2];

    let fft = RealFft::<Complex32>::new(&mut twiddles, &mut bitrev, n).unwrap();

    fft.process(&mut buffer, false).unwrap();

    for (i, &val) in buffer.iter().enumerate() {
        assert_float_close(val, expected_fft[i]);
    }
}

#[test]
fn test_fft_reverse() {
    let n = 16;
    let expected_input: [f32; 16] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
    ];
    let mut buffer: [f32; 16] = [
        0.0,
        -8.0,
        9.0,
        -45.24605542913263,
        -8.0,
        19.31370849898476,
        9.0,
        -13.469451863989402,
        -8.0,
        8.0,
        9.0,
        -6.013607741273688,
        -8.0,
        3.313708498984761,
        9.0,
        -1.7902113064169214,
    ];

    let mut twiddles = vec![Complex32::new(0., 0.); n];
    let mut bitrev = vec![0; n / 2];

    let fft = RealFft::<Complex32>::new(&mut twiddles, &mut bitrev, n).unwrap();

    fft.process(&mut buffer, true).unwrap();

    for (i, &val) in buffer.iter().enumerate() {
        assert_float_close(val, expected_input[i]);
    }
}

#[test]
fn test_unpack_pack_spectrum_float() {
    // Simulate a packed buffer from RFFT of size 8
    // Layout:
    // [0]: DC
    // [1]: Nyquist
    // [2,3]: F1.re, F1.im
    // [4,5]: F2.re, F2.im
    // [6,7]: F3.re, F3.im
    let mut packed = [0.0f32; 8];
    packed[0] = 10.0; // DC
    packed[1] = 2.0; // Nyquist

    // F1 = 3 + 4i
    packed[2] = 3.0;
    packed[3] = 4.0;

    // F2 = 5 + 6i
    packed[4] = 5.0;
    packed[5] = 6.0;

    // F3 = 7 + 8i
    packed[6] = 7.0;
    packed[7] = 8.0;

    let mut spectrum = [Complex32::new(0.0, 0.0); 8];
    unpack_rfft_spectrum(&packed, &mut spectrum);

    // Check DC (Index 0)
    assert_float_close(spectrum[0].re, 10.0);
    assert_float_close(spectrum[0].im, 0.0);

    // Check Nyquist (Index 4)
    assert_float_close(spectrum[4].re, 2.0);
    assert_float_close(spectrum[4].im, 0.0);

    // Check F1 (Index 1) and its Conjugate (Index 7)
    assert_float_close(spectrum[1].re, 3.0);
    assert_float_close(spectrum[1].im, 4.0);
    assert_float_close(spectrum[7].re, 3.0);
    assert_float_close(spectrum[7].im, -4.0);

    // Check F2 (Index 2) and Conjugate (Index 6)
    assert_float_close(spectrum[2].re, 5.0);
    assert_float_close(spectrum[2].im, 6.0);
    assert_float_close(spectrum[6].re, 5.0);
    assert_float_close(spectrum[6].im, -6.0);

    // Check F3 (Index 3) and Conjugate (Index 5)
    assert_float_close(spectrum[3].re, 7.0);
    assert_float_close(spectrum[3].im, 8.0);
    assert_float_close(spectrum[5].re, 7.0);
    assert_float_close(spectrum[5].im, -8.0);

    // Test Round Trip (Pack back)
    let mut packed_back = [0.0f32; 8];
    pack_rfft_spectrum(&spectrum, &mut packed_back);

    for i in 0..8 {
        assert_float_close(packed_back[i], packed[i]);
    }
}
