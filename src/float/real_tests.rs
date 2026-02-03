use num_complex::Complex32;
use super::RealFft;

fn assert_float_close(val: f32, expected: f32) {
    let tolerance = 1e-4;
    let diff = (val - expected).abs();
    assert!(
        diff < tolerance,
        "Error. Expected: {}, Got: {}", expected, val
    );
}

#[test]
fn test_fft_forward() {
    let n = 16;
    let input: [f32; 16] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        -8.0,-7.0,-6.0,-5.0,-4.0,-3.0,-2.0,-1.0
    ];
    let expected_fft: [f32; 16] = [
        0.0,-8.0,
        9.0,-45.24605542913263,
        -8.0,19.31370849898476,
        9.0,-13.469451863989402,
        -8.0,8.0,
        9.0,-6.013607741273688,
        -8.0,3.313708498984761,
        9.0,-1.7902113064169214
    ];

    let mut buffer = input.to_vec();
    let mut twiddles = vec![Complex32::new(0., 0.); n ];
    let mut bitrev = vec![0; n/2];

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
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        -8.0,-7.0,-6.0,-5.0,-4.0,-3.0,-2.0,-1.0
    ];
    let mut buffer: [f32; 16] = [
        0.0,-8.0,
        9.0,-45.24605542913263,
        -8.0,19.31370849898476,
        9.0,-13.469451863989402,
        -8.0,8.0,
        9.0,-6.013607741273688,
        -8.0,3.313708498984761,
        9.0,-1.7902113064169214
    ];

    let mut twiddles = vec![Complex32::new(0., 0.); n ];
    let mut bitrev = vec![0; n/2];

    let fft = RealFft::<Complex32>::new(&mut twiddles, &mut bitrev, n).unwrap();

    fft.process(&mut buffer, true).unwrap();

    for (i, &val) in buffer.iter().enumerate() {
        assert_float_close(val, expected_input[i]);
    }
}