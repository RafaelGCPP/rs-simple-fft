use super::fixed::Fixed;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComplexFixed<const FRAC: u32> {
    pub re: Fixed<FRAC>,
    pub im: Fixed<FRAC>,
}

impl<const FRAC: u32> ComplexFixed<FRAC> {
    pub fn new(re: Fixed<FRAC>, im: Fixed<FRAC>) -> Self {
        Self { re, im }
    }

    /// Returns the complex conjugate (a - bi)
    #[inline]
    pub fn conj(self) -> Self {
        ComplexFixed {
            re: self.re,
            im: Fixed::from_bits(self.im.to_bits().saturating_neg()),
        }
    }

    /// Scales both real and imaginary parts by 0.5 (right shift by 1).
    /// Used for stage normalization in inverse FFT to avoid overflow.
    #[inline]
    pub fn scale_half(self) -> Self {
        ComplexFixed {
            re: Fixed::from_bits(self.re.to_bits() >> 1),
            im: Fixed::from_bits(self.im.to_bits() >> 1),
        }
    }
}

use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};

// Addition: ComplexFixed<F1> + ComplexFixed<F2> -> ComplexFixed<F1>
impl<const F1: u32, const F2: u32> Add<ComplexFixed<F2>> for ComplexFixed<F1> {
    type Output = ComplexFixed<F1>;

    #[inline]
    fn add(self, rhs: ComplexFixed<F2>) -> Self::Output {
        ComplexFixed {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl<const F1: u32, const F2: u32> AddAssign<ComplexFixed<F2>> for ComplexFixed<F1> {
    #[inline]
    fn add_assign(&mut self, rhs: ComplexFixed<F2>) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

// Subtraction: ComplexFixed<F1> - ComplexFixed<F2> -> ComplexFixed<F1>
impl<const F1: u32, const F2: u32> Sub<ComplexFixed<F2>> for ComplexFixed<F1> {
    type Output = ComplexFixed<F1>;

    #[inline]
    fn sub(self, rhs: ComplexFixed<F2>) -> Self::Output {
        ComplexFixed {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl<const F1: u32, const F2: u32> SubAssign<ComplexFixed<F2>> for ComplexFixed<F1> {
    #[inline]
    fn sub_assign(&mut self, rhs: ComplexFixed<F2>) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

// Multiplication: ComplexFixed<F1> * ComplexFixed<F2> -> ComplexFixed<F1>
impl<const F1: u32, const F2: u32> Mul<ComplexFixed<F2>> for ComplexFixed<F1> {
    type Output = ComplexFixed<F1>;

    #[inline]
    fn mul(self, rhs: ComplexFixed<F2>) -> Self::Output {
        // (ac - bd)
        let re = (self.re * rhs.re) - (self.im * rhs.im);
        // (ad + bc)
        let im = (self.re * rhs.im) + (self.im * rhs.re);
        
        ComplexFixed { re, im }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let re = Fixed::<16>::from_int(3);
        let im = Fixed::<16>::from_int(4);
        let c = ComplexFixed::new(re, im);
        
        assert_eq!(c.re, re);
        assert_eq!(c.im, im);
    }

    #[test]
    fn test_addition_same_precision() {
        // (1 + 2i) + (3 + 4i) = (4 + 6i)
        let a = ComplexFixed::new(
            Fixed::<16>::from_int(1),
            Fixed::<16>::from_int(2)
        );
        let b = ComplexFixed::new(
            Fixed::<16>::from_int(3),
            Fixed::<16>::from_int(4)
        );
        
        let result = a + b;
        
        assert_eq!(result.re, Fixed::<16>::from_int(4));
        assert_eq!(result.im, Fixed::<16>::from_int(6));
    }

    #[test]
    fn test_addition_mixed_precision() {
        // (1 + 2i) [Q16] + (0.5 + 0.5i) [Q31] = (1.5 + 2.5i) [Q16]
        let a = ComplexFixed::new(
            Fixed::<16>::from_int(1),
            Fixed::<16>::from_int(2)
        );
        let b = ComplexFixed::new(
            Fixed::<31>::from_bits(1 << 30), // 0.5 in Q31
            Fixed::<31>::from_bits(1 << 30)
        );
        
        let result = a + b;
        
        assert_eq!(result.re, Fixed::<16>::from_f64(1.5));
        assert_eq!(result.im, Fixed::<16>::from_f64(2.5));
    }

    #[test]
    fn test_addition_with_negative() {
        // (5 + 3i) + (-2 - 1i) = (3 + 2i)
        let a = ComplexFixed::new(
            Fixed::<16>::from_int(5),
            Fixed::<16>::from_int(3)
        );
        let b = ComplexFixed::new(
            Fixed::<16>::from_int(-2),
            Fixed::<16>::from_int(-1)
        );
        
        let result = a + b;
        
        assert_eq!(result.re, Fixed::<16>::from_int(3));
        assert_eq!(result.im, Fixed::<16>::from_int(2));
    }

    #[test]
    fn test_add_assign() {
        // Test += operator
        let mut a = ComplexFixed::new(
            Fixed::<16>::from_int(1),
            Fixed::<16>::from_int(2)
        );
        let b = ComplexFixed::new(
            Fixed::<16>::from_int(3),
            Fixed::<16>::from_int(4)
        );
        
        a += b;
        
        assert_eq!(a.re, Fixed::<16>::from_int(4));
        assert_eq!(a.im, Fixed::<16>::from_int(6));
    }

    #[test]
    fn test_multiplication_same_precision() {
        // (1 + 2i) * (3 + 4i) = (1*3 - 2*4) + (1*4 + 2*3)i = -5 + 10i
        let a = ComplexFixed::new(
            Fixed::<16>::from_int(1),
            Fixed::<16>::from_int(2)
        );
        let b = ComplexFixed::new(
            Fixed::<16>::from_int(3),
            Fixed::<16>::from_int(4)
        );
        
        let result = a * b;
        
        assert_eq!(result.re, Fixed::<16>::from_int(-5));
        assert_eq!(result.im, Fixed::<16>::from_int(10));
    }

    #[test]
    fn test_multiplication_mixed_precision() {
        // (2 + 0i) * (0.5 + 0i) = (1 + 0i)
        let a = ComplexFixed::new(
            Fixed::<16>::from_int(2),
            Fixed::<16>::from_int(0)
        );
        let b = ComplexFixed::new(
            Fixed::<31>::from_bits(1 << 30), // 0.5 in Q31
            Fixed::<31>::from_int(0)
        );
        
        let result = a * b;
        
        assert_eq!(result.re, Fixed::<16>::from_int(1));
        assert_eq!(result.im, Fixed::<16>::from_int(0));
    }

    #[test]
    fn test_multiplication_by_i() {
        // (3 + 4i) * (0 + 1i) = (0 - 4) + (3 + 0)i = -4 + 3i
        let a = ComplexFixed::new(
            Fixed::<16>::from_int(3),
            Fixed::<16>::from_int(4)
        );
        let i = ComplexFixed::new(
            Fixed::<16>::from_int(0),
            Fixed::<16>::from_int(1)
        );
        
        let result = a * i;
        
        assert_eq!(result.re, Fixed::<16>::from_int(-4));
        assert_eq!(result.im, Fixed::<16>::from_int(3));
    }

    #[test]
    fn test_multiplication_by_conjugate() {
        // (3 + 4i) * (3 - 4i) = (9 + 16) + 0i = 25 + 0i
        let a = ComplexFixed::new(
            Fixed::<16>::from_int(3),
            Fixed::<16>::from_int(4)
        );
        let conj = ComplexFixed::new(
            Fixed::<16>::from_int(3),
            Fixed::<16>::from_int(-4)
        );
        
        let result = a * conj;
        
        assert_eq!(result.re, Fixed::<16>::from_int(25));
        assert_eq!(result.im, Fixed::<16>::from_int(0));
    }

    #[test]
    fn test_fractional_values() {
        // (0.5 + 0.5i) * (0.5 + 0.5i) = (0 + 0.5i)
        let a = ComplexFixed::new(
            Fixed::<16>::from_f64(0.5),
            Fixed::<16>::from_f64(0.5)
        );
        
        let result = a * a;
        
        // re = 0.5*0.5 - 0.5*0.5 = 0.25 - 0.25 = 0
        // im = 0.5*0.5 + 0.5*0.5 = 0.25 + 0.25 = 0.5
        assert_eq!(result.re, Fixed::<16>::from_int(0));
        assert_eq!(result.im, Fixed::<16>::from_f64(0.5));
    }

    // --- Subtraction tests ---

    #[test]
    fn test_subtraction_same_precision() {
        // (5 + 7i) - (2 + 3i) = (3 + 4i)
        let a = ComplexFixed::new(
            Fixed::<16>::from_int(5),
            Fixed::<16>::from_int(7)
        );
        let b = ComplexFixed::new(
            Fixed::<16>::from_int(2),
            Fixed::<16>::from_int(3)
        );
        
        let result = a - b;
        
        assert_eq!(result.re, Fixed::<16>::from_int(3));
        assert_eq!(result.im, Fixed::<16>::from_int(4));
    }

    #[test]
    fn test_subtraction_mixed_precision() {
        // (2 + 3i) [Q16] - (0.5 + 0.5i) [Q31] = (1.5 + 2.5i) [Q16]
        let a = ComplexFixed::new(
            Fixed::<16>::from_int(2),
            Fixed::<16>::from_int(3)
        );
        let b = ComplexFixed::new(
            Fixed::<31>::from_bits(1 << 30), // 0.5 in Q31
            Fixed::<31>::from_bits(1 << 30)
        );
        
        let result = a - b;
        
        assert_eq!(result.re, Fixed::<16>::from_f64(1.5));
        assert_eq!(result.im, Fixed::<16>::from_f64(2.5));
    }

    #[test]
    fn test_subtraction_resulting_negative() {
        // (1 + 2i) - (3 + 5i) = (-2 - 3i)
        let a = ComplexFixed::new(
            Fixed::<16>::from_int(1),
            Fixed::<16>::from_int(2)
        );
        let b = ComplexFixed::new(
            Fixed::<16>::from_int(3),
            Fixed::<16>::from_int(5)
        );
        
        let result = a - b;
        
        assert_eq!(result.re, Fixed::<16>::from_int(-2));
        assert_eq!(result.im, Fixed::<16>::from_int(-3));
    }

    #[test]
    fn test_sub_assign() {
        // Test -= operator
        let mut a = ComplexFixed::new(
            Fixed::<16>::from_int(5),
            Fixed::<16>::from_int(7)
        );
        let b = ComplexFixed::new(
            Fixed::<16>::from_int(2),
            Fixed::<16>::from_int(3)
        );
        
        a -= b;
        
        assert_eq!(a.re, Fixed::<16>::from_int(3));
        assert_eq!(a.im, Fixed::<16>::from_int(4));
    }

    // --- Conjugate tests ---

    #[test]
    fn test_conj_positive() {
        // conj(3 + 4i) = (3 - 4i)
        let a = ComplexFixed::new(
            Fixed::<16>::from_int(3),
            Fixed::<16>::from_int(4)
        );
        
        let result = a.conj();
        
        assert_eq!(result.re, Fixed::<16>::from_int(3));
        assert_eq!(result.im, Fixed::<16>::from_int(-4));
    }

    #[test]
    fn test_conj_negative_imaginary() {
        // conj(2 - 5i) = (2 + 5i)
        let a = ComplexFixed::new(
            Fixed::<16>::from_int(2),
            Fixed::<16>::from_int(-5)
        );
        
        let result = a.conj();
        
        assert_eq!(result.re, Fixed::<16>::from_int(2));
        assert_eq!(result.im, Fixed::<16>::from_int(5));
    }

    #[test]
    fn test_conj_zero_imaginary() {
        // conj(7 + 0i) = (7 + 0i)
        let a = ComplexFixed::new(
            Fixed::<16>::from_int(7),
            Fixed::<16>::from_int(0)
        );
        
        let result = a.conj();
        
        assert_eq!(result.re, Fixed::<16>::from_int(7));
        assert_eq!(result.im, Fixed::<16>::from_int(0));
    }

    #[test]
    fn test_conj_fractional() {
        // conj(0.5 + 0.25i) = (0.5 - 0.25i)
        let a = ComplexFixed::new(
            Fixed::<16>::from_f64(0.5),
            Fixed::<16>::from_f64(0.25)
        );
        
        let result = a.conj();
        
        assert_eq!(result.re, Fixed::<16>::from_f64(0.5));
        assert_eq!(result.im, Fixed::<16>::from_f64(-0.25));
    }

    // --- scale_half tests ---

    #[test]
    fn test_scale_half_integer() {
        // scale_half(4 + 6i) = (2 + 3i)
        let a = ComplexFixed::new(
            Fixed::<16>::from_int(4),
            Fixed::<16>::from_int(6)
        );
        
        let result = a.scale_half();
        
        assert_eq!(result.re, Fixed::<16>::from_int(2));
        assert_eq!(result.im, Fixed::<16>::from_int(3));
    }

    #[test]
    fn test_scale_half_fractional() {
        // scale_half(1 + 1i) = (0.5 + 0.5i)
        let a = ComplexFixed::new(
            Fixed::<16>::from_int(1),
            Fixed::<16>::from_int(1)
        );
        
        let result = a.scale_half();
        
        assert_eq!(result.re, Fixed::<16>::from_f64(0.5));
        assert_eq!(result.im, Fixed::<16>::from_f64(0.5));
    }

    #[test]
    fn test_scale_half_negative() {
        // scale_half(-4 - 8i) = (-2 - 4i)
        let a = ComplexFixed::new(
            Fixed::<16>::from_int(-4),
            Fixed::<16>::from_int(-8)
        );
        
        let result = a.scale_half();
        
        assert_eq!(result.re, Fixed::<16>::from_int(-2));
        assert_eq!(result.im, Fixed::<16>::from_int(-4));
    }

    #[test]
    fn test_scale_half_twice() {
        // scale_half(scale_half(8 + 4i)) = (2 + 1i)
        let a = ComplexFixed::new(
            Fixed::<16>::from_int(8),
            Fixed::<16>::from_int(4)
        );
        
        let result = a.scale_half().scale_half();
        
        assert_eq!(result.re, Fixed::<16>::from_int(2));
        assert_eq!(result.im, Fixed::<16>::from_int(1));
    }

    #[test]
    fn test_scale_half_zero() {
        // scale_half(0 + 0i) = (0 + 0i)
        let a = ComplexFixed::new(
            Fixed::<16>::from_int(0),
            Fixed::<16>::from_int(0)
        );
        
        let result = a.scale_half();
        
        assert_eq!(result.re, Fixed::<16>::from_int(0));
        assert_eq!(result.im, Fixed::<16>::from_int(0));
    }
}
