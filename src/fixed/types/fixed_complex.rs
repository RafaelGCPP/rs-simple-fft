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
}


use std::ops::{Add, AddAssign, Mul};

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
}
