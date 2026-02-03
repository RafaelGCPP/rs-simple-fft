// src/fixed/types.rs
/// Generic fixed-point structure based on the number of fractional bits (FRAC).
/// The internal value is stored as a signed 32-bit integer.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Fixed<const FRAC: u32>(i32);

impl<const FRAC: u32> Fixed<FRAC> {
    /// Creates a Fixed from the raw integer value (without shift).
    #[inline]
    pub const fn from_bits(bits: i32) -> Self {
        Self(bits)
    }

    /// Creates a Fixed from an integer, applying the necessary shift.
    /// E.g.: Fixed::<8>::from_int(1) will result in internal value 256.
    #[inline]
    pub fn from_int(value: i32) -> Self {
        Self(value << FRAC)
    }


    /// Converts an f64 to Fixed, applying correct rounding.
    /// Useful for initializing constants and Twiddle Factors.
    pub fn from_f64(value: f64) -> Self {
        // Multiply the float by 2^FRAC and round to the nearest integer
        let scaling_factor = (1u64 << FRAC) as f64;
        let bits = (value * scaling_factor).round() as i32;
        Self::from_bits(bits)
    }

    /// Returns the stored raw value.
    #[inline]
    pub fn to_bits(self) -> i32 {
        self.0
    }

    /// Scales the value by 0.5 (shifts right by 1).
    #[inline]
    pub fn scale_half(self) -> Self {
        Self(self.0 >> 1)
    }
}

impl<const FRAC: u32> Fixed<FRAC> {
    #[inline]
    pub fn convert<const TO_FRAC: u32>(self) -> Fixed<TO_FRAC> {
        if TO_FRAC > FRAC {
            Fixed::from_bits(self.0 << (TO_FRAC - FRAC))
        } else {
            Fixed::from_bits(self.0 >> (FRAC - TO_FRAC))
        }
    }
}

use std::ops::Add;

impl<const F1: u32, const F2: u32> Add<Fixed<F2>> for Fixed<F1> {
    type Output = Fixed<F1>;

    #[inline]
    fn add(self, rhs: Fixed<F2>) -> Self::Output {
        let rhs_converted: Fixed<F1> = rhs.convert();
        // When F1 == F2, convert is a no-op and we just add the raw values
        Fixed(self.0 + rhs_converted.0)
    }
}

use std::ops::AddAssign;

impl<const F1: u32, const F2: u32> AddAssign<Fixed<F2>> for Fixed<F1> {
    #[inline]
    fn add_assign(&mut self, rhs: Fixed<F2>) {
        // Use the convert method to match rhs scale to self scale (F1)
        let adjusted_rhs = rhs.convert::<F1>();

        // Add the raw internal value
        self.0 += adjusted_rhs.to_bits();
    }
}

use std::ops::Mul;

impl<const F1: u32, const F2: u32> Mul<Fixed<F2>> for Fixed<F1> {
    type Output = Fixed<F1>;

    #[inline]
    fn mul(self, rhs: Fixed<F2>) -> Self::Output {
        let a = self.0 as i64;
        let b = rhs.0 as i64;
        
        let product = a * b;
        
        // If F2 > 0, add 2^(F2-1) for rounding
        let rounded = if F2 > 0 {
            let offset = 1i64 << (F2 - 1);
            (product + offset) >> F2
        } else {
            product // If FRAC is 0, nothing to round
        };
        
        Fixed::from_bits(rounded as i32)
    }
}

use std::ops::MulAssign;

impl<const F1: u32, const F2: u32> MulAssign<Fixed<F2>> for Fixed<F1> {
    #[inline]
    fn mul_assign(&mut self, rhs: Fixed<F2>) {
        // Reuse the Mul logic we just created
        *self = *self * rhs;
    }
}

use std::fmt;

impl<const FRAC: u32> fmt::Display for Fixed<FRAC> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Calculate the real value: raw_value / 2^FRAC
        let val = self.0 as f64 / (1i64 << FRAC) as f64;
        // Format with desired number of decimal places
        write!(f, "{:.6}", val)
    }
}

impl<const FRAC: u32> fmt::Debug for Fixed<FRAC> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let val = self.0 as f64 / (1i64 << FRAC) as f64;
        // In Debug, show both the decimal value and the raw value in parentheses
        write!(f, "{:.6} (raw: {})", val, self.0)
    }
}

impl<const FRAC: u32> Fixed<FRAC> {
    pub fn new(bits: i32) -> Self {
        assert!(FRAC <= 31, "FRAC cannot be greater than 31 bits for i32");
        Self(bits)
    }
}

use std::ops::Sub;

impl<const F1: u32, const F2: u32> Sub<Fixed<F2>> for Fixed<F1> {
    type Output = Fixed<F1>;

    #[inline]
    fn sub(self, rhs: Fixed<F2>) -> Self::Output {
        let rhs_converted = rhs.convert::<F1>();
        Fixed::from_bits(self.0 - rhs_converted.to_bits())
    }
}


use std::ops::SubAssign;
impl<const F1: u32, const F2: u32> SubAssign<Fixed<F2>> for Fixed<F1> {
    #[inline]
    fn sub_assign(&mut self, rhs: Fixed<F2>) {
        let rhs_converted = rhs.convert::<F1>();
        self.0 -= rhs_converted.to_bits();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_same_scale() {
        let a = Fixed::<23>::from_int(10);
        let b = Fixed::<23>::from_int(5);
        assert_eq!((a + b).to_bits(), Fixed::<23>::from_int(15).to_bits());
    }

    #[test]
    fn test_sum_different_scales() {
        let a = Fixed::<16>::from_int(1); // 1.0 in Q16
        let b = Fixed::<8>::from_int(2);  // 2.0 in Q8
        let res = a + b;                  // Result should be 3.0 in Q16
        assert_eq!(res.to_bits(), 3 << 16);
    }

    #[test]
    fn test_multiplication_with_rounding() {
        // 0.5 (Q31) * 0.5 (Q31) = 0.25
        let a = Fixed::<31>::from_bits(1 << 30); 
        let b = Fixed::<31>::from_bits(1 << 30);
        let res = a * b;
        assert_eq!(res.to_bits(), 1 << 29); // 0.25 in Q31
    }

    #[test]
    fn test_mixed_precision_multiplication() {
        // 2.0 (Q16) * 0.5 (Q31) = 1.0 (Q16)
        let a = Fixed::<16>::from_int(2);
        let b = Fixed::<31>::from_bits(1 << 30);
        let res = a * b;
        assert_eq!(res, Fixed::<16>::from_int(1));
    }

    #[test]
    fn test_debug_display() {
        let val = Fixed::<23>::from_bits(1 << 22); // 0.5
        assert_eq!(format!("{}", val), "0.500000");
    }

    #[test]
    fn test_from_f64() {
        // Test conversion of 0.5 to Q23
        let val = Fixed::<23>::from_f64(0.5);
        assert_eq!(val.to_bits(), 1 << 22);

        // Test conversion of 1.0 to Q16
        let one = Fixed::<16>::from_f64(1.0);
        assert_eq!(one.to_bits(), 1 << 16);

        // Test negative value
        let neg = Fixed::<8>::from_f64(-2.5);
        let expected = Fixed::<8>::from_bits((-2.5f64 * 256.0).round() as i32);
        assert_eq!(neg.to_bits(), expected.to_bits());

        // Test rounding
        let rounded = Fixed::<16>::from_f64(1.0 / 3.0);
        let approx = rounded.to_bits() as f64 / (1 << 16) as f64;
        assert!((approx - 1.0 / 3.0).abs() < 0.0001);
    }
}