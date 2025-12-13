/* e8/gf8/src/bitcodec/lossy.rs */
//! Lossy quantization for `Gf8` using sign-based encoding.
//!
//! # Lossy Quantization
//!
//! This module provides the original lossy quantization approach that maps arbitrary
//! `Gf8` vectors to 8-bit codes by extracting sign patterns. This is inherently lossy
//! because it discards magnitude information and can only represent 256 discrete states.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::Gf8;

/// A compact 8-bit code representing a `Gf8` direction.
///
/// This is the "E8B-like" binary form. It is a type-safe wrapper around a `u8`
/// that can be losslessly converted to and from a `Gf8` direction that follows
/// the even-parity ±1 pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct Gf8BitSig(pub u8);

impl From<u8> for Gf8BitSig {
    #[inline]
    fn from(value: u8) -> Self {
        Gf8BitSig(value)
    }
}

impl From<Gf8BitSig> for u8 {
    #[inline]
    fn from(code: Gf8BitSig) -> Self {
        code.0
    }
}

/// Converts a `u8` into an array of 8 bits, with the least significant bit at index 0.
#[inline]
pub fn bits_from_u8_le(value: u8) -> [u8; 8] {
    let mut bits = [0u8; 8];
    for (i, bit) in bits.iter_mut().enumerate() {
        *bit = (value >> i) & 0x01;
    }
    bits
}

/// Converts an array of 8 bits (0 or 1) into a single `u8`, with the bit at index 0 as the LSB.
#[inline]
pub fn bits_to_u8_le(bits: [u8; 8]) -> u8 {
    let mut value = 0u8;
    for (i, &bit) in bits.iter().enumerate() {
        if bit != 0 {
            value |= 1 << i;
        }
    }
    value
}

/// Decodes a `Gf8` direction from a compact `Gf8BitSig `.
///
/// This function expands the `u8` into its 8 bits, constructs a ±1 vector with
/// an even parity constraint, and normalizes it to create a valid `Gf8`.
#[inline]
pub fn gf8_from_code(code: Gf8BitSig) -> Gf8 {
    let bits = bits_from_u8_le(code.0);
    Gf8::from_bits_even_parity(bits)
}

/// Encodes a `Gf8` direction into a compact 8-bit `Gf8BitSig `.
///
/// This function performs a lossy quantization. It reads the sign pattern of the `Gf8`'s
/// coordinates (`< 0.0` -> bit 1, `>= 0.0` -> bit 0), enforces an even number of set bits
/// by potentially flipping the last bit, and packs the result into a `u8`.
#[inline]
pub fn gf8_to_code(gf: &Gf8) -> Gf8BitSig {
    let mut bits = [0u8; 8];
    let mut set_bits = 0u32;

    // Determine bits from the sign of each coordinate.
    for (i, &c) in gf.coords().iter().enumerate() {
        if c < 0.0 {
            bits[i] = 1;
            set_bits += 1;
        } else {
            bits[i] = 0;
        }
    }

    // Enforce even parity of set bits (1s) by flipping the last bit if necessary.
    // This ensures the bit pattern corresponds to a valid E8-like state.
    if set_bits % 2 == 1 {
        bits[7] ^= 1;
    }

    Gf8BitSig(bits_to_u8_le(bits))
}
