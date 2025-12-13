/* rune-xero/src/types/value.rs */
//!▫~•◦-----------------------------‣
//! # RUNE-Xero – Zero-Copy Value
//!▫~•◦-----------------------------‣
//!
//! Defines the `Value<'a>` enum, a zero-copy AST that borrows string data
//! directly from the source input using `Cow<'a, str>`.
//!
//! Includes a `Number` type that preserves the raw text representation
//! for lossless storage and lazy parsing.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::{
    borrow::Cow,
    fmt,
    ops::{Index, IndexMut},
};
use indexmap::IndexMap;

/// A zero-copy number type that preserves the original text representation.
///
/// Instead of eagerly parsing to lossy `f64`, this stores the raw string slice.
/// Parsing happens strictly on-demand.
#[derive(Clone, Debug, PartialEq)]
pub enum Number<'a> {
    /// Zero-copy reference to the numeric string in the source.
    Raw(Cow<'a, str>),
    /// Owned integer (created programmatically).
    PosInt(u64),
    /// Owned negative integer (created programmatically).
    NegInt(i64),
    /// Owned float (created programmatically).
    Float(f64),
}

impl<'a> Number<'a> {
    /// Create a number from a string slice (zero-copy).
    pub fn from_raw(s: &'a str) -> Self {
        Number::Raw(Cow::Borrowed(s))
    }

    /// Convert to `f64` (lossy).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Number::PosInt(u) => Some(*u as f64),
            Number::NegInt(i) => Some(*i as f64),
            Number::Float(f) => Some(*f),
            Number::Raw(s) => s.parse().ok(),
        }
    }

    /// Convert to `i64`.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Number::PosInt(u) => {
                if *u <= i64::MAX as u64 {
                    Some(*u as i64)
                } else {
                    None
                }
            }
            Number::NegInt(i) => Some(*i),
            Number::Float(f) => {
                // Check if float represents an integer exactly
                if f.fract() == 0.0 && *f >= i64::MIN as f64 && *f <= i64::MAX as f64 {
                    Some(*f as i64)
                } else {
                    None
                }
            }
            Number::Raw(s) => {
                // Try parsing as integer first
                if let Ok(i) = s.parse::<i64>() {
                    return Some(i);
                }
                // Fallback: parse as float and check if integer
                let f: f64 = s.parse().ok()?;
                if f.fract() == 0.0 && f >= i64::MIN as f64 && f <= i64::MAX as f64 {
                    Some(f as i64)
                } else {
                    None
                }
            }
        }
    }

    /// Convert to `u64`.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Number::PosInt(u) => Some(*u),
            Number::NegInt(_) => None,
            Number::Float(f) => {
                if f.fract() == 0.0 && *f >= 0.0 && *f <= u64::MAX as f64 {
                    Some(*f as u64)
                } else {
                    None
                }
            }
            Number::Raw(s) => {
                if let Ok(u) = s.parse::<u64>() {
                    return Some(u);
                }
                let f: f64 = s.parse().ok()?;
                if f.fract() == 0.0 && f >= 0.0 && f <= u64::MAX as f64 {
                    Some(f as u64)
                } else {
                    None
                }
            }
        }
    }

    /// Returns true if the number represents an integer.
    pub fn is_integer(&self) -> bool {
        match self {
            Number::PosInt(_) | Number::NegInt(_) => true,
            Number::Float(f) => f.fract() == 0.0,
            Number::Raw(s) => {
                // Check if it parses as integer directly
                if s.parse::<i64>().is_ok() || s.parse::<u64>().is_ok() {
                    return true;
                }
                // Check if it parses as float with no fractional part
                if let Ok(f) = s.parse::<f64>() {
                    return f.fract() == 0.0;
                }
                false
            }
        }
    }
}

impl<'a> fmt::Display for Number<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Number::Raw(s) => f.write_str(s),
            Number::PosInt(u) => write!(f, "{u}"),
            Number::NegInt(i) => write!(f, "{i}"),
            Number::Float(n) => write!(f, "{n}"),
        }
    }
}

// Zero-copy primitive conversions
impl<'a> From<i64> for Number<'a> {
    fn from(n: i64) -> Self { 
        if n >= 0 { Number::PosInt(n as u64) } else { Number::NegInt(n) }
    }
}
impl<'a> From<u64> for Number<'a> {
    fn from(n: u64) -> Self { Number::PosInt(n) }
}
impl<'a> From<f64> for Number<'a> {
    fn from(n: f64) -> Self { Number::Float(n) }
}
impl<'a> From<&'a str> for Number<'a> {
    fn from(s: &'a str) -> Self { Number::Raw(Cow::Borrowed(s)) }
}
impl<'a> From<String> for Number<'a> {
    fn from(s: String) -> Self { Number::Raw(Cow::Owned(s)) }
}

/// Object type (preserves order).
pub type Object<'a> = IndexMap<String, Value<'a>>;

/// Zero-Copy RUNE Value.
///
/// Uses `Cow<'a, str>` to borrow strings from input where possible.
#[derive(Clone, Debug, PartialEq)]
pub enum Value<'a> {
    Null,
    Bool(bool),
    Number(Number<'a>),
    String(Cow<'a, str>),
    Array(Vec<Value<'a>>),
    Object(Object<'a>),
}

// Alias for compatibility with code expecting "JsonValue"
pub type JsonValue = Value<'static>;

impl<'a> Default for Value<'a> {
    fn default() -> Self {
        Value::Null
    }
}

impl<'a> Value<'a> {
    // Type checks
    pub const fn is_null(&self) -> bool { matches!(self, Value::Null) }
    pub const fn is_bool(&self) -> bool { matches!(self, Value::Bool(_)) }
    pub const fn is_number(&self) -> bool { matches!(self, Value::Number(_)) }
    pub const fn is_string(&self) -> bool { matches!(self, Value::String(_)) }
    pub const fn is_array(&self) -> bool { matches!(self, Value::Array(_)) }
    pub const fn is_object(&self) -> bool { matches!(self, Value::Object(_)) }

    // Zero-Copy Accessors
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s.as_ref()),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Vec<Value<'a>>> {
        match self {
            Value::Array(arr) => Some(arr),
            _ => None,
        }
    }

    pub fn as_array_mut(&mut self) -> Option<&mut Vec<Value<'a>>> {
        match self {
            Value::Array(arr) => Some(arr),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&Object<'a>> {
        match self {
            Value::Object(obj) => Some(obj),
            _ => None,
        }
    }

    pub fn as_object_mut(&mut self) -> Option<&mut Object<'a>> {
        match self {
            Value::Object(obj) => Some(obj),
            _ => None,
        }
    }

    /// Takes the value, leaving Null in its place.
    pub fn take(&mut self) -> Value<'a> {
        std::mem::replace(self, Value::Null)
    }

    /// Converts this Value<'a> into a Value<'static>, ensuring all borrowed data is owned.
    pub fn into_static(self) -> Value<'static> {
        match self {
            Value::Null => Value::Null,
            Value::Bool(b) => Value::Bool(b),
            Value::Number(n) => Value::Number(n.into_static()),
            Value::String(s) => Value::String(Cow::Owned(s.into_owned())),
            Value::Array(arr) => Value::Array(arr.into_iter().map(|v| v.into_static()).collect()),
            Value::Object(obj) => Value::Object(
                obj.into_iter()
                    .map(|(k, v)| (Cow::Owned(k.into_owned()), v.into_static()))
                    .collect(),
            ),
        }
    }
}

impl<'a> Number<'a> {
    // ... existing methods ...

    /// Converts this Number<'a> into a Number<'static>, ensuring any borrowed data is owned.
    pub fn into_static(self) -> Number<'static> {
        match self {
            Number::Raw(s) => Number::Raw(Cow::Owned(s.into_owned())),
            Number::PosInt(u) => Number::PosInt(u),
            Number::NegInt(i) => Number::NegInt(i),
            Number::Float(f) => Number::Float(f),
        }
    }
}

impl<'a> fmt::Display for Value<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => write!(f, "null"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Number(n) => write!(f, "{n}"),
            Value::String(s) => write!(f, "{:?}", s), // Debug quotes the string
            Value::Array(arr) => {
                write!(f, "[")?;
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{v}")?;
                }
                write!(f, "]")
            }
            Value::Object(obj) => {
                write!(f, "{{")?;
                for (i, (k, v)) in obj.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{:?}: {v}", k)?;
                }
                write!(f, "}}")
            }
        }
    }
}

// Indexing implementations
impl<'a> Index<usize> for Value<'a> {
    type Output = Value<'a>;
    fn index(&self, index: usize) -> &Self::Output {
        match self {
            Value::Array(arr) => &arr[index],
            _ => panic!("Index out of bounds or not an array"),
        }
    }
}

impl<'a> IndexMut<usize> for Value<'a> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self {
            Value::Array(arr) => &mut arr[index],
            _ => panic!("Index out of bounds or not an array"),
        }
    }
}

impl<'a> Index<&str> for Value<'a> {
    type Output = Value<'a>;
    fn index(&self, key: &str) -> &Self::Output {
        match self {
            Value::Object(obj) => &obj[key],
            _ => panic!("Key not found or not an object"),
        }
    }
}

impl<'a> IndexMut<&str> for Value<'a> {
    fn index_mut(&mut self, key: &str) -> &mut Self::Output {
        match self {
            Value::Object(obj) => obj.get_mut(key).expect("Key not found"),
            _ => panic!("Not an object"),
        }
    }
}

// Conversion trait for owned types
pub trait IntoValue<'a> {
    fn into_value(self) -> Value<'a>;
}

impl<'a> IntoValue<'a> for Value<'a> {
    fn into_value(self) -> Value<'a> { self }
}

impl<'a> IntoValue<'a> for &'a str {
    fn into_value(self) -> Value<'a> { Value::String(Cow::Borrowed(self)) }
}

impl<'a> IntoValue<'a> for String {
    fn into_value(self) -> Value<'a> { Value::String(Cow::Owned(self)) }
}

impl<'a> IntoValue<'a> for i32 {
    fn into_value(self) -> Value<'a> { Value::Number(Number::from(self as i64)) }
}

impl<'a> IntoValue<'a> for f64 {
    fn into_value(self) -> Value<'a> { Value::Number(Number::from(self)) }
}

impl<'a> IntoValue<'a> for bool {
    fn into_value(self) -> Value<'a> { Value::Bool(self) }
}

// Enable conversion from serde_json::Value for test compatibility
use serde_json;

impl<'a> From<serde_json::Value> for Value<'a> {
    fn from(v: serde_json::Value) -> Self {
        match v {
            serde_json::Value::Null => Value::Null,
            serde_json::Value::Bool(b) => Value::Bool(b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Value::Number(Number::from(i))
                } else if let Some(f) = n.as_f64() {
                    Value::Number(Number::from(f))
                } else {
                    Value::String(Cow::Owned(n.to_string()))
                }
            }
            serde_json::Value::String(s) => Value::String(Cow::Owned(s)),
            serde_json::Value::Array(arr) => Value::Array(
                arr.into_iter().map(Value::from).collect()
            ),
            serde_json::Value::Object(obj) => Value::Object(
                obj.into_iter().map(|(k, v)| (Cow::Owned(k), Value::from(v))).collect()
            ),
        }
    }
}
