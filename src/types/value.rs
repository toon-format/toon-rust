use std::{
    fmt,
    ops::{Index, IndexMut},
};

use indexmap::IndexMap;

/// Numeric representation used by TOON.
///
/// # Examples
/// ```
/// use toon_format::types::Number;
///
/// let n = Number::from(42i64);
/// assert!(n.is_i64());
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum Number {
    PosInt(u64),
    NegInt(i64),
    Float(f64),
}

impl Number {
    /// Create a floating-point number when the value is finite.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::Number;
    ///
    /// let n = Number::from_f64(3.14).unwrap();
    /// assert!(n.is_f64());
    /// ```
    pub fn from_f64(f: f64) -> Option<Self> {
        if f.is_finite() {
            Some(Number::Float(f))
        } else {
            None
        }
    }

    /// Returns true if the number can be represented as `i64`.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::Number;
    ///
    /// let n = Number::from(-5i64);
    /// assert!(n.is_i64());
    /// ```
    pub fn is_i64(&self) -> bool {
        match self {
            Number::NegInt(_) => true,
            Number::PosInt(u) => *u <= i64::MAX as u64,
            Number::Float(f) => {
                let i = *f as i64;
                i as f64 == *f && i != i64::MAX
            }
        }
    }

    /// Returns true if the number can be represented as `u64`.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::Number;
    ///
    /// let n = Number::from(5u64);
    /// assert!(n.is_u64());
    /// ```
    pub fn is_u64(&self) -> bool {
        match self {
            Number::PosInt(_) => true,
            Number::NegInt(_) => false,
            Number::Float(f) => {
                let u = *f as u64;
                u as f64 == *f
            }
        }
    }

    /// Returns true if the number is stored as an `f64`.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::Number;
    ///
    /// let n = Number::from(1.5f64);
    /// assert!(n.is_f64());
    /// ```
    pub fn is_f64(&self) -> bool {
        matches!(self, Number::Float(_))
    }

    /// Return the number as `i64` when possible.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::Number;
    ///
    /// let n = Number::from(5i64);
    /// assert_eq!(n.as_i64(), Some(5));
    /// ```
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
                let i = *f as i64;
                if i as f64 == *f {
                    Some(i)
                } else {
                    None
                }
            }
        }
    }

    /// Return the number as `u64` when possible.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::Number;
    ///
    /// let n = Number::from(5u64);
    /// assert_eq!(n.as_u64(), Some(5));
    /// ```
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Number::PosInt(u) => Some(*u),
            Number::NegInt(_) => None,
            Number::Float(f) => {
                if *f >= 0.0 {
                    let u = *f as u64;
                    if u as f64 == *f {
                        Some(u)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        }
    }

    /// Return the number as `f64`.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::Number;
    ///
    /// let n = Number::from(5u64);
    /// assert_eq!(n.as_f64(), Some(5.0));
    /// ```
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Number::PosInt(u) => Some(*u as f64),
            Number::NegInt(i) => Some(*i as f64),
            Number::Float(f) => Some(*f),
        }
    }

    /// Returns true if the number has no fractional component.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::Number;
    ///
    /// let n = Number::from(2.0f64);
    /// assert!(n.is_integer());
    /// ```
    pub fn is_integer(&self) -> bool {
        match self {
            Number::PosInt(_) | Number::NegInt(_) => true,
            Number::Float(f) => f.fract() == 0.0,
        }
    }
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s_json_num = match self {
            Number::PosInt(u) => serde_json::Number::from(*u),
            Number::NegInt(i) => serde_json::Number::from(*i),
            Number::Float(fl) => {
                serde_json::Number::from_f64(*fl).unwrap_or_else(|| serde_json::Number::from(0))
            }
        };
        write!(f, "{s_json_num}")
    }
}

impl From<i8> for Number {
    fn from(n: i8) -> Self {
        Number::NegInt(n as i64)
    }
}

impl From<i16> for Number {
    fn from(n: i16) -> Self {
        Number::NegInt(n as i64)
    }
}

impl From<i32> for Number {
    fn from(n: i32) -> Self {
        Number::NegInt(n as i64)
    }
}

impl From<i64> for Number {
    fn from(n: i64) -> Self {
        if n >= 0 {
            Number::PosInt(n as u64)
        } else {
            Number::NegInt(n)
        }
    }
}

impl From<isize> for Number {
    fn from(n: isize) -> Self {
        Number::from(n as i64)
    }
}

impl From<u8> for Number {
    fn from(n: u8) -> Self {
        Number::PosInt(n as u64)
    }
}

impl From<u16> for Number {
    fn from(n: u16) -> Self {
        Number::PosInt(n as u64)
    }
}

impl From<u32> for Number {
    fn from(n: u32) -> Self {
        Number::PosInt(n as u64)
    }
}

impl From<u64> for Number {
    fn from(n: u64) -> Self {
        Number::PosInt(n)
    }
}

impl From<usize> for Number {
    fn from(n: usize) -> Self {
        Number::PosInt(n as u64)
    }
}

impl From<f32> for Number {
    fn from(n: f32) -> Self {
        Number::Float(n as f64)
    }
}

impl From<f64> for Number {
    fn from(n: f64) -> Self {
        Number::Float(n)
    }
}

/// Map type used for TOON objects.
///
/// # Examples
/// ```
/// use indexmap::IndexMap;
/// use toon_format::types::JsonValue;
///
/// let mut obj: IndexMap<String, JsonValue> = IndexMap::new();
/// obj.insert("key".to_string(), JsonValue::Null);
/// assert!(obj.contains_key("key"));
/// ```
pub type Object = IndexMap<String, JsonValue>;

/// TOON value representation.
///
/// # Examples
/// ```
/// use toon_format::types::JsonValue;
///
/// let value = JsonValue::String("hello".to_string());
/// assert!(value.is_string());
/// ```
#[derive(Clone, Debug, PartialEq, Default)]
pub enum JsonValue {
    #[default]
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Array(Vec<JsonValue>),
    Object(Object),
}

impl JsonValue {
    /// Returns true if the value is `null`.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::JsonValue;
    ///
    /// let value = JsonValue::Null;
    /// assert!(value.is_null());
    /// ```
    pub const fn is_null(&self) -> bool {
        matches!(self, JsonValue::Null)
    }

    /// Returns true if the value is a boolean.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::JsonValue;
    ///
    /// let value = JsonValue::Bool(true);
    /// assert!(value.is_bool());
    /// ```
    pub const fn is_bool(&self) -> bool {
        matches!(self, JsonValue::Bool(_))
    }

    /// Returns true if the value is a number.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::{JsonValue, Number};
    ///
    /// let value = JsonValue::Number(Number::from(1u64));
    /// assert!(value.is_number());
    /// ```
    pub const fn is_number(&self) -> bool {
        matches!(self, JsonValue::Number(_))
    }

    /// Returns true if the value is a string.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::JsonValue;
    ///
    /// let value = JsonValue::String("hi".to_string());
    /// assert!(value.is_string());
    /// ```
    pub const fn is_string(&self) -> bool {
        matches!(self, JsonValue::String(_))
    }

    /// Returns true if the value is an array.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::JsonValue;
    ///
    /// let value = JsonValue::Array(vec![JsonValue::Null]);
    /// assert!(value.is_array());
    /// ```
    pub const fn is_array(&self) -> bool {
        matches!(self, JsonValue::Array(_))
    }

    /// Returns true if the value is an object.
    ///
    /// # Examples
    /// ```
    /// use indexmap::IndexMap;
    /// use toon_format::types::JsonValue;
    ///
    /// let mut obj: IndexMap<String, JsonValue> = IndexMap::new();
    /// obj.insert("a".to_string(), JsonValue::Null);
    /// let value = JsonValue::Object(obj);
    /// assert!(value.is_object());
    /// ```
    pub const fn is_object(&self) -> bool {
        matches!(self, JsonValue::Object(_))
    }

    /// Returns true if the value is a number that can be represented as i64.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::{JsonValue, Number};
    ///
    /// let value = JsonValue::Number(Number::from(1i64));
    /// assert!(value.is_i64());
    /// ```
    pub fn is_i64(&self) -> bool {
        match self {
            JsonValue::Number(n) => n.is_i64(),
            _ => false,
        }
    }

    /// Returns true if the value is a number that can be represented as u64.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::{JsonValue, Number};
    ///
    /// let value = JsonValue::Number(Number::from(1u64));
    /// assert!(value.is_u64());
    /// ```
    pub fn is_u64(&self) -> bool {
        match self {
            JsonValue::Number(n) => n.is_u64(),
            _ => false,
        }
    }

    /// Returns true if the value is stored as a floating point number.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::{JsonValue, Number};
    ///
    /// let value = JsonValue::Number(Number::from(1.5f64));
    /// assert!(value.is_f64());
    /// ```
    pub fn is_f64(&self) -> bool {
        match self {
            JsonValue::Number(n) => n.is_f64(),
            _ => false,
        }
    }

    /// If the value is a Bool, returns the associated bool. Returns None
    /// otherwise.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::JsonValue;
    ///
    /// let value = JsonValue::Bool(true);
    /// assert_eq!(value.as_bool(), Some(true));
    /// ```
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// If the value is a number, represent it as i64 if possible. Returns None
    /// otherwise.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::{JsonValue, Number};
    ///
    /// let value = JsonValue::Number(Number::from(5i64));
    /// assert_eq!(value.as_i64(), Some(5));
    /// ```
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            JsonValue::Number(n) => n.as_i64(),
            _ => None,
        }
    }

    /// If the value is a number, represent it as u64 if possible. Returns None
    /// otherwise.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::{JsonValue, Number};
    ///
    /// let value = JsonValue::Number(Number::from(5u64));
    /// assert_eq!(value.as_u64(), Some(5));
    /// ```
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            JsonValue::Number(n) => n.as_u64(),
            _ => None,
        }
    }

    /// If the value is a number, represent it as f64 if possible. Returns None
    /// otherwise.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::{JsonValue, Number};
    ///
    /// let value = JsonValue::Number(Number::from(5u64));
    /// assert_eq!(value.as_f64(), Some(5.0));
    /// ```
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Number(n) => n.as_f64(),
            _ => None,
        }
    }

    /// If the value is a string, returns its contents.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::JsonValue;
    ///
    /// let value = JsonValue::String("hi".to_string());
    /// assert_eq!(value.as_str(), Some("hi"));
    /// ```
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// If the value is an array, returns a shared reference.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::JsonValue;
    ///
    /// let value = JsonValue::Array(vec![JsonValue::Null]);
    /// assert!(value.as_array().is_some());
    /// ```
    pub fn as_array(&self) -> Option<&Vec<JsonValue>> {
        match self {
            JsonValue::Array(arr) => Some(arr),
            _ => None,
        }
    }

    /// If the value is an array, returns a mutable reference.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::JsonValue;
    ///
    /// let mut value = JsonValue::Array(vec![JsonValue::Null]);
    /// assert!(value.as_array_mut().is_some());
    /// ```
    pub fn as_array_mut(&mut self) -> Option<&mut Vec<JsonValue>> {
        match self {
            JsonValue::Array(arr) => Some(arr),
            _ => None,
        }
    }

    /// If the value is an object, returns a shared reference.
    ///
    /// # Examples
    /// ```
    /// use indexmap::IndexMap;
    /// use toon_format::types::JsonValue;
    ///
    /// let mut obj: IndexMap<String, JsonValue> = IndexMap::new();
    /// obj.insert("a".to_string(), JsonValue::Null);
    /// let value = JsonValue::Object(obj);
    /// assert!(value.as_object().is_some());
    /// ```
    pub fn as_object(&self) -> Option<&Object> {
        match self {
            JsonValue::Object(obj) => Some(obj),
            _ => None,
        }
    }

    /// If the value is an object, returns a mutable reference.
    ///
    /// # Examples
    /// ```
    /// use indexmap::IndexMap;
    /// use toon_format::types::JsonValue;
    ///
    /// let mut obj: IndexMap<String, JsonValue> = IndexMap::new();
    /// obj.insert("a".to_string(), JsonValue::Null);
    /// let mut value = JsonValue::Object(obj);
    /// assert!(value.as_object_mut().is_some());
    /// ```
    pub fn as_object_mut(&mut self) -> Option<&mut Object> {
        match self {
            JsonValue::Object(obj) => Some(obj),
            _ => None,
        }
    }

    /// Return a value by key when the value is an object.
    ///
    /// # Examples
    /// ```
    /// use indexmap::IndexMap;
    /// use toon_format::types::JsonValue;
    ///
    /// let mut obj: IndexMap<String, JsonValue> = IndexMap::new();
    /// obj.insert("a".to_string(), JsonValue::Bool(true));
    /// let value = JsonValue::Object(obj);
    /// assert!(value.get("a").is_some());
    /// ```
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(obj) => obj.get(key),
            _ => None,
        }
    }

    /// Return a value by index when the value is an array.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::JsonValue;
    ///
    /// let value = JsonValue::Array(vec![JsonValue::Bool(true)]);
    /// assert!(value.get_index(0).is_some());
    /// ```
    pub fn get_index(&self, index: usize) -> Option<&JsonValue> {
        match self {
            JsonValue::Array(arr) => arr.get(index),
            _ => None,
        }
    }

    /// Takes the value, leaving Null in its place.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::JsonValue;
    ///
    /// let mut value = JsonValue::Bool(true);
    /// let taken = value.take();
    /// assert!(value.is_null());
    /// assert!(matches!(taken, JsonValue::Bool(true)));
    /// ```
    pub fn take(&mut self) -> JsonValue {
        std::mem::replace(self, JsonValue::Null)
    }

    /// Return the type name used in error messages.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::JsonValue;
    ///
    /// let value = JsonValue::Array(vec![]);
    /// assert_eq!(value.type_name(), "array");
    /// ```
    pub fn type_name(&self) -> &'static str {
        match self {
            JsonValue::Null => "null",
            JsonValue::Bool(_) => "boolean",
            JsonValue::Number(_) => "number",
            JsonValue::String(_) => "string",
            JsonValue::Array(_) => "array",
            JsonValue::Object(_) => "object",
        }
    }
}

impl fmt::Display for JsonValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JsonValue::Null => write!(f, "null"),
            JsonValue::Bool(b) => write!(f, "{b}"),
            JsonValue::Number(n) => write!(f, "{n}"),
            JsonValue::String(s) => write!(f, "\"{s}\""),
            JsonValue::Array(arr) => {
                write!(f, "[")?;
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "]")
            }
            JsonValue::Object(obj) => {
                write!(f, "{{")?;
                for (i, (k, v)) in obj.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "\"{k}\": {v}")?;
                }
                write!(f, "}}")
            }
        }
    }
}

impl Index<usize> for JsonValue {
    type Output = JsonValue;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            JsonValue::Array(arr) => arr.get(index).unwrap_or_else(|| {
                panic!(
                    "index {index} out of bounds for array of length {}",
                    arr.len()
                )
            }),
            _ => panic!(
                "cannot index into non-array value of type {}",
                self.type_name()
            ),
        }
    }
}

impl IndexMut<usize> for JsonValue {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self {
            JsonValue::Array(arr) => {
                let len = arr.len();
                arr.get_mut(index).unwrap_or_else(|| {
                    panic!("index {index} out of bounds for array of length {len}")
                })
            }
            _ => panic!(
                "cannot index into non-array value of type {}",
                self.type_name()
            ),
        }
    }
}

impl Index<&str> for JsonValue {
    type Output = JsonValue;

    fn index(&self, key: &str) -> &Self::Output {
        match self {
            JsonValue::Object(obj) => obj.get(key).unwrap_or_else(|| {
                panic!("key '{key}' not found in object with {} entries", obj.len())
            }),
            _ => panic!(
                "cannot index into non-object value of type {}",
                self.type_name()
            ),
        }
    }
}

impl IndexMut<&str> for JsonValue {
    fn index_mut(&mut self, key: &str) -> &mut Self::Output {
        match self {
            JsonValue::Object(obj) => {
                let len = obj.len();
                obj.get_mut(key)
                    .unwrap_or_else(|| panic!("key '{key}' not found in object with {len} entries"))
            }
            _ => panic!(
                "cannot index into non-object value of type {}",
                self.type_name()
            ),
        }
    }
}

impl Index<String> for JsonValue {
    type Output = JsonValue;

    fn index(&self, key: String) -> &Self::Output {
        self.index(key.as_str())
    }
}

impl IndexMut<String> for JsonValue {
    fn index_mut(&mut self, key: String) -> &mut Self::Output {
        self.index_mut(key.as_str())
    }
}

impl From<serde_json::Value> for JsonValue {
    fn from(value: serde_json::Value) -> Self {
        match value {
            serde_json::Value::Null => JsonValue::Null,
            serde_json::Value::Bool(b) => JsonValue::Bool(b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    JsonValue::Number(Number::from(i))
                } else if let Some(u) = n.as_u64() {
                    JsonValue::Number(Number::from(u))
                } else if let Some(f) = n.as_f64() {
                    JsonValue::Number(Number::from(f))
                } else {
                    JsonValue::Null
                }
            }
            serde_json::Value::String(s) => JsonValue::String(s),
            serde_json::Value::Array(arr) => {
                JsonValue::Array(arr.into_iter().map(JsonValue::from).collect())
            }
            serde_json::Value::Object(obj) => {
                let mut new_obj = Object::new();
                for (k, v) in obj {
                    new_obj.insert(k, JsonValue::from(v));
                }
                JsonValue::Object(new_obj)
            }
        }
    }
}

impl From<&serde_json::Value> for JsonValue {
    fn from(value: &serde_json::Value) -> Self {
        value.clone().into()
    }
}

impl From<JsonValue> for serde_json::Value {
    fn from(value: JsonValue) -> Self {
        match value {
            JsonValue::Null => serde_json::Value::Null,
            JsonValue::Bool(b) => serde_json::Value::Bool(b),
            JsonValue::Number(n) => {
                if let Some(i) = n.as_i64() {
                    serde_json::Value::Number(i.into())
                } else if let Some(u) = n.as_u64() {
                    serde_json::Value::Number(u.into())
                } else if let Some(f) = n.as_f64() {
                    serde_json::Number::from_f64(f)
                        .map(serde_json::Value::Number)
                        .unwrap_or(serde_json::Value::Null)
                } else {
                    serde_json::Value::Null
                }
            }
            JsonValue::String(s) => serde_json::Value::String(s),
            JsonValue::Array(arr) => {
                serde_json::Value::Array(arr.into_iter().map(Into::into).collect())
            }
            JsonValue::Object(obj) => {
                let mut new_obj = serde_json::Map::new();
                for (k, v) in obj {
                    new_obj.insert(k, v.into());
                }
                serde_json::Value::Object(new_obj)
            }
        }
    }
}

impl From<&JsonValue> for serde_json::Value {
    fn from(value: &JsonValue) -> Self {
        value.clone().into()
    }
}

/// Convert common value types into TOON's `JsonValue`.
///
/// # Examples
/// ```
/// use toon_format::types::{IntoJsonValue, JsonValue};
///
/// let value: JsonValue = serde_json::json!({"a": 1}).into_json_value();
/// assert!(value.is_object());
/// ```
pub trait IntoJsonValue {
    /// Convert the value into a `JsonValue`.
    ///
    /// # Examples
    /// ```
    /// use toon_format::types::{IntoJsonValue, JsonValue};
    ///
    /// let value: JsonValue = serde_json::json!({"a": 1}).into_json_value();
    /// assert!(value.is_object());
    /// ```
    fn into_json_value(self) -> JsonValue;
}

impl IntoJsonValue for &JsonValue {
    fn into_json_value(self) -> JsonValue {
        self.clone()
    }
}

impl IntoJsonValue for JsonValue {
    fn into_json_value(self) -> JsonValue {
        self
    }
}

impl IntoJsonValue for &serde_json::Value {
    fn into_json_value(self) -> JsonValue {
        self.into()
    }
}

impl IntoJsonValue for serde_json::Value {
    fn into_json_value(self) -> JsonValue {
        (&self).into()
    }
}
