use std::{
    fmt,
    ops::{Index, IndexMut},
};

use indexmap::IndexMap;

#[derive(Clone, Debug, PartialEq)]
pub enum Number {
    PosInt(u64),
    NegInt(i64),
    Float(f64),
}

impl Number {
    pub fn from_f64(f: f64) -> Option<Self> {
        if f.is_finite() {
            Some(Number::Float(f))
        } else {
            None
        }
    }

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

    pub fn is_f64(&self) -> bool {
        matches!(self, Number::Float(_))
    }

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

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Number::PosInt(u) => Some(*u as f64),
            Number::NegInt(i) => Some(*i as f64),
            Number::Float(f) => Some(*f),
        }
    }

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

pub type Object = IndexMap<String, JsonValue>;

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
    pub const fn is_null(&self) -> bool {
        matches!(self, JsonValue::Null)
    }

    pub const fn is_bool(&self) -> bool {
        matches!(self, JsonValue::Bool(_))
    }

    pub const fn is_number(&self) -> bool {
        matches!(self, JsonValue::Number(_))
    }

    pub const fn is_string(&self) -> bool {
        matches!(self, JsonValue::String(_))
    }

    pub const fn is_array(&self) -> bool {
        matches!(self, JsonValue::Array(_))
    }

    pub const fn is_object(&self) -> bool {
        matches!(self, JsonValue::Object(_))
    }

    /// Returns true if the value is a number that can be represented as i64
    pub fn is_i64(&self) -> bool {
        match self {
            JsonValue::Number(n) => n.is_i64(),
            _ => false,
        }
    }

    /// Returns true if the value is a number that can be represented as u64
    pub fn is_u64(&self) -> bool {
        match self {
            JsonValue::Number(n) => n.is_u64(),
            _ => false,
        }
    }

    pub fn is_f64(&self) -> bool {
        match self {
            JsonValue::Number(n) => n.is_f64(),
            _ => false,
        }
    }

    /// If the value is a Bool, returns the associated bool. Returns None
    /// otherwise.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// If the value is a number, represent it as i64 if possible. Returns None
    /// otherwise.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            JsonValue::Number(n) => n.as_i64(),
            _ => None,
        }
    }

    /// If the value is a number, represent it as u64 if possible. Returns None
    /// otherwise.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            JsonValue::Number(n) => n.as_u64(),
            _ => None,
        }
    }

    /// If the value is a number, represent it as f64 if possible. Returns None
    /// otherwise.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Number(n) => n.as_f64(),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Vec<JsonValue>> {
        match self {
            JsonValue::Array(arr) => Some(arr),
            _ => None,
        }
    }

    pub fn as_array_mut(&mut self) -> Option<&mut Vec<JsonValue>> {
        match self {
            JsonValue::Array(arr) => Some(arr),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&Object> {
        match self {
            JsonValue::Object(obj) => Some(obj),
            _ => None,
        }
    }

    pub fn as_object_mut(&mut self) -> Option<&mut Object> {
        match self {
            JsonValue::Object(obj) => Some(obj),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(obj) => obj.get(key),
            _ => None,
        }
    }

    pub fn get_index(&self, index: usize) -> Option<&JsonValue> {
        match self {
            JsonValue::Array(arr) => arr.get(index),
            _ => None,
        }
    }

    /// Takes the value, leaving Null in its place.
    pub fn take(&mut self) -> JsonValue {
        std::mem::replace(self, JsonValue::Null)
    }

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

pub trait IntoJsonValue {
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
