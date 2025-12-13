![Yoshi logo](assets/YoshiLogo.png)

[![crates.io](https://img.shields.io/crates/v/yoshi-derive.svg)](https://crates.io/crates/yoshi-derive)
[![docs.rs](https://docs.rs/yoshi-derive/badge.svg)](https://docs.rs/yoshi-derive)

# yoshi-derive

Procedural macros to derive lightweight, ergonomic error types for Yoshi.

Provides `#[derive(AnyError)]` with per-variant `#[anyerror(...)]` configuration.

## Quick example

```rust
use yoshi_derive::AnyError;

#[derive(Debug, AnyError)]
pub enum YoError {
    #[anyerror("I/O failure: {0}")]
    Io(#[source] std::io::Error),

    #[anyerror(transparent, from)]
    Other(String),
}

fn test() -> Result<(), YoError> {
    Err(YoError::Io(std::io::Error::new(std::io::ErrorKind::Other, "oops")))
}
```

## Features
- Zero-dependency derive: small, no runtime cost
- Named/tuple struct and unit variants supported
- `from`, `source`, `transparent` helpers supported via attributes

## License
Dual-licensed under MIT OR Apache-2.0.
