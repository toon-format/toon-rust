#[cfg(feature = "hydron")]
use crate::rune::hydron::values::Value;
#[cfg(feature = "hydron")]
use crate::{DecodeOptions, decode};

#[cfg(not(feature = "hydron"))]
pub fn decode_strict() {
    println!("Hydron feature required for dynamic decode strict");
}

#[cfg(feature = "hydron")]
pub fn decode_strict() {
    // Malformed: header says 2 rows, but only 1 provided
    let malformed = "items[2]{id,name}:\n  1,Ada";

    let opts = DecodeOptions::new().with_strict(true);
    match decode::<Value>(malformed, &opts) {
        Ok(val) => println!("Unexpectedly decoded: {:?}", val),
        Err(err) => println!("Strict decode error: {err}"),
    }
}
