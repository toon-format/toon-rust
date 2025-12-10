use rune_format::rune::parts::{
    arrays, arrays_of_arrays, decode_strict, delimiters, empty_and_root, mixed_arrays, objects,
    round_trip, structs, tabular,
};

fn main() {
    println!("=== R-Toon Consolidated Examples ===\n");

    println!("-- array examples --");
    arrays::arrays();

    println!("\n-- arrays of arrays --");
    arrays_of_arrays::arrays_of_arrays();

    println!("\n-- objects --");
    objects::objects();

    println!("\n-- delimiters --");
    delimiters::delimiters();

    println!("\n-- mixed arrays --");
    mixed_arrays::mixed_arrays();

    println!("\n-- round trip --");
    round_trip::round_trip();

    println!("\n-- tabular --");
    tabular::tabular();

    println!("\n-- empty and root --");
    empty_and_root::empty_and_root();

    println!("\n-- decode strict --");
    decode_strict::decode_strict();

    println!("\n-- serde structs --");
    structs::serde_structs();

    println!("\n=== Examples Complete ===");
}
