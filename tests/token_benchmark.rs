/// Token counting benchmark to compare RUNE vs TOON vs JSON formats
///
/// This test calculates actual token counts using a simple tokenization approach
/// to demonstrate RUNE's efficiency with semantic prefixes and array literals.
use serde_json::json;

fn count_tokens(text: &str) -> usize {
    text.split_whitespace()
        .flat_map(|word| {
            // Split on punctuation and operators
            word.split(|c: char| {
                matches!(
                    c,
                    ',' | ':'
                        | '{'
                        | '}'
                        | '['
                        | ']'
                        | '('
                        | ')'
                        | '/'
                        | '\\'
                        | '|'
                        | '*'
                        | '+'
                        | '-'
                        | '='
                        | '>'
                        | '<'
                )
            })
            .filter(|s| !s.is_empty())
        })
        .count()
}

#[test]
fn test_rune_vs_toon_token_efficiency() {
    // Sample data: E8 continuum configuration
    let json_data = json!({
        "tensor_Gf8": 2.5,
        "vector_velocity": [1.0, 2.0, 3.0],
        "matrix_transform": [[1,0,0],[0,1,0],[0,0,1]],
        "layers": {
            "config": [
                {"id": 1, "type": "Lattice", "active": true},
                {"id": 2, "type": "Projection", "active": true},
                {"id": 3, "type": "Transform", "active": false}
            ]
        }
    });

    // JSON format (standard)
    let json_str = serde_json::to_string_pretty(&json_data).unwrap();
    let json_tokens = count_tokens(&json_str);
    let json_bytes = json_str.len();

    // JSON compact format
    let json_compact = serde_json::to_string(&json_data).unwrap();
    let json_compact_tokens = count_tokens(&json_compact);
    let json_compact_bytes = json_compact.len();

    // TOON format (traditional without semantic prefixes)
    let toon_str = r#"tensor_Gf8: 2.5
vector_velocity: [1.0, 2.0, 3.0]
matrix_transform: [[1,0,0],[0,1,0],[0,0,1]]
layers:
  config[3]{id, type, active}:
    1,Lattice,true
    2,Projection,true
    3,Transform,false"#;
    let toon_tokens = count_tokens(toon_str);
    let toon_bytes = toon_str.len();

    // RUNE format (with semantic prefixes)
    let rune_str = r#"T:Gf8: 2.5
V:velocity: [1.0, 2.0, 3.0]
M:transform: [[1,0,0],[0,1,0],[0,0,1]]
layers:
  config[3]{id, type, active}:
    1,Lattice,true
    2,Projection,true
    3,Transform,false"#;
    let rune_tokens = count_tokens(rune_str);
    let rune_bytes = rune_str.len();

    // Calculate savings
    let rune_token_savings = ((json_tokens - rune_tokens) as f64 / json_tokens as f64) * 100.0;
    let rune_byte_savings = ((json_bytes - rune_bytes) as f64 / json_bytes as f64) * 100.0;

    let toon_token_savings = ((json_tokens - toon_tokens) as f64 / json_tokens as f64) * 100.0;
    let toon_byte_savings = ((json_bytes - toon_bytes) as f64 / json_bytes as f64) * 100.0;

    println!("\n=== TOKEN BENCHMARK RESULTS ===");
    println!("\nJSON (pretty):");
    println!("  Tokens: {}", json_tokens);
    println!("  Bytes:  {}", json_bytes);

    println!("\nJSON (compact):");
    println!("  Tokens: {}", json_compact_tokens);
    println!("  Bytes:  {}", json_compact_bytes);

    println!("\nTOON:");
    println!(
        "  Tokens: {} ({:.1}% savings vs JSON)",
        toon_tokens, toon_token_savings
    );
    println!(
        "  Bytes:  {} ({:.1}% savings vs JSON)",
        toon_bytes, toon_byte_savings
    );

    println!("\nRUNE (with semantic prefixes):");
    println!(
        "  Tokens: {} ({:.1}% savings vs JSON)",
        rune_tokens, rune_token_savings
    );
    println!(
        "  Bytes:  {} ({:.1}% savings vs JSON)",
        rune_bytes, rune_byte_savings
    );

    println!("\nRUNE vs TOON:");
    println!(
        "  Token difference: {} tokens",
        rune_tokens as i32 - toon_tokens as i32
    );
    println!(
        "  Byte difference: {} bytes",
        rune_bytes as i32 - toon_bytes as i32
    );
    println!("  Semantic clarity: +{} semantic markers", 3); // T:, V:, M:

    // Verify RUNE maintains efficiency despite semantic additions
    // RUNE adds semantic clarity with minimal token overhead (~3 tokens for semantic prefixes)
    assert!(rune_token_savings > 0.0, "RUNE should save tokens vs JSON");
    assert!(
        rune_byte_savings > 60.0,
        "RUNE should save >60% bytes vs JSON"
    );

    // The semantic prefixes add ~3 tokens but provide explicit domain context
    let semantic_overhead = rune_tokens - toon_tokens;
    assert!(
        semantic_overhead <= 5,
        "Semantic prefixes should add minimal token overhead"
    );
}

#[test]
fn test_complex_rune_efficiency() {
    // More complex example with arrays and semantic operations
    let json_data = json!({
        "tensor_network": {
            "Gf8_primary": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            "velocities": [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [1.5, 1.5, 1.5]],
            "metadata": {
                "dimensions": [3, 3],
                "timestamp": 1638835200,
                "layers": [
                    {"id": 1, "name": "input", "type": "vector"},
                    {"id": 2, "name": "hidden1", "type": "tensor"},
                    {"id": 3, "name": "hidden2", "type": "tensor"},
                    {"id": 4, "name": "hidden3", "type": "tensor"},
                    {"id": 5, "name": "output", "type": "scalar"}
                ]
            }
        }
    });

    let json_str = serde_json::to_string_pretty(&json_data).unwrap();
    let json_tokens = count_tokens(&json_str);
    let json_bytes = json_str.len();

    let rune_str = r#"T:Gf8_primary: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
V:velocities: [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [1.5, 1.5, 1.5]]
metadata:
  dimensions: [3, 3]
  timestamp: 1638835200
  layers[5]{id, name, type}:
    1,input,vector
    2,hidden1,tensor
    3,hidden2,tensor
    4,hidden3,tensor
    5,output,scalar"#;

    let rune_tokens = count_tokens(rune_str);
    let rune_bytes = rune_str.len();

    let token_savings = ((json_tokens - rune_tokens) as f64 / json_tokens as f64) * 100.0;
    let byte_savings = ((json_bytes - rune_bytes) as f64 / json_bytes as f64) * 100.0;

    println!("\n=== COMPLEX STRUCTURE BENCHMARK ===");
    println!("\nJSON:");
    println!("  Tokens: {}", json_tokens);
    println!("  Bytes:  {}", json_bytes);

    println!("\nRUNE:");
    println!("  Tokens: {} ({:.1}% savings)", rune_tokens, token_savings);
    println!("  Bytes:  {} ({:.1}% savings)", rune_bytes, byte_savings);

    // RUNE's semantic prefixes add clarity with modest token overhead
    assert!(token_savings > 10.0, "Complex RUNE should save >10% tokens");
    assert!(byte_savings > 60.0, "Complex RUNE should save >60% bytes");
}
