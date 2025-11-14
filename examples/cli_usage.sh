#!/bin/bash
# TOON CLI Usage Examples
# Demonstrates various command-line options and use cases

set -e  # Exit on error

echo "=== TOON CLI Usage Examples ==="
echo ""

# 1. Basic encoding
echo "1. Basic Encoding"
echo '{"name": "Alice", "age": 30}' | toon -e
echo ""

# 2. Basic decoding
echo "2. Basic Decoding"
echo 'name: Alice
age: 30' | toon -d
echo ""

# 3. Auto-detect from file extension
echo "3. Auto-detect from extension"
echo '{"users": [{"id": 1, "name": "Alice"}]}' > /tmp/test.json
toon /tmp/test.json
rm /tmp/test.json
echo ""

# 4. Custom delimiter - Pipe
echo "4. Custom Delimiter (Pipe)"
echo '{"tags": ["a", "b", "c"]}' | toon -e --delimiter pipe
echo ""

# 5. Custom delimiter - Tab
echo "5. Custom Delimiter (Tab)"
echo '{"items": ["x", "y", "z"]}' | toon -e --delimiter tab
echo ""

# 6. Custom indentation
echo "6. Custom Indentation (4 spaces)"
echo '{"data": {"nested": {"value": 42}}}' | toon -e --indent 4
echo ""

# 7. Key folding
echo "7. Key Folding (v1.5)"
echo '{"data": {"meta": {"items": ["x", "y"]}}}' | toon -e --fold-keys
echo ""

# 8. Key folding with depth limit
echo "8. Key Folding with Depth Limit"
echo '{"a": {"b": {"c": {"d": 1}}}}' | toon -e --fold-keys --flatten-depth 2
echo ""

# 9. Path expansion
echo "9. Path Expansion (v1.5)"
echo 'a.b.c: 1
a.b.d: 2
a.e: 3' | toon -d --expand-paths
echo ""

# 10. Statistics
echo "10. Encoding with Statistics"
echo '{"data": {"meta": {"items": ["x", "y"]}}}' | toon -e --fold-keys --stats
echo ""

# 11. Pretty-print JSON output
echo "11. Pretty-print JSON"
echo 'users[2]{id,name}:
  1,Alice
  2,Bob' | toon -d --json-indent 2
echo ""

# 12. Disable type coercion
echo "12. Disable Type Coercion"
echo 'value: true
number: 123' | toon -d --no-coerce
echo ""

# 13. Round-trip with folding and expansion
echo "13. Round-trip with Key Folding and Path Expansion"
ORIGINAL='{"user": {"profile": {"name": "Alice", "age": 30}}}'
echo "Original JSON:"
echo "$ORIGINAL"
echo ""
echo "Encoded with key folding:"
TOON=$(echo "$ORIGINAL" | toon -e --fold-keys)
echo "$TOON"
echo ""
echo "Decoded with path expansion:"
echo "$TOON" | toon -d --expand-paths --json-indent 2
echo ""

# 14. Tabular array
echo "14. Tabular Array"
echo '{
  "users": [
    {"id": 1, "name": "Alice", "role": "admin"},
    {"id": 2, "name": "Bob", "role": "user"},
    {"id": 3, "name": "Carol", "role": "user"}
  ]
}' | toon -e
echo ""

# 15. Nested arrays
echo "15. Nested Arrays"
echo '{
  "matrix": [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
  ]
}' | toon -e
echo ""

# 16. Complex nested structure
echo "16. Complex Nested Structure"
echo '{
  "company": {
    "name": "Acme Corp",
    "departments": [
      {
        "name": "Engineering",
        "employees": [
          {"id": 1, "name": "Alice"},
          {"id": 2, "name": "Bob"}
        ]
      },
      {
        "name": "Sales",
        "employees": [
          {"id": 3, "name": "Carol"}
        ]
      }
    ]
  }
}' | toon -e --indent 2
echo ""

# 17. Using stdin and stdout in pipeline
echo "17. Pipeline Usage"
curl -s https://api.github.com/repos/rust-lang/rust 2>/dev/null | \
  head -20 | \
  toon -e --stats 2>/dev/null || echo "Note: Requires internet connection"
echo ""

# 18. File I/O
echo "18. File I/O"
echo '{"status": "ok", "message": "Hello from TOON"}' > /tmp/data.json
echo "Encoding file to TOON:"
toon /tmp/data.json > /tmp/data.toon
cat /tmp/data.toon
echo ""
echo "Decoding file back to JSON:"
toon /tmp/data.toon --json-indent 2
rm /tmp/data.json /tmp/data.toon
echo ""

# 19. Help and version
echo "19. Help and Version"
echo "Get help:"
toon --help | head -5
echo "..."
echo ""
echo "Get version:"
toon --version
echo ""

echo "=== Examples Complete ==="

