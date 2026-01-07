# CLI Usage

The `toon` binary encodes JSON to TOON or decodes TOON to JSON.

## Input Selection and Modes

- `--interactive` / `-i` launches the TUI (requires the `tui` feature).
- `--encode` / `-e` forces JSON -> TOON.
- `--decode` / `-d` forces TOON -> JSON.
- If no mode is provided and the input path ends with `.json`, encode.
- If no mode is provided and the input path ends with `.toon`, decode.
- If reading from stdin (no input path or `-`), encode by default.

## Output Behavior

- Use `--output` / `-o` to write to a file; otherwise output goes to stdout.
- Decode writes a trailing newline to stdout if one is missing.
- Empty decode input prints `{}` and returns success.

## Flags

- `-i`, `--interactive`: Launch interactive TUI mode.
- `-o`, `--output <path>`: Output file path.
- `-e`, `--encode`: Force encode mode (JSON -> TOON).
- `-d`, `--decode`: Force decode mode (TOON -> JSON).
- `--stats`: Show token count and savings (encode only, requires `cli-stats` feature).
- `--delimiter <comma|tab|pipe>`: Set array delimiter (encode only).
- `--indent <N>`: Set TOON indentation spaces (encode only).
- `--fold-keys`: Enable key folding (encode only).
- `--flatten-depth <N>`: Limit key folding depth (requires `--fold-keys`).
- `--json-indent <N>`: Pretty-print JSON with N spaces (decode only).
- `--no-strict`: Disable strict validation (decode only).
- `--no-coerce`: Disable type coercion (decode only).
- `--expand-paths`: Enable path expansion for dotted keys (decode only).

## Examples

```bash
# Auto-detect from extension
toon data.json        # Encode
toon data.toon        # Decode

# Force mode
toon -e data.txt      # Force encode
toon -d data.txt      # Force decode

# Output file
toon input.json -o output.toon

# Pipe from stdin
cat data.json | toon
echo '{"name": "Alice"}' | toon -e

# Encode options
toon data.json --delimiter pipe
toon data.json --indent 4
toon data.json --fold-keys --flatten-depth 2
toon data.json --stats

# Decode options
toon data.toon --json-indent 2
toon data.toon --no-strict
toon data.toon --no-coerce
toon data.toon --expand-paths
```
