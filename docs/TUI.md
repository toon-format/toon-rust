# TOON Interactive TUI

## Overview

The TOON Format includes a full-featured, interactive Terminal User Interface (TUI) powered by [Ratatui](https://github.com/ratatui/ratatui). This provides a powerful, real-time environment for working with TOON format conversions.

## Launching the TUI

```bash
# Launch interactive mode
toon --interactive

# Or use the short flag
toon -i
```

## Features

### Dual-Mode Conversion
- **Encode Mode**: JSON → TOON
- **Decode Mode**: TOON → JSON
- Toggle between modes with `Ctrl+E` or `Ctrl+M`

### Split-Screen Editor
- Real-time conversion as you type
- Syntax highlighting support
- Input panel (left) and output panel (right)
- Switch between panels with `Tab`
- Line numbers for easy navigation

### Live Statistics
- Token count comparison (JSON vs TOON)
- Byte size comparison
- Percentage savings calculation
- Real-time updates as you edit

### Interactive Settings Panel (`Ctrl+P`)
Configure encoding and decoding options on-the-fly:

**Encode Settings:**
- **Delimiter**: Comma (`,`), Tab (`\t`), or Pipe (`|`) - Press `d` to cycle
- **Indentation**: Adjust spaces with `+` / `-` keys
- **Key Folding**: Toggle with `f` - Collapses `{a:{b:1}}` → `a.b: 1`
- **Flatten Depth**: Control folding depth with `[` / `]` keys, `u` for unlimited

**Decode Settings:**
- **Strict Mode**: Toggle with `s` - Enforce spec compliance
- **Type Coercion**: Toggle with `c` - Auto-convert strings to types
- **Path Expansion**: Toggle with `p` - Expand `a.b:1` → `{"a":{"b":1}}`

### File Browser (`Ctrl+F`)
- Visual file navigation
- File type indicators (JSON, TOON, directories)
- Navigate with arrow keys
- Open files with `Enter`
- Multi-file selection with `Space`

### Conversion History (`Ctrl+H`)
- Track all conversions in the session
- View timestamps
- See token savings for each conversion
- File names and modes

### Side-by-Side Diff Viewer (`Ctrl+D`)
- Compare input and output side-by-side
- Line numbers for easy reference
- Perfect for understanding transformations

### File Operations
- `Ctrl+O` - Open file
- `Ctrl+S` - Save output to file
- `Ctrl+N` - New file (clear editor)
- Auto-detection of file types (`.json`, `.toon`)

### Clipboard Integration
- `Ctrl+K` - Copy selection to clipboard
- `Ctrl+Y` - Copy output to clipboard
- `Ctrl+V` - Paste into input
- Seamless integration with system clipboard

### Round-Trip Testing (`Ctrl+B`)
- Test conversion fidelity
- Automatically converts output back to input
- Toggles mode and re-converts
- Validates that data survives round-trip

### REPL Mode (`Ctrl+R`)
- Interactive command-line interface
- Execute commands like `encode`, `decode`, `help`
- Store variables with `let $var = data`
- View command history with Up/Down arrows
- Scrollable output
- Full-screen mode for focused work

### Theme Support (`Ctrl+T`)
- **Dark Theme** (default)
- **Light Theme**
- Toggle between themes on-the-fly

### Built-in Help (`F1`)
- Complete keyboard shortcuts reference
- Feature documentation
- Quick examples

## Keyboard Shortcuts

### Global Commands
| Shortcut | Action |
|----------|--------|
| `Ctrl+C` / `Ctrl+Q` | Quit application |
| `Ctrl+E` / `Ctrl+M` | Toggle mode (Encode ⇄ Decode) |
| `Tab` | Switch between input/output panels |
| `F1` | Show/hide help screen |
| `Esc` | Close overlays (help, settings, REPL, etc.) |

### File Operations
| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open file browser |
| `Ctrl+S` | Save output to file |
| `Ctrl+N` | New file (clear editor) |

### View Controls
| Shortcut | Action |
|----------|--------|
| `Ctrl+P` | Toggle settings panel |
| `Ctrl+F` | Toggle file browser |
| `Ctrl+H` | Toggle conversion history |
| `Ctrl+D` | Toggle diff viewer |
| `Ctrl+T` | Toggle theme (Dark/Light) |
| `Ctrl+R` | Open REPL |

### Editing & Testing
| Shortcut | Action |
|----------|--------|
| `Ctrl+K` | Copy selection to clipboard |
| `Ctrl+Y` | Copy output to clipboard |
| `Ctrl+V` | Paste into input |
| `Ctrl+L` | Clear input and output |
| `Ctrl+B` | Round-trip test |

### Settings Panel (when `Ctrl+P` is active)
| Shortcut | Action |
|----------|--------|
| `d` | Cycle delimiter (Comma → Tab → Pipe) |
| `+` / `=` | Increase indentation |
| `-` / `_` | Decrease indentation |
| `f` | Toggle key folding |
| `[` / `]` | Adjust flatten depth |
| `u` | Toggle unlimited flatten depth |
| `p` | Toggle path expansion |
| `s` | Toggle strict mode |
| `c` | Toggle type coercion |

### File Browser (when `Ctrl+F` is active)
| Shortcut | Action |
|----------|--------|
| `↑` / `↓` | Navigate files |
| `Enter` | Open file |
| `Space` | Select/deselect file |
| `Esc` | Close browser |

### REPL (when `Ctrl+R` is active)
| Shortcut | Action |
|----------|--------|
| `↑` / `↓` | Navigate command history |
| `PageUp` / `PageDown` | Scroll output |
| `Enter` | Execute command |
| `Esc` | Close REPL |

## REPL Commands

| Command | Description | Example |
|---------|-------------|---------|
| `encode <data>` | Encode JSON to TOON | `encode {"name": "Alice"}` |
| `decode <data>` | Decode TOON to JSON | `decode name: Alice` |
| `let $var = <data>` | Store variable | `let $config = {"port": 8080}` |
| `vars` | List all variables | `vars` |
| `clear` | Clear output | `clear` |
| `help` | Show help | `help` |
| `exit` | Close REPL | `exit` |

**Variable Usage:**
- Use `$_` to reference the last result
- Use `$varname` to reference stored variables
- Example: `let $data = {...}` then `encode $data`

## Use Cases

### Learning & Exploration
Perfect for understanding how TOON format works:
- Experiment with different JSON structures
- See real-time token savings
- Understand key folding and path expansion
- Learn by doing

### Development & Testing
Ideal for developers working with TOON:
- Test conversions interactively
- Debug format issues with immediate feedback
- Compare JSON and TOON side-by-side
- Validate strict mode compliance
- Round-trip testing with one keystroke

### Data Conversion
Streamlined workflow for data transformation:
- Open JSON files quickly
- Convert with live preview
- Adjust settings on-the-fly
- Save with one keystroke

### Teaching & Demos
Great for presentations and education:
- Visual, intuitive interface
- Real-time demonstrations
- Clear before/after comparisons
- Token savings visualization

## Example Workflow

1. **Launch TUI**: `toon -i`
2. **Paste JSON** into input panel (left side)
3. **See TOON output** instantly (right side)
4. **View statistics** at the bottom (tokens saved, bytes saved)
5. **Adjust settings** with `Ctrl+P` (try enabling key folding!)
6. **Save output** with `Ctrl+S`
7. **Test round-trip** with `Ctrl+B` to verify data fidelity
8. **Switch to decode mode** with `Ctrl+E` for manual testing

## Requirements

The TUI is included in the standard `toon` binary with no additional setup required. Dependencies are handled automatically by Cargo.

## Tips & Tricks

1. **Quick Start**: Use `Ctrl+V` to paste clipboard content directly
2. **Settings Memory**: Settings persist during your session
3. **Theme Preference**: Toggle theme with `Ctrl+T` for your environment
4. **Learning Mode**: Keep `F1` (Help) open while learning shortcuts
5. **Performance**: The TUI handles large JSON files efficiently with scrolling
6. **REPL Variables**: Store frequently used data in variables for quick testing
7. **Round-Trip**: Always test with `Ctrl+B` to ensure your data survives conversion

## Feedback & Contributions

Found a bug or have a feature request? Please open an issue on GitHub!


