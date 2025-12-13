#!/usr/bin/env python3
"""
Markdown normalization script.

This script normalizes various markdown formatting issues:
- Converts setext headings to ATX headings
- Converts indented code blocks to fenced code blocks
- Handles doc-comment fence markers
- Ensures proper spacing around code fences
- Handles file headers with language-specific fenced blocks (optional)

Usage:
    python normalize_md.py [OPTIONS] file.md [file2.md ...]

Options:
    --debug-indented, -d    Debug indented code blocks (show first occurrence)
    --analyze-fences, -a    Analyze fence markers in the file
    --help, -h              Show this help message

Examples:
    python normalize_md.py README.md
    python normalize_md.py --analyze-fences docs/*.md
    python normalize_md.py --debug-indented myfile.md
"""

import os
import re
import sys

def usage():
    print(__doc__.strip())

def ext_to_lang(ext):
    """Map file extension to language identifier for code blocks."""
    mapping = {
        'rs': 'rust',
        'toml': 'toml',
        'txt': 'text',
        'js': 'javascript',
        'json': 'json',
        'c': 'c',
        'cpp': 'cpp',
        'h': 'c',
        'wasm': 'wasm',
        'd.ts': 'typescript',
        'ts': 'typescript',
        'md': 'markdown',
        'sh': 'bash',
        'py': 'python',
    }
    return mapping.get(ext.lower(), '')

def debug_indented(file_path):
    """Debug function to find the first indented code block line."""
    print(f"Debugging indented code blocks in {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    in_fenced = False
    for idx, line in enumerate(lines):
        if re.match(r"^(\s*)```", line):
            in_fenced = not in_fenced
        if not in_fenced and re.match(r"^(\s{4,}).*", line):
            print(f'Found indented line at {idx+1}: {repr(line[:80])}')
            # Print context
            print("Context:")
            for i in range(max(0, idx-3), min(len(lines), idx+4)):
                marker = "-->" if i == idx else "   "
                print(f"{i+1:3d}: {marker} {repr(lines[i][:120])}")
            break
    else:
        print("No indented code blocks found outside fenced blocks.")

def analyze_fences(file_path):
    """Analyze fence markers in the file."""
    print(f"Analyzing fences in {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    fence_indices = []
    for idx, ln in enumerate(lines):
        if re.match(r"^\s*```", ln):
            fence_indices.append(idx+1)

    print(f'Total fence lines: {len(fence_indices)}')

    # Show first 20 fence contexts
    for i in fence_indices[:20]:
        start = max(1, i-2)
        end = min(len(lines), i+2)
        print(f'\nFence at line {i}:')
        for j in range(start, end+1):
            if j <= len(lines):
                marker = "-->" if j == i else "   "
                print(f"{j:3d}: {marker} {repr(lines[j-1][:120])}")
    if len(fence_indices) > 20:
        print(f"\n... and {len(fence_indices) - 20} more fences")

def normalize_md(content):
    """Normalize markdown content through multiple passes."""
    lines = content.splitlines(keepends=True)

    # Pass 1: Convert doc-comment fences, setext headings, indented code blocks to fenced
    in_fenced = False

    out_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect fenced blocks start/end
        fence_match = re.match(r"^(\s*)```", line)
        if fence_match:
            in_fenced = not in_fenced
            out_lines.append(line)
            i += 1
            continue

        if not in_fenced:
            # Replace doc-comment code fence markers like '//! ```' and '/// ```' with plain '```'
            doc_fence_match = re.match(r"^(\s*)(//+\s*|//!\s*|///\s*)(```.*)$", line)
            if doc_fence_match:
                leading_space, comment_token, fence = doc_fence_match.groups()
                # If previous line is not blank, ensure blank line before fence
                if len(out_lines) > 0 and out_lines[-1].strip() != '':
                    out_lines.append('\n')
                out_lines.append(leading_space + fence + '\n')
                in_fenced = True
                i += 1
                continue

            # Convert setext headings: lookahead to next non-empty line
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                # if next line is all '=' or all '-'
                if re.match(r"^[ ]*={3,}[ ]*$", next_line):
                    # Convert current line to '## ' heading
                    new_line = re.sub(r"^#*\s*", '', line).rstrip('\n')
                    out_lines.append('## ' + new_line + '\n')
                    i += 2
                    continue
                if re.match(r"^[ ]*-{3,}[ ]*$", next_line):
                    # Convert current line to '### ' heading
                    new_line = re.sub(r"^#*\s*", '', line).rstrip('\n')
                    out_lines.append('### ' + new_line + '\n')
                    i += 2
                    continue

            # Detect indented code block (4 spaces) start
            if re.match(r"^(\s{4,}).*", line):
                # Collect block of consecutive indented lines
                code_block_lines = []
                while i < len(lines) and re.match(r"^(\s{4,}).*", lines[i]):
                    code_block_lines.append(re.sub(r"^\s{4}", '', lines[i]))
                    i += 1
                # Output fenced block; insert blank line above if needed
                if len(out_lines) > 0 and out_lines[-1].strip() != '':
                    out_lines.append('\n')
                out_lines.append('```\n')
                out_lines.extend(code_block_lines)
                out_lines.append('```\n')
                # skip normal increment
                continue

        # Default: just copy line
        out_lines.append(line)
        i += 1

    content = ''.join(out_lines)

    # Second-pass replace: Convert File: ... setext headings to ATX globally (catch missed cases)
    content = re.sub(r"(?m)^(File: .+)\r?\n^[=]{3,}\r?\n", r"## \1\n\n", content)
    content = re.sub(r"(?m)^(File: .+)\r?\n^[-]{3,}\r?\n", r"## \1\n\n", content)

    # Third pass: Ensure fenced code blocks are surrounded by blank lines and strip inlined doc-svg fences
    lines = content.splitlines(keepends=True)
    out_lines2 = []
    in_fence = False
    for idx, ln in enumerate(lines):
        striped = ln.rstrip('\n')
        if re.match(r"^\s*```", striped):
            # Ensure previous line blank
            if not in_fence and len(out_lines2) > 0 and out_lines2[-1].strip() != '':
                out_lines2.append('\n')
            out_lines2.append(ln)
            in_fence = not in_fence
            continue

        # For doc comment fence closing like '//! ```' - normalized earlier, but handle just in case
        if not in_fence and re.match(r"^\s*(//+\s*|///\s*)(```.*)", ln):
            if len(out_lines2) > 0 and out_lines2[-1].strip() != '':
                out_lines2.append('\n')
            out_lines2.append(re.sub(r"^\s*(//+\s*|///\s*)", '', ln))
            in_fence = True
            continue

        out_lines2.append(ln)

    content = ''.join(out_lines2)

    # Fourth pass: Close previous code blocks before file headers and open new fenced block with language info
    lines = content.splitlines(keepends=True)

    out_lines3 = []
    i = 0
    open_fence = False
    while i < len(lines):
        ln = lines[i]
        # Track fence toggling to keep open_fence accurate
        if re.match(r"^\s*```", ln):
            out_lines3.append(ln)
            open_fence = not open_fence
            i += 1
            continue

        # Identify headers like '## File: src\gf8.rs'
        m = re.match(r"^(##\s+File:\s+(.+))\r?\n$", ln)
        if m:
            path = m.group(2)
            # If a fence is currently open, close it before the header
            if open_fence:
                out_lines3.append('```\n')
                open_fence = False
            # Append header
            out_lines3.append(ln)
            # Ensure blank line after header
            if i+1 < len(lines) and lines[i+1].strip() != '':
                out_lines3.append('\n')
            # Add opening fence if next non-blank isn't a fence
            k = i+1
            while k < len(lines) and lines[k].strip() == '':
                k += 1
            if k < len(lines) and not re.match(r"^\s*```", lines[k]):
                # get extension
                ext = ''
                if '.' in path:
                    ext = os.path.splitext(path)[1].lstrip('.')
                lang = ext_to_lang(ext)
                if lang:
                    out_lines3.append(f'```{lang}\n')
                else:
                    out_lines3.append('```\n')
                open_fence = True
            i += 1
            continue

        out_lines3.append(ln)
        i += 1

    return ''.join(out_lines3)

def process_file(file_path, debug_indented_flag=False, analyze_fences_flag=False):
    """Process a single markdown file."""
    print(f"Processing {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    if debug_indented_flag:
        debug_indented(file_path)
        return

    if analyze_fences_flag:
        analyze_fences(file_path)
        return

    # Perform normalization
    normalized_content = normalize_md(original_content)

    if normalized_content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(normalized_content)
            print(f"Normalized {file_path}: converted setext headings to ATX, indented code blocks to fenced, and processed file headers.")
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
    else:
        print(f"{file_path} is already normalized.")

def main():
    if len(sys.argv) < 2 or '--help' in sys.argv or '-h' in sys.argv:
        usage()
        sys.exit(0)

    debug_indented_flag = '--debug-indented' in sys.argv or '-d' in sys.argv
    analyze_fences_flag = '--analyze-fences' in sys.argv or '-a' in sys.argv

    # Remove flags from arguments
    args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]

    if not args:
        print("Error: No files specified")
        usage()
        sys.exit(1)

    for file_path in args:
        if not file_path.endswith('.md'):
            print(f"Warning: {file_path} does not have .md extension. Processing anyway.")
        process_file(file_path, debug_indented_flag, analyze_fences_flag)

if __name__ == '__main__':
    main()
