#!/usr/bin/env python3
"""
Refactor RUNE ontology coordinates: add raw + normalized for each mapping.
Writes a new file `hex.rune.full.normalized.txt` alongside source.
"""
import re
import math
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'examples' / 'hex.rune.full.txt'
OUT = Path(__file__).resolve().parents[1] / 'examples' / 'hex.rune.full.normalized.txt'

VECTOR_RE = re.compile(r"^(\s+)([A-Za-z0-9_]+):\s*\[(.*?)\]\s*(#.*)?$")

EPS = 1e-12


def normalize(values):
    s = sum(x * x for x in values)
    if s <= EPS:
        return values
    n = math.sqrt(s)
    return [x / n for x in values]


with SRC.open('r', encoding='utf-8') as f:
    lines = f.readlines()

out_lines = []

in_coords_block = False
# Coordinates block starts with a line that contains 'Coordinates:' (exact)
for line in lines:
    if not in_coords_block and 'Coordinates:' in line:
        in_coords_block = True
        out_lines.append(line)
        continue
    if in_coords_block and line.strip().startswith('T:') and 'Type' not in line:
        # It's a type header inside Coordinates, we keep it
        out_lines.append(line)
        continue
    if in_coords_block and line.strip().startswith('Ontology:Palette'):
        # End of Coordinates block
        in_coords_block = False
        out_lines.append(line)
        continue

    if in_coords_block:
        m = VECTOR_RE.match(line)
        if m:
            indent, label, vec_text, comment = m.groups()
            try:
                nums = [float(x.strip()) for x in vec_text.split(',') if x.strip()]
            except Exception:
                # Not a vector line we can parse, copy through
                out_lines.append(line)
                continue
            normed = normalize(nums)
            # Format floats consistently; keep up to 4 decimals, but preserve integers
            def fmt(x):
                if abs(x - round(x)) < 1e-10:
                    return str(int(round(x))) + '.0'  # make it explicit float
                return f'{x:.4f}'
            raw_str = ', '.join(fmt(x) for x in nums)
            norm_str = ', '.join(fmt(x) for x in normed)
            # Build new mapping with raw + normalized object
            new_line = f"{indent}{label}: {{ raw: [{raw_str}], normalized: [{norm_str}] }}"
            if comment:
                new_line += ' ' + comment
            new_line += '\n'
            out_lines.append(new_line)
            continue
        else:
            out_lines.append(line)
            continue
    else:
        out_lines.append(line)

with OUT.open('w', encoding='utf-8') as f:
    f.writelines(out_lines)

print(f'Wrote normalized file: {OUT}')
