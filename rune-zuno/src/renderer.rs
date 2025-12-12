//! Renderer for the borrowed AST back into `.rune` text.

use crate::ast::{Document, Item, Value};

pub fn render(doc: &Document<'_>) -> String {
    let mut out = String::new();
    for (idx, item) in doc.items.iter().enumerate() {
        if idx > 0 {
            out.push('\n');
        }
        render_item(item, &mut out, 0);
    }
    out
}

fn render_item(item: &Item<'_>, out: &mut String, indent: usize) {
    match item {
        Item::KeyValue { key, value } => {
            indent_spaces(out, indent);
            out.push_str(key);
            out.push(':');
            out.push(' ');
            render_value(value, out, indent);
        }
        Item::Block(value) => {
            indent_spaces(out, indent);
            render_value(value, out, indent);
        }
    }
}

fn render_value(value: &Value<'_>, out: &mut String, indent: usize) {
    match value {
        Value::Str(s) => {
            out.push('"');
            out.push_str(s);
            out.push('"');
        }
        Value::Num(n) => out.push_str(n),
        Value::Array(arr) => {
            out.push('[');
            for (i, v) in arr.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                render_value(v, out, indent);
            }
            out.push(']');
        }
        Value::Object(items) => {
            out.push('{');
            if !items.is_empty() {
                out.push('\n');
                for (i, it) in items.iter().enumerate() {
                    render_item(it, out, indent + 2);
                    if i + 1 < items.len() {
                        out.push_str(",\n");
                    } else {
                        out.push('\n');
                    }
                }
                indent_spaces(out, indent);
            }
            out.push('}');
        }
    }
}

fn indent_spaces(out: &mut String, n: usize) {
    for _ in 0..n {
        out.push(' ');
    }
}
