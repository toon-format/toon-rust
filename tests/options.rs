use toon_format::types::{DecodeOptions, EncodeOptions, Indent, KeyFoldingMode, PathExpansionMode};
use toon_format::Delimiter;

#[test]
fn test_indent_helpers() {
    let indent = Indent::Spaces(2);
    assert_eq!(indent.get_string(0), "");
    assert_eq!(indent.get_string(3).len(), 6);
    assert_eq!(indent.get_spaces(), 2);

    let indent = Indent::Spaces(0);
    assert_eq!(indent.get_string(2), "");
}

#[test]
fn test_encode_options_setters() {
    let opts = EncodeOptions::new()
        .with_delimiter(Delimiter::Pipe)
        .with_key_folding(KeyFoldingMode::Safe)
        .with_flatten_depth(2)
        .with_spaces(4);

    assert_eq!(opts.delimiter, Delimiter::Pipe);
    assert_eq!(opts.key_folding, KeyFoldingMode::Safe);
    assert_eq!(opts.flatten_depth, 2);
    assert_eq!(opts.indent, Indent::Spaces(4));
}

#[test]
fn test_decode_options_setters() {
    let opts = DecodeOptions::new()
        .with_strict(false)
        .with_delimiter(Delimiter::Pipe)
        .with_coerce_types(false)
        .with_indent(Indent::Spaces(4))
        .with_expand_paths(PathExpansionMode::Safe);

    assert!(!opts.strict);
    assert_eq!(opts.delimiter, Some(Delimiter::Pipe));
    assert!(!opts.coerce_types);
    assert_eq!(opts.indent, Indent::Spaces(4));
    assert_eq!(opts.expand_paths, PathExpansionMode::Safe);
}
