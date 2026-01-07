use std::sync::Arc;

use toon_format::types::{ErrorContext, ToonError};

#[test]
fn test_error_context_inline_rendering() {
    let ctx = ErrorContext::new("line 2")
        .with_preceding_lines(vec!["line 1".to_string()])
        .with_following_lines(vec!["line 3".to_string()])
        .with_indicator(3)
        .with_suggestion("use a colon");

    let rendered = format!("{ctx}");
    assert!(rendered.contains("line 1"));
    assert!(rendered.contains("> line 2"));
    assert!(rendered.contains("^"));
    assert!(rendered.contains("line 3"));
    assert!(rendered.contains("Suggestion: use a colon"));
}

#[test]
fn test_error_context_lazy_rendering_and_indicator() {
    let input: Arc<str> = Arc::from("one\ntwo\nthree");
    let ctx = ErrorContext::from_shared_input(Arc::clone(&input), 2, 2, 1).unwrap();

    let rendered = format!("{ctx}");
    assert!(rendered.contains("> two"));
    assert!(rendered.contains("^"));
    assert!(rendered.contains("one"));
    assert!(rendered.contains("three"));

    let inline = ctx.with_indicator(2);
    let rendered_inline = format!("{inline}");
    assert!(rendered_inline.contains("^"));
}

#[test]
fn test_error_context_invalid_line_returns_none() {
    let input: Arc<str> = Arc::from("line 1");
    assert!(ErrorContext::from_shared_input(Arc::clone(&input), 0, 1, 1).is_none());
    assert!(ErrorContext::from_shared_input(Arc::clone(&input), 2, 1, 1).is_none());
}

#[test]
fn test_toon_error_helpers() {
    let err = ToonError::invalid_char('@', 4);
    match err {
        ToonError::InvalidCharacter { char, position } => {
            assert_eq!(char, '@');
            assert_eq!(position, 4);
        }
        _ => panic!("Expected InvalidCharacter error"),
    }

    let err = ToonError::type_mismatch("string", "number");
    match err {
        ToonError::TypeMismatch { expected, found } => {
            assert_eq!(expected, "string");
            assert_eq!(found, "number");
        }
        _ => panic!("Expected TypeMismatch error"),
    }

    let ctx = ErrorContext::new("bad input");
    let err = ToonError::parse_error(1, 2, "oops").with_context(ctx.clone());
    match err {
        ToonError::ParseError {
            context: Some(context),
            ..
        } => {
            assert_eq!(context, Box::new(ctx.clone()));
        }
        _ => panic!("Expected ParseError with context"),
    }

    let err = ToonError::length_mismatch(2, 1).with_context(ctx.clone());
    match err {
        ToonError::LengthMismatch {
            context: Some(context),
            ..
        } => {
            assert_eq!(context, Box::new(ctx));
        }
        _ => panic!("Expected LengthMismatch with context"),
    }

    let err = ToonError::InvalidInput("nope".to_string());
    let untouched = err.clone().with_context(ErrorContext::new("unused"));
    assert_eq!(err, untouched);

    let err = ToonError::parse_error(1, 1, "bad").with_suggestion("fix it");
    match err {
        ToonError::ParseError {
            context: Some(context),
            ..
        } => {
            assert_eq!(context.suggestion.as_deref(), Some("fix it"));
        }
        _ => panic!("Expected ParseError with suggestion"),
    }
}
