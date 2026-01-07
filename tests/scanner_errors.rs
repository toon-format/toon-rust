use toon_format::decode::scanner::{Scanner, Token};
use toon_format::ToonError;

#[test]
fn test_tabs_in_indentation_rejected() {
    let mut scanner = Scanner::new("\tkey: value");
    let err = scanner.scan_token().unwrap_err();
    assert!(err
        .to_string()
        .contains("Tabs are not allowed in indentation"));
}

#[test]
fn test_scan_quoted_string_invalid_escape() {
    let mut scanner = Scanner::new(r#""bad\x""#);
    let err = scanner.scan_token().unwrap_err();
    assert!(err.to_string().contains("Invalid escape sequence"));
}

#[test]
fn test_scan_quoted_string_unterminated() {
    let mut scanner = Scanner::new("\"unterminated");
    let err = scanner.scan_token().unwrap_err();
    assert!(matches!(err, ToonError::UnexpectedEof));
}

#[test]
fn test_parse_value_string_invalid_escape() {
    let scanner = Scanner::new("");
    let err = scanner.parse_value_string(r#""bad\x""#).unwrap_err();
    assert!(err.to_string().contains("Invalid escape sequence"));
}

#[test]
fn test_parse_value_string_unexpected_trailing_chars() {
    let scanner = Scanner::new("");
    let err = scanner
        .parse_value_string(r#""hello" trailing"#)
        .unwrap_err();
    assert!(err
        .to_string()
        .contains("Unexpected characters after closing quote"));
}

#[test]
fn test_parse_value_string_unterminated() {
    let scanner = Scanner::new("");
    let err = scanner.parse_value_string(r#""missing"#).unwrap_err();
    assert!(err.to_string().contains("Unterminated string"));
}

#[test]
fn test_scan_number_leading_zero_string() {
    let mut scanner = Scanner::new("05");
    assert_eq!(
        scanner.scan_token().unwrap(),
        Token::String("05".to_string(), false)
    );
}

#[test]
fn test_scan_number_trailing_char_string() {
    let mut scanner = Scanner::new("1x");
    assert_eq!(
        scanner.scan_token().unwrap(),
        Token::String("1".to_string(), false)
    );
}
