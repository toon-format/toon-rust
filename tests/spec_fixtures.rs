use datatest_stable::Utf8Path;
use serde::Deserialize;
use serde_json::Value;
use toon_format::{
    decode,
    encode,
    DecodeOptions,
    Delimiter,
    EncodeOptions,
    Indent,
};

#[derive(Deserialize, Debug)]
struct FixtureFile {
    tests: Vec<TestCase>,
}

#[derive(Deserialize, Debug, Clone)]
struct TestCase {
    name: String,
    input: Value,
    expected: Value,
    #[serde(default)]
    options: TestOptions,
    #[serde(default, rename = "shouldError")]
    should_error: bool,
}

#[derive(Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
struct TestOptions {
    // Decode options
    strict: Option<bool>,

    // Encode options
    delimiter: Option<String>,
    indent: Option<usize>,
    length_marker: Option<String>,
}

fn test_decode_fixtures(path: &Utf8Path, contents: String) -> datatest_stable::Result<()> {
    let file_data: FixtureFile = serde_json::from_str(&contents)
        .map_err(|e| format!("Failed to parse JSON fixture [{path}]: {e}"))?;

    let file_name = path.file_stem().unwrap_or("unknown");

    for test in file_data.tests {
        let test_name = format!("[decode] {}: {}", file_name, test.name);

        let mut opts = DecodeOptions::new();
        if let Some(strict) = test.options.strict {
            opts = opts.with_strict(strict);
        }
        if let Some(indent) = test.options.indent {
            opts = opts.with_indent(Indent::Spaces(indent));
        }

        let toon_input = test
            .input
            .as_str()
            .ok_or_else(|| format!("Test '{test_name}': input field is not a string"))?;

        let result = decode(toon_input, &opts);

        if test.should_error {
            if result.is_ok() {
                return Err(format!(
                    "Test '{}' should have FAILED, but it succeeded with: {:?}",
                    test_name,
                    result.unwrap()
                )
                .into());
            }
        } else {
            let actual_json = result.map_err(|e| {
                format!("Test '{test_name}' should have SUCCEEDED, but it FAILED with: {e:?}",)
            })?;

            if actual_json != test.expected {
                return Err(format!(
                    "Test '{test_name}' succeeded, but the JSON output was incorrect.\nExpected: \
                     {:?}\nActual: {actual_json:?}",
                    test.expected,
                )
                .into());
            }
        }
    }

    Ok(())
}

fn test_encode_fixtures(path: &Utf8Path, contents: String) -> datatest_stable::Result<()> {
    let file_data: FixtureFile = serde_json::from_str(&contents)
        .map_err(|e| format!("Failed to parse JSON fixture [{path}]: {e}"))?;

    let file_name = path.file_stem().unwrap_or("unknown");

    for test in file_data.tests {
        let test_name = format!("[encode] {}: {}", file_name, test.name);

        let mut opts = EncodeOptions::new();
        if let Some(indent) = test.options.indent {
            opts = opts.with_indent(Indent::Spaces(indent));
        }
        if let Some(marker) = &test.options.length_marker {
            if let Some(c) = marker.chars().next() {
                opts = opts.with_length_marker(c);
            }
        }
        if let Some(delim_str) = &test.options.delimiter {
            let delim = match delim_str.as_str() {
                "," => Delimiter::Comma,
                "\t" => Delimiter::Tab,
                "|" => Delimiter::Pipe,
                _ => return Err(format!("Invalid delimiter in fixture: {delim_str}").into()),
            };
            opts = opts.with_delimiter(delim);
        }

        let result = encode(&test.input, &opts);

        let expected_toon = test
            .expected
            .as_str()
            .ok_or_else(|| format!("Test '{test_name}': expected field is not a string",))?;

        let encoded_toon = result.map_err(|e| {
            format!("Test '{test_name}' should have SUCCEEDED, but it FAILED with: {e:?}",)
        })?;

        let normalized_result = encoded_toon.replace("\r\n", "\n");
        let normalized_expected = expected_toon.replace("\r\n", "\n");

        if normalized_result != normalized_expected {
            return Err(format!(
                "Test '{test_name}' succeeded, but the TOON output was \
                 incorrect.\nExpected:\n{normalized_expected}\nActual:\n{normalized_result}",
            )
            .into());
        }
    }

    Ok(())
}

datatest_stable::harness! {
    { test = test_decode_fixtures, root = "spec/tests/fixtures/decode", pattern = r"^.*\.json$" },
    { test = test_encode_fixtures, root = "spec/tests/fixtures/encode", pattern = r"^.*\.json$" },
}
