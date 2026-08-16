use datatest_stable::Utf8Path;
use serde::Deserialize;
use serde_json::Value;
use toon_format::{
    decode,
    encode,
    types::{
        DecodeOptions,
        Delimiter,
        EncodeOptions,
        Indent,
    },
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

    // Shared options
    indent_size: Option<usize>,
}

fn parse_delimiter(delim_str: &str) -> Result<Delimiter, String> {
    match delim_str {
        "," => Ok(Delimiter::Comma),
        "\t" => Ok(Delimiter::Tab),
        "|" => Ok(Delimiter::Pipe),
        _ => Err(format!("Invalid delimiter in fixture: {delim_str}")),
    }
}

fn report(failures: Vec<String>) -> datatest_stable::Result<()> {
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} case(s) failed:\n\n{}",
            failures.len(),
            failures.join("\n\n")
        )
        .into())
    }
}

fn test_decode_fixtures(path: &Utf8Path, contents: String) -> datatest_stable::Result<()> {
    let file_data: FixtureFile = serde_json::from_str(&contents)
        .map_err(|e| format!("Failed to parse JSON fixture [{path}]: {e}"))?;

    let file_name = path.file_stem().unwrap_or("unknown");
    let mut failures = Vec::new();

    for test in file_data.tests {
        let test_name = format!("[decode] {}: {}", file_name, test.name);

        let mut opts = DecodeOptions::new();
        if let Some(strict) = test.options.strict {
            opts = opts.with_strict(strict);
        }
        if let Some(indent_size) = test.options.indent_size {
            opts = opts.with_indent(Indent::Spaces(indent_size));
        }

        let toon_input = match test.input.as_str() {
            Some(s) => s,
            None => {
                failures.push(format!("Test '{test_name}': input field is not a string"));
                continue;
            }
        };

        let result = decode::<Value>(toon_input, &opts);

        if test.should_error {
            if let Ok(actual_json) = result {
                failures.push(format!(
                    "Test '{test_name}' should have FAILED, but it succeeded with: {actual_json:?}"
                ));
            }
        } else {
            match result {
                Err(e) => failures.push(format!(
                    "Test '{test_name}' should have SUCCEEDED, but it FAILED with: {e:?}"
                )),
                Ok(actual_json) => {
                    if actual_json != test.expected {
                        failures.push(format!(
                            "Test '{test_name}' succeeded, but the JSON output was \
                             incorrect.\nExpected: {:?}\nActual: {actual_json:?}",
                            test.expected,
                        ));
                    }
                }
            }
        }
    }

    report(failures)
}

fn test_encode_fixtures(path: &Utf8Path, contents: String) -> datatest_stable::Result<()> {
    let file_data: FixtureFile = serde_json::from_str(&contents)
        .map_err(|e| format!("Failed to parse JSON fixture [{path}]: {e}"))?;

    let file_name = path.file_stem().unwrap_or("unknown");
    let mut failures = Vec::new();

    for test in file_data.tests {
        let test_name = format!("[encode] {}: {}", file_name, test.name);

        let mut opts = EncodeOptions::new();
        if let Some(indent_size) = test.options.indent_size {
            opts = opts.with_indent(Indent::Spaces(indent_size));
        }
        if let Some(delim_str) = &test.options.delimiter {
            match parse_delimiter(delim_str) {
                Ok(delim) => opts = opts.with_delimiter(delim),
                Err(e) => {
                    failures.push(format!("Test '{test_name}': {e}"));
                    continue;
                }
            }
        }

        let result = encode(&test.input, &opts);

        if test.should_error {
            if let Ok(actual_toon) = result {
                failures.push(format!(
                    "Test '{test_name}' should have FAILED, but it succeeded with:\n{actual_toon}"
                ));
            }
            continue;
        }

        let expected_toon = match test.expected.as_str() {
            Some(s) => s,
            None => {
                failures.push(format!(
                    "Test '{test_name}': expected field is not a string"
                ));
                continue;
            }
        };

        match result {
            Err(e) => failures.push(format!(
                "Test '{test_name}' should have SUCCEEDED, but it FAILED with: {e:?}"
            )),
            Ok(encoded_toon) => {
                if encoded_toon != expected_toon {
                    failures.push(format!(
                        "Test '{test_name}' succeeded, but the TOON output was \
                         incorrect.\nExpected:\n{expected_toon}\nActual:\n{encoded_toon}",
                    ));
                }
            }
        }
    }

    report(failures)
}

datatest_stable::harness! {
    { test = test_decode_fixtures, root = "spec/tests/fixtures/decode", pattern = r"^.*\.json$" },
    { test = test_encode_fixtures, root = "spec/tests/fixtures/encode", pattern = r"^.*\.json$" },
}
