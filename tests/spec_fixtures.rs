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

/// The spec version this crate targets, mirroring the `toon-spec` line in
/// `src/lib.rs`. A fixture tagged with a newer `minSpecVersion` means the
/// `spec` submodule has moved ahead of the implementation.
const TARGET_SPEC_VERSION: (u32, u32) = (4, 4);

#[derive(Deserialize, Debug)]
struct FixtureFile {
    tests: Vec<TestCase>,
}

/// Unknown fields are denied here as well as on `TestOptions`, and for the
/// same reason: a spec bump that adds a case-level field must fail loudly
/// rather than be dropped. `minSpecVersion` went unread for exactly that
/// reason when the fixtures gained it.
#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct TestCase {
    name: String,
    input: Value,
    expected: Value,
    #[serde(default)]
    options: TestOptions,
    #[serde(default)]
    should_error: bool,
    /// Lowest spec version whose contract this case expresses, checked
    /// against `TARGET_SPEC_VERSION`.
    min_spec_version: Option<String>,
    /// Carried by the fixtures to locate the rule a case pins; both are
    /// folded into the failure label so a red case names its rule.
    spec_section: Option<String>,
    note: Option<String>,
}

impl TestCase {
    /// Failure label carrying the spec section and any fixture note.
    fn label(&self, kind: &str, file_name: &str) -> String {
        let section = self.spec_section.as_deref().unwrap_or("?");
        match &self.note {
            Some(note) => format!("[{kind}] {file_name} (§{section}): {} - {note}", self.name),
            None => format!("[{kind}] {file_name} (§{section}): {}", self.name),
        }
    }
}

/// Parses a `MAJOR.MINOR` spec version as `minSpecVersion` spells it.
fn parse_spec_version(version: &str) -> Result<(u32, u32), String> {
    let malformed = || format!("Malformed minSpecVersion: {version:?}");
    let (major, minor) = version.split_once('.').ok_or_else(malformed)?;
    Ok((
        major.parse().map_err(|_| malformed())?,
        minor.parse().map_err(|_| malformed())?,
    ))
}

/// A case whose contract postdates `TARGET_SPEC_VERSION` is a signal to act
/// on, not a case to skip quietly: the fixtures describe a spec this crate
/// has not implemented yet.
fn check_spec_version(test: &TestCase, test_name: &str) -> Option<String> {
    let required = test.min_spec_version.as_deref()?;
    match parse_spec_version(required) {
        Err(e) => Some(format!("Test '{test_name}': {e}")),
        Ok(version) if version > TARGET_SPEC_VERSION => Some(format!(
            "Test '{test_name}' requires spec v{required}, but this crate targets v{}.{}: the \
             `spec` submodule has moved ahead of the implementation.",
            TARGET_SPEC_VERSION.0, TARGET_SPEC_VERSION.1
        )),
        Ok(_) => None,
    }
}

/// Unknown fields are denied: a spec bump that introduces a new fixture
/// option must fail loudly here rather than be silently decoded as a default,
/// which would run the case against the wrong contract.
#[derive(Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
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
        let test_name = test.label("decode", file_name);
        if let Some(failure) = check_spec_version(&test, &test_name) {
            failures.push(failure);
            continue;
        }

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
                    // `Value` equality ignores key order even with
                    // `preserve_order`; comparing the serializations also
                    // checks the key order §2 and §9.3 require.
                    if serde_json::to_string(&actual_json)?
                        != serde_json::to_string(&test.expected)?
                    {
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
        let test_name = test.label("encode", file_name);
        if let Some(failure) = check_spec_version(&test, &test_name) {
            failures.push(failure);
            continue;
        }

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
