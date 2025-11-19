use std::{
    fs,
    io::{
        self,
        Read,
        Write,
    },
    path::{
        Path,
        PathBuf,
    },
};

use anyhow::{
    bail,
    Context,
    Result,
};
use clap::Parser;
use comfy_table::Table;
use serde::Serialize;
use tiktoken_rs::cl100k_base;
use toon_format::{
    decode,
    encode,
    types::{
        DecodeOptions,
        Delimiter,
        EncodeOptions,
        Indent,
        KeyFoldingMode,
        PathExpansionMode,
    },
};

#[derive(Parser, Debug)]
#[command(
    name = "toon",
    version = env!("CARGO_PKG_VERSION"),
    author = env!("CARGO_PKG_AUTHORS"),
    about = "Encode JSON to TOON or decode TOON to JSON",
    long_about = "TOON Format CLI - Token-efficient JSON alternative for LLMs

EXAMPLES:
  toon --interactive                       # Launch interactive TUI
  toon -i                                  # Short flag for interactive mode
  
  toon input.json -o output.toon
  toon input.toon --json-indent 2
  cat data.json | toon -e --stats
  toon input.json --delimiter pipe
  toon input.toon -d --no-coerce
  
  toon input.json --fold-keys              # Collapse {a:{b:1}} to a.b: 1
  toon input.json --fold-keys --flatten-depth 2
  toon input.toon --expand-paths           # Expand a.b:1 to {\"a\":{\"b\":1}}",
    disable_help_subcommand = true
)]
struct Cli {
    input: Option<String>,

    #[arg(short, long, help = "Launch interactive TUI mode")]
    interactive: bool,

    #[arg(short, long, help = "Output file path")]
    output: Option<PathBuf>,

    #[arg(short, long, help = "Force encode mode (JSON → TOON)")]
    encode: bool,

    #[arg(short, long, help = "Force decode mode (TOON → JSON)")]
    decode: bool,

    #[arg(long, help = "Show token count and savings")]
    stats: bool,

    #[arg(long, value_parser = parse_delimiter, help = "Delimiter: comma, tab, or pipe")]
    delimiter: Option<Delimiter>,

    #[arg(short, long, value_parser = parse_indent, help = "Indentation spaces")]
    indent: Option<usize>,

    #[arg(long, help = "Disable strict validation (decode)")]
    no_strict: bool,

    #[arg(long, help = "Disable type coercion (decode)")]
    no_coerce: bool,

    #[arg(long, help = "Indent output JSON with N spaces")]
    json_indent: Option<usize>,

    #[arg(
        long,
        help = "Enable key folding (encode): collapse {a:{b:1}} → a.b: 1"
    )]
    fold_keys: bool,

    #[arg(long, help = "Max depth for key folding (default: unlimited)")]
    flatten_depth: Option<usize>,

    #[arg(
        long,
        help = "Enable path expansion (decode): expand a.b:1 → {\"a\":{\"b\":1}}"
    )]
    expand_paths: bool,
}

#[derive(Debug, PartialEq)]
enum Operation {
    Encode,
    Decode,
}

fn parse_indent(s: &str) -> Result<usize, String> {
    s.parse::<usize>()
        .map_err(|_| format!("'{s}' is not a valid number"))
}

fn parse_delimiter(s: &str) -> Result<Delimiter, String> {
    match s.to_lowercase().as_str() {
        "comma" | "," => Ok(Delimiter::Comma),
        "tab" | "\t" => Ok(Delimiter::Tab),
        "pipe" | "|" => Ok(Delimiter::Pipe),
        _ => Err(format!(
            "'{s}' is not a valid delimiter. Use 'comma', 'tab', or 'pipe'",
        )),
    }
}

fn get_input(file_arg: Option<&str>) -> Result<String> {
    let mut input_str = String::new();
    let mut reader: Box<dyn Read> = match file_arg {
        Some(path_str) if path_str != "-" => {
            let path = Path::new(path_str);
            Box::new(
                fs::File::open(path)
                    .with_context(|| format!("Failed to open: {}", path.display()))?,
            )
        }
        _ => Box::new(io::stdin()),
    };
    reader
        .read_to_string(&mut input_str)
        .context("Failed to read input")?;
    Ok(input_str)
}

fn write_output(output_path: Option<PathBuf>, content: &str) -> Result<()> {
    let mut writer: Box<dyn Write> = match output_path {
        Some(path) => Box::new(
            fs::File::create(&path)
                .with_context(|| format!("Failed to create: {}", path.display()))?,
        ),
        None => Box::new(io::stdout()),
    };
    writer
        .write_all(content.as_bytes())
        .context("Failed to write output")?;
    Ok(())
}

fn run_encode(cli: &Cli, input: &str) -> Result<()> {
    if input.trim().is_empty() {
        bail!("Input is empty. Provide JSON data via file or stdin");
    }

    let json_value: serde_json::Value =
        serde_json::from_str(input).context("Failed to parse input as JSON")?;

    let mut opts = EncodeOptions::new();
    if let Some(d) = cli.delimiter {
        opts = opts.with_delimiter(d);
    }
    if let Some(i) = cli.indent {
        opts = opts.with_indent(Indent::Spaces(i));
    }

    if cli.fold_keys {
        opts = opts.with_key_folding(KeyFoldingMode::Safe);
        if let Some(depth) = cli.flatten_depth {
            opts = opts.with_flatten_depth(depth);
        }
    }

    let toon_str = encode(&json_value, &opts).context("Failed to encode to TOON")?;

    write_output(cli.output.clone(), &toon_str)?;

    if cli.output.is_none() && !toon_str.ends_with('\n') {
        io::stdout().write_all(b"\n")?;
    }

    if cli.stats {
        let json_bytes = input.len();
        let toon_bytes = toon_str.len();
        let size_savings = 100.0 * (1.0 - (toon_bytes as f64 / json_bytes as f64));

        let bpe = cl100k_base().context("Failed to load tokenizer")?;
        let json_tokens = bpe.encode_with_special_tokens(input).len();
        let toon_tokens = bpe.encode_with_special_tokens(&toon_str).len();
        let token_savings = 100.0 * (1.0 - (toon_tokens as f64 / json_tokens as f64));

        eprintln!("\nStats:");
        let mut table = Table::new();
        table.set_header(vec!["Metric", "JSON", "TOON", "Savings"]);

        table.add_row(vec![
            "Tokens",
            &json_tokens.to_string(),
            &toon_tokens.to_string(),
            &format!("{token_savings:.2}%"),
        ]);

        table.add_row(vec![
            "Size (bytes)",
            &json_bytes.to_string(),
            &toon_bytes.to_string(),
            &format!("{size_savings:.2}%"),
        ]);

        eprintln!("\n{table}\n");
    }

    Ok(())
}

fn run_decode(cli: &Cli, input: &str) -> Result<()> {
    if input.trim().is_empty() {
        write_output(cli.output.clone(), "{}\n")?;
        return Ok(());
    }

    let mut opts = DecodeOptions::new();
    if cli.no_strict {
        opts = opts.with_strict(false);
    }
    if cli.no_coerce {
        opts = opts.with_coerce_types(false);
    }

    if cli.expand_paths {
        opts = opts.with_expand_paths(PathExpansionMode::Safe);
    }

    let json_value: serde_json::Value = decode(input, &opts).context("Failed to decode TOON")?;

    let output_json = match cli.json_indent {
        Some(n) if n > 0 => {
            let mut buf = Vec::new();
            let indent_str = " ".repeat(n);
            let formatter = serde_json::ser::PrettyFormatter::with_indent(indent_str.as_bytes());
            let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
            json_value
                .serialize(&mut ser)
                .context("Failed to serialize JSON")?;
            String::from_utf8(buf).context("Invalid UTF-8 in JSON output")?
        }
        _ => serde_json::to_string(&json_value).context("Failed to serialize JSON")?,
    };

    write_output(cli.output.clone(), &output_json)?;
    if cli.output.is_none() && !output_json.ends_with('\n') {
        io::stdout().write_all(b"\n")?;
    }
    Ok(())
}

fn determine_operation(cli: &Cli) -> Result<(Operation, bool)> {
    let mut from_stdin = false;
    let mut operation: Option<Operation> = None;

    if cli.interactive {
        bail!("Interactive mode cannot be combined with other operations");
    }

    if cli.encode && cli.decode {
        bail!("Cannot use --encode and --decode simultaneously");
    }

    if cli.encode {
        operation = Some(Operation::Encode);
    } else if cli.decode {
        operation = Some(Operation::Decode);
    }

    let file_arg = cli.input.as_deref();

    match file_arg {
        None | Some("-") => {
            from_stdin = true;
            if operation.is_none() {
                operation = Some(Operation::Encode);
            }
        }
        Some(path_str) => {
            if operation.is_none() {
                let path = Path::new(path_str);
                let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
                match ext {
                    "json" => operation = Some(Operation::Encode),
                    "toon" => operation = Some(Operation::Decode),
                    _ => bail!(
                        "Cannot auto-detect operation for file: {}\nUse -e to encode or -d to \
                         decode",
                        path.display()
                    ),
                }
            }
        }
    }

    Ok((operation.unwrap(), from_stdin))
}

fn validate_flags(cli: &Cli, operation: &Operation) -> Result<()> {
    match operation {
        Operation::Encode => {
            if cli.no_strict {
                bail!("--no-strict is only valid for decode mode");
            }
            if cli.no_coerce {
                bail!("--no-coerce is only valid for decode mode");
            }
            if cli.json_indent.is_some() {
                bail!("--json-indent is only valid for decode mode");
            }
            if cli.expand_paths {
                bail!("--expand-paths is only valid for decode mode");
            }
        }
        Operation::Decode => {
            if cli.delimiter.is_some() {
                bail!("--delimiter is only valid for encode mode");
            }
            if cli.stats {
                bail!("--stats is only valid for encode mode");
            }
            if cli.indent.is_some() {
                bail!("--indent is only valid for encode mode");
            }
            if cli.fold_keys {
                bail!("--fold-keys is only valid for encode mode");
            }
            if cli.flatten_depth.is_some() {
                bail!("--flatten-depth is only valid for encode mode (use with --fold-keys)");
            }
        }
    }

    // Additional validation: flatten-depth requires fold-keys
    if cli.flatten_depth.is_some() && !cli.fold_keys {
        bail!("--flatten-depth requires --fold-keys to be enabled");
    }

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Check if interactive mode is requested
    if cli.interactive {
        return run_interactive();
    }

    let (operation, from_stdin) = determine_operation(&cli)?;
    validate_flags(&cli, &operation)?;

    let input = get_input(cli.input.as_deref()).with_context(|| {
        if from_stdin {
            "Failed to read from stdin"
        } else {
            "Failed to read input file"
        }
    })?;

    match operation {
        Operation::Encode => run_encode(&cli, &input)?,
        Operation::Decode => run_decode(&cli, &input)?,
    }

    Ok(())
}

fn run_interactive() -> Result<()> {
    toon_format::tui::run().context("Failed to run interactive TUI")?;
    Ok(())
}
