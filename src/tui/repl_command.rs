//! REPL command parser with inline data support

use anyhow::{
    bail,
    Result,
};

/// Parsed REPL command with inline data
#[derive(Debug, Clone)]
pub struct ReplCommand {
    pub name: String,
    pub inline_data: Option<String>,
    pub args: Vec<String>,
}

impl ReplCommand {
    /// Parse command input, extracting inline data if present.
    ///
    /// Handles patterns like:
    /// - `encode {"data": true}` - JSON inline
    /// - `decode name: Alice` - TOON inline
    /// - `encode $var` - Variable reference
    pub fn parse(input: &str) -> Result<Self> {
        let input = input.trim();
        if input.is_empty() {
            bail!("Empty command");
        }

        let parts: Vec<&str> = input.splitn(2, ' ').collect();
        let cmd_name = parts[0].to_string();

        let (inline_data, remaining_args) = if parts.len() > 1 {
            let rest = parts[1].trim();

            // Check if input looks like data rather than flags/args
            if rest.starts_with('{')
                || rest.starts_with('"')
                || rest.starts_with('$')
                || rest.contains(':')
            {
                let data_end = if rest.starts_with('{') {
                    find_matching_brace(rest)  // Handle nested braces
                } else if rest.starts_with('$') {
                    rest.find(' ').unwrap_or(rest.len())  // Variable name
                } else {
                    rest.find(" --").unwrap_or(rest.len())  // Until flag or end
                };

                let data = rest[..data_end].trim().to_string();
                let remaining = rest[data_end..].trim();

                (
                    Some(data),
                    if remaining.is_empty() {
                        vec![]
                    } else {
                        remaining
                            .split_whitespace()
                            .map(|s| s.to_string())
                            .collect()
                    },
                )
            } else {
                (
                    None,
                    rest.split_whitespace().map(|s| s.to_string()).collect(),
                )
            }
        } else {
            (None, vec![])
        };

        Ok(ReplCommand {
            name: cmd_name,
            inline_data,
            args: remaining_args,
        })
    }

    pub fn has_flag(&self, flag: &str) -> bool {
        self.args.iter().any(|a| a == flag)
    }

    pub fn get_option(&self, option: &str) -> Option<&str> {
        self.args
            .iter()
            .position(|a| a == option)
            .and_then(|i| self.args.get(i + 1))
            .map(|s| s.as_str())
    }
}

fn find_matching_brace(s: &str) -> usize {
    let mut depth = 0;
    for (i, ch) in s.chars().enumerate() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return i + 1;
                }
            }
            _ => {}
        }
    }
    s.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inline_json() {
        let cmd = ReplCommand::parse(r#"encode {"test": true}"#).unwrap();
        assert_eq!(cmd.name, "encode");
        assert_eq!(cmd.inline_data, Some(r#"{"test": true}"#.to_string()));
    }

    #[test]
    fn test_inline_toon() {
        let cmd = ReplCommand::parse("decode name: Alice").unwrap();
        assert_eq!(cmd.name, "decode");
        assert_eq!(cmd.inline_data, Some("name: Alice".to_string()));
    }

    #[test]
    fn test_with_flags() {
        let cmd = ReplCommand::parse(r#"encode {"test": true} --fold-keys"#).unwrap();
        assert_eq!(cmd.name, "encode");
        assert!(cmd.inline_data.is_some());
        assert!(cmd.has_flag("--fold-keys"));
    }
}
