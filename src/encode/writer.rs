use crate::{
    types::{
        Delimiter,
        EncodeOptions,
        ToonResult,
    },
    utils::string::{
        is_valid_unquoted_key,
        needs_quoting,
        quote_string,
    },
};

/// Writer that builds TOON output string from JSON values.
pub struct Writer {
    buffer: String,
    options: EncodeOptions,
}

impl Writer {
    /// Create a new writer with the given options.
    pub fn new(options: EncodeOptions) -> Self {
        Self {
            buffer: String::new(),
            options,
        }
    }

    /// Finish writing and return the complete TOON string.
    pub fn finish(self) -> String {
        self.buffer
    }

    pub fn write_str(&mut self, s: &str) -> ToonResult<()> {
        self.buffer.push_str(s);
        Ok(())
    }

    pub fn write_char(&mut self, ch: char) -> ToonResult<()> {
        self.buffer.push(ch);
        Ok(())
    }

    pub fn write_newline(&mut self) -> ToonResult<()> {
        self.buffer.push('\n');
        Ok(())
    }

    pub fn write_indent(&mut self, depth: usize) -> ToonResult<()> {
        let indent_string = self.options.indent.get_string(depth);
        if !indent_string.is_empty() {
            self.buffer.push_str(&indent_string);
        }
        Ok(())
    }

    pub fn write_delimiter(&mut self) -> ToonResult<()> {
        self.buffer.push(self.options.delimiter.as_char());
        Ok(())
    }

    pub fn write_key(&mut self, key: &str) -> ToonResult<()> {
        if is_valid_unquoted_key(key) {
            self.write_str(key)
        } else {
            self.write_quoted_string(key)
        }
    }

    /// Write an array header with key, length, and optional field list.
    pub fn write_array_header(
        &mut self,
        key: Option<&str>,
        length: usize,
        fields: Option<&[String]>,
        depth: usize,
    ) -> ToonResult<()> {
        if let Some(k) = key {
            if depth > 0 {
                self.write_indent(depth)?;
            }
            self.write_key(k)?;
        }

        self.write_char('[')?;

        let length_str = self.options.format_length(length);
        self.write_str(&length_str)?;

        if self.options.delimiter != Delimiter::Comma {
            self.write_delimiter()?;
        }

        self.write_char(']')?;

        if let Some(field_list) = fields {
            self.write_char('{')?;
            for (i, field) in field_list.iter().enumerate() {
                if i > 0 {
                    self.write_delimiter()?;
                }
                self.write_key(field)?;
            }
            self.write_char('}')?;
        }

        self.write_char(':')
    }

    /// Write an empty array header.
    pub fn write_empty_array_with_key(&mut self, key: Option<&str>) -> ToonResult<()> {
        if let Some(k) = key {
            self.write_key(k)?;
        }
        self.write_char('[')?;

        let length_str = self.options.format_length(0);
        self.write_str(&length_str)?;

        if self.options.delimiter != Delimiter::Comma {
            self.write_delimiter()?;
        }

        self.write_char(']')?;
        self.write_char(':')
    }

    pub fn needs_quoting(&self, s: &str) -> bool {
        needs_quoting(s, self.options.delimiter)
    }

    pub fn write_quoted_string(&mut self, s: &str) -> ToonResult<()> {
        self.write_str(&quote_string(s))
    }

    pub fn write_value(&mut self, s: &str) -> ToonResult<()> {
        if self.needs_quoting(s) {
            self.write_quoted_string(s)
        } else {
            self.write_str(s)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_writer_basic() {
        let opts = EncodeOptions::default();
        let mut writer = Writer::new(opts);

        writer.write_str("hello").unwrap();
        writer.write_str(" ").unwrap();
        writer.write_str("world").unwrap();

        assert_eq!(writer.finish(), "hello world");
    }

    #[test]
    fn test_write_delimiter() {
        let mut opts = EncodeOptions::default();
        let mut writer = Writer::new(opts.clone());

        writer.write_str("a").unwrap();
        writer.write_delimiter().unwrap();
        writer.write_str("b").unwrap();

        assert_eq!(writer.finish(), "a,b");

        opts = opts.with_delimiter(Delimiter::Pipe);
        let mut writer = Writer::new(opts);

        writer.write_str("a").unwrap();
        writer.write_delimiter().unwrap();
        writer.write_str("b").unwrap();

        assert_eq!(writer.finish(), "a|b");
    }

    #[test]
    fn test_write_indent() {
        let opts = EncodeOptions::default();
        let mut writer = Writer::new(opts);

        writer.write_indent(0).unwrap();
        writer.write_str("a").unwrap();
        writer.write_newline().unwrap();

        writer.write_indent(1).unwrap();
        writer.write_str("b").unwrap();
        writer.write_newline().unwrap();

        writer.write_indent(2).unwrap();
        writer.write_str("c").unwrap();

        assert_eq!(writer.finish(), "a\n  b\n    c");
    }

    #[test]
    fn test_write_array_header() {
        let opts = EncodeOptions::default();
        let mut writer = Writer::new(opts);

        writer
            .write_array_header(Some("items"), 3, None, 0)
            .unwrap();
        assert_eq!(writer.finish(), "items[3]:");

        let opts = EncodeOptions::default();
        let mut writer = Writer::new(opts);
        let fields = vec!["id".to_string(), "name".to_string()];

        writer
            .write_array_header(Some("users"), 2, Some(&fields), 0)
            .unwrap();
        assert_eq!(writer.finish(), "users[2]{id,name}:");
    }

    #[test]
    fn test_write_array_header_with_length_marker() {
        let opts = EncodeOptions::new().with_length_marker('#');
        let mut writer = Writer::new(opts);

        writer
            .write_array_header(Some("items"), 3, None, 0)
            .unwrap();
        assert_eq!(writer.finish(), "items[#3]:");
    }

    #[test]
    fn test_write_array_header_with_pipe_delimiter() {
        let opts = EncodeOptions::new().with_delimiter(Delimiter::Pipe);
        let mut writer = Writer::new(opts);

        writer
            .write_array_header(Some("items"), 3, None, 0)
            .unwrap();
        assert_eq!(writer.finish(), "items[3|]:");

        let opts = EncodeOptions::new().with_delimiter(Delimiter::Pipe);
        let mut writer = Writer::new(opts);
        let fields = vec!["id".to_string(), "name".to_string()];

        writer
            .write_array_header(Some("users"), 2, Some(&fields), 0)
            .unwrap();
        assert_eq!(writer.finish(), "users[2|]{id|name}:");
    }

    #[test]
    fn test_write_key_with_special_chars() {
        let opts = EncodeOptions::default();
        let mut writer = Writer::new(opts);

        writer.write_key("normal_key").unwrap();
        assert_eq!(writer.finish(), "normal_key");

        let opts = EncodeOptions::default();
        let mut writer = Writer::new(opts);

        writer.write_key("key:with:colons").unwrap();
        assert_eq!(writer.finish(), "\"key:with:colons\"");
    }

    #[test]
    fn test_write_quoted_string() {
        let opts = EncodeOptions::default();
        let mut writer = Writer::new(opts);

        writer.write_quoted_string("hello world").unwrap();
        assert_eq!(writer.finish(), "\"hello world\"");

        let opts = EncodeOptions::default();
        let mut writer = Writer::new(opts);

        writer.write_quoted_string("say \"hi\"").unwrap();
        assert_eq!(writer.finish(), r#""say \"hi\"""#);
    }

    #[test]
    fn test_needs_quoting() {
        let opts = EncodeOptions::default();
        let writer = Writer::new(opts);

        assert!(!writer.needs_quoting("hello"));
        assert!(writer.needs_quoting("hello,world"));
        assert!(writer.needs_quoting("true"));
        assert!(writer.needs_quoting("false"));
        assert!(writer.needs_quoting("null"));
        assert!(writer.needs_quoting("123"));
        assert!(writer.needs_quoting(""));
        assert!(writer.needs_quoting("hello:world"));
    }

    #[test]
    fn test_write_empty_array() {
        let opts = EncodeOptions::default();
        let mut writer = Writer::new(opts);

        writer.write_empty_array_with_key(Some("items")).unwrap();
        assert_eq!(writer.finish(), "items[0]:");
    }
}
