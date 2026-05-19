use std::io::{
    Read,
    Write,
};

use serde::{
    de::{
        self,
        DeserializeSeed,
        Error as _,
        MapAccess,
        SeqAccess,
        Visitor,
    },
    Deserialize,
};
use serde_json::Value as SerdeValue;

use super::writer::Writer;
use crate::{
    types::{
        EncodeOptions,
        JsonValue,
        KeyFoldingMode,
        ToonError,
        ToonResult,
    },
    utils::{
        normalize,
        QuotingContext,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingEncodeOptions {
    pub streaming_depth: usize,
}

impl Default for StreamingEncodeOptions {
    fn default() -> Self {
        Self { streaming_depth: 2 }
    }
}

impl StreamingEncodeOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_streaming_depth(mut self, depth: usize) -> Self {
        self.streaming_depth = depth;
        self
    }
}

/// Encode JSON from a reader into a TOON `String`.
///
/// This avoids materializing the input as a single `serde_json::Value`, but it
/// still buffers the full TOON output in memory. For the lowest-memory path,
/// prefer [`encode_json_stream`].
///
/// When options such as key folding require whole-document sibling inspection,
/// this function falls back to the existing in-memory encoder to preserve
/// correctness.
pub fn encode_json_reader<R: Read>(
    reader: R,
    encode_options: &EncodeOptions,
    streaming_options: &StreamingEncodeOptions,
) -> ToonResult<String> {
    let mut output = Vec::new();
    encode_json_stream(reader, &mut output, encode_options, streaming_options)?;
    String::from_utf8(output).map_err(|e| ToonError::SerializationError(e.to_string()))
}

/// Encode JSON from a reader into a TOON `String` using default encode and
/// streaming options.
pub fn encode_json_reader_default<R: Read>(reader: R) -> ToonResult<String> {
    let encode_options = EncodeOptions::default();
    let streaming_options = StreamingEncodeOptions::default();
    encode_json_reader(reader, &encode_options, &streaming_options)
}

/// Encode JSON from a reader into TOON, writing the result to the supplied
/// writer.
///
/// This is the lowest-memory JSON streaming encode API exposed by the crate.
/// It avoids materializing the input as a single `serde_json::Value` and writes
/// output incrementally to the provided writer.
///
/// When encode options require whole-document sibling inspection, it falls back
/// to the existing in-memory encoder to preserve correctness.
pub fn encode_json_stream<R: Read, W: Write>(
    reader: R,
    mut writer: W,
    encode_options: &EncodeOptions,
    streaming_options: &StreamingEncodeOptions,
) -> ToonResult<()> {
    if requires_whole_document(encode_options) {
        return encode_via_in_memory_fallback(reader, &mut writer, encode_options);
    }

    let mut deserializer = serde_json::Deserializer::from_reader(reader);
    let seed = WriteSeed {
        sink: &mut writer,
        options: encode_options,
        stream_depth: streaming_options.streaming_depth,
        context: RenderContext::Root,
    };
    seed.deserialize(&mut deserializer)
        .map_err(|e| ToonError::SerializationError(e.to_string()))?;
    deserializer
        .end()
        .map_err(|e| ToonError::SerializationError(e.to_string()))
}

/// Encode JSON from a reader into TOON, writing to the supplied writer using
/// default encode and streaming options.
pub fn encode_json_stream_default<R: Read, W: Write>(reader: R, writer: W) -> ToonResult<()> {
    let encode_options = EncodeOptions::default();
    let streaming_options = StreamingEncodeOptions::default();
    encode_json_stream(reader, writer, &encode_options, &streaming_options)
}

fn requires_whole_document(options: &EncodeOptions) -> bool {
    options.key_folding != KeyFoldingMode::Off
}

fn encode_via_in_memory_fallback<R: Read, W: Write>(
    reader: R,
    writer: &mut W,
    options: &EncodeOptions,
) -> ToonResult<()> {
    let value: SerdeValue = serde_json::from_reader(reader)
        .map_err(|e| ToonError::SerializationError(e.to_string()))?;
    let output = super::encode_impl(&JsonValue::from(value), options)?;
    writer
        .write_all(output.as_bytes())
        .map_err(|e| ToonError::SerializationError(e.to_string()))
}

#[derive(Debug, Clone)]
enum RenderContext {
    Root,
    ObjectField { key: String, depth: usize },
}

struct WriteSeed<'a, W: Write + ?Sized> {
    sink: &'a mut W,
    options: &'a EncodeOptions,
    stream_depth: usize,
    context: RenderContext,
}

impl<'de, W: Write + ?Sized> DeserializeSeed<'de> for WriteSeed<'_, W> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        if self.stream_depth == 0 {
            let value = SerdeValue::deserialize(deserializer)?;
            let value = normalize(JsonValue::from(value));
            return write_rendered_value(self.sink, &value, self.options, &self.context)
                .map_err(D::Error::custom);
        }

        deserializer.deserialize_any(WriteVisitor {
            sink: self.sink,
            options: self.options,
            stream_depth: self.stream_depth,
            context: self.context,
        })
    }
}

struct WriteVisitor<'a, W: Write + ?Sized> {
    sink: &'a mut W,
    options: &'a EncodeOptions,
    stream_depth: usize,
    context: RenderContext,
}

impl<'de, W: Write + ?Sized> Visitor<'de> for WriteVisitor<'_, W> {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_map<A>(self, map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let child_depth = self.stream_depth.saturating_sub(1);
        match self.context {
            RenderContext::Root => {
                write_root_object_from_map(map, self.sink, self.options, child_depth)
            }
            RenderContext::ObjectField { key, depth } => {
                write_object_field_from_map(map, self.sink, self.options, key, depth, child_depth)
            }
        }
    }

    fn visit_seq<A>(self, seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        stream_array_from_seq_to_sink(seq, self.sink, self.options, self.context)
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        write_rendered_value(
            self.sink,
            &JsonValue::Bool(value),
            self.options,
            &self.context,
        )
        .map_err(E::custom)
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        write_rendered_value(
            self.sink,
            &JsonValue::from(SerdeValue::from(value)),
            self.options,
            &self.context,
        )
        .map_err(E::custom)
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        write_rendered_value(
            self.sink,
            &JsonValue::from(SerdeValue::from(value)),
            self.options,
            &self.context,
        )
        .map_err(E::custom)
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        let json_number = serde_json::Number::from_f64(value)
            .ok_or_else(|| E::custom(format!("invalid JSON number: {value}")))?;
        write_rendered_value(
            self.sink,
            &normalize(JsonValue::from(SerdeValue::Number(json_number))),
            self.options,
            &self.context,
        )
        .map_err(E::custom)
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        write_rendered_value(
            self.sink,
            &JsonValue::String(value.to_string()),
            self.options,
            &self.context,
        )
        .map_err(E::custom)
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        write_rendered_value(
            self.sink,
            &JsonValue::String(value),
            self.options,
            &self.context,
        )
        .map_err(E::custom)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        write_rendered_value(self.sink, &JsonValue::Null, self.options, &self.context)
            .map_err(E::custom)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        write_rendered_value(self.sink, &JsonValue::Null, self.options, &self.context)
            .map_err(E::custom)
    }
}

fn write_root_object_from_map<'de, A, W>(
    mut map: A,
    sink: &mut W,
    options: &EncodeOptions,
    stream_depth: usize,
) -> Result<(), A::Error>
where
    A: MapAccess<'de>,
    W: Write + ?Sized,
{
    let mut first = true;

    while let Some(key) = map.next_key::<String>()? {
        if !first {
            write_str(sink, "\n").map_err(A::Error::custom)?;
        }
        first = false;

        map.next_value_seed(WriteSeed {
            sink,
            options,
            stream_depth,
            context: RenderContext::ObjectField { key, depth: 0 },
        })?;
    }

    Ok(())
}

fn write_object_field_from_map<'de, A, W>(
    mut map: A,
    sink: &mut W,
    options: &EncodeOptions,
    key: String,
    depth: usize,
    stream_depth: usize,
) -> Result<(), A::Error>
where
    A: MapAccess<'de>,
    W: Write + ?Sized,
{
    let Some(first_key) = map.next_key::<String>()? else {
        write_rendered_value(
            sink,
            &JsonValue::Object(indexmap::IndexMap::new()),
            options,
            &RenderContext::ObjectField { key, depth },
        )
        .map_err(A::Error::custom)?;
        return Ok(());
    };

    write_str(sink, &options.indent.get_string(depth)).map_err(A::Error::custom)?;
    write_str(sink, &render_key(&key, options).map_err(A::Error::custom)?)
        .map_err(A::Error::custom)?;
    write_str(sink, ":\n").map_err(A::Error::custom)?;

    map.next_value_seed(WriteSeed {
        sink,
        options,
        stream_depth,
        context: RenderContext::ObjectField {
            key: first_key,
            depth: depth + 1,
        },
    })?;

    while let Some(next_key) = map.next_key::<String>()? {
        write_str(sink, "\n").map_err(A::Error::custom)?;
        map.next_value_seed(WriteSeed {
            sink,
            options,
            stream_depth,
            context: RenderContext::ObjectField {
                key: next_key,
                depth: depth + 1,
            },
        })?;
    }

    Ok(())
}

fn write_rendered_value<W: Write + ?Sized>(
    sink: &mut W,
    value: &JsonValue,
    options: &EncodeOptions,
    context: &RenderContext,
) -> ToonResult<()> {
    let rendered = render_value_in_context(value, options, context)?;
    write_str(sink, &rendered)
}

fn write_str<W: Write + ?Sized>(sink: &mut W, value: &str) -> ToonResult<()> {
    sink.write_all(value.as_bytes())
        .map_err(|e| ToonError::SerializationError(e.to_string()))
}

fn stream_array_from_seq_to_sink<'de, A, W>(
    mut seq: A,
    sink: &mut W,
    options: &EncodeOptions,
    context: RenderContext,
) -> Result<(), A::Error>
where
    A: SeqAccess<'de>,
    W: Write + ?Sized,
{
    let array_depth = match &context {
        RenderContext::Root => 0,
        RenderContext::ObjectField { depth, .. } => *depth,
    };

    let mut encoder = StreamArrayEncoder::new(options, array_depth);
    while let Some(value) = seq.next_element::<SerdeValue>()? {
        encoder
            .push(normalize(JsonValue::from(value)))
            .map_err(A::Error::custom)?;
    }

    let array = encoder.render().map_err(A::Error::custom)?;
    match context {
        RenderContext::Root => write_str(sink, &array).map_err(A::Error::custom),
        RenderContext::ObjectField { key, depth } => {
            let prefixed = prefix_first_line(
                &format!(
                    "{}{}",
                    options.indent.get_string(depth),
                    render_key(&key, options).map_err(A::Error::custom)?
                ),
                &array,
            );
            write_str(sink, &prefixed).map_err(A::Error::custom)
        }
    }
}

fn render_value_in_context(
    value: &JsonValue,
    options: &EncodeOptions,
    context: &RenderContext,
) -> ToonResult<String> {
    match context {
        RenderContext::Root => render_root_value(value, options),
        RenderContext::ObjectField { key, depth } => {
            render_object_field_value(key, *depth, value, options)
        }
    }
}

fn render_root_value(value: &JsonValue, options: &EncodeOptions) -> ToonResult<String> {
    let mut writer = Writer::new(options.clone());
    match value {
        JsonValue::Array(arr) => super::write_array(&mut writer, None, arr, 0)?,
        JsonValue::Object(obj) => super::write_object(&mut writer, obj, 0)?,
        _ => super::write_primitive_value(&mut writer, value, QuotingContext::ObjectValue)?,
    }
    Ok(writer.finish())
}

fn render_object_field_value(
    key: &str,
    depth: usize,
    value: &JsonValue,
    options: &EncodeOptions,
) -> ToonResult<String> {
    let mut writer = Writer::new(options.clone());
    match value {
        JsonValue::Array(arr) => super::write_array(&mut writer, Some(key), arr, depth)?,
        JsonValue::Object(obj) => {
            if depth > 0 {
                writer.write_indent(depth)?;
            }
            writer.write_key(key)?;
            writer.write_char(':')?;
            if !obj.is_empty() {
                writer.write_newline()?;
                super::write_object(&mut writer, obj, depth + 1)?;
            }
        }
        _ => {
            if depth > 0 {
                writer.write_indent(depth)?;
            }
            writer.write_key(key)?;
            writer.write_char(':')?;
            writer.write_char(' ')?;
            super::write_primitive_value(&mut writer, value, QuotingContext::ObjectValue)?;
        }
    }

    Ok(writer.finish())
}

fn render_array_element_value(
    array_depth: usize,
    value: &JsonValue,
    options: &EncodeOptions,
) -> ToonResult<String> {
    let mut writer = Writer::new(options.clone());
    writer.write_indent(array_depth + 1)?;
    writer.write_char('-')?;

    match value {
        JsonValue::Array(inner_arr) => {
            writer.write_char(' ')?;
            super::write_array(&mut writer, None, inner_arr, array_depth + 1)?;
        }
        JsonValue::Object(obj) => {
            let keys: Vec<&String> = obj.keys().collect();
            if let Some(first_key) = keys.first() {
                writer.write_char(' ')?;
                let first_val = &obj[*first_key];

                match first_val {
                    JsonValue::Array(arr) => {
                        writer.write_key(first_key)?;

                        if let Some(keys) = super::is_tabular_array(arr) {
                            super::encode_list_item_tabular_array(
                                &mut writer,
                                arr,
                                &keys,
                                array_depth + 1,
                            )?;
                        } else {
                            super::write_array(&mut writer, None, arr, array_depth + 2)?;
                        }
                    }
                    JsonValue::Object(nested_obj) => {
                        writer.write_key(first_key)?;
                        writer.write_char(':')?;
                        if !nested_obj.is_empty() {
                            writer.write_newline()?;
                            super::write_object(&mut writer, nested_obj, array_depth + 3)?;
                        }
                    }
                    _ => {
                        writer.write_key(first_key)?;
                        writer.write_char(':')?;
                        writer.write_char(' ')?;
                        super::write_primitive_value(
                            &mut writer,
                            first_val,
                            QuotingContext::ObjectValue,
                        )?;
                    }
                }

                for key in keys.iter().skip(1) {
                    writer.write_newline()?;
                    writer.write_indent(array_depth + 2)?;

                    let field_value = &obj[*key];
                    match field_value {
                        JsonValue::Array(arr) => {
                            writer.write_key(key)?;
                            super::write_array(&mut writer, None, arr, array_depth + 2)?;
                        }
                        JsonValue::Object(nested_obj) => {
                            writer.write_key(key)?;
                            writer.write_char(':')?;
                            if !nested_obj.is_empty() {
                                writer.write_newline()?;
                                super::write_object(&mut writer, nested_obj, array_depth + 3)?;
                            }
                        }
                        _ => {
                            writer.write_key(key)?;
                            writer.write_char(':')?;
                            writer.write_char(' ')?;
                            super::write_primitive_value(
                                &mut writer,
                                field_value,
                                QuotingContext::ObjectValue,
                            )?;
                        }
                    }
                }
            }
        }
        _ => {
            writer.write_char(' ')?;
            super::write_primitive_value(&mut writer, value, QuotingContext::ArrayValue)?;
        }
    }

    Ok(writer.finish())
}

fn render_key(key: &str, options: &EncodeOptions) -> ToonResult<String> {
    let mut writer = Writer::new(options.clone());
    writer.write_key(key)?;
    Ok(writer.finish())
}

fn prefix_first_line(prefix: &str, value: &str) -> String {
    if let Some((first, rest)) = value.split_once('\n') {
        format!("{prefix}{first}\n{rest}")
    } else {
        format!("{prefix}{value}")
    }
}

struct StreamArrayEncoder<'a> {
    options: &'a EncodeOptions,
    array_depth: usize,
    len: usize,
    primitive_chunks: Vec<String>,
    nested_chunks: Vec<String>,
    tabular_rows: Vec<String>,
    tabular_keys: Option<Vec<String>>,
    all_primitives: bool,
    all_tabular: bool,
}

impl<'a> StreamArrayEncoder<'a> {
    fn new(options: &'a EncodeOptions, array_depth: usize) -> Self {
        Self {
            options,
            array_depth,
            len: 0,
            primitive_chunks: Vec::new(),
            nested_chunks: Vec::new(),
            tabular_rows: Vec::new(),
            tabular_keys: None,
            all_primitives: true,
            all_tabular: true,
        }
    }

    fn push(&mut self, value: JsonValue) -> ToonResult<()> {
        self.len += 1;

        if super::is_primitive(&value) {
            let primitive = render_primitive_array_value(&value, self.options)?;
            if !self.all_primitives {
                self.nested_chunks.push(render_primitive_nested_chunk(
                    self.array_depth,
                    &primitive,
                    self.options,
                ));
            }
            self.primitive_chunks.push(primitive);
        } else {
            if self.all_primitives {
                self.nested_chunks = self
                    .primitive_chunks
                    .iter()
                    .map(|chunk| {
                        render_primitive_nested_chunk(self.array_depth, chunk, self.options)
                    })
                    .collect();
            }
            self.all_primitives = false;
            self.nested_chunks.push(render_array_element_value(
                self.array_depth,
                &value,
                self.options,
            )?);
        }

        if self.all_tabular {
            if let Some(obj) = value.as_object() {
                if obj.values().all(super::is_primitive) {
                    let current_keys: Vec<String> = obj.keys().cloned().collect();
                    match &self.tabular_keys {
                        None => {
                            self.tabular_rows.push(render_tabular_row(
                                obj,
                                &current_keys,
                                self.options,
                            )?);
                            self.tabular_keys = Some(current_keys);
                        }
                        Some(expected_keys)
                            if expected_keys.len() == obj.len()
                                && expected_keys.iter().all(|key| obj.contains_key(key)) =>
                        {
                            self.tabular_rows.push(render_tabular_row(
                                obj,
                                expected_keys,
                                self.options,
                            )?);
                        }
                        _ => {
                            self.all_tabular = false;
                            self.tabular_rows.clear();
                            self.tabular_keys = None;
                        }
                    }
                } else {
                    self.all_tabular = false;
                    self.tabular_rows.clear();
                    self.tabular_keys = None;
                }
            } else {
                self.all_tabular = false;
                self.tabular_rows.clear();
                self.tabular_keys = None;
            }
        }

        Ok(())
    }

    fn render(&self) -> ToonResult<String> {
        if self.len == 0 {
            let mut writer = Writer::new(self.options.clone());
            writer.write_empty_array_with_key(None, self.array_depth)?;
            return Ok(writer.finish());
        }

        if self.all_tabular {
            let keys = self
                .tabular_keys
                .as_ref()
                .ok_or_else(|| ToonError::SerializationError("tabular keys missing".to_string()))?;
            let mut writer = Writer::new(self.options.clone());
            writer.write_array_header(None, self.len, Some(keys), self.array_depth)?;
            writer.write_newline()?;

            for (i, row) in self.tabular_rows.iter().enumerate() {
                if i > 0 {
                    writer.write_newline()?;
                }
                writer.write_indent(self.array_depth + 1)?;
                writer.write_str(row)?;
            }

            return Ok(writer.finish());
        }

        if self.all_primitives {
            let mut writer = Writer::new(self.options.clone());
            writer.write_array_header(None, self.len, None, self.array_depth)?;
            writer.write_char(' ')?;
            for (i, item) in self.primitive_chunks.iter().enumerate() {
                if i > 0 {
                    writer.write_delimiter()?;
                }
                writer.write_str(item)?;
            }
            return Ok(writer.finish());
        }

        let mut writer = Writer::new(self.options.clone());
        writer.write_array_header(None, self.len, None, self.array_depth)?;
        writer.write_newline()?;

        for (i, item) in self.nested_chunks.iter().enumerate() {
            if i > 0 {
                writer.write_newline()?;
            }
            writer.write_str(item)?;
        }

        Ok(writer.finish())
    }
}

fn render_primitive_array_value(value: &JsonValue, options: &EncodeOptions) -> ToonResult<String> {
    let mut writer = Writer::new(options.clone());
    super::write_primitive_value(&mut writer, value, QuotingContext::ArrayValue)?;
    Ok(writer.finish())
}

fn render_tabular_row(
    obj: &indexmap::IndexMap<String, JsonValue>,
    keys: &[String],
    options: &EncodeOptions,
) -> ToonResult<String> {
    let mut writer = Writer::new(options.clone());
    writer.push_active_delimiter(writer.options.delimiter);

    for (i, key) in keys.iter().enumerate() {
        if i > 0 {
            writer.write_delimiter()?;
        }

        if let Some(value) = obj.get(key) {
            super::write_primitive_value(&mut writer, value, QuotingContext::ArrayValue)?;
        } else {
            writer.write_str("null")?;
        }
    }

    writer.pop_active_delimiter();
    Ok(writer.finish())
}

fn render_primitive_nested_chunk(
    array_depth: usize,
    primitive: &str,
    options: &EncodeOptions,
) -> String {
    format!(
        "{}- {}",
        options.indent.get_string(array_depth + 1),
        primitive
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        encode,
        encode_default,
        types::{
            Delimiter,
            EncodeOptions,
            Indent,
            KeyFoldingMode,
        },
    };

    #[test]
    fn test_streaming_root_object_matches_in_memory() {
        let input = br#"{"name":"Alice","age":30,"tags":["a","b"]}"#;
        let options = EncodeOptions::new().with_indent(Indent::Spaces(2));

        let streaming =
            encode_json_reader(&input[..], &options, &StreamingEncodeOptions::default()).unwrap();
        let in_memory =
            encode(&json!({"name":"Alice","age":30,"tags":["a","b"]}), &options).unwrap();

        assert_eq!(streaming, in_memory);
    }

    #[test]
    fn test_streaming_reader_default_uses_default_options() {
        let input = br#"{"name":"Alice","age":30}"#;

        let streaming = encode_json_reader_default(&input[..]).unwrap();
        let in_memory = encode_default(&json!({"name":"Alice","age":30})).unwrap();

        assert_eq!(streaming, in_memory);
    }

    #[test]
    fn test_streaming_root_primitive_array_matches_in_memory() {
        let input = br#"["reading","gaming","coding"]"#;
        let options = EncodeOptions::new().with_delimiter(Delimiter::Pipe);

        let streaming =
            encode_json_reader(&input[..], &options, &StreamingEncodeOptions::default()).unwrap();
        let in_memory = encode(&json!(["reading", "gaming", "coding"]), &options).unwrap();

        assert_eq!(streaming, in_memory);
    }

    #[test]
    fn test_streaming_root_tabular_array_matches_in_memory() {
        let input = br#"[{"id":1,"name":"Alice"},{"id":2,"name":"Bob"}]"#;
        let options = EncodeOptions::default();

        let streaming =
            encode_json_reader(&input[..], &options, &StreamingEncodeOptions::default()).unwrap();
        let in_memory = encode(
            &json!([
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"}
            ]),
            &options,
        )
        .unwrap();

        assert_eq!(streaming, in_memory);
    }

    #[test]
    fn test_streaming_writer_default_uses_default_options() {
        let input = br#"[{"id":1,"name":"Alice"},{"id":2,"name":"Bob"}]"#;
        let mut output = Vec::new();

        encode_json_stream_default(&input[..], &mut output).unwrap();

        let streaming = String::from_utf8(output).unwrap();
        let in_memory = encode_default(&json!([
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"}
        ]))
        .unwrap();

        assert_eq!(streaming, in_memory);
    }

    #[test]
    fn test_streaming_key_folding_falls_back_to_in_memory_output() {
        let input = br#"{"a":{"b":1}}"#;
        let options = EncodeOptions::new().with_key_folding(KeyFoldingMode::Safe);

        let streaming =
            encode_json_reader(&input[..], &options, &StreamingEncodeOptions::default()).unwrap();
        let in_memory = encode(&json!({"a": {"b": 1}}), &options).unwrap();

        assert_eq!(streaming, in_memory);
    }

    #[test]
    fn test_streaming_root_object_preserves_order() {
        let input = br#"{"b":1,"a":2,"c":3}"#;

        let output = encode_json_reader(
            &input[..],
            &EncodeOptions::default(),
            &StreamingEncodeOptions::default(),
        )
        .unwrap();
        assert_eq!(output, "b: 1\na: 2\nc: 3");
    }

    #[test]
    fn test_streaming_depth_defaults_to_two() {
        assert_eq!(StreamingEncodeOptions::default().streaming_depth, 2);
    }

    #[test]
    fn test_streaming_depth_zero_gracefully_falls_back_per_subtree() {
        let input = br#"{"meta":{"count":1},"groups":[[1,2],[3,4]]}"#;
        let options = EncodeOptions::default();
        let streaming_options = StreamingEncodeOptions::new().with_streaming_depth(0);

        let streaming = encode_json_reader(&input[..], &options, &streaming_options).unwrap();
        let in_memory = encode(
            &json!({
                "meta": {"count": 1},
                "groups": [[1, 2], [3, 4]]
            }),
            &options,
        )
        .unwrap();

        assert_eq!(streaming, in_memory);
    }

    #[test]
    fn test_streaming_nested_object_with_depth_two_matches_in_memory() {
        let input = br#"{"meta":{"count":1},"indices":{"a":{"size":1},"b":{"size":2}}}"#;
        let options = EncodeOptions::default();
        let streaming_options = StreamingEncodeOptions::new().with_streaming_depth(2);

        let streaming = encode_json_reader(&input[..], &options, &streaming_options).unwrap();
        let in_memory = encode(
            &json!({
                "meta": {"count": 1},
                "indices": {
                    "a": {"size": 1},
                    "b": {"size": 2}
                }
            }),
            &options,
        )
        .unwrap();

        assert_eq!(streaming, in_memory);
    }

    #[test]
    fn test_streaming_nested_array_with_depth_two_matches_in_memory() {
        let input = br#"{"meta":{"count":1},"groups":[[1,2],[3,4]]}"#;
        let options = EncodeOptions::default();
        let streaming_options = StreamingEncodeOptions::new().with_streaming_depth(2);

        let streaming = encode_json_reader(&input[..], &options, &streaming_options).unwrap();
        let in_memory = encode(
            &json!({
                "meta": {"count": 1},
                "groups": [[1, 2], [3, 4]]
            }),
            &options,
        )
        .unwrap();

        assert_eq!(streaming, in_memory);
    }
}
