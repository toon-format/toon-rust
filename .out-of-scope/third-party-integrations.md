# Integrations Into Other Tools

`toon-format` is a codec crate. It doesn't integrate itself into proxies, agents, editors, or other tools – those tools add the crate as a dependency and call it.

## Why this is out of scope

The public API is the integration point: `encode` and `encode_default` take any `serde::Serialize` value, `decode` returns any `serde::Deserialize` type. Which payloads get converted, above what size, and what happens when TOON doesn't save tokens are decisions for the tool that sees the traffic, not for the codec. The spec frames TOON the same way: "produce data as JSON in code, encode to TOON for downstream consumption, and decode back to JSON if needed" ([Purpose and Scope](https://github.com/toon-format/spec/blob/main/SPEC.md#purpose-and-scope)).

Integration code in this repo would also tie the crate to another project's internals and release cycle, when the only thing it needs to track is the spec.

For a Rust tool the integration is a few lines on its side:

```rust
let toon = toon_format::encode_default(&value)?;
```

Tools in other languages use the TOON implementation for their language, listed in [`docs/ecosystem/implementations.md`](https://github.com/toon-format/toon/blob/main/docs/ecosystem/implementations.md).

## Prior requests

- [#73](https://github.com/toon-format/toon-rust/issues/73) – "lean-ctx Integration"
