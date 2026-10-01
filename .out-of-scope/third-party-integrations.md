# Integrations Into Other Tools

`toon-format` is a codec crate. It doesn't integrate itself into proxies, agents, editors, or other tools – those tools add the crate as a dependency and call it.

## Why this is out of scope

The serde-based `encode` and `decode` functions are the integration point. Which payloads to convert, above what size, and what to do when TOON saves no tokens are decisions for the tool that sees the traffic. Integration code here would also tie the crate to another project's internals and release cycle, when the only thing it should track is the spec.

## Prior requests

- [#73](https://github.com/toon-format/toon-rust/issues/73) – "lean-ctx Integration"
