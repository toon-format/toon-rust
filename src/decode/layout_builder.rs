use crate::layout::{
    Layout,
    NodeLayout,
};

pub(crate) struct LayoutBuilder {
    stack: Vec<String>,
    layout: Layout,
}

impl LayoutBuilder {
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            layout: Layout::new(),
        }
    }

    pub fn push(&mut self, segment: impl Into<String>) {
        self.stack.push(segment.into());
    }

    pub fn pop(&mut self) {
        self.stack.pop();
    }

    pub fn record(&mut self, node: NodeLayout) {
        self.layout.insert(self.current_path(), node);
    }

    pub fn finish(self) -> Layout {
        self.layout
    }

    fn current_path(&self) -> String {
        if self.stack.is_empty() {
            return String::new();
        }
        let mut out = String::with_capacity(self.stack.iter().map(|s| s.len() + 1).sum());
        for segment in &self.stack {
            out.push('/');
            for ch in segment.chars() {
                match ch {
                    '~' => out.push_str("~0"),
                    '/' => out.push_str("~1"),
                    _ => out.push(ch),
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Delimiter;

    #[test]
    fn root_path_is_empty() {
        let b = LayoutBuilder::new();
        assert_eq!(b.current_path(), "");
    }

    #[test]
    fn path_joins_segments_with_slash() {
        let mut b = LayoutBuilder::new();
        b.push("users");
        b.push("0");
        b.push("name");
        assert_eq!(b.current_path(), "/users/0/name");
    }

    #[test]
    fn path_escapes_rfc6901_specials() {
        let mut b = LayoutBuilder::new();
        b.push("a/b");
        b.push("c~d");
        assert_eq!(b.current_path(), "/a~1b/c~0d");
    }

    #[test]
    fn pop_balances_push() {
        let mut b = LayoutBuilder::new();
        b.push("a");
        b.push("b");
        b.pop();
        assert_eq!(b.current_path(), "/a");
    }

    #[test]
    fn record_writes_at_current_path() {
        let mut b = LayoutBuilder::new();
        b.push("users");
        b.record(NodeLayout::List { declared_len: 2 });
        let layout = b.finish();
        assert_eq!(
            layout.get("/users"),
            Some(&NodeLayout::List { declared_len: 2 })
        );
    }

    #[test]
    fn record_root() {
        let mut b = LayoutBuilder::new();
        b.record(NodeLayout::InlineArray {
            declared_len: 3,
            delimiter: Delimiter::Comma,
        });
        let layout = b.finish();
        assert!(layout.get("").is_some());
    }
}
