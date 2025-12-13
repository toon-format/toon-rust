
#[derive(Debug, Clone)]
pub struct MermaidNode {
    pub id: String,
    pub label: String,
}

#[derive(Debug, Clone)]
pub struct MermaidEdge {
    pub from: String,
    pub to: String,
    pub kind: String,      // e.g. "->", "-->", etc.
}

#[derive(Debug, Clone)]
pub struct MermaidGraph {
    pub nodes: Vec<MermaidNode>,
    pub edges: Vec<MermaidEdge>,
}

pub trait MermaidParser {
    fn parse_mermaid(&self, source: &str) -> anyhow::Result<MermaidGraph>;
}
