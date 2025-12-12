
use std::path::Path;

#[derive(Debug, Clone)]
pub enum DocumentKind {
    Markdown,
    Pdf,
    Mermaid,
}

#[derive(Debug, Clone)]
pub struct DocumentSummary {
    pub title: String,
    pub kind: DocumentKind,
    pub page_count: usize,        // for PDF, 1 for md/mmd
    pub path: std::path::PathBuf,
}

#[derive(Debug, Clone)]
pub struct DocumentPage {
    pub index: usize,
    pub text: String,             // for md/pdf: extracted text
    // optional later: images, layout info, etc.
}

pub struct DocumentHandle {
    pub summary: DocumentSummary,
    pub pages: Vec<DocumentPage>,
}

pub trait DocumentLoader {
    fn load_document(&self, path: &Path) -> anyhow::Result<DocumentHandle>;
}
