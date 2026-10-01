/// One entry of a tabular or keyed header's field list (§6).
///
/// A leaf field (no children) maps to one row cell; a nested field group
/// carries its subfields and declares a nested-uniform column (§9.3).
#[derive(Debug, Clone)]
pub(crate) struct FieldNode {
    pub name: String,
    pub children: Option<Vec<FieldNode>>,
}
