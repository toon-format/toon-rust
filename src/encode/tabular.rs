//! Tabular and keyed tabular form detection (§9.3, §9.5).

use indexmap::IndexMap;

use crate::types::JsonValue as Value;

/// One entry of a tabular header's field list. A leaf field (no children)
/// maps to one row cell; a nested field group carries its subfields and
/// declares a nested-uniform column (§9.3).
#[derive(Debug, Clone)]
pub(crate) struct FieldNode {
    pub name: String,
    pub children: Option<Vec<FieldNode>>,
}

/// Classifies array elements into a tabular field list, or `None` when they
/// are not uniformly tabular (§9.3).
pub(crate) fn extract_tabular_fields(rows: &[Value]) -> Option<Vec<FieldNode>> {
    if rows.is_empty() {
        return None;
    }

    let mut objects = Vec::with_capacity(rows.len());
    for row in rows {
        let Value::Object(map) = row else {
            return None;
        };
        objects.push(map);
    }

    extract_fields_from_objects(&objects)
}

/// Classifies an object's values as a keyed tabular field list – at least
/// two entries, every value a non-empty object, uniform columns – or `None`
/// (§9.5).
pub(crate) fn extract_keyed_tabular_fields(
    obj: &IndexMap<String, Value>,
) -> Option<Vec<FieldNode>> {
    if obj.len() < 2 {
        return None;
    }

    let mut entry_values = Vec::with_capacity(obj.len());
    for value in obj.values() {
        let Value::Object(map) = value else {
            return None;
        };
        if map.is_empty() {
            return None;
        }
        entry_values.push(map);
    }

    extract_fields_from_objects(&entry_values)
}

fn extract_fields_from_objects(objects: &[&IndexMap<String, Value>]) -> Option<Vec<FieldNode>> {
    let first_keys: Vec<&String> = objects[0].keys().collect();
    if first_keys.is_empty() {
        return None;
    }

    // All objects must have the same set of keys (order per object may vary).
    for object in objects {
        if object.len() != first_keys.len() {
            return None;
        }
        for key in &first_keys {
            if !object.contains_key(*key) {
                return None;
            }
        }
    }

    let mut fields = Vec::with_capacity(first_keys.len());
    for key in first_keys {
        let values: Vec<&Value> = objects
            .iter()
            .map(|object| object.get(key).expect("key presence checked above"))
            .collect();
        fields.push(classify_column(key, &values)?);
    }

    Some(fields)
}

fn classify_column(name: &str, values: &[&Value]) -> Option<FieldNode> {
    // Uniform-primitive column: a bare leaf field.
    if values.iter().all(|value| is_primitive(value)) {
        return Some(FieldNode {
            name: name.to_string(),
            children: None,
        });
    }

    // Nested-uniform column: non-empty objects sharing one key set,
    // classified recursively.
    let mut objects = Vec::with_capacity(values.len());
    for value in values {
        let Value::Object(map) = value else {
            return None;
        };
        if map.is_empty() {
            return None;
        }
        objects.push(map);
    }

    let children = extract_fields_from_objects(&objects)?;
    Some(FieldNode {
        name: name.to_string(),
        children: Some(children),
    })
}

/// Reads one row's leaf cells in the depth-first order the field list
/// declares (§9.3).
pub(crate) fn collect_row_leaves<'v>(
    row: &'v IndexMap<String, Value>,
    fields: &[FieldNode],
    leaves: &mut Vec<&'v Value>,
) {
    for field in fields {
        let value = row
            .get(&field.name)
            .expect("tabular detection guarantees field presence");
        match &field.children {
            Some(children) => {
                let Value::Object(nested) = value else {
                    unreachable!("nested-uniform column holds objects");
                };
                collect_row_leaves(nested, children, leaves);
            }
            None => leaves.push(value),
        }
    }
}

/// Check if a value is a primitive (not array or object).
pub(crate) fn is_primitive(value: &Value) -> bool {
    !matches!(value, Value::Array(_) | Value::Object(_))
}
