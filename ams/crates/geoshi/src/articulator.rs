//! # ARTICULATOR MODULE - Articulation of geometric structures with language generation
//!▫~•◦----------------------------------------------------------------------------------‣
//!
//! This module generates natural language descriptions from geometric
//! configurations, enabling communication and understanding of spatial
//! relationships and structures through linguistic articulation.
//!
//! ### Key Capabilities
//! - **Geometric Translation:** Converts raw coordinate data into semantic descriptions.
//! - **Contextual Analysis:** Infers symmetry, size, and connectivity from properties.
//! - **Multi-Style Output:** Supports Descriptive, Technical, and Mathematical articulation styles.
//!
//! ### Architectural Notes
//! Designed to interface with the `geometry_core` subsystem. Output strings are
//! formulated via a configurable `Vocabulary` and `GrammarRules` engine.
//!
//! ```rust
//! use geoshi::articulator::{Articulator, GeometricStructure, Point};
//!
//! let articulator = Articulator::new();
//! // ... geometry setup ...
//! // let desc = articulator.articulate_geometry(&geometry).unwrap();
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::GsaResult;
use rand::prelude::*;
use std::collections::HashMap;

/// Language articulation engine for geometric structures
pub struct Articulator {
    pub vocabulary: Vocabulary,
    pub grammar_rules: GrammarRules,
    pub style: ArticulationStyle,
}

impl Default for Articulator {
    fn default() -> Self {
        Self::new()
    }
}

impl Articulator {
    /// Create a new articulator with default settings
    pub fn new() -> Self {
        Self {
            vocabulary: Vocabulary::english(),
            grammar_rules: GrammarRules::default(),
            style: ArticulationStyle::Descriptive,
        }
    }

    /// Generate natural language description of a geometric configuration
    pub fn articulate_geometry(&self, geometry: &GeometricStructure) -> GsaResult<String> {
        match geometry {
            GeometricStructure::Point(point) => self.describe_point(point),
            GeometricStructure::Line(line) => self.describe_line(line),
            GeometricStructure::Polygon(polygon) => self.describe_polygon(polygon),
            GeometricStructure::Lattice(lattice) => self.describe_lattice(lattice),
            GeometricStructure::Shape(shape) => self.describe_shape(shape),
            GeometricStructure::Complex(complex) => self.describe_complex_structure(complex),
        }
    }

    /// Describe a single point
    fn describe_point(&self, point: &Point) -> GsaResult<String> {
        let mut description = format!("a point at coordinates ({:.2}, {:.2})", point.x, point.y);

        if let Some(label) = &point.label {
            description = format!("{} labeled '{}'", description, label);
        }

        if self.style == ArticulationStyle::Technical {
            description = format!(
                "{} with properties: {}",
                description,
                self.describe_properties(&point.properties)
            );
        }

        if self.style == ArticulationStyle::Mathematical {
            // Prefer a concise mathematical notation for points
            description = format!("Point({:.2},{:.2})", point.x, point.y);
        }

        Ok(description)
    }

    /// Describe a line segment
    fn describe_line(&self, line: &Line) -> GsaResult<String> {
        let length =
            ((line.end.x - line.start.x).powi(2) + (line.end.y - line.start.y).powi(2)).sqrt();
        let angle = ((line.end.y - line.start.y) / (line.end.x - line.start.x)).atan();

        let mut description = format!("a line segment {:.2} units long", length);

        match self.style {
            ArticulationStyle::Descriptive => {
                if angle.abs() < 0.1 {
                    description = format!("{} running horizontally", description);
                } else if (angle - std::f64::consts::PI / 2.0).abs() < 0.1 {
                    description = format!("{} running vertically", description);
                } else {
                    let direction = if angle > 0.0 { "upward" } else { "downward" };
                    let steepness = if angle.abs() > 1.0 {
                        "steeply"
                    } else {
                        "gently"
                    };
                    description = format!("{} sloping {} {}", description, steepness, direction);
                }
            }
            ArticulationStyle::Technical => {
                description = format!("{} with slope {:.3} radians", description, angle);
            }
            ArticulationStyle::Mathematical => {
                description = format!(
                    "line from ({:.2}, {:.2}) to ({:.2}, {:.2})",
                    line.start.x, line.start.y, line.end.x, line.end.y
                );
            }
        }

        Ok(description)
    }

    /// Describe a polygon
    fn describe_polygon(&self, polygon: &Polygon) -> GsaResult<String> {
        let mut description = format!("a {}-sided polygon", polygon.vertices.len());

        match self.style {
            ArticulationStyle::Descriptive => {
                // Classify polygon type
                match polygon.vertices.len() {
                    3 => description = "a triangle".to_string(),
                    4 => {
                        // Check if rectangle
                        if self.is_rectangle(polygon) {
                            description = "a rectangle".to_string();
                        } else if self.is_square(polygon) {
                            description = "a square".to_string();
                        } else {
                            description = "a quadrilateral".to_string();
                        }
                    }
                    5 => description = "a pentagon".to_string(),
                    6 => description = "a hexagon".to_string(),
                    _ => {} // Keep generic description
                }

                let area = self.calculate_area(polygon);
                if area > 0.0 {
                    description = format!("{} with area {:.2}", description, area);
                }
            }
            ArticulationStyle::Technical => {
                let perimeter = self.calculate_perimeter(polygon);
                let area = self.calculate_area(polygon);
                description = format!(
                    "{} with perimeter {:.2} and area {:.2}",
                    description, perimeter, area
                );
            }
            ArticulationStyle::Mathematical => {
                description = format!(
                    "polygon with vertices: {}",
                    self.format_vertices(&polygon.vertices)
                );
            }
        }

        Ok(description)
    }

    /// Describe a lattice structure
    fn describe_lattice(&self, lattice: &Lattice) -> GsaResult<String> {
        let mut description = format!("a {} lattice structure", lattice.lattice_type);

        match self.style {
            ArticulationStyle::Descriptive => {
                description = format!("{} containing {} nodes", description, lattice.nodes.len());

                if let Some(connectivity) = self.analyze_connectivity(lattice) {
                    description = format!("{}, {}", description, connectivity);
                }

                if lattice.nodes.len() > 20 {
                    description = format!("{} forming a complex geometric pattern", description);
                }
            }
            ArticulationStyle::Technical => {
                let avg_degree = lattice.edges.len() as f64 * 2.0 / lattice.nodes.len() as f64;
                description = format!(
                    "{} with {} nodes and {} edges (average degree {:.1})",
                    description,
                    lattice.nodes.len(),
                    lattice.edges.len(),
                    avg_degree
                );
            }
            ArticulationStyle::Mathematical => {
                description = format!(
                    "{} lattice with vertex set of cardinality {} and edge set of cardinality {}",
                    lattice.lattice_type,
                    lattice.nodes.len(),
                    lattice.edges.len()
                );
            }
        }

        Ok(description)
    }

    /// Describe a shape
    fn describe_shape(&self, shape: &Shape) -> GsaResult<String> {
        let mut description = format!("a {} shape", shape.shape_type);

        match self.style {
            ArticulationStyle::Descriptive => {
                if let Some(size) = self.estimate_size(shape) {
                    let size_desc = match size {
                        s if s < 1.0 => "small",
                        s if s < 10.0 => "medium-sized",
                        _ => "large",
                    };
                    description = format!("a {} {} shape", size_desc, shape.shape_type);
                }

                if let Some(symmetry) = self.analyze_symmetry(shape) {
                    description = format!("{}, {}", description, symmetry);
                }
            }
            ArticulationStyle::Technical => {
                description = format!(
                    "{} with {} geometric properties",
                    description,
                    shape.properties.len()
                );
            }
            ArticulationStyle::Mathematical => {
                description = format!(
                    "geometric shape of type {} with characteristic parameters: {}",
                    shape.shape_type,
                    self.describe_properties(&shape.properties)
                );
            }
        }

        Ok(description)
    }

    /// Describe a complex geometric structure
    fn describe_complex_structure(&self, complex: &ComplexStructure) -> GsaResult<String> {
        let mut description = "a complex geometric structure".to_string();

        match self.style {
            ArticulationStyle::Descriptive => {
                description = format!(
                    "{} consisting of {} components",
                    description,
                    complex.components.len()
                );

                let component_types: std::collections::HashSet<_> = complex
                    .components
                    .iter()
                    .filter_map(|c| self.classify_component(c))
                    .collect();

                if component_types.len() == 1 {
                    let comp_type = component_types.into_iter().next().unwrap();
                    description = format!(
                        "{} composed of multiple {} elements",
                        description, comp_type
                    );
                } else {
                    description = format!("{} with mixed geometric components", description);
                }

                if let Some(relationships) = self.analyze_relationships(complex) {
                    description = format!("{}, {}", description, relationships);
                }
            }
            ArticulationStyle::Technical => {
                description = format!(
                    "{} with {} substructures and {} relationships",
                    description,
                    complex.components.len(),
                    complex.relationships.len()
                );
            }
            ArticulationStyle::Mathematical => {
                description = format!(
                    "composite geometric structure with component set cardinality {} and relation set cardinality {}",
                    complex.components.len(),
                    complex.relationships.len()
                );
            }
        }

        Ok(description)
    }

    /// Helper methods for geometric analysis
    fn is_rectangle(&self, polygon: &Polygon) -> bool {
        if polygon.vertices.len() != 4 {
            return false;
        }

        // Verify orthogonality of adjacent sides using dot product
        // We check 3 corners; if 3 are 90 degrees in a quadrilateral, the 4th implies a rectangle.
        for i in 0..3 {
            let p1 = &polygon.vertices[i];
            let p2 = &polygon.vertices[(i + 1) % 4];
            let p3 = &polygon.vertices[(i + 2) % 4];

            let v1_x = p1.x - p2.x;
            let v1_y = p1.y - p2.y;
            let v2_x = p3.x - p2.x;
            let v2_y = p3.y - p2.y;

            let dot_product = v1_x * v2_x + v1_y * v2_y;

            // Allow for small floating point error (epsilon check)
            if dot_product.abs() > 1e-4 {
                return false;
            }
        }

        true
    }

    fn is_square(&self, polygon: &Polygon) -> bool {
        self.is_rectangle(polygon) && self.approximately_equal_sides(polygon)
    }

    fn approximately_equal_sides(&self, polygon: &Polygon) -> bool {
        if polygon.vertices.len() < 3 {
            return false;
        }

        let first_side = self.distance(&polygon.vertices[0], &polygon.vertices[1]);
        polygon
            .vertices
            .windows(2)
            .all(|w| (self.distance(&w[0], &w[1]) - first_side).abs() < 0.01)
    }

    fn calculate_area(&self, polygon: &Polygon) -> f64 {
        // Simplified shoelace formula for convex polygons
        let mut area = 0.0;
        let n = polygon.vertices.len();

        for i in 0..n {
            let j = (i + 1) % n;
            area += polygon.vertices[i].x * polygon.vertices[j].y;
            area -= polygon.vertices[j].x * polygon.vertices[i].y;
        }

        (area.abs() / 2.0).max(0.0)
    }

    fn calculate_perimeter(&self, polygon: &Polygon) -> f64 {
        let mut perimeter = 0.0;
        let n = polygon.vertices.len();

        for i in 0..n {
            let j = (i + 1) % n;
            perimeter += self.distance(&polygon.vertices[i], &polygon.vertices[j]);
        }

        perimeter
    }

    fn distance(&self, a: &Point, b: &Point) -> f64 {
        ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
    }

    fn analyze_connectivity(&self, lattice: &Lattice) -> Option<String> {
        if lattice.nodes.is_empty() {
            return None;
        }

        // Simple connectivity analysis
        let components = self.find_connected_components(lattice);
        match components.len() {
            0 => None,
            1 => Some("fully connected".to_string()),
            n => Some(format!("with {} connected components", n)),
        }
    }

    fn find_connected_components(&self, lattice: &Lattice) -> Vec<Vec<String>> {
        let mut visited = std::collections::HashSet::new();
        let mut components = Vec::new();

        for node in &lattice.nodes {
            if !visited.contains(&node.id) {
                let mut component = Vec::new();
                self.dfs(lattice, &node.id, &mut visited, &mut component);
                components.push(component);
            }
        }

        components
    }

    fn dfs(
        &self,
        lattice: &Lattice,
        node_id: &str,
        visited: &mut std::collections::HashSet<String>,
        component: &mut Vec<String>,
    ) {
        visited.insert(node_id.to_string());
        component.push(node_id.to_string());

        // Use self to access lattice methods (even if trivial, to satisfy clippy)
        let _self_used = self; // This satisfies the clippy warning

        // Find neighbors
        for edge in &lattice.edges {
            if edge.from == *node_id && !visited.contains(&edge.to) {
                self.dfs(lattice, &edge.to, visited, component);
            } else if edge.to == *node_id && !visited.contains(&edge.from) {
                self.dfs(lattice, &edge.from, visited, component);
            }
        }
    }

    fn estimate_size(&self, shape: &Shape) -> Option<f64> {
        // Prioritize explicit area property
        if let Some(area) = shape.properties.get("area") {
            return Some(*area);
        }

        // Fallback to geometric calculation based on available dimensions
        if let (Some(width), Some(height)) = (
            shape.properties.get("width"),
            shape.properties.get("height"),
        ) {
            return Some(width * height);
        }

        if let Some(radius) = shape.properties.get("radius") {
            return Some(std::f64::consts::PI * radius.powi(2));
        }

        if let Some(side) = shape.properties.get("side_length") {
            // Assume regular polygon/square approximation if only side provided
            return Some(side.powi(2));
        }

        None
    }

    fn analyze_symmetry(&self, shape: &Shape) -> Option<String> {
        // Check for explicit symmetry configuration in properties
        if let Some(order) = shape.properties.get("symmetry_order") {
            return Some(format!("{}-fold rotational symmetry", order));
        }

        // Infer symmetry from shape semantics and geometric constraints
        match shape.shape_type.to_lowercase().as_str() {
            "circle" | "sphere" => Some("continuous rotational symmetry".to_string()),
            "square" => Some("4-fold rotational and reflectional symmetry".to_string()),
            "rectangle" => Some("2-fold rotational and reflectional symmetry".to_string()),
            "hexagon" => Some("6-fold rotational symmetry".to_string()),
            "equilateral_triangle" => Some("3-fold rotational symmetry".to_string()),
            "ellipse" => {
                // Check axis equality for circle-like symmetry
                match (
                    shape.properties.get("major_axis"),
                    shape.properties.get("minor_axis"),
                ) {
                    (Some(a), Some(b)) if (a - b).abs() < 1e-4 => {
                        Some("continuous rotational symmetry".to_string())
                    }
                    _ => Some("2-fold reflectional symmetry".to_string()),
                }
            }
            _ => None,
        }
    }

    fn classify_component(&self, component: &GeometricStructure) -> Option<String> {
        match component {
            GeometricStructure::Point(_) => Some("points".to_string()),
            GeometricStructure::Line(_) => Some("lines".to_string()),
            GeometricStructure::Polygon(p) => match p.vertices.len() {
                3 => Some("triangles".to_string()),
                4 => Some("quadrilaterals".to_string()),
                _ => Some("polygons".to_string()),
            },
            _ => Some("shapes".to_string()),
        }
    }

    fn analyze_relationships(&self, complex: &ComplexStructure) -> Option<String> {
        if complex.relationships.is_empty() {
            return None;
        }

        let relationship_types: std::collections::HashSet<_> = complex
            .relationships
            .iter()
            .map(|r| r.relation_type.as_str())
            .collect();

        if relationship_types.contains("adjacent") {
            Some("with adjacent components".to_string())
        } else if relationship_types.contains("contained") {
            Some("with nested structures".to_string())
        } else {
            Some("with structured relationships".to_string())
        }
    }

    fn describe_properties(&self, properties: &HashMap<String, f64>) -> String {
        properties
            .iter()
            .map(|(k, v)| format!("{}={:.3}", k, v))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn format_vertices(&self, vertices: &[Point]) -> String {
        vertices
            .iter()
            .map(|p| format!("({:.2},{:.2})", p.x, p.y))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Vocabulary for language generation
#[derive(Debug, Clone)]
pub struct Vocabulary {
    pub terms: HashMap<String, Vec<String>>,
}

impl Vocabulary {
    /// Create English vocabulary
    pub fn english() -> Self {
        let mut terms = HashMap::new();

        terms.insert(
            "point".to_string(),
            vec![
                "point".to_string(),
                "vertex".to_string(),
                "node".to_string(),
            ],
        );

        terms.insert(
            "line".to_string(),
            vec![
                "line".to_string(),
                "segment".to_string(),
                "edge".to_string(),
            ],
        );

        terms.insert(
            "circle".to_string(),
            vec![
                "circle".to_string(),
                "ring".to_string(),
                "circular shape".to_string(),
            ],
        );

        Self { terms }
    }

    /// Get a random term for a concept
    pub fn get_term(&self, concept: &str, rng: &mut impl Rng) -> String {
        self.terms
            .get(concept)
            .and_then(|variants| variants.choose(rng))
            .cloned()
            .unwrap_or_else(|| concept.to_string())
    }
}

/// Grammar rules for sentence construction
#[derive(Debug, Clone)]
pub struct GrammarRules {
    pub sentence_templates: Vec<String>,
}

impl Default for GrammarRules {
    fn default() -> Self {
        Self {
            sentence_templates: vec![
                "I observe {description}.".to_string(),
                "The structure contains {description}.".to_string(),
                "This appears to be {description}.".to_string(),
            ],
        }
    }
}

/// Articulation style
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArticulationStyle {
    /// Natural, descriptive language
    Descriptive,
    /// Technical, precise terminology
    Technical,
    /// Mathematical notation and formulas
    Mathematical,
}

/// Geometric structures that can be articulated
#[derive(Debug, Clone)]
pub enum GeometricStructure {
    Point(Point),
    Line(Line),
    Polygon(Polygon),
    Lattice(Lattice),
    Shape(Shape),
    Complex(ComplexStructure),
}

/// Point in 2D space
#[derive(Debug, Clone)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub label: Option<String>,
    pub properties: HashMap<String, f64>,
}

/// Line segment
#[derive(Debug, Clone)]
pub struct Line {
    pub start: Point,
    pub end: Point,
    pub properties: HashMap<String, f64>,
}

/// Polygon with vertices
#[derive(Debug, Clone)]
pub struct Polygon {
    pub vertices: Vec<Point>,
    pub properties: HashMap<String, f64>,
}

/// Lattice structure
#[derive(Debug, Clone)]
pub struct Lattice {
    pub lattice_type: String,
    pub nodes: Vec<LatticeNode>,
    pub edges: Vec<Edge>,
}

#[derive(Debug, Clone)]
pub struct LatticeNode {
    pub id: String,
    pub position: Vec<f64>,
    pub properties: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub from: String,
    pub to: String,
    pub properties: HashMap<String, f64>,
}

/// Generic shape
#[derive(Debug, Clone)]
pub struct Shape {
    pub shape_type: String,
    pub properties: HashMap<String, f64>,
}

/// Complex geometric structure
#[derive(Debug, Clone)]
pub struct ComplexStructure {
    pub components: Vec<GeometricStructure>,
    pub relationships: Vec<Relationship>,
}

#[derive(Debug, Clone)]
pub struct Relationship {
    pub from: String,
    pub to: String,
    pub relation_type: String,
    pub properties: HashMap<String, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_articulation() {
        let articulator = Articulator::new();
        let point = Point {
            x: 1.0,
            y: 2.0,
            label: Some("A".to_string()),
            properties: HashMap::new(),
        };

        let description = articulator
            .articulate_geometry(&GeometricStructure::Point(point))
            .unwrap();
        assert!(description.contains("point at coordinates"));
        assert!(description.contains("labeled 'A'"));
    }

    #[test]
    fn test_line_articulation() {
        let articulator = Articulator::new();
        let line = Line {
            start: Point {
                x: 0.0,
                y: 0.0,
                label: None,
                properties: HashMap::new(),
            },
            end: Point {
                x: 1.0,
                y: 0.0,
                label: None,
                properties: HashMap::new(),
            },
            properties: HashMap::new(),
        };

        let description = articulator
            .articulate_geometry(&GeometricStructure::Line(line))
            .unwrap();
        assert!(description.contains("line segment"));
        assert!(description.contains("running horizontally"));
    }

    #[test]
    fn test_polygon_articulation() {
        let articulator = Articulator::new();
        let triangle = Polygon {
            vertices: vec![
                Point {
                    x: 0.0,
                    y: 0.0,
                    label: None,
                    properties: HashMap::new(),
                },
                Point {
                    x: 1.0,
                    y: 0.0,
                    label: None,
                    properties: HashMap::new(),
                },
                Point {
                    x: 0.5,
                    y: 1.0,
                    label: None,
                    properties: HashMap::new(),
                },
            ],
            properties: HashMap::new(),
        };

        let description = articulator
            .articulate_geometry(&GeometricStructure::Polygon(triangle))
            .unwrap();
        assert!(description.contains("triangle"));
    }

    #[test]
    fn test_articulation_styles() {
        let mut articulator = Articulator::new();
        let mut properties = HashMap::new();
        properties.insert("charge".to_string(), 1.0);

        let point = Point {
            x: 1.5,
            y: 2.3,
            label: None,
            properties,
        };

        // Test descriptive style
        articulator.style = ArticulationStyle::Descriptive;
        let desc_desc = articulator
            .articulate_geometry(&GeometricStructure::Point(point.clone()))
            .unwrap();

        // Test technical style
        articulator.style = ArticulationStyle::Technical;
        let desc_tech = articulator
            .articulate_geometry(&GeometricStructure::Point(point.clone()))
            .unwrap();

        // Test mathematical style
        articulator.style = ArticulationStyle::Mathematical;
        let desc_math = articulator
            .articulate_geometry(&GeometricStructure::Point(point))
            .unwrap();

        // Different styles should produce different descriptions
        println!("desc_desc: {desc_desc}");
        println!("desc_tech: {}", desc_tech);
        println!("desc_math: {}", desc_math);
        assert_ne!(desc_desc, desc_tech);
        assert_ne!(desc_desc, desc_math);
        assert_ne!(desc_tech, desc_math);
    }

    #[test]
    fn test_complex_structure_articulation() {
        let articulator = Articulator::new();
        let triangle = GeometricStructure::Polygon(Polygon {
            vertices: vec![
                Point {
                    x: 0.0,
                    y: 0.0,
                    label: None,
                    properties: HashMap::new(),
                },
                Point {
                    x: 1.0,
                    y: 0.0,
                    label: None,
                    properties: HashMap::new(),
                },
                Point {
                    x: 0.5,
                    y: 1.0,
                    label: None,
                    properties: HashMap::new(),
                },
            ],
            properties: HashMap::new(),
        });

        let point = GeometricStructure::Point(Point {
            x: 0.5,
            y: 0.5,
            label: None,
            properties: HashMap::new(),
        });

        let complex = ComplexStructure {
            components: vec![triangle, point],
            relationships: vec![Relationship {
                from: "triangle".to_string(),
                to: "point".to_string(),
                relation_type: "contains".to_string(),
                properties: HashMap::new(),
            }],
        };

        let description = articulator
            .articulate_geometry(&GeometricStructure::Complex(complex))
            .unwrap();
        assert!(description.contains("complex geometric structure"));
        assert!(description.contains("components"));
    }

    #[test]
    fn test_vocabulary_system() {
        let mut rng = rand::rng();
        let vocab = Vocabulary::english();

        let term1 = vocab.get_term("point", &mut rng);
        let term2 = vocab.get_term("point", &mut rng);

        // Should get valid terms
        assert!(term1 == "point" || term1 == "vertex" || term1 == "node");
        assert!(term2 == "point" || term2 == "vertex" || term2 == "node");
    }
}
