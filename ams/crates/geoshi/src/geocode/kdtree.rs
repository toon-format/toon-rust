/* src/geocode/kdtree.rs */
//! Lightweight KD-tree for E8 embeddings to accelerate neighbor queries
//! used by the codebase scanner when constructing topologies.

#[derive(Debug, Clone)]
pub struct KdNode {
    pub point: Vec<f64>,
    pub data_index: usize, // Index into the original data array
    pub left: Option<Box<KdNode>>,
    pub right: Option<Box<KdNode>>,
    pub axis: usize,
}

pub struct KdTree {
    pub root: Option<Box<KdNode>>,
    pub dimensions: usize,
}

impl KdTree {
    pub fn new(points: &[(usize, Vec<f64>)], dimensions: usize) -> Self {
        let mut nodes: Vec<_> = points.to_vec();
        let root = Self::build_recursive(&mut nodes, 0, dimensions);
        Self { root, dimensions }
    }

    fn build_recursive(
        points: &mut [(usize, Vec<f64>)],
        depth: usize,
        k: usize,
    ) -> Option<Box<KdNode>> {
        if points.is_empty() {
            return None;
        }

        let axis = depth % k;
        points.sort_by(|a, b| a.1[axis].partial_cmp(&b.1[axis]).unwrap_or(std::cmp::Ordering::Equal));
        let mid = points.len() / 2;

        let (left_slice, right_slice) = points.split_at_mut(mid);
        let (pivot, right_remainder) = right_slice.split_first_mut().unwrap();

        Some(Box::new(KdNode {
            point: pivot.1.clone(),
            data_index: pivot.0,
            left: Self::build_recursive(left_slice, depth + 1, k),
            right: Self::build_recursive(right_remainder, depth + 1, k),
            axis,
        }))
    }

    /// Find all points within `radius` of `target`.
    pub fn radius_search(&self, target: &[f64], radius: f64) -> Vec<usize> {
        let mut results = Vec::new();
        let r_sq = radius * radius;
        self.search_recursive(self.root.as_ref(), target, r_sq, &mut results);
        results
    }

    fn search_recursive(
        &self,
        node: Option<&Box<KdNode>>,
        target: &[f64],
        r_sq: f64,
        results: &mut Vec<usize>,
    ) {
        let Some(node) = node else { return };

        let dist_sq = node
            .point
            .iter()
            .zip(target)
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>();
        if dist_sq <= r_sq {
            results.push(node.data_index);
        }

        let axis = node.axis;
        let diff = target[axis] - node.point[axis];

        let (near, far) = if diff <= 0.0 {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        self.search_recursive(near.as_ref(), target, r_sq, results);

        if diff * diff < r_sq {
            self.search_recursive(far.as_ref(), target, r_sq, results);
        }
    }
}
