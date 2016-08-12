//! A two-class decision tree classifer.
//!
//! This model implements the CART (Classification and Regression Trees)
//! algorithm for both dense and sparse data. The tree is split by
//! randomly sampling `max_features` candidate features, then choosing
//! the best split amongst those features using reduction in Gini impurity.
//!
//! Both binary and numeric features are supported; categorical features
//! without a clear ordering should be one-hot encoded for best results.
//!
//! The model is specified using [hyperparameters](./struct.Hyperparameters.html)
//!
//! # Examples
//!
//! Fitting the model on the iris dataset is straightforward:
//!
//! ```
//! use rustlearn::prelude::*;
//! use rustlearn::trees::decision_tree::Hyperparameters;
//! use rustlearn::datasets::iris;
//!
//! let (X, y) = iris::load_data();
//!
//! let mut model = Hyperparameters::new(4)
//!                                 .min_samples_split(5)
//!                                 .max_depth(40)
//!                                 .one_vs_rest();
//!
//! model.fit(&X, &y).unwrap();
//!
//! let prediction = model.predict(&X).unwrap();
//! ```

use std::cmp::{min, Ordering};
use std::f32;
use std::usize;

use prelude::*;

use multiclass::OneVsRestWrapper;
use utils::{check_data_dimensionality, check_matched_dimensions, check_valid_labels, EncodableRng};

use rand;
use rand::{Rng, StdRng};
use rand::distributions::{IndependentSample, Range};


fn sample_without_replacement<T: Copy, R: Rng>(from: &mut [T],
                                               to: &mut Vec<T>,
                                               number: usize,
                                               rng: &mut R) {

    // A partial Fisher-Yates shuffle for sampling without
    // replacement
    for num_sampled in 0..number {
        let idx = Range::new(num_sampled, from.len()).ind_sample(rng);

        let sampled = from[idx];
        to.push(sampled);

        from.swap(idx, num_sampled);
    }
}


struct FeatureIndices {
    num_used: usize,
    candidate_indices: Vec<usize>,
}


impl FeatureIndices {
    fn new(candidate_indices: Vec<usize>) -> FeatureIndices {
        FeatureIndices {
            num_used: 0,
            candidate_indices: candidate_indices,
        }
    }

    fn sample_indices(&mut self, to: &mut Vec<usize>, number: usize, rng: &mut StdRng) {

        to.clear();

        let number = min(number, self.candidate_indices.len() - self.num_used);

        sample_without_replacement(&mut self.candidate_indices[self.num_used..],
                                   to,
                                   number,
                                   rng);
    }

    fn mark_as_used(&mut self, feature_idx: usize) {

        // Relies on the fact that that features are
        // tested in the same order they are organised
        // in self.candidate_indices (due to the sampling
        // process).

        let indices = &mut self.candidate_indices[self.num_used..];

        // Swap current feature with an unused feature
        // at the beginning of the vector
        // and advance the used features marker
        indices.swap(0, feature_idx);
        self.num_used += 1;
    }
}


#[derive(Debug)]
struct FeatureValues {
    xy_pairs: Vec<(f32, f32)>,
    zero_count: usize,
    zero_y: f32,
    count: usize,
    total_y: f32,
}


impl FeatureValues {
    fn with_capacity(capacity: usize) -> FeatureValues {
        FeatureValues {
            xy_pairs: Vec::with_capacity(capacity),
            zero_count: 0,
            zero_y: 0.0,
            count: 0,
            total_y: 0.0,
        }
    }

    fn push(&mut self, x: f32, y: f32) {

        self.count += 1;
        self.total_y += y;

        if x == 0.0 {

            // We need that one zero value
            // for later finding the optimal split
            if self.zero_count == 0 {
                self.xy_pairs.push((x, 0.0))
            }

            self.zero_count += 1;
            self.zero_y += y;
        } else {
            self.xy_pairs.push((x, y))
        }
    }

    fn fill_remaining_zeros(&mut self, remaining_zero_count: usize, remaining_positives: f32) {

        if self.zero_count == 0 {
            self.xy_pairs.push((0.0, 0.0));
        }

        self.zero_count += remaining_zero_count;
        self.zero_y += remaining_positives;
        self.total_y += remaining_positives;

        self.count += remaining_zero_count;
    }

    fn sort(&mut self) {
        self.xy_pairs.sort_by(|a, b| {
            a.0
                .partial_cmp(&b.0)
                .unwrap_or(Ordering::Equal)
        });
    }

    fn value_bounds(&self) -> (f32, f32) {
        match self.xy_pairs.len() {
            0 => (0.0, 0.0),
            _ => (self.xy_pairs[0].0, self.xy_pairs[self.xy_pairs.len() - 1].0),
        }
    }

    fn feature_type(&self) -> FeatureType {
        match self.xy_pairs.len() {
            1 => FeatureType::Constant,
            2 => FeatureType::Binary,
            _ => FeatureType::Continuous,
        }
    }

    fn clear(&mut self) {
        unsafe {
            self.xy_pairs.set_len(0);
        }
        self.zero_count = 0;
        self.zero_y = 0.0;
        self.total_y = 0.0;
        self.count = 0;
    }
}


#[derive(Clone)]
#[derive(RustcEncodable, RustcDecodable)]
enum FeatureType {
    Constant,
    Binary,
    Continuous,
}


/// Hyperparameters for a `DecisionTree` model.
#[derive(RustcEncodable, RustcDecodable)]
#[derive(Clone)]
pub struct Hyperparameters {
    dim: usize,

    max_features: usize,
    min_samples_split: usize,
    max_depth: usize,

    rng: EncodableRng,
}


impl Hyperparameters {
    /// Creates new Hyperparameters
    ///
    /// # Examples
    ///
    /// ```
    /// use rustlearn::trees::decision_tree::Hyperparameters;
    ///
    /// let model = Hyperparameters::new(4)
    ///                             .min_samples_split(5)
    ///                             .max_depth(40)
    ///                             .build();
    /// ```
    pub fn new(dim: usize) -> Hyperparameters {
        Hyperparameters {
            dim: dim,
            max_features: (dim as f32).sqrt() as usize,
            min_samples_split: 2,
            max_depth: usize::MAX,
            rng: EncodableRng::new(),
        }
    }

    /// Set the maximum number of features to be considered when
    /// finding the best split for the decision tree.
    ///
    /// Defaults to `sqrt(self.dim)`
    pub fn max_features(&mut self, max_features: usize) -> &mut Hyperparameters {
        self.max_features = max_features;
        self
    }
    /// Set the minimum number of samples that must be present
    /// in order for further splitting to take place.
    ///
    /// Defaults to 2.
    pub fn min_samples_split(&mut self, min_samples_split: usize) -> &mut Hyperparameters {
        self.min_samples_split = min_samples_split;
        self
    }
    /// Set the maximum depth of the tree.
    ///
    /// Defaults to `usize::MAX`.
    pub fn max_depth(&mut self, max_depth: usize) -> &mut Hyperparameters {
        self.max_depth = max_depth;
        self
    }
    /// Set the random number generator used for sampling features
    /// to consider at each split.
    pub fn rng(&mut self, rng: rand::StdRng) -> &mut Hyperparameters {
        self.rng.rng = rng;
        self
    }
    /// Build a binary decision tree model.
    pub fn build(&self) -> DecisionTree {
        DecisionTree {
            dim: self.dim,
            max_features: self.max_features,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            root: None,
            feature_types: Vec::new(),
            rng: self.rng.clone(),
        }
    }
    /// Build a one-vs-rest multi-class decision tree model.
    pub fn one_vs_rest(&self) -> OneVsRestWrapper<DecisionTree> {
        let base_model = self.build();

        OneVsRestWrapper::new(base_model)
    }
}


#[derive(RustcEncodable, RustcDecodable)]
#[derive(Clone)]
enum Node {
    Interior {
        feature: usize,
        threshold: f32,
        children: Box<(Node, Node)>,
    },
    Leaf {
        probability: f32,
    },
}


/// A two-class decision tree.
#[derive(RustcEncodable, RustcDecodable)]
#[derive(Clone)]
pub struct DecisionTree {
    dim: usize,

    max_features: usize,
    max_depth: usize,
    min_samples_split: usize,

    root: Option<Node>,
    feature_types: Vec<FeatureType>,
    rng: EncodableRng,
}


impl<'a> SupervisedModel<&'a Array> for DecisionTree {
    fn fit(&mut self, X: &Array, y: &Array) -> Result<(), &'static str> {

        try!(check_data_dimensionality(self.dim, X));
        try!(check_matched_dimensions(X, y));
        try!(check_valid_labels(y));

        self.feature_types = DecisionTree::analyze_features(X);

        let mut feature_values = FeatureValues::with_capacity(X.rows());
        let mut feature_indices = FeatureIndices::new(self.get_nonconstant_feature_indices());
        let mut candidate_features = Vec::with_capacity(self.max_features);

        self.root = Some(self.build_tree(X,
                                         y,
                                         &mut (0..X.rows()).collect::<Vec<usize>>()[..],
                                         &mut feature_indices,
                                         &mut candidate_features,
                                         0,
                                         &mut feature_values,
                                         &DecisionTree::get_values,
                                         &DecisionTree::split_indices));

        Ok(())
    }

    fn decision_function(&self, X: &Array) -> Result<Array, &'static str> {

        try!(check_data_dimensionality(self.dim, X));

        match self.root {
            Some(ref node) => {
                let mut data = Vec::with_capacity(X.rows());
                for row_idx in 0..X.rows() {
                    data.push(self.query_tree(&node, X, row_idx));
                }
                Ok(Array::from(data))
            }
            None => Err("Tree must be built before predicting"),
        }
    }
}


impl<'a> SupervisedModel<&'a SparseColumnArray> for DecisionTree {
    fn fit(&mut self, X: &SparseColumnArray, y: &Array) -> Result<(), &'static str> {

        try!(check_data_dimensionality(self.dim, X));
        try!(check_matched_dimensions(X, y));
        try!(check_valid_labels(y));

        self.feature_types = DecisionTree::analyze_features_sparse(X);

        let mut feature_values = FeatureValues::with_capacity(X.rows());
        let mut feature_indices = FeatureIndices::new(self.get_nonconstant_feature_indices());
        let mut candidate_features = Vec::with_capacity(self.max_features);

        self.root = Some(self.build_tree(X,
                                         y,
                                         &mut (0..X.rows()).collect::<Vec<usize>>()[..],
                                         &mut feature_indices,
                                         &mut candidate_features,
                                         0,
                                         &mut feature_values,
                                         &DecisionTree::get_values_sparse,
                                         &DecisionTree::split_indices_sparse));

        Ok(())
    }

    fn decision_function(&self, X: &SparseColumnArray) -> Result<Array, &'static str> {

        try!(check_data_dimensionality(self.dim, X));

        match self.root {
            Some(ref node) => {
                let mut data = Vec::with_capacity(X.rows());
                for row_idx in 0..X.rows() {
                    data.push(self.query_tree_sparse(node, X, row_idx));
                }
                Ok(Array::from(data))
            }
            None => Err("Tree must be built before predicting"),
        }
    }
}


impl DecisionTree {
    fn analyze_features(X: &Array) -> Vec<FeatureType> {

        let mut features = Vec::with_capacity(X.cols());

        for col in X.iter_columns() {
            let mut values = col.iter().collect::<Vec<_>>();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            values.dedup();

            features.push(match values.len() {
                1 => FeatureType::Constant,
                2 => FeatureType::Binary,
                _ => FeatureType::Continuous,
            });
        }

        features
    }

    fn analyze_features_sparse(X: &SparseColumnArray) -> Vec<FeatureType> {

        let mut features = Vec::with_capacity(X.cols());

        for col in X.iter_columns() {

            let mut values = col.iter_nonzero()
                .map(|(_, val)| val)
                .collect::<Vec<_>>();

            // Add zero as a distrinct value
            if values.len() < X.rows() {
                values.push(0.0);
            }

            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            values.dedup();

            features.push(match values.len() {
                1 => FeatureType::Constant,
                2 => FeatureType::Binary,
                _ => FeatureType::Continuous,
            });
        }

        features
    }

    fn get_nonconstant_feature_indices(&self) -> Vec<usize> {
        self.feature_types
            .iter()
            .enumerate()
            .filter(|&(_, feature_type)| {
                match *feature_type {
                    FeatureType::Constant => false,
                    _ => true,
                }
            })
            .map(|(i, _)| i)
            .collect::<Vec<_>>()
    }

    fn count_positives(y: &Array, indices: &mut [usize]) -> usize {

        let mut count = 0;

        let data = y.data();

        for &row_idx in indices.iter() {
            if data[row_idx] > 0.0 {
                count += 1;
            }
        }

        count
    }

    fn build_tree<T, F, G>(&mut self,
                           X: T,
                           y: &Array,
                           indices: &mut [usize],
                           feature_indices: &mut FeatureIndices,
                           candidate_features: &mut Vec<usize>,
                           depth: usize,
                           feature_values: &mut FeatureValues,
                           get_values: &F,
                           split_indices: &G)
                           -> Node
        where T: Copy,
              F: Fn(T, &Array, usize, usize, &[usize], &mut FeatureValues) -> (),
              G: Fn(T, &mut [usize], usize, f32) -> (&mut [usize], &mut [usize])
    {

        let num_positives = DecisionTree::count_positives(y, indices);
        let probability = num_positives as f32 / indices.len() as f32;

        if probability == 0.0 || probability == 1.0 || depth > self.max_depth ||
           indices.len() < self.min_samples_split {
            return Node::Leaf { probability: probability };
        }

        // Multiple attemps to perform a split.
        for _ in 0..10 {

            feature_indices.sample_indices(candidate_features,
                                           self.max_features,
                                           &mut self.rng.rng);

            let mut best_feature_position = 0;
            let mut best_feature_idx = 0;
            let mut best_feature_threshold = 0.0;
            let mut best_impurity = f32::INFINITY;

            for (feature_position, &feature_idx) in candidate_features.iter().enumerate() {
                get_values(X, y, num_positives, feature_idx, indices, feature_values);

                if let FeatureType::Constant = feature_values.feature_type() {
                    feature_indices.mark_as_used(feature_position);
                    continue;
                }

                let (threshold, impurity) = DecisionTree::calculate_split(feature_values);

                if impurity < best_impurity {
                    best_feature_position = feature_position;
                    best_feature_idx = feature_idx;
                    best_feature_threshold = threshold;
                    best_impurity = impurity;
                }
            }

            let (left_indices, right_indices) =
                split_indices(X, indices, best_feature_idx, best_feature_threshold);

            if left_indices.len() > 0 && right_indices.len() > 0 {

                // Cannot split on binary feature more than one time
                if let FeatureType::Binary = self.feature_types[best_feature_idx] {
                    feature_indices.mark_as_used(best_feature_position);
                }

                let num_used_features = feature_indices.num_used;

                let left = self.build_tree(X,
                                           y,
                                           left_indices,
                                           feature_indices,
                                           candidate_features,
                                           depth + 1,
                                           feature_values,
                                           get_values,
                                           split_indices);

                feature_indices.num_used = num_used_features;

                let right = self.build_tree(X,
                                            y,
                                            right_indices,
                                            feature_indices,
                                            candidate_features,
                                            depth + 1,
                                            feature_values,
                                            get_values,
                                            split_indices);

                return Node::Interior {
                    feature: best_feature_idx,
                    threshold: best_feature_threshold,
                    children: Box::new((left, right)),
                };
            }
        }

        Node::Leaf { probability: probability }
    }

    fn split_indices<'a>(X: &Array,
                         indices: &'a mut [usize],
                         feature_idx: usize,
                         threshold: f32)
                         -> (&'a mut [usize], &'a mut [usize]) {

        let mut num_left = 0;

        for i in 0..indices.len() {

            let row_idx = indices[i];

            if X.get(row_idx, feature_idx) <= threshold {
                indices.swap(i, num_left);
                num_left += 1;
            }
        }

        let (left, right) = indices.split_at_mut(num_left);

        // Need to keep the indices sorted
        right.sort();

        (left, right)
    }

    fn split_indices_sparse<'a>(x: &SparseColumnArray,
                                indices: &'a mut [usize],
                                feature_idx: usize,
                                threshold: f32)
                                -> (&'a mut [usize], &'a mut [usize]) {

        let mut num_left = 0;
        let indices_len = indices.len();

        let x = x.view_column(feature_idx);

        macro_rules! assign {
            ($i:expr, $row_idx:expr, $value:expr) => {{
                if $value <= threshold {
                    indices[$i] = indices[num_left];
                    indices[num_left] = $row_idx;
                    num_left += 1;
                }
            }}
        }

        let mut nonzero_iter = x.iter_nonzero();
        let mut nonzero_option = nonzero_iter.next();

        let mut i = 0;

        while i < indices_len {

            let indices_idx = indices[i];

            if let Some((nonzero_idx, nonzero_value)) = nonzero_option {
                match indices_idx.cmp(&nonzero_idx) {
                    // Haven't reached the first nonzero in column
                    Ordering::Less => {
                        assign!(i, indices_idx, 0.0);
                        i += 1;
                    }
                    // Reached a nonzero at index we are interested in
                    Ordering::Equal => {
                        assign!(i, indices_idx, nonzero_value);
                        i += 1;
                        nonzero_option = nonzero_iter.next();
                    }
                    // Move to the next nonzero value
                    Ordering::Greater => {
                        nonzero_option = nonzero_iter.next();
                    }
                }
            } else {
                assign!(i, indices_idx, 0.0);
                i += 1;
            }
        }

        let (left, right) = indices.split_at_mut(num_left);

        right.sort();

        (left, right)
    }

    fn calculate_split(values: &FeatureValues) -> (f32, f32) {
        let (_, max_value) = values.value_bounds();

        let mut split_impurity = f32::INFINITY;
        let mut split_x = 0.0;

        let total_count = values.count as f32;
        let total_y = values.total_y;

        let mut cumulative_count = 0.0;
        let mut cumulative_y = 0.0;

        for &(x, y) in &values.xy_pairs {

            if x == 0.0 {
                cumulative_count += values.zero_count as f32;
                cumulative_y += values.zero_y;
            } else {
                cumulative_count += 1.0;
                cumulative_y += y;
            }

            if x == max_value {
                continue;
            }

            let left_child_proportion = cumulative_count / total_count;
            let left_child_positive_probability = cumulative_y / cumulative_count;
            let right_child_positive_probability = (total_y - cumulative_y) /
                                                   (total_count - cumulative_count);

            let impurity = DecisionTree::proxy_gini_impurity(left_child_proportion,
                                                             left_child_positive_probability,
                                                             right_child_positive_probability);

            // It's important that this is less than or equal rather
            // than less than: subject to no decrease in impurity
            // it's always good to move to a split at a higher value.
            if impurity <= split_impurity {
                split_impurity = impurity;
                split_x = x;
            }
        }

        (split_x, split_impurity)
    }

    fn proxy_gini_impurity(left_child_proportion: f32,
                           left_child_probability: f32,
                           right_child_probability: f32)
                           -> f32 {

        let right_child_proportion = 1.0 - left_child_proportion;

        let left_impurity = 1.0 - left_child_probability.powi(2) -
                            (1.0 - left_child_probability).powi(2);
        let right_impurity = 1.0 - right_child_probability.powi(2) -
                             (1.0 - right_child_probability).powi(2);

        left_child_proportion * left_impurity + right_child_proportion * right_impurity
    }

    #[allow(unused_variables)]
    fn get_values(X: &Array,
                  y: &Array,
                  num_positives: usize,
                  feature_idx: usize,
                  indices: &[usize],
                  values: &mut FeatureValues) {

        values.clear();

        for &row_idx in indices.iter() {
            values.push(X.get(row_idx, feature_idx), y.get(row_idx, 0));
        }

        values.sort();
    }

    fn get_values_sparse(x: &SparseColumnArray,
                         y: &Array,
                         num_positives: usize,
                         feature_idx: usize,
                         indices: &[usize],
                         values: &mut FeatureValues) {

        let x = x.view_column(feature_idx);

        values.clear();

        let x_nnz = x.nnz() as f32;

        if x_nnz * (indices.len() as f32).log(2.0) < x_nnz + indices.len() as f32 {
            DecisionTree::get_values_sparse_by_search(x, y, num_positives, indices, values);
        } else {
            DecisionTree::get_values_sparse_by_iteration(x, y, num_positives, indices, values);
        }

        values.sort();
    }

    fn get_values_sparse_by_iteration(x: SparseArrayView,
                                      y: &Array,
                                      num_positives: usize,
                                      indices: &[usize],
                                      values: &mut FeatureValues) {

        let mut indices_iter = indices.iter();
        let mut nonzero_iter = x.iter_nonzero();

        let mut indices_option = indices_iter.next();
        let mut nonzero_option = nonzero_iter.next();

        while let Some(&indices_idx) = indices_option {
            if let Some((nonzero_idx, nonzero_value)) = nonzero_option {
                match indices_idx.cmp(&nonzero_idx) {
                    // Haven't reached the first nonzero in column
                    Ordering::Less => {
                        values.push(0.0, y.get(indices_idx, 0));
                        indices_option = indices_iter.next();
                    }
                    // Reached a nonzero at index we are interested in
                    Ordering::Equal => {
                        values.push(nonzero_value, y.get(nonzero_idx, 0));
                        indices_option = indices_iter.next();
                        nonzero_option = nonzero_iter.next();
                    }
                    // Move to the next nonzero value
                    Ordering::Greater => {
                        nonzero_option = nonzero_iter.next();
                    }
                }
            } else {

                // We've exhausted all nonzero indices
                let remaining_zeros = indices.len() - values.count;
                let remaining_positives = num_positives as f32 - values.total_y;

                values.fill_remaining_zeros(remaining_zeros, remaining_positives);
                break;
            }
        }
    }

    fn get_values_sparse_by_search(x: SparseArrayView,
                                   y: &Array,
                                   num_positives: usize,
                                   indices: &[usize],
                                   values: &mut FeatureValues) {

        let y = y.data();

        for (row_idx, value) in x.iter_nonzero() {
            if let Ok(_) = indices.binary_search(&row_idx) {
                values.push(value, y[row_idx]);
            }
        }

        let remaining_zeros = indices.len() - values.count;
        let remaining_positives = num_positives as f32 - values.total_y;

        values.fill_remaining_zeros(remaining_zeros, remaining_positives);
    }

    fn query_tree(&self, node: &Node, x: &Array, row_idx: usize) -> f32 {
        match *node {
            Node::Interior { feature, threshold, ref children } => {
                if x.get(row_idx, feature) <= threshold {
                    self.query_tree(&children.0, x, row_idx)
                } else {
                    self.query_tree(&children.1, x, row_idx)
                }
            }
            Node::Leaf { probability } => probability,
        }
    }

    fn query_tree_sparse(&self, node: &Node, x: &SparseColumnArray, row_idx: usize) -> f32 {
        match *node {
            Node::Interior { feature, threshold, ref children } => {
                if x.get(row_idx, feature) <= threshold {
                    self.query_tree_sparse(&children.0, x, row_idx)
                } else {
                    self.query_tree_sparse(&children.1, x, row_idx)
                }
            }
            Node::Leaf { probability } => probability,
        }
    }
}


#[cfg(test)]
mod tests {
    use prelude::*;

    use cross_validation::cross_validation::CrossValidation;
    use datasets::iris::load_data;
    use metrics::accuracy_score;
    use multiclass::OneVsRestWrapper;
    use super::*;
    use super::FeatureValues;

    use rand::{StdRng, SeedableRng};

    use bincode;

    #[cfg(feature = "all_tests")]
    use datasets::newsgroups;

    extern crate time;

    #[test]
    fn test_gini_impurity() {
        let impurity = DecisionTree::proxy_gini_impurity(0.5, 0.5, 0.5);
        let expected = 0.5;
        assert!(impurity == expected);

        let impurity = DecisionTree::proxy_gini_impurity(0.5, 1.0, 0.0);
        let expected = 0.0;
        assert!(impurity == expected);

        let impurity = DecisionTree::proxy_gini_impurity(0.2, 1.0, 0.5);
        let expected = 0.8 * 0.5;
        assert!(impurity == expected);
    }

    #[test]
    fn calculate_split_1() {

        let x = SparseColumnArray::from(&Array::from(&vec![vec![-1.0], vec![-0.5], vec![0.0],
                                                           vec![0.0], vec![0.0], vec![0.5],
                                                           vec![1.0]]));
        let y = Array::from(vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut values = FeatureValues::with_capacity(y.rows());
        let mut indices = (0..y.rows()).collect::<Vec<_>>();

        DecisionTree::get_values_sparse(&x, &y, 2, 0, &mut indices[..], &mut values);

        let (threshold, impurity) = DecisionTree::calculate_split(&values);

        assert!(threshold == -0.5);
        assert!(impurity == 0.0);
    }

    #[test]
    fn calculate_split_2() {

        let x = SparseColumnArray::from(&Array::from(&vec![vec![-1.0], vec![-0.5], vec![0.0],
                                                           vec![0.0], vec![0.0], vec![0.5],
                                                           vec![1.0]]));
        let y = Array::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]);
        let mut values = FeatureValues::with_capacity(y.rows());
        let mut indices = (0..y.rows()).collect::<Vec<_>>();

        DecisionTree::get_values_sparse(&x, &y, 5, 0, &mut indices[..], &mut values);

        let (threshold, impurity) = DecisionTree::calculate_split(&values);

        assert!(threshold == 0.0);
        assert!(impurity == 0.0);
    }

    #[test]
    fn test_indices_split() {
        let x =
            Array::from(&vec![vec![-1.0, 1.0], vec![-0.5, 1.0], vec![0.5, 0.0], vec![1.0, 0.0]]);
        let mut indices = vec![0, 1, 2, 3];
        let (left, right) = DecisionTree::split_indices(&x, &mut indices[..], 0, -0.5);
        assert!(left.to_owned() == vec![0, 1]);
        assert!(right.to_owned() == vec![2, 3]);
    }

    #[test]
    fn test_indices_split_sparse() {
        let x = SparseColumnArray::from(&Array::from(&vec![vec![-1.0, 1.0],
                                                           vec![0.0, 2.0],
                                                           vec![0.0, 3.0],
                                                           vec![1.0, 0.0],
                                                           vec![0.0, 0.0]]));
        let mut indices = (0..5).collect::<Vec<_>>();
        let ind_bor = &mut indices[..];

        let (left, right) = DecisionTree::split_indices_sparse(&x, ind_bor, 0, 0.0);

        assert!(left.to_owned() == vec![0, 1, 2, 4]);
        assert!(right.to_owned() == vec![3]);

        let mut indices = (0..5).collect::<Vec<_>>();
        let ind_bor = &mut indices[..];

        let (left, right) = DecisionTree::split_indices_sparse(&x, ind_bor, 1, 0.5);

        assert!(left.to_owned() == vec![3, 4]);
        assert!(right.to_owned() == vec![0, 1, 2]);
    }


    #[test]
    fn test_get_values_sparse() {
        let x = SparseColumnArray::from(&Array::from(&vec![vec![-1.0, 1.0],
                                                           vec![0.0, 2.0],
                                                           vec![0.0, 3.0],
                                                           vec![1.0, 0.0],
                                                           vec![0.0, 0.0]]));
        let y = Array::from(vec![0.0, 1.0, 0.0, 1.0, 1.0]);
        let mut values = FeatureValues::with_capacity(5);
        let mut indices = (0..5).collect::<Vec<_>>();

        DecisionTree::get_values_sparse(&x, &y, 3, 0, &mut indices[..], &mut values);

        assert!(values.xy_pairs == vec![(-1.0, 0.0), (0.0, 0.0), (1.0, 1.0)]);
        assert!(values.zero_count == 3);
        assert!(values.zero_y == 2.0);
        assert!(values.total_y == 3.0);

        values.clear();

        DecisionTree::get_values_sparse(&x, &y, 3, 1, &mut indices[..], &mut values);

        assert!(values.xy_pairs == vec![(0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (3.0, 0.0)]);
        assert!(values.zero_count == 2);
        assert!(values.zero_y == 2.0);
        assert!(values.total_y == 3.0);
    }

    #[test]
    fn test_basic_tree_building() {
        let X = Array::from(&vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        let y = Array::from(vec![1.0, 0.0]);

        let params = Hyperparameters::new(2);
        let mut model = params.build();

        model.fit(&X, &y).unwrap();

        assert!(allclose(&y, &model.decision_function(&X).unwrap()));
    }

    #[test]
    fn test_decision_tree_iris() {
        let (data, target) = load_data();

        let mut test_accuracy = 0.0;

        let no_splits = 10;

        let mut cv = CrossValidation::new(data.rows(), no_splits);
        cv.set_rng(StdRng::from_seed(&[100]));

        for (train_idx, test_idx) in cv {

            let x_train = data.get_rows(&train_idx);
            let x_test = data.get_rows(&test_idx);

            let y_train = target.get_rows(&train_idx);

            let mut model = Hyperparameters::new(data.cols())
                .min_samples_split(5)
                .max_features(4)
                .rng(StdRng::from_seed(&[100]))
                .one_vs_rest();

            model.fit(&x_train, &y_train).unwrap();

            let test_prediction = model.predict(&x_test).unwrap();

            test_accuracy += accuracy_score(&target.get_rows(&test_idx), &test_prediction);
        }

        test_accuracy /= no_splits as f32;

        println!("Accuracy {}", test_accuracy);

        assert!(test_accuracy > 0.96);
    }

    #[test]
    fn test_decision_tree_iris_sparse() {
        let (data, target) = load_data();

        let mut test_accuracy = 0.0;

        let no_splits = 10;

        let mut cv = CrossValidation::new(data.rows(), no_splits);
        cv.set_rng(StdRng::from_seed(&[100]));

        for (train_idx, test_idx) in cv {

            let x_train = SparseColumnArray::from(&data.get_rows(&train_idx));
            let x_test = SparseColumnArray::from(&data.get_rows(&test_idx));

            let y_train = target.get_rows(&train_idx);

            let mut model = Hyperparameters::new(data.cols())
                .min_samples_split(5)
                .max_features(4)
                .rng(StdRng::from_seed(&[100]))
                .one_vs_rest();

            model.fit(&x_train, &y_train).unwrap();

            let test_prediction = model.predict(&x_test).unwrap();

            test_accuracy += accuracy_score(&target.get_rows(&test_idx), &test_prediction);
        }

        test_accuracy /= no_splits as f32;

        println!("Accuracy {}", test_accuracy);

        assert!(test_accuracy > 0.96);
    }

    #[test]
    fn serialization() {
        let (data, target) = load_data();

        let mut test_accuracy = 0.0;

        let no_splits = 10;

        let mut cv = CrossValidation::new(data.rows(), no_splits);
        cv.set_rng(StdRng::from_seed(&[100]));

        for (train_idx, test_idx) in cv {

            let x_train = data.get_rows(&train_idx);
            let x_test = data.get_rows(&test_idx);

            let y_train = target.get_rows(&train_idx);

            let mut model = Hyperparameters::new(data.cols())
                .min_samples_split(5)
                .max_features(4)
                .rng(StdRng::from_seed(&[100]))
                .one_vs_rest();

            model.fit(&x_train, &y_train).unwrap();

            let encoded = bincode::rustc_serialize::encode(&model, bincode::SizeLimit::Infinite)
                .unwrap();
            let decoded: OneVsRestWrapper<DecisionTree> =
                bincode::rustc_serialize::decode(&encoded).unwrap();

            let test_prediction = decoded.predict(&x_test).unwrap();

            test_accuracy += accuracy_score(&target.get_rows(&test_idx), &test_prediction);
        }

        test_accuracy /= no_splits as f32;

        println!("Accuracy {}", test_accuracy);

        assert!(test_accuracy > 0.96);
    }

    #[test]
    #[cfg(feature = "all_tests")]
    fn test_decision_tree_newsgroups() {

        let (X, target) = newsgroups::load_data();

        let no_splits = 2;

        let mut test_accuracy = 0.0;
        let mut train_accuracy = 0.0;

        let mut cv = CrossValidation::new(X.rows(), no_splits);
        cv.set_rng(StdRng::from_seed(&[100]));

        for (train_idx, test_idx) in cv {

            let x_train = SparseColumnArray::from(&X.get_rows(&train_idx));

            let x_test = SparseColumnArray::from(&X.get_rows(&test_idx));
            let y_train = target.get_rows(&train_idx);

            let mut model = Hyperparameters::new(X.cols())
                .min_samples_split(5)
                .rng(StdRng::from_seed(&[100]))
                .one_vs_rest();

            let start = time::precise_time_ns();
            model.fit(&x_train, &y_train).unwrap();
            println!("Elapsed {}", time::precise_time_ns() - start);

            let y_hat = model.predict(&x_test).unwrap();
            let y_hat_train = model.predict(&x_train).unwrap();

            test_accuracy += accuracy_score(&target.get_rows(&test_idx), &y_hat);

            train_accuracy += accuracy_score(&target.get_rows(&train_idx), &y_hat_train);
        }

        test_accuracy /= no_splits as f32;
        train_accuracy /= no_splits as f32;
        println!("{}", test_accuracy);
        println!("train accuracy {}", train_accuracy);

        assert!(train_accuracy > 0.95);
    }

    #[test]
    #[cfg(feature = "all_tests")]
    fn test_decision_tree_newsgroups_parallel() {

        let (X, target) = newsgroups::load_data();

        let no_splits = 2;

        let mut test_accuracy = 0.0;
        let mut train_accuracy = 0.0;

        let mut cv = CrossValidation::new(X.rows(), no_splits);
        cv.set_rng(StdRng::from_seed(&[100]));

        for (train_idx, test_idx) in cv {

            let x_train = SparseColumnArray::from(&X.get_rows(&train_idx));

            let x_test = SparseColumnArray::from(&X.get_rows(&test_idx));
            let y_train = target.get_rows(&train_idx);

            let mut model = Hyperparameters::new(X.cols())
                .min_samples_split(5)
                .rng(StdRng::from_seed(&[100]))
                .one_vs_rest();

            let start = time::precise_time_ns();
            for _ in 0..2 {
                model.fit_parallel(&x_train, &y_train, 2).unwrap();
            }
            println!("Elapsed {}", time::precise_time_ns() - start);

            let y_hat = model.predict_parallel(&x_test, 2).unwrap();
            let y_hat_train = model.predict_parallel(&x_train, 2).unwrap();

            test_accuracy += accuracy_score(&target.get_rows(&test_idx), &y_hat);

            train_accuracy += accuracy_score(&target.get_rows(&train_idx), &y_hat_train);
        }

        test_accuracy /= no_splits as f32;
        train_accuracy /= no_splits as f32;
        println!("{}", test_accuracy);
        println!("train accuracy {}", train_accuracy);

        assert!(train_accuracy > 0.95);
    }
}


#[cfg(feature = "bench")]
#[allow(unused_imports)]
mod bench {

    use prelude::*;

    use datasets::iris::load_data;
    use datasets::newsgroups;
    use super::Hyperparameters;

    use rand::{Rng, StdRng, SeedableRng};

    use test::Bencher;

    #[bench]
    fn bench_wide(b: &mut Bencher) {

        let rows = 100;
        let cols = 5000;

        let mut rng = StdRng::new().unwrap();

        let mut X = Array::from((0..(rows * cols))
            .map(|_| rng.next_f32())
            .collect::<Vec<_>>());
        X.reshape(rows, cols);

        let y = Array::from((0..rows)
            .map(|_| *rng.choose(&vec![0.0, 1.0][..]).unwrap())
            .collect::<Vec<_>>());

        let mut model = Hyperparameters::new(cols)
            .min_samples_split(5)
            .rng(StdRng::from_seed(&[100]))
            .build();

        b.iter(|| {
            model.fit(&X, &y).unwrap();
        });
    }

    #[bench]
    fn bench_tall(b: &mut Bencher) {

        let rows = 5000;
        let cols = 10;

        let mut rng = StdRng::new().unwrap();

        let mut X = Array::from((0..(rows * cols))
            .map(|_| rng.next_f32())
            .collect::<Vec<_>>());
        X.reshape(rows, cols);

        let y = Array::from((0..rows)
            .map(|_| *rng.choose(&vec![0.0, 1.0][..]).unwrap())
            .collect::<Vec<_>>());


        let mut model = Hyperparameters::new(cols)
            .min_samples_split(5)
            .rng(StdRng::from_seed(&[100]))
            .build();

        b.iter(|| {
            model.fit(&X, &y).unwrap();
        });
    }

    #[bench]
    fn bench_wide_sparse(b: &mut Bencher) {

        let rows = 100;
        let cols = 5000;

        let mut rng = StdRng::new().unwrap();

        let mut X = Array::from((0..(rows * cols))
            .map(|_| {
                match rng.gen_weighted_bool(4) {
                    true => rng.next_f32(),
                    false => 0.0,
                }
            })
            .collect::<Vec<_>>());
        X.reshape(rows, cols);

        let X = SparseColumnArray::from(&X);

        let y = Array::from((0..rows)
            .map(|_| *rng.choose(&vec![0.0, 1.0][..]).unwrap())
            .collect::<Vec<_>>());

        let mut model = Hyperparameters::new(cols)
            .min_samples_split(5)
            .rng(StdRng::from_seed(&[100]))
            .build();

        b.iter(|| {
            model.fit(&X, &y).unwrap();
        });
    }

    #[bench]
    fn bench_tall_sparse(b: &mut Bencher) {

        let rows = 5000;
        let cols = 100;

        let mut rng = StdRng::new().unwrap();

        let mut X = Array::from((0..(rows * cols))
            .map(|_| {
                match rng.gen_weighted_bool(4) {
                    true => rng.next_f32(),
                    false => 0.0,
                }
            })
            .collect::<Vec<_>>());
        X.reshape(rows, cols);

        let X = SparseColumnArray::from(&X);

        let y = Array::from((0..rows)
            .map(|_| *rng.choose(&vec![0.0, 1.0][..]).unwrap())
            .collect::<Vec<_>>());

        let mut model = Hyperparameters::new(cols)
            .min_samples_split(5)
            .rng(StdRng::from_seed(&[100]))
            .build();

        b.iter(|| {
            model.fit(&X, &y).unwrap();
        });
    }

    #[bench]
    fn bench_decision_tree_iris(b: &mut Bencher) {
        let (data, target) = load_data();

        let mut model = Hyperparameters::new(data.cols())
            .min_samples_split(5)
            .max_features(4)
            .rng(StdRng::from_seed(&[100]))
            .one_vs_rest();

        b.iter(|| {
            model.fit(&data, &target).unwrap();
        });
    }

    #[bench]
    fn bench_decision_tree_newsgroups(b: &mut Bencher) {

        let (X, target) = newsgroups::load_data();

        let x_train = SparseColumnArray::from(&X.get_rows(&(..500)));
        let target = target.get_rows(&(..500));

        let mut model = Hyperparameters::new(X.cols())
            .min_samples_split(5)
            .rng(StdRng::from_seed(&[100]))
            .one_vs_rest();

        b.iter(|| {
            model.fit(&x_train, &target).unwrap();
        });
    }

    #[bench]
    fn bench_decision_tree_newsgroups_parallel(b: &mut Bencher) {

        let (X, target) = newsgroups::load_data();

        let x_train = SparseColumnArray::from(&X.get_rows(&(..500)));
        let target = target.get_rows(&(..500));

        let mut model = Hyperparameters::new(X.cols())
            .min_samples_split(5)
            .rng(StdRng::from_seed(&[100]))
            .one_vs_rest();

        b.iter(|| {
            model.fit_parallel(&x_train, &target, 2).unwrap();
        });
    }
}
