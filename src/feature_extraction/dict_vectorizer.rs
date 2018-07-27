//! `DictVectorizer` vectorizers a set of named features into a sparse array
//! via one-hot encoding.
//!
//! # Examples
//!
//! ```
//! use rustlearn::prelude::*;
//! use rustlearn::feature_extraction::DictVectorizer;
//!
//! let features = vec![vec!["feature_one", "feature_two"],
//!                     vec!["feature_two"]];
//!
//! let mut vectorizer = DictVectorizer::new();
//!
//! for (row_idx, row) in features.iter().enumerate() {
//!     for feature in row.iter() {
//!         vectorizer.partial_fit(row_idx, feature, 1.0);
//!     }
//! }
//!
//! let X = vectorizer.transform();
//!
//! assert!(X.rows() == 2 && X.cols() == 2);
//! ```

use std::collections::HashMap;

use prelude::*;

#[derive(Serialize, Deserialize, Default)]
pub struct DictVectorizer {
    dictionary: HashMap<String, (usize, usize)>,
    data: Vec<(usize, usize, f32)>,
}

impl DictVectorizer {
    /// Create a new `DictVectorizer`.
    pub fn new() -> DictVectorizer {
        DictVectorizer {
            dictionary: HashMap::new(),
            data: Vec::new(),
        }
    }

    /// Set the feature value of a named feature in a given row.
    pub fn partial_fit(&mut self, row: usize, name: &str, value: f32) {
        // All of the below is due to the borrow checker's insanity
        // in match statements.
        let mut insert = false;
        let dict_len = self.dictionary.len();

        let col = match self.dictionary.get_mut(name) {
            Some(value) => {
                value.1 += 1;
                value.0
            }
            None => {
                insert = true;
                dict_len
            }
        };

        if insert {
            self.dictionary.insert(name.to_string(), (col, 1));
        }

        self.data.push((row, col, value));
    }

    /// Transform the accumulated data into a sparse array.
    pub fn transform(&self) -> SparseRowArray {
        let rows = self.data.iter().map(|x| x.0).max().unwrap() + 1;
        let cols = self.dictionary.len();

        let mut array = SparseRowArray::zeros(rows, cols);

        for &(row, col, value) in &self.data {
            array.set(row, col, value);
        }

        array
    }

    /// Return a reference to the feature dictionary, mapping
    /// feature names to their (column index, occurrence count).
    pub fn dictionary(&self) -> &HashMap<String, (usize, usize)> {
        &self.dictionary
    }
}
