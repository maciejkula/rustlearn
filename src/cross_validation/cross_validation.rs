//! K-fold cross validation.
//!
//! # Examples
//!
//! ```
//! use rustlearn::prelude::*;
//! use rustlearn::datasets::iris;
//! use rustlearn::cross_validation::CrossValidation;
//!
//!
//! let (X, y) = iris::load_data();
//!
//! let num_splits = 10;
//!
//! for (train_idx, test_idx) in CrossValidation::new(X.rows(), num_splits) {
//!
//!     let X_train = X.get_rows(&train_idx);
//!     let y_train = y.get_rows(&train_idx);
//!     let X_test = X.get_rows(&test_idx);
//!     let y_test = y.get_rows(&test_idx);
//!
//!     // Model fitting happens here
//! }
//! ```

use std::iter::Iterator;

extern crate rand;

use rand::{Rng, StdRng};


pub struct CrossValidation {
    n_samples: usize,
    n_folds: usize,
    indices: Vec<usize>,
    iter: usize,
    rng: StdRng
}


impl CrossValidation {
    /// Create a new instance of the cross validation utility.
    ///
    /// # Panics
    /// Panics if `n_folds < n_samples` or `n_folds <= 1`.
    pub fn new(n_samples: usize, n_folds: usize) -> CrossValidation {

        assert!(n_folds < n_samples, "Number of folds must be smaller than number of samples");
        assert!(n_folds > 1, "Number of folds must be greater than one");

        let mut indices = (0..n_samples).collect::<Vec<_>>();
        let mut rng = rand::StdRng::new().unwrap();

        rng.shuffle(&mut indices);
        
        CrossValidation {n_samples: n_samples,
                         n_folds: n_folds,
                         indices: indices,
                         iter: 0,
                         rng: rng}
    }

    /// Fix the random number generator.
    pub fn set_rng(&mut self, rng: rand::StdRng) {

        self.rng = rng;

        self.indices = (0..self.n_samples).collect::<Vec<_>>();
        self.rng.shuffle(&mut self.indices);
    }
}


impl Iterator for CrossValidation {
    type Item = (Vec<usize>, Vec<usize>);

    fn next(&mut self) -> Option<(Vec<usize>, Vec<usize>)> {

        let ret = match self.iter < self.n_folds {
            true => {
                let fold_step = self.n_samples / self.n_folds;

                let validation_start = self.iter * fold_step;
                let validation_stop = (self.iter + 1) * fold_step;

                let train = (0..validation_start)
                    .chain(validation_stop..self.indices.len())
                    .map(|i| self.indices[i]).collect::<Vec<_>>();
                let test = (validation_start..validation_stop)
                    .map(|i| self.indices[i]).collect::<Vec<_>>();

                Some((train, test))
            },
            false => None,
        };

        self.iter += 1;

        ret
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    extern crate rand;

    #[test]
    fn cross_validate_iteration() {
        let split = CrossValidation::new(100, 4);
        let mut count = 0;

        for _ in split {
            count += 1;
        }

        assert!(count == 4);
    }


    #[test]
    fn cross_validate_size_split() {
        let split = CrossValidation::new(100, 4);

        for (train, test) in split {

            let mut set = HashSet::new();
            
            assert!(train.len() == 75);
            assert!(test.len() == 25);

            for idx in train.iter().chain(test.iter()) {
                set.insert(idx);
            }

            assert!(set.len() == 100);
        }
    }
}
