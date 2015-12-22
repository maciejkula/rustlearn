//! Validation via repeated random shuffling
//! of the data and splitting into a training and test set.
//!
//! # Examples
//!
//! ```
//! use rustlearn::prelude::*;
//! use rustlearn::datasets::iris;
//! use rustlearn::cross_validation::ShuffleSplit;
//!
//!
//! let (X, y) = iris::load_data();
//!
//! let num_splits = 10;
//! let test_percentage = 0.2;
//!
//! for (train_idx, test_idx) in ShuffleSplit::new(X.rows(), num_splits, test_percentage) {
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

use rand;
use rand::{Rng};


pub struct ShuffleSplit {
    n: usize,
    n_iter: usize,
    test_size: f32,
    rng: rand::StdRng,
    iter: usize,
}


impl ShuffleSplit {
    /// Create a new instance of the shuffle split utility.
    ///
    /// Iterating over it will split the dataset of size `n_samples`
    /// into a train set of `(1.0 - test_size) * n_samples` rows
    /// and a test set of `test_size * n_samples` rows, `n_iter` times.
    pub fn new(n_samples: usize, n_iter: usize, test_size: f32) -> ShuffleSplit {
        ShuffleSplit {
            n: n_samples,
            n_iter: n_iter,
            test_size: test_size,
            rng: rand::StdRng::new().unwrap(),
            iter: 0,
        }
    }

    /// Set the random number generator.
    pub fn set_rng(&mut self, rng: rand::StdRng) {
        self.rng = rng;
    }

    fn get_shuffled_indices(&mut self) -> Vec<usize> {
        let mut indices = (0..self.n).collect::<Vec<usize>>();
        self.rng.shuffle(&mut indices);

        indices
    }
}


impl Iterator for ShuffleSplit {
    type Item = (Vec<usize>, Vec<usize>);
    fn next(&mut self) -> Option<(Vec<usize>, Vec<usize>)> {

        let ret = match self.iter < self.n_iter {
            true => {
                let split_idx: usize = (self.n as f32 * (1.0 - self.test_size)).floor() as usize;
                let shuffled_indices = self.get_shuffled_indices();
                let (train, test) = shuffled_indices.split_at(split_idx);
                Some((train.to_owned(), test.to_owned()))
            }
            false => None,
        };

        self.iter += 1;
        ret
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    extern crate rand;

    use rand::{SeedableRng, StdRng};

    #[test]
    fn iteration() {
        let split = ShuffleSplit::new(100, 4, 0.2);
        let mut count = 0;

        for _ in split {
            count += 1;
        }

        assert!(count == 4);
    }


    #[test]
    fn size_split() {
        let split = ShuffleSplit::new(100, 4, 0.2);

        for (train, test) in split {
            assert!(train.len() == 80);
            assert!(test.len() == 20);
        }
    }


    #[test]
    #[should_panic]
    fn shuffle_differs() {
        let set1 = ShuffleSplit::new(1000, 1, 0.2).collect::<Vec<_>>();
        let set2 = ShuffleSplit::new(1000, 1, 0.2).collect::<Vec<_>>();

        assert!(set1[0].0 == set2[0].0);
    }


    #[test]
    fn set_rng() {

        let seed: &[_] = &[1, 2, 3, 4];
        let rng1: StdRng = SeedableRng::from_seed(seed);
        let rng2: StdRng = SeedableRng::from_seed(seed);

        let mut split1 = ShuffleSplit::new(1000, 1, 0.2);
        let mut split2 = ShuffleSplit::new(1000, 1, 0.2);

        split1.set_rng(rng1);
        split2.set_rng(rng2);

        let set1 = split1.collect::<Vec<_>>();
        let set2 = split2.collect::<Vec<_>>();

        assert!(set1[0].0 == set2[0].0);
    }

}
