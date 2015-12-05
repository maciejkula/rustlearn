//! A two-class logistic regression classifier implemented using stochastic gradient descent.
//!
//! This model implements a two-class logistic regression classifier, using stochastic
//! gradient descent with an adaptive per-parameter learning rate (Adagrad). The model
//! can be regularized using L2 and L1 regularization, and supports fitting on both
//! dense and sparse data.
//!
//! Repeated calls to the `fit` function are equivalent to running
//! multiple epochs of training.
//!
//! # Examples
//!
//! Fitting the model on the iris dataset is straightforward:
//!
//! ```
//! use rustlearn::prelude::*;
//! use rustlearn::linear_models::sgdclassifier::Hyperparameters;
//! use rustlearn::datasets::iris;
//!
//! let (X, y) = iris::load_data();
//!
//! let mut model = Hyperparameters::new(4)
//!                                 .learning_rate(1.0)
//!                                 .l2_penalty(0.5)
//!                                 .l1_penalty(0.0)
//!                                 .one_vs_rest();
//!
//! model.fit(&X, &y).unwrap();
//!
//! let prediction = model.predict(&X).unwrap();
//! ```
//!
//! To run multiple epochs of training, use repeated calls to
//! `fit`:
//!
//! ```
//! use rustlearn::prelude::*;
//! use rustlearn::linear_models::sgdclassifier::Hyperparameters;
//! use rustlearn::datasets::iris;
//!
//! let (X, y) = iris::load_data();
//!
//! let mut model = Hyperparameters::new(4)
//!                                 .learning_rate(1.0)
//!                                 .l2_penalty(0.5)
//!                                 .l1_penalty(0.0)
//!                                 .one_vs_rest();
//!
//! let num_epochs = 20;
//!
//! for _ in 0..num_epochs {
//!     model.fit(&X, &y).unwrap();
//! }
//!
//! let prediction = model.predict(&X).unwrap();
//! ```

use std::iter::Iterator;

use prelude::*;

use multiclass::OneVsRestWrapper;


/// Hyperparameters for a SGDClassifier model.
#[derive(RustcEncodable, RustcDecodable)]
pub struct Hyperparameters {
    dim: usize,

    learning_rate: f32,
    l2_penalty: f32,
    l1_penalty: f32,
}


impl Hyperparameters {
    /// Creates new Hyperparameters.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustlearn::prelude::*;
    /// use rustlearn::linear_models::sgdclassifier::Hyperparameters;
    ///
    ///
    /// let mut model = Hyperparameters::new(4)
    ///                                 .learning_rate(1.0)
    ///                                 .l2_penalty(0.5)
    ///                                 .l1_penalty(0.0)
    ///                                 .build();
    /// ```
    pub fn new(dim: usize) -> Hyperparameters {
        Hyperparameters {dim: dim,
                         learning_rate: 0.05,
                         l2_penalty: 0.0,
                         l1_penalty: 0.0}
    }
    /// Set the initial learning rate.
    ///
    /// During fitting, the learning rate decreases more for parameters which have
    /// have received larger gradient updates. This maintains more stable estimates
    /// for common features while allowing fast learning for rare features.
    pub fn learning_rate(&mut self, learning_rate: f32) -> &mut Hyperparameters {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the L2 penalty.
    pub fn l2_penalty(&mut self, l2_penalty: f32) -> &mut Hyperparameters {
        self.l2_penalty = l2_penalty;
        self
    }

    /// Set the L1 penalty.
    ///
    /// Coefficient sparsity is achieved by truncating at zero whenever
    /// a coefficient update would change its sign.
    pub fn l1_penalty(&mut self, l1_penalty: f32) -> &mut Hyperparameters {
        self.l1_penalty = l1_penalty;
        self
    }

    /// Build a two-class model.
    pub fn build(&self) -> SGDClassifier {
        SGDClassifier {dim: self.dim,
                       learning_rate: self.learning_rate,
                       l2_penalty: self.l2_penalty,
                       l1_penalty: self.l1_penalty,
                       coefficients: Array::zeros(self.dim, 1),
                       gradsq: Array::ones(self.dim, 1),
                       applied_l1: Array::zeros(self.dim, 1),
                       applied_l2: Array::ones(self.dim, 1),
                       accumulated_l1: 0.0,
                       accumulated_l2: 1.0}
    }
    
    /// Build a one-vs-rest multiclass model.
    pub fn one_vs_rest(&self) -> OneVsRestWrapper<SGDClassifier> {
        let base_model = self.build();

        OneVsRestWrapper::new(base_model)
    }
}

/// A two-class linear regression classifier implemented using stochastic gradient descent.
#[derive(RustcEncodable, RustcDecodable)]
#[derive(Clone)]
pub struct SGDClassifier {
    dim: usize,

    learning_rate: f32,
    l2_penalty: f32,
    l1_penalty: f32,

    coefficients: Array,
    gradsq: Array,
    applied_l1: Array,
    applied_l2: Array,
    accumulated_l1: f32,
    accumulated_l2: f32,
}


fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}


fn logistic_loss(y: f32, y_hat: f32) -> f32 {
    y_hat - y
}


macro_rules! adagrad_updates {
    ($coefficients:expr, $x:expr, $gradsq:expr) => {{
        $coefficients.iter_mut()
            .zip($x.iter())
            .zip(($gradsq.iter_mut()))
        }}
}


macro_rules! max {
    ($x:expr, $y:expr) => {{
        match $x > $y {
            true => $x,
            false => $y,
        }
    }}
}


macro_rules! min {
    ($x:expr, $y:expr) => {{
        match $x < $y {
            true => $x,
            false => $y,
        }
    }}
}


impl SupervisedModel<Array> for SGDClassifier {
    
    fn fit(&mut self, X: &Array, y: &Array) -> Result<(), &'static str> {

        assert!(X.cols() == self.dim);
        assert!(X.rows() == y.rows());
        assert!(y.cols() == 1);

        for (row, &true_y) in X.iter_rows().zip(y.data().iter()) {
            let y_hat = self.compute_prediction(&row);
            let loss = logistic_loss(true_y, y_hat);
            self.update(&row, loss);
        }

        for idx in 0..self.dim {
            self.apply_regularization(idx);
        }

        Ok(())
    }

    fn decision_function(&self, x: &Array) -> Result<Array, &'static str> {

        assert!(x.cols() == self.dim);

        let mut data = Vec::with_capacity(x.rows());

        for row in x.iter_rows() {
            data.push(self.compute_prediction(&row));
        }

        Ok(Array::from(data))
    }
}


impl SupervisedModel<SparseRowArray> for SGDClassifier {

    fn fit(&mut self, X: &SparseRowArray, y: &Array) -> Result<(), &'static str> {

        assert!(X.cols() == self.dim);
        assert!(X.rows() == y.rows());
        assert!(y.cols() == 1);

        for (row, &true_y) in X.iter_rows().zip(y.data().iter()) {
            let y_hat = self.compute_prediction(&row);
            let loss = logistic_loss(true_y, y_hat);
            self.update(&row, loss);
        }

        for idx in 0..self.dim {
            self.apply_regularization(idx);
        }

        Ok(())
    }

    fn decision_function(&self, x: &SparseRowArray) -> Result<Array, &'static str> {

        assert!(x.cols() == self.dim);

        let mut data = Vec::with_capacity(x.rows());

        for row in x.iter_rows() {
            data.push(self.compute_prediction(&row));
        }

        Ok(Array::from(data))
    }
}


impl SGDClassifier {

    /// Returns a reference to the estimated coefficients vector.
    pub fn get_coefficients(&self) -> &Array {
        &self.coefficients
    }

    fn update_at_idx(&mut self, idx: usize, update: f32) {

        let gradsq = self.gradsq.get(idx, 0);
        
        let local_learning_rate = self.learning_rate / gradsq.sqrt();
        
        *self.coefficients.get_mut(idx, 0) -= local_learning_rate * update;
        *self.gradsq.get_mut(idx, 0) += update.powi(2);
    }

    fn update<T: NonzeroIterable>(&mut self, x: T, loss: f32) {

        for (idx, gradient) in x.iter_nonzero() {
            
            self.update_at_idx(idx, loss * gradient);
            self.apply_regularization(idx);
        }

        self.accumulate_regularization();
    }

    fn accumulate_regularization(&mut self) {
        self.accumulated_l1 += self.l1_penalty;
        self.accumulated_l2 *= 1.0 - self.l2_penalty;
    }

    fn apply_regularization(&mut self, coefficient_index: usize) {

        let idx = coefficient_index;
        let coefficient = self.coefficients.get_mut(idx, 0);
        let applied_l2 = self.applied_l2.get_mut(idx, 0);
        let applied_l1 = self.applied_l1.get_mut(idx, 0);

        let local_learning_rate = self.learning_rate / self.gradsq.get(idx, 0).sqrt();
        let l2_update = self.accumulated_l2 / *applied_l2;

        *coefficient *= 1.0 - (1.0 - l2_update) * local_learning_rate;
        *applied_l2 *= l2_update;

        let pre_update_coeff = coefficient.clone();
        let l1_potential_update = self.accumulated_l1 - *applied_l1;

        match *coefficient > 0.0 {
            true => {
                *coefficient = max!(0.0,
                                    *coefficient - local_learning_rate * l1_potential_update);
            },
            false => {
                *coefficient = min!(0.0,
                                    *coefficient + local_learning_rate * l1_potential_update)
            }
        }

        let l1_actual_update = (pre_update_coeff - *coefficient).abs();
        *applied_l1 += l1_actual_update;
    }

    fn compute_prediction<T: NonzeroIterable>(&self, row: T) -> f32 {

        let mut prediction = 0.0;

        for (idx, value) in row.iter_nonzero() {
            prediction += self.coefficients.get(idx, 0) * value;
        }

        sigmoid(prediction)
    }

}


#[cfg(test)]
mod tests {
    use rand::{StdRng, SeedableRng};

    use prelude::*;

    use cross_validation::cross_validation::CrossValidation;
    use datasets::iris::load_data;
    use metrics::accuracy_score;
    use multiclass::OneVsRestWrapper;

    use super::*;

    use bincode;

    #[cfg(feature = "all_tests")]
    use csv;

    #[test]
    fn basic_updating() {

        let mut model = Hyperparameters::new(2)
            .learning_rate(0.01)
            .l2_penalty(0.0)
            .l1_penalty(0.0)
            .build();

        let y = Array::ones(1, 1);
        let X = Array::from(&vec![vec![1.0, -0.1]]);

        model.fit(&X, &y).unwrap();

        assert!(model.gradsq.data()[0] > 1.0);
        assert!(model.gradsq.data()[1] > 1.0);

        assert!(model.coefficients.data()[0] == 0.005);
        assert!(model.coefficients.data()[1] == -0.0005);

        model.fit(&X, &y).unwrap();

        assert!(model.coefficients.data()[0] == 0.009460844);
        assert!(model.coefficients.data()[1] == -0.0009981153);
    }

    #[test]
    fn basic_regularization() {
        let mut model = Hyperparameters::new(2)
            .learning_rate(1.0)
            .l2_penalty(0.5)
            .l1_penalty(0.0)
            .build();

        let y = Array::ones(1, 1);
        let X = Array::from(&vec![vec![0.0, 0.0]]);

        model.coefficients.as_mut_slice()[0] = 1.0;
        model.coefficients.as_mut_slice()[1] = -1.0;

        model.fit(&X, &y).unwrap();

        assert!(model.coefficients.data()[0] == 0.5);
        assert!(model.coefficients.data()[1] == -0.5);

        let mut model = Hyperparameters::new(2)
            .learning_rate(1.0)
            .l2_penalty(0.0)
            .l1_penalty(0.5)
            .build();

        model.coefficients.as_mut_slice()[0] = 1.0;
        model.coefficients.as_mut_slice()[1] = -1.0;

        model.fit(&X, &y).unwrap();

        assert!(model.coefficients.data()[0] == 0.5);
        assert!(model.coefficients.data()[1] == -0.5);

        // Should be regularised away to zero
        for _ in 0..10 {
            model.fit(&X, &y).unwrap();
        }

        assert!(model.coefficients.data()[0] == 0.0);
        assert!(model.coefficients.data()[1] == 0.0);
    }

    #[test]
    fn test_sparse_regularization() {
        let mut model = Hyperparameters::new(2)
            .learning_rate(1.0)
            .l2_penalty(0.001)
            .l1_penalty(0.00001)
            .build();

        let y = Array::ones(1, 1);
        let X = SparseRowArray::from(&Array::from(&vec![vec![1.0, 0.0]]));

        for _ in 0..10 {
            model.fit(&X, &y).unwrap();
        }

        // Feature 0 appeared many times, its coefficient
        // should be high.
        assert!(model.coefficients.get(0, 0) > 1.0);

        // Make the feature disappear. It should be regularized away.
        let X = SparseRowArray::zeros(1, 2);

        for _ in 0..10000 {
            model.fit(&X, &y).unwrap();
        }

        assert!(model.coefficients.get(0, 0) == 0.0);
    }

    #[test]
    fn test_iris() {
        let (data, target) = load_data();

        let mut test_accuracy = 0.0;

        let no_splits = 10;

        let mut cv = CrossValidation::new(data.rows(),
                                          no_splits);
        cv.set_rng(StdRng::from_seed(&[100]));

        for (train_idx, test_idx) in cv {

            let x_train = data.get_rows(&train_idx);
            let x_test = data.get_rows(&test_idx);

            let y_train = target.get_rows(&train_idx);

            let mut model = Hyperparameters::new(data.cols())
                .learning_rate(0.5)
                .l2_penalty(0.0)
                .l1_penalty(0.0)
                .one_vs_rest();

            for _ in 0..20 {
                model.fit(&x_train, &y_train).unwrap();
            }

            let y_hat = model.predict(&x_test).unwrap();

            test_accuracy += accuracy_score(
                &target.get_rows(&test_idx),
                &y_hat);
        }

        test_accuracy /= no_splits as f32;

        println!("Accuracy {}", test_accuracy);

        assert!(test_accuracy > 0.9);
    }

    #[test]
    fn serialization() {
        let (data, target) = load_data();

        let mut test_accuracy = 0.0;

        let no_splits = 10;

        let mut cv = CrossValidation::new(data.rows(),
                                          no_splits);
        cv.set_rng(StdRng::from_seed(&[100]));

        for (train_idx, test_idx) in cv {

            let x_train = data.get_rows(&train_idx);
            let x_test = data.get_rows(&test_idx);

            let y_train = target.get_rows(&train_idx);

            let mut model = Hyperparameters::new(data.cols())
                .learning_rate(0.5)
                .l2_penalty(0.0)
                .l1_penalty(0.0)
                .one_vs_rest();

            for _ in 0..20 {
                model.fit(&x_train, &y_train).unwrap();
            }

            let encoded = bincode::rustc_serialize::encode(&model,
                                                           bincode::SizeLimit::Infinite).unwrap();
            let decoded: OneVsRestWrapper<SGDClassifier> = bincode::rustc_serialize::decode(&encoded).unwrap();

            let y_hat = decoded.predict(&x_test).unwrap();

            test_accuracy += accuracy_score(
                &target.get_rows(&test_idx),
                &y_hat);
        }

        test_accuracy /= no_splits as f32;

        println!("Accuracy {}", test_accuracy);

        assert!(test_accuracy > 0.9);
    }

    #[test]
    fn test_iris_sparse() {
        let (dense_data, target) = load_data();
        let data = SparseRowArray::from(&dense_data);

        let mut test_accuracy = 0.0;

        let no_splits = 10;

        let mut cv = CrossValidation::new(data.rows(),
                                          no_splits);
        cv.set_rng(StdRng::from_seed(&[100]));

        for (train_idx, test_idx) in cv {

            let x_train = data.get_rows(&train_idx);
            let x_test = data.get_rows(&test_idx);

            let y_train = target.get_rows(&train_idx);

            let mut model = Hyperparameters::new(data.cols())
                .learning_rate(0.5)
                .l2_penalty(0.0)
                .l1_penalty(0.0)
                .one_vs_rest();

            for _ in 0..20 {
                model.fit(&x_train, &y_train).unwrap();
            }

            let y_hat = model.predict(&x_test).unwrap();

            test_accuracy += accuracy_score(
                &target.get_rows(&test_idx),
                &y_hat);
        }

        test_accuracy /= no_splits as f32;

        println!("Accuracy {}", test_accuracy);

        assert!(test_accuracy > 0.9);
    }

    #[test]
    #[cfg(feature = "all_tests")]
    fn test_sgdclassifier_newsgroups() {

        use feature_extraction::dict_vectorizer::*;

        let mut rdr = csv::Reader::from_file("./test_data/newsgroups/data.csv")
            .unwrap()
            .has_headers(false);

        let mut vectorizer = DictVectorizer::new();
        let mut target = Vec::new();

        for (row, record) in rdr.decode().enumerate() {
            let (y, data): (f32, String) = record.unwrap();

            for token in data.split_whitespace() {
                vectorizer.partial_fit(row, token, 1.0);
            }

            target.push(y);
        }

        let target = Array::from(target);

        let X = vectorizer.transform();

        let no_splits = 2;

        let mut test_accuracy = 0.0;

        let mut cv = CrossValidation::new(X.rows(),
                                          no_splits);
        cv.set_rng(StdRng::from_seed(&[100]));

        for (train_idx, test_idx) in cv {

            let x_train = X.get_rows(&train_idx);
            let x_test = X.get_rows(&test_idx);

            let y_train = target.get_rows(&train_idx);

            let mut model = Hyperparameters::new(X.cols())
                .learning_rate(0.05)
                .l2_penalty(0.000001)
                .l1_penalty(0.000001)
                .one_vs_rest();

            // Run 5 epochs of training
            for _ in 0..5 {
                model.fit(&x_train, &y_train).unwrap();
            }

            let y_hat = model.predict(&x_test).unwrap();

            test_accuracy += accuracy_score(
                &target.get_rows(&test_idx),
                &y_hat);
        }

        test_accuracy /= no_splits as f32;
        println!("{}", test_accuracy);

        assert!(test_accuracy > 0.88);
    }
}
