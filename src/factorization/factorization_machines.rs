//! A factorization machine model implemented using stochastic gradient descent.
//!
//! A [factorization machine](http://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) (Rendle 2008)
//! model combines the advantages of linear and factorization models. In this implementation, it approximates
//! second-order feature interactions (as in a quadratic SVM) via reduced-rank matrix factorization.
//! This allows it to estimate feature interactions even in sparse datasets (like recommender systems) where
//! traditional polynomial SVMs fail.
//!
//! The complexity of the model is controlled by the dimensionality of the factorization matrix:
//! a higher setting will make the model more expressive at the expense of training time and
//! risk of overfitting.
//!
//! # Parallelism
//!
//! The model supports multithreaded model fitting via asynchronous stochastic
//! gradient descent (Hogwild).
//!
//! # Examples
//!
//! ```
//! use rustlearn::prelude::*;
//! use rustlearn::factorization::factorization_machines::Hyperparameters;
//! use rustlearn::datasets::iris;
//!
//! let (X, y) = iris::load_data();
//!
//! let mut model = Hyperparameters::new(4, 10)
//!                                 .one_vs_rest();
//!
//! model.fit(&X, &y).unwrap();
//!
//! let prediction = model.predict(&X).unwrap();
//! ```
#![allow(non_snake_case)]

use std::cmp;

use prelude::*;

use multiclass::OneVsRestWrapper;
use utils::{
    check_data_dimensionality, check_matched_dimensions, check_valid_labels, EncodableRng,
};

use rand;
use rand::distributions::IndependentSample;

use crossbeam;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn logistic_loss(y: f32, y_hat: f32) -> f32 {
    y_hat - y
}

macro_rules! max {
    ($x:expr, $y:expr) => {{
        match $x > $y {
            true => $x,
            false => $y,
        }
    }};
}

macro_rules! min {
    ($x:expr, $y:expr) => {{
        match $x < $y {
            true => $x,
            false => $y,
        }
    }};
}

/// Hyperparameters for a FactorizationMachine
#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    dim: usize,
    num_components: usize,

    learning_rate: f32,
    l2_penalty: f32,
    l1_penalty: f32,
    rng: EncodableRng,
}

impl Hyperparameters {
    /// Creates new Hyperparameters.
    ///
    /// The complexity of the model is controlled by the dimensionality of the factorization matrix:
    /// a higher `num_components` setting will make the model more expressive
    /// at the expense of training time and risk of overfitting.
    pub fn new(dim: usize, num_components: usize) -> Hyperparameters {
        Hyperparameters {
            dim: dim,
            num_components: num_components,
            learning_rate: 0.05,
            l2_penalty: 0.0,
            l1_penalty: 0.0,
            rng: EncodableRng::new(),
        }
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
    pub fn l1_penalty(&mut self, l1_penalty: f32) -> &mut Hyperparameters {
        self.l1_penalty = l1_penalty;
        self
    }

    pub fn rng(&mut self, rng: rand::StdRng) -> &mut Hyperparameters {
        self.rng.rng = rng;
        self
    }

    /// Build a two-class model.
    pub fn build(&self) -> FactorizationMachine {
        let mut rng = self.rng.clone();

        FactorizationMachine {
            dim: self.dim,
            num_components: self.num_components,

            learning_rate: self.learning_rate,
            l2_penalty: self.l2_penalty,
            l1_penalty: self.l1_penalty,

            coefficients: Array::zeros(self.dim, 1),
            latent_factors: self.init_latent_factors_array(&mut rng),
            gradsq: Array::ones(self.dim, 1),
            latent_gradsq: Array::ones(self.dim, self.num_components),
            applied_l2: Array::ones(self.dim, 1),
            applied_l1: Array::zeros(self.dim, 1),
            latent_applied_l2: Array::ones(self.dim, self.num_components),
            latent_applied_l1: Array::zeros(self.dim, self.num_components),
            accumulated_l2: 1.0,
            accumulated_l1: 0.0,

            rng: rng,
        }
    }

    /// Initialize the latent factors.
    fn init_latent_factors_array(&self, rng: &mut EncodableRng) -> Array {
        let mut data = Vec::with_capacity(self.dim * self.num_components);
        // let normal = rand::distributions::normal::Normal::new(0.0, 0.1 / ((self.dim * self.num_components) as f64).sqrt());
        let normal =
            rand::distributions::normal::Normal::new(0.0, 1.0 / self.num_components as f64);

        for _ in 0..(self.dim * self.num_components) {
            data.push(normal.ind_sample(&mut rng.rng) as f32)
        }

        let mut array = Array::from(data);
        array.reshape(self.dim, self.num_components);
        array
    }

    /// Build a one-vs-rest multiclass model.
    #[allow(dead_code)]
    pub fn one_vs_rest(&self) -> OneVsRestWrapper<FactorizationMachine> {
        let base_model = self.build();

        OneVsRestWrapper::new(base_model)
    }
}

/// A two-class factorization machine implemented using stochastic gradient descent.
#[derive(Clone, Serialize, Deserialize)]
pub struct FactorizationMachine {
    dim: usize,
    num_components: usize,

    learning_rate: f32,
    l2_penalty: f32,
    l1_penalty: f32,

    coefficients: Array,
    latent_factors: Array,
    gradsq: Array,
    latent_gradsq: Array,
    applied_l2: Array,
    applied_l1: Array,
    latent_applied_l2: Array,
    latent_applied_l1: Array,
    accumulated_l2: f32,
    accumulated_l1: f32,

    rng: EncodableRng,
}

impl FactorizationMachine {
    fn compute_prediction<T: NonzeroIterable>(&self, row: &T, component_sum: &mut [f32]) -> f32 {
        let mut result = 0.0;

        for (feature_idx, feature_value) in row.iter_nonzero() {
            result += feature_value * self.coefficients.get(feature_idx, 0);
        }

        for component_idx in 0..self.num_components {
            let mut component_sum_elem = 0.0;
            let mut component_sum_sq_elem = 0.0;

            for (feature_idx, feature_value) in row.iter_nonzero() {
                let val = self.latent_factors.get(feature_idx, component_idx) * feature_value;
                component_sum_elem += val;
                component_sum_sq_elem += val.powi(2);
            }

            component_sum[component_idx] = component_sum_elem;

            result += 0.5 * (component_sum_elem.powi(2) - component_sum_sq_elem);
        }

        result
    }

    fn apply_regularization(
        parameter_value: &mut f32,
        applied_l2: &mut f32,
        applied_l1: &mut f32,
        local_learning_rate: f32,
        accumulated_l2: f32,
        accumulated_l1: f32,
    ) {
        let l2_update = accumulated_l2 / *applied_l2;

        *parameter_value *= 1.0 - (1.0 - l2_update) * local_learning_rate;
        *applied_l2 *= l2_update;

        let l1_potential_update = accumulated_l1 - *applied_l1;
        let pre_update_coeff = parameter_value.clone();

        if *parameter_value > 0.0 {
            *parameter_value = max!(
                0.0,
                *parameter_value - l1_potential_update * local_learning_rate
            );
        } else {
            *parameter_value = min!(
                0.0,
                *parameter_value + l1_potential_update * local_learning_rate
            );
        }

        let l1_actual_update = (pre_update_coeff - *parameter_value).abs();
        *applied_l1 += l1_actual_update;
    }

    fn update<T: NonzeroIterable>(&mut self, row: T, loss: f32, component_sum: &[f32]) {
        for (feature_idx, feature_value) in (&row).iter_nonzero() {
            // Update coefficients
            let gradsq = self.gradsq.get_mut(feature_idx, 0);
            let local_learning_rate = self.learning_rate / gradsq.sqrt();
            let coefficient_value = self.coefficients.get_mut(feature_idx, 0);

            let applied_l2 = self.applied_l2.get_mut(feature_idx, 0);
            let applied_l1 = self.applied_l1.get_mut(feature_idx, 0);

            let gradient = loss * feature_value;

            *coefficient_value -= local_learning_rate * gradient;
            *gradsq += gradient.powi(2);

            FactorizationMachine::apply_regularization(
                coefficient_value,
                applied_l2,
                applied_l1,
                local_learning_rate,
                self.accumulated_l2,
                self.accumulated_l1,
            );

            // Update latent factors
            let slice_start = feature_idx * self.num_components;
            let slice_stop = slice_start + self.num_components;

            let mut component_row =
                &mut self.latent_factors.as_mut_slice()[slice_start..slice_stop];
            let mut gradsq_row = &mut self.latent_gradsq.as_mut_slice()[slice_start..slice_stop];
            let mut applied_l2_row =
                &mut self.latent_applied_l2.as_mut_slice()[slice_start..slice_stop];
            let mut applied_l1_row =
                &mut self.latent_applied_l1.as_mut_slice()[slice_start..slice_stop];

            for (component_value, (gradsq, (applied_l2, (applied_l1, component_sum_value)))) in
                component_row.iter_mut().zip(
                    gradsq_row.iter_mut().zip(
                        applied_l2_row
                            .iter_mut()
                            .zip(applied_l1_row.iter_mut().zip(component_sum.iter())),
                    ),
                ) {
                let local_learning_rate = self.learning_rate / gradsq.sqrt();
                let update = loss * ((component_sum_value * feature_value)
                    - (*component_value * feature_value.powi(2)));

                *component_value -= local_learning_rate * update;
                *gradsq += update.powi(2);

                FactorizationMachine::apply_regularization(
                    component_value,
                    applied_l2,
                    applied_l1,
                    local_learning_rate,
                    self.accumulated_l2,
                    self.accumulated_l1,
                );
            }
        }
    }

    fn accumulate_regularization(&mut self) {
        self.accumulated_l2 *= 1.0 - self.l2_penalty;
        self.accumulated_l1 += self.l1_penalty;
    }

    fn fit_sigmoid<'a, T>(&mut self, X: &'a T, y: &Array) -> Result<(), &'static str>
    where
        T: IndexableMatrix,
        &'a T: RowIterable,
    {
        let mut component_sum = &mut vec![0.0; self.num_components][..];

        for (row, &true_y) in X.iter_rows().zip(y.data().iter()) {
            let y_hat = sigmoid(self.compute_prediction(&row, component_sum));

            let loss = logistic_loss(true_y, y_hat);

            self.update(row, loss, component_sum);

            self.accumulate_regularization();
        }

        self.regularize_all();

        Ok(())
    }

    /// Perform a dummy update pass over all features to force regularization to be applied.
    fn regularize_all(&mut self) {
        if self.l1_penalty == 0.0 && self.l2_penalty == 0.0 {
            return;
        }

        let array = Array::ones(1, self.dim);
        let num_components = self.num_components;

        self.update(
            &array.view_row(0),
            0.0,
            &vec![0.0; num_components.clone()][..],
        );

        self.accumulated_l2 = 1.0;
        self.accumulated_l1 = 0.0;
    }

    pub fn get_coefficients(&self) -> &Array {
        &self.coefficients
    }

    pub fn get_latent_factors(&self) -> &Array {
        &self.latent_factors
    }
}

impl<'a, T> SupervisedModel<&'a T> for FactorizationMachine
where
    &'a T: RowIterable,
    T: IndexableMatrix,
{
    fn fit(&mut self, X: &'a T, y: &Array) -> Result<(), &'static str> {
        try!(check_data_dimensionality(self.dim, X));
        try!(check_matched_dimensions(X, y));
        try!(check_valid_labels(y));

        self.fit_sigmoid(X, y)
    }

    fn decision_function(&self, X: &'a T) -> Result<Array, &'static str> {
        try!(check_data_dimensionality(self.dim, X));

        let mut data = Vec::with_capacity(X.rows());

        let mut component_sum = &mut vec![0.0; self.num_components][..];

        for row in X.iter_rows() {
            let prediction = self.compute_prediction(&row, component_sum);
            data.push(sigmoid(prediction));
        }

        Ok(Array::from(data))
    }
}

impl<'a, T> ParallelSupervisedModel<&'a T> for FactorizationMachine
where
    &'a T: RowIterable,
    T: IndexableMatrix + Sync,
{
    fn fit_parallel(
        &mut self,
        X: &'a T,
        y: &Array,
        num_threads: usize,
    ) -> Result<(), &'static str> {
        try!(check_data_dimensionality(self.dim, X));
        try!(check_matched_dimensions(X, y));
        try!(check_valid_labels(y));

        let rows_per_thread = X.rows() / num_threads + 1;
        let num_components = self.num_components;

        let model_ptr = unsafe { &*(self as *const FactorizationMachine) };

        crossbeam::scope(|scope| {
            for thread_num in 0..num_threads {
                scope.spawn(move || {
                    let start = thread_num * rows_per_thread;
                    let stop = cmp::min((thread_num + 1) * rows_per_thread, X.rows());

                    let mut component_sum = vec![0.0; num_components];

                    let model = unsafe {
                        &mut *(model_ptr as *const FactorizationMachine
                            as *mut FactorizationMachine)
                    };

                    for (row, &true_y) in X
                        .iter_rows_range(start..stop)
                        .zip(y.data()[start..stop].iter())
                    {
                        let y_hat = sigmoid(model.compute_prediction(&row, &mut component_sum[..]));
                        let loss = logistic_loss(true_y, y_hat);
                        model.update(row, loss, &mut component_sum[..]);
                        model.accumulate_regularization();
                    }
                });
            }
        });

        self.regularize_all();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, StdRng};

    use prelude::*;

    use cross_validation::cross_validation::CrossValidation;
    use datasets::iris::load_data;
    use metrics::accuracy_score;
    use multiclass::OneVsRest;

    #[cfg(feature = "all_tests")]
    use datasets::newsgroups;

    use super::*;

    #[test]
    fn basic_updating() {
        let mut model = Hyperparameters::new(2, 2)
            .learning_rate(0.01)
            .l2_penalty(0.0)
            .l1_penalty(0.0)
            .build();

        // Set to zero to allow coefficient update tests
        // to be straightforward
        for elem in model.latent_factors.as_mut_slice().iter_mut() {
            *elem = 0.0;
        }

        let y = Array::ones(1, 1);
        let X = Array::from(&vec![vec![1.0, -0.1]]);

        model.fit(&X, &y).unwrap();

        assert!(model.gradsq.data()[0] > 1.0);
        assert!(model.gradsq.data()[1] > 1.0);

        assert!(model.coefficients.data()[0] == 0.005);
        assert!(model.coefficients.data()[1] == -0.0005);

        model.fit(&X, &y).unwrap();
        println!("model coefficients {:?}", model.coefficients.data());

        assert!(model.coefficients.data()[0] == 0.009460844);
        assert!(model.coefficients.data()[1] == -0.0009981153);
    }

    #[test]
    fn test_basic_l1() {
        let mut model = Hyperparameters::new(2, 2)
            .learning_rate(0.01)
            .l2_penalty(0.0)
            .l1_penalty(100.0)
            .rng(StdRng::from_seed(&[100]))
            .build();

        let y = Array::ones(1, 1);
        let X = Array::from(&vec![vec![1.0, -0.1]]);

        for &elem in model.latent_factors.data() {
            assert!(elem != 0.0);
        }

        model.fit(&X, &y).unwrap();

        assert!(model.gradsq.data()[0] > 1.0);
        assert!(model.gradsq.data()[1] > 1.0);

        // All the coefficients/factors should
        // have been regularized away to zero.
        for &elem in model.coefficients.data() {
            assert!(elem == 0.0);
        }

        for &elem in model.latent_factors.data() {
            assert!(elem == 0.0);
        }
    }

    #[test]
    fn test_iris() {
        let (data, target) = load_data();

        let mut test_accuracy = 0.0;
        let mut train_accuracy = 0.0;

        let no_splits = 10;

        let mut cv = CrossValidation::new(data.rows(), no_splits);
        cv.set_rng(StdRng::from_seed(&[100]));

        for (train_idx, test_idx) in cv {
            let x_train = data.get_rows(&train_idx);
            let x_test = data.get_rows(&test_idx);

            let y_train = target.get_rows(&train_idx);

            let mut model = Hyperparameters::new(data.cols(), 5)
                .learning_rate(0.05)
                .l2_penalty(0.0)
                .rng(StdRng::from_seed(&[100]))
                .one_vs_rest();

            for _ in 0..20 {
                model.fit(&x_train, &y_train).unwrap();
            }

            let y_hat = model.predict(&x_test).unwrap();
            let y_hat_train = model.predict(&x_train).unwrap();

            test_accuracy += accuracy_score(&target.get_rows(&test_idx), &y_hat);
            train_accuracy += accuracy_score(&target.get_rows(&train_idx), &y_hat_train);
        }

        test_accuracy /= no_splits as f32;
        train_accuracy /= no_splits as f32;

        println!("Accuracy {}", test_accuracy);
        println!("Train accuracy {}", train_accuracy);

        assert!(test_accuracy > 0.94);
    }

    #[test]
    fn test_iris_parallel() {
        let (data, target) = load_data();

        // Get a binary target so that the parallelism
        // goes through the FM model and not through the
        // OvR wrapper.
        let (_, target) = OneVsRest::split(&target).next().unwrap();

        let mut test_accuracy = 0.0;
        let mut train_accuracy = 0.0;

        let no_splits = 10;

        let mut cv = CrossValidation::new(data.rows(), no_splits);
        cv.set_rng(StdRng::from_seed(&[100]));

        for (train_idx, test_idx) in cv {
            let x_train = data.get_rows(&train_idx);
            let x_test = data.get_rows(&test_idx);

            let y_train = target.get_rows(&train_idx);

            let mut model = Hyperparameters::new(data.cols(), 5)
                .learning_rate(0.05)
                .l2_penalty(0.0)
                .rng(StdRng::from_seed(&[100]))
                .build();

            for _ in 0..20 {
                model.fit_parallel(&x_train, &y_train, 4).unwrap();
            }

            let y_hat = model.predict(&x_test).unwrap();
            let y_hat_train = model.predict(&x_train).unwrap();

            test_accuracy += accuracy_score(&target.get_rows(&test_idx), &y_hat);
            train_accuracy += accuracy_score(&target.get_rows(&train_idx), &y_hat_train);
        }

        test_accuracy /= no_splits as f32;
        train_accuracy /= no_splits as f32;

        println!("Accuracy {}", test_accuracy);
        println!("Train accuracy {}", train_accuracy);

        assert!(test_accuracy > 0.94);
    }

    #[test]
    #[cfg(feature = "all_tests")]
    fn test_fm_newsgroups() {
        let (X, target) = newsgroups::load_data();

        let no_splits = 2;

        let mut test_accuracy = 0.0;
        let mut train_accuracy = 0.0;

        let mut cv = CrossValidation::new(X.rows(), no_splits);
        cv.set_rng(StdRng::from_seed(&[100]));

        for (train_idx, test_idx) in cv {
            let x_train = X.get_rows(&train_idx);

            let x_test = X.get_rows(&test_idx);
            let y_train = target.get_rows(&train_idx);

            let mut model = Hyperparameters::new(X.cols(), 10)
                .learning_rate(0.005)
                .rng(StdRng::from_seed(&[100]))
                .one_vs_rest();

            for _ in 0..5 {
                model.fit(&x_train, &y_train).unwrap();
                println!("fit");
            }

            let y_hat = model.predict(&x_test).unwrap();
            let y_hat_train = model.predict(&x_train).unwrap();

            test_accuracy += accuracy_score(&target.get_rows(&test_idx), &y_hat);
            train_accuracy += accuracy_score(&target.get_rows(&train_idx), &y_hat_train);
        }

        train_accuracy /= no_splits as f32;
        test_accuracy /= no_splits as f32;

        println!("Train accuracy {}", train_accuracy);
        println!("Test accuracy {}", test_accuracy);

        assert!(train_accuracy > 0.95);
        assert!(test_accuracy > 0.65);
    }
}

#[cfg(feature = "bench")]
#[allow(unused_imports)]
mod bench {
    use rand::{SeedableRng, StdRng};

    use test::Bencher;

    use prelude::*;

    use cross_validation::cross_validation::CrossValidation;
    use datasets::iris::load_data;
    use datasets::newsgroups;
    use metrics::{accuracy_score, roc_auc_score};
    use multiclass::OneVsRest;

    use super::*;

    #[bench]
    fn bench_iris_sparse(b: &mut Bencher) {
        let (data, target) = load_data();

        let sparse_data = SparseRowArray::from(&data);

        let mut model = Hyperparameters::new(data.cols(), 5)
            .learning_rate(0.05)
            .l2_penalty(0.0)
            .rng(StdRng::from_seed(&[100]))
            .one_vs_rest();

        b.iter(|| {
            model.fit(&sparse_data, &target).unwrap();
        });
    }

    #[bench]
    fn bench_fm_newsgroups(b: &mut Bencher) {
        let (X, target) = newsgroups::load_data();
        let (_, target) = OneVsRest::split(&target).next().unwrap();

        let X = X.get_rows(&(..500));
        let target = target.get_rows(&(..500));

        let mut model = Hyperparameters::new(X.cols(), 10)
            .rng(StdRng::from_seed(&[100]))
            .build();

        b.iter(|| {
            model.fit(&X, &target).unwrap();
        });
    }

    #[bench]
    fn bench_fm_newsgroups_parallel(b: &mut Bencher) {
        let (X, target) = newsgroups::load_data();
        let (_, target) = OneVsRest::split(&target).next().unwrap();

        let X = X.get_rows(&(..500));
        let target = target.get_rows(&(..500));

        let mut model = Hyperparameters::new(X.cols(), 10)
            .rng(StdRng::from_seed(&[100]))
            .build();

        b.iter(|| {
            model.fit_parallel(&X, &target, 2).unwrap();
        });
    }
}
