//! Support Vector Classifier using the `libsvm` library.
//!
//! Both dense and sparse models are supported.

use prelude::*;

use super::ffi;
pub use super::ffi::KernelType;

use utils::{check_data_dimensionality,
            check_matched_dimensions};


#[derive(Clone)]
#[derive(RustcEncodable, RustcDecodable)]
/// Hyperparameters for the SVC model.
pub struct Hyperparameters {
    dim: usize,
    num_classes: usize,
    svm_parameter: ffi::SvmParameter
}


impl Hyperparameters {
    pub fn new(dim: usize, kernel: KernelType, num_classes: usize) -> Hyperparameters {
        Hyperparameters { dim: dim,
                          num_classes: num_classes,
                          svm_parameter: ffi::SvmParameter::new(ffi::SvmType::C_SVC,
                                                                kernel,
                                                                num_classes,
                                                                dim) }
    }

    /// Set the regularization parameter `C`; smaller values
    /// mean more regularization.
    /// Default is `1.0`.
    pub fn C(&mut self, C: f64) -> &mut Hyperparameters {
        self.svm_parameter.C = C;
        self
    }

    /// Set the degree of the polynomial kernel. No effect on other
    /// kernels. Default: 3.
    pub fn degree(&mut self, degree: i32) -> &mut Hyperparameters {
        self.svm_parameter.degree = degree;
        self
    }

    /// Set the gamma parameter of the RBF kernel.
    /// Default is `1 / self.dim`.
    pub fn gamma(&mut self, gamma: f64) -> &mut Hyperparameters {
        self.svm_parameter.gamma = gamma;
        self
    }

    /// Set the coef0 parameter for the sigmoid kernel.
    /// Default is `0.0`.
    pub fn coef0(&mut self, coef0: f64) -> &mut Hyperparameters {
        self.svm_parameter.coef0 = coef0;
        self
    }

    /// Set the `libsvm` cache size, in megabytes.
    /// Default is `100.0`.
    pub fn cache_size(&mut self, cache_size: f64) -> &mut Hyperparameters {
        self.svm_parameter.cache_size = cache_size;
        self
    }

    /// Build the SVC model. `libsvm` natively supports multiclass
    /// problems via one-vs-one (OvO) estimation, so no one-vs-rest
    /// wrapper is provided.
    pub fn build(&self) -> SVC {
        SVC { dim: self.dim,
              hyperparams: self.to_owned(),
              model: None }
    }

    fn svm_parameter(&self) -> &ffi::SvmParameter {
        &self.svm_parameter
    }
}

/// Support Vector Classifier provided by the `libsvm` library.
#[derive(Clone)]
#[derive(RustcEncodable, RustcDecodable)]
pub struct SVC {
    dim: usize,
    hyperparams: Hyperparameters,
    model: Option<ffi::SvmModel>
}


macro_rules! impl_supervised_model {
    ($x_type:ty) => {
        impl SupervisedModel<$x_type> for SVC {
            fn fit(&mut self, X: &$x_type, y: &Array) -> Result<(), &'static str> {

                try!(check_data_dimensionality(self.dim, X));
                try!(check_matched_dimensions(X, y));

                let svm_params = self.hyperparams.svm_parameter();

                self.model = Some(try!(ffi::fit(X, y, &svm_params)));

                Ok(())
            }

            fn decision_function(&self, X: &$x_type) -> Result<Array, &'static str> {

                try!(check_data_dimensionality(self.dim, X));

                match self.model {
                    Some(ref model) => {
                        let (decision_function, _)
                            = ffi::predict(model, X);
                        Ok(decision_function)
                    },
                    None => Err("Model must be fit before predicting.")
                }
            }

            fn predict(&self, X: &$x_type) -> Result<Array, &'static str> {

                match self.model {
                    Some(ref model) => {
                        let (_, predicted_class)
                            = ffi::predict(model, X);
                        Ok(predicted_class)
                    },
                    None => Err("Model must be fit before predicting.")
                }
            }
        }
    }
}


impl_supervised_model!(Array);
impl_supervised_model!(SparseRowArray);


#[cfg(test)]
mod tests {
    use super::*;

    use rand::{StdRng, SeedableRng};

    use prelude::*;

    use cross_validation::cross_validation::CrossValidation;
    use datasets::iris::load_data;
    use metrics::accuracy_score;

    use bincode;

    #[cfg(feature = "all_tests")]
    use csv;

    macro_rules! test_iris_kernel {
        ($kernel:expr, $fn_name:ident) => {
            #[test]
            fn $fn_name() {
                let (data, target) = load_data();

                let mut test_accuracy = 0.0;
                let mut train_accuracy = 0.0;

                let no_splits = 10;

                let mut cv = CrossValidation::new(data.rows(),
                                                  no_splits);
                cv.set_rng(StdRng::from_seed(&[100]));

                for (train_idx, test_idx) in cv {

                    let x_train = data.get_rows(&train_idx);
                    let x_test = data.get_rows(&test_idx);

                    let y_train = target.get_rows(&train_idx);

                    let mut model = Hyperparameters::new(data.cols(), $kernel, 3)
                        .build();

                    model.fit(&x_train, &y_train).unwrap();

                    let y_hat = model.predict(&x_test).unwrap();

                    test_accuracy += accuracy_score(
                        &target.get_rows(&test_idx),
                        &y_hat);
                    train_accuracy += accuracy_score(
                        &y_train,
                        &model.predict(&x_train).unwrap());
                }

                test_accuracy /= no_splits as f32;
                train_accuracy /= no_splits as f32;

                println!("Accuracy {}", test_accuracy);
                println!("Train accuracy {}", train_accuracy);
                assert!(test_accuracy > 0.97);
            }
        }
    }

    test_iris_kernel!(KernelType::Linear, test_iris_linear);
    test_iris_kernel!(KernelType::Polynomial, test_iris_polynomial);
    test_iris_kernel!(KernelType::RBF, test_iris_rbf);

    #[test]
    fn test_sparse_iris() {
        let (dense_data, target) = load_data();
        let data = SparseRowArray::from(&dense_data);

        let mut test_accuracy = 0.0;
        let mut train_accuracy = 0.0;

        let no_splits = 10;

        let mut cv = CrossValidation::new(data.rows(),
                                          no_splits);
        cv.set_rng(StdRng::from_seed(&[100]));

        for (train_idx, test_idx) in cv {

            let x_train = data.get_rows(&train_idx);
            let x_test = data.get_rows(&test_idx);

            let y_train = target.get_rows(&train_idx);

            let mut model = Hyperparameters::new(data.cols(), KernelType::Linear, 3)
                .build();

            model.fit(&x_train, &y_train).unwrap();

            let y_hat = model.predict(&x_test).unwrap();

            test_accuracy += accuracy_score(
                &target.get_rows(&test_idx),
                &y_hat);
            train_accuracy += accuracy_score(
                &y_train,
                &model.predict(&x_train).unwrap());
        }

        test_accuracy /= no_splits as f32;
        train_accuracy /= no_splits as f32;

        println!("Accuracy {}", test_accuracy);
        println!("Train accuracy {}", train_accuracy);
        assert!(test_accuracy > 0.97);
    }

    #[test]
    fn serialization() {
        let (data, target) = load_data();

        let mut test_accuracy = 0.0;
        let mut train_accuracy = 0.0;

        let no_splits = 10;

        let mut cv = CrossValidation::new(data.rows(),
                                          no_splits);
        cv.set_rng(StdRng::from_seed(&[100]));

        for (train_idx, test_idx) in cv {

            let x_train = data.get_rows(&train_idx);
            let x_test = data.get_rows(&test_idx);

            let y_train = target.get_rows(&train_idx);

            let mut model = Hyperparameters::new(data.cols(), KernelType::Linear, 3)
                .build();

            model.fit(&x_train, &y_train).unwrap();

            let encoded = bincode::rustc_serialize::encode(&model,
                                                           bincode::SizeLimit::Infinite).unwrap();
            let decoded: SVC = bincode::rustc_serialize::decode(&encoded).unwrap();

            let y_hat = decoded.predict(&x_test).unwrap();

            test_accuracy += accuracy_score(
                &target.get_rows(&test_idx),
                &y_hat);
            train_accuracy += accuracy_score(
                &y_train,
                &decoded.predict(&x_train).unwrap());
        }

        test_accuracy /= no_splits as f32;
        train_accuracy /= no_splits as f32;

        println!("Accuracy {}", test_accuracy);
        println!("Train accuracy {}", train_accuracy);
        assert!(test_accuracy > 0.97);
    }

    #[test]
    #[cfg(feature = "all_tests")]
    fn test_newsgroups() {

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

            let mut model = Hyperparameters::new(X.cols(),
                                                 KernelType::Linear,
                                                 20)
                .build();
            

            model.fit(&x_train, &y_train).unwrap();

            let y_hat = model.predict(&x_test).unwrap();

            test_accuracy += accuracy_score(
                &target.get_rows(&test_idx),
                &y_hat);
        }

        test_accuracy /= no_splits as f32;
        println!("{}", test_accuracy);

        // This could definitely be improved
        // with better hyperparameter choice.
        assert!(test_accuracy > 0.8);
    }
}
