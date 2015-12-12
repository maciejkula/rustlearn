use std::cmp::Ordering;

use super::ffi;
use prelude::*;

pub use super::ffi::KernelType;

use multiclass::OneVsRestWrapper;
use utils::{check_data_dimensionality,
            check_matched_dimensions,
            check_valid_labels};


#[derive(Clone)]
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

    pub fn build(&self) -> SVC {
        SVC { dim: self.dim,
              hyperparams: self.to_owned(),
              model: None }
    }

    pub fn svm_parameter(&self) -> &ffi::SvmParameter {
        &self.svm_parameter
    }
}


#[derive(Clone)]
pub struct SVC {
    dim: usize,
    hyperparams: Hyperparameters,
    model: Option<ffi::SvmModel>
}


impl SupervisedModel<Array> for SVC {
    fn fit(&mut self, X: &Array, y: &Array) -> Result<(), &'static str> {

        try!(check_data_dimensionality(self.dim, X));
        try!(check_matched_dimensions(X, y));

        let svm_problem = ffi::SvmProblem::new_dense(X, y);
        let svm_params = self.hyperparams.svm_parameter();

        self.model = Some(ffi::svm_train_safe(&svm_problem, &svm_params));

        Ok(())
    }

    fn decision_function(&self, X: &Array) -> Result<Array, &'static str> {

        try!(check_data_dimensionality(self.dim, X));

        match self.model {
            Some(ref model) => {
                let (decision_function, predicted_class)
                    = ffi::svm_decision_function_dense(model,
                                                       self.hyperparams.num_classes,
                                                       X);
                Ok(decision_function)
            },
            None => Err("Model must be fit before predicting.")
        }
    }

    fn predict(&self, X: &Array) -> Result<Array, &'static str> {

        match self.model {
            Some(ref model) => {
                let (decision_function, predicted_class)
                    = ffi::svm_decision_function_dense(model,
                                                       self.hyperparams.num_classes,
                                                       X);
                Ok(predicted_class)
            },
            None => Err("Model must be fit before predicting.")
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    use rand::{StdRng, SeedableRng};

    use prelude::*;

    use cross_validation::cross_validation::CrossValidation;
    use datasets::iris::load_data;
    use metrics::accuracy_score;
    use multiclass::OneVsRestWrapper;

    use super::super::ffi;

    use bincode;

    #[test]
    fn test_iris() {
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
    }
}
