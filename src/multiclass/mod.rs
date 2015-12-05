//! Utilities for mutliclass classifiers.

use std::f32;
use std::cmp::{Ordering, PartialOrd};
use std::iter::Iterator;

use array::dense::*;
use array::sparse::*;
use array::traits::*;

use traits::*;


pub struct OneVsRest<'a> {
    y: &'a Array,
    classes: Vec<f32>,
    iter: usize,
}


impl<'a> OneVsRest<'a> {
    pub fn split(y: &'a Array) -> OneVsRest {

        let mut classes = y.data().clone();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        classes.dedup();
        
        OneVsRest {y: y, classes: classes, iter: 0}
    }

    pub fn merge(class_labels: &Vec<f32>, predictions: &Vec<Array>) -> Array {

        assert!(class_labels.len() > 0);
        assert!(class_labels.len() == predictions.len());

        let no_rows = predictions[0].rows();

        let mut prediction = Array::zeros(no_rows, 1);

        for i in 0..no_rows {

            let mut decision_func_val = 0.0;
            
            for (&label, prediction_arr) in class_labels.iter()
                .zip(predictions.iter()) {
                    if prediction_arr.get(i, 0) > decision_func_val {
                        *prediction.get_mut(i, 0) = label;
                        decision_func_val = prediction_arr.get(i, 0);
                    }
                }
        }

        prediction
    }
}


impl<'a> Iterator for OneVsRest<'a> {
    type Item = (f32, Array);
    fn next(&mut self) -> Option<(f32, Array)> {

        let ret = match self.iter < self.classes.len() {
            true => {
                let target_class = self.classes[self.iter];
                let binary_target = Array::from(
                    self.y.data().iter()
                        .map(|&v| if v == target_class {1.0} else {0.0})
                        .collect::<Vec<_>>());
                Some((target_class, binary_target))
            },
            false => None
        };

        self.iter += 1;
        ret
    }
}

/// Wraps simple two-class classifiers to implement one-vs-rest strategies.
#[derive(RustcEncodable, RustcDecodable)]
pub struct OneVsRestWrapper<T> {
    base_model: T,
    models: Vec<T>,
    class_labels: Vec<f32>
}


impl<T: Clone> OneVsRestWrapper<T> {
    pub fn new(base_model: T) -> OneVsRestWrapper<T> {
        OneVsRestWrapper {base_model: base_model,
                          models: Vec::new(),
                          class_labels: Vec::new()}
    }

    fn get_model(&mut self, class_label: f32) -> &mut T {
        for (idx, label) in self.class_labels.iter().enumerate() {
            match class_label.partial_cmp(label) {
                Some(Ordering::Equal) => return &mut self.models[idx],
                _ => {},
            }
        }

        self.class_labels.push(class_label);
        self.models.push(self.base_model.clone());

        &mut self.models[self.class_labels.len() - 1]
    }

    pub fn models(&self) -> &Vec<T> {
        &self.models
    }

    pub fn class_labels(&self) -> &Vec<f32> {
        &self.class_labels
    }
}


impl<T: SupervisedModel<Array> + Clone> SupervisedModel<Array> for OneVsRestWrapper<T> {
    fn fit(&mut self, X: &Array, y: &Array) -> Result<(), &'static str> {

        for (class_label, binary_target)
            in OneVsRest::split(y) {
                
                let model = self.get_model(class_label);
                try!(model.fit(X, &binary_target));
            }
        Ok(())
    }

    fn decision_function(&self, X: &Array) -> Result<Array, &'static str> {

        let mut out = Array::zeros(X.rows(),
                                   self.class_labels.len());

        for (col_idx, model) in self.models.iter().enumerate() {
            let values = try!(model.decision_function(X));
            for (row_idx, &val) in values.data().iter().enumerate() {
                *out.get_mut(row_idx, col_idx) = val;
            }
        }

        Ok(out)
    }

    fn predict(&self, X: &Array) -> Result<Array, &'static str> {

        let decision = try!(self.decision_function(X));
        let mut predictions = Vec::with_capacity(X.rows());

        for row in decision.iter_rows() {

            let mut max_value = f32::NEG_INFINITY;
            let mut max_class = 0;
            
            for (class_idx, val) in row.iter_nonzero() {
                if val > max_value {
                    max_value = val;
                    max_class = class_idx;
                }
            }

            predictions.push(self.class_labels[max_class]);
        }

        Ok(Array::from(predictions))
    }
}


impl<T: SupervisedModel<SparseRowArray> + Clone> SupervisedModel<SparseRowArray> for OneVsRestWrapper<T> {
    fn fit(&mut self, X: &SparseRowArray, y: &Array) -> Result<(), &'static str> {
        for (class_label, binary_target)
            in OneVsRest::split(y) {
                let model = self.get_model(class_label);
                try!(model.fit(X, &binary_target));
            }
        Ok(())
    }

    fn decision_function(&self, X: &SparseRowArray) -> Result<Array, &'static str> {

        let mut out = Array::zeros(X.rows(),
                                   self.class_labels.len());

        for (col_idx, model) in self.models.iter().enumerate() {
            let values = try!(model.decision_function(X));
            for (row_idx, &val) in values.data().iter().enumerate() {
                *out.get_mut(row_idx, col_idx) = val;
            }
        }

        Ok(out)
    }

    fn predict(&self, X: &SparseRowArray) -> Result<Array, &'static str> {

        let decision = try!(self.decision_function(X));
        let mut predictions = Vec::with_capacity(X.rows());

        for row in decision.iter_rows() {

            let mut max_value = f32::NEG_INFINITY;
            let mut max_class = 0;
            
            for (class_idx, val) in row.iter_nonzero() {
                if val > max_value {
                    max_value = val;
                    max_class = class_idx;
                }
            }

            predictions.push(self.class_labels[max_class]);
        }

        Ok(Array::from(predictions))
    }
}


impl<T: SupervisedModel<SparseColumnArray> + Clone> SupervisedModel<SparseColumnArray> for OneVsRestWrapper<T> {
    fn fit(&mut self, X: &SparseColumnArray, y: &Array) -> Result<(), &'static str> {
        for (class_label, binary_target)
            in OneVsRest::split(y) {
                let model = self.get_model(class_label);
                try!(model.fit(X, &binary_target));
            }
        Ok(())
    }

    fn decision_function(&self, X: &SparseColumnArray) -> Result<Array, &'static str> {

        let mut out = Array::zeros(X.rows(),
                                   self.class_labels.len());

        for (col_idx, model) in self.models.iter().enumerate() {
            let values = try!(model.decision_function(X));
            for (row_idx, &val) in values.data().iter().enumerate() {
                *out.get_mut(row_idx, col_idx) = val;
            }
        }

        Ok(out)
    }

    fn predict(&self, X: &SparseColumnArray) -> Result<Array, &'static str> {

        let decision = try!(self.decision_function(X));
        let mut predictions = Vec::with_capacity(X.rows());

        for row in decision.iter_rows() {

            let mut max_value = f32::NEG_INFINITY;
            let mut max_class = 0;
            
            for (class_idx, val) in row.iter_nonzero() {
                if val > max_value {
                    max_value = val;
                    max_class = class_idx;
                }
            }

            predictions.push(self.class_labels[max_class]);
        }

        Ok(Array::from(predictions))
    }
}
