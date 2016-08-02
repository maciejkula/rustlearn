//! Utilities for mutliclass classifiers.

use std::f32;
use std::cmp::{Ordering, PartialOrd};
use std::iter::Iterator;

use array::dense::*;
use array::sparse::*;
use array::traits::*;

use traits::*;

use crossbeam;


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

        OneVsRest {
            y: y,
            classes: classes,
            iter: 0,
        }
    }

    pub fn merge(class_labels: &[f32], predictions: &[Array]) -> Array {

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

        let ret = if self.iter < self.classes.len() {
            let target_class = self.classes[self.iter];
            let binary_target = Array::from(self.y
                .data()
                .iter()
                .map(|&v| {
                    if v == target_class {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>());
            Some((target_class, binary_target))
        } else {
            None
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
    class_labels: Vec<f32>,
}


impl<T: Clone> OneVsRestWrapper<T> {
    pub fn new(base_model: T) -> OneVsRestWrapper<T> {
        OneVsRestWrapper {
            base_model: base_model,
            models: Vec::new(),
            class_labels: Vec::new(),
        }
    }

    fn get_model(&mut self, class_label: f32) -> &mut T {
        for (idx, label) in self.class_labels.iter().enumerate() {
            if let Some(Ordering::Equal) = class_label.partial_cmp(label) {
                return &mut self.models[idx];
            }
        }

        self.class_labels.push(class_label);
        self.models.push(self.base_model.clone());

        &mut self.models[self.class_labels.len() - 1]
    }

    fn extract_model(&mut self, class_label: f32) -> T {

        let mut model_idx = None;

        for (idx, label) in self.class_labels.iter().enumerate() {
            if let Some(Ordering::Equal) = class_label.partial_cmp(label) {
                model_idx = Some(idx);
            }
        }

        if let Some(idx) = model_idx {
            self.class_labels.remove(idx);
            return self.models.remove(idx);
        }

        self.base_model.clone()
    }

    pub fn models(&self) -> &Vec<T> {
        &self.models
    }

    pub fn class_labels(&self) -> &Vec<f32> {
        &self.class_labels
    }
}


macro_rules! impl_multiclass_supervised_model {
    ($t:ty) => {
        impl<T: SupervisedModel<$t> + Clone> SupervisedModel<$t> for OneVsRestWrapper<T> {
            fn fit(&mut self, X: &$t, y: &Array) -> Result<(), &'static str> {

                for (class_label, binary_target) in OneVsRest::split(y) {

                    let model = self.get_model(class_label);
                    try!(model.fit(X, &binary_target));
                }
                Ok(())
            }

            fn decision_function(&self, X: &$t) -> Result<Array, &'static str> {

                let mut out = Array::zeros(X.rows(), self.class_labels.len());

                for (col_idx, model) in self.models.iter().enumerate() {
                    let values = try!(model.decision_function(X));
                    for (row_idx, &val) in values.data().iter().enumerate() {
                        *out.get_mut(row_idx, col_idx) = val;
                    }
                }

                Ok(out)
            }

            fn predict(&self, X: &$t) -> Result<Array, &'static str> {

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
    };
}


macro_rules! impl_multiclass_parallel_predict {
    ($t:ty) => {
        impl<T: SupervisedModel<$t> + Clone + Sync> ParallelPredict<$t> for OneVsRestWrapper<T> {
            fn decision_function_parallel(&self, X: &$t, num_threads: usize) -> Result<Array, &'static str> {

                let mut out = Array::zeros(X.rows(), self.class_labels.len());

                let numbered_models = self.models.iter().enumerate().collect::<Vec<_>>();

                for slc in numbered_models.chunks(num_threads) {

                    let mut guards = Vec::new();
                    
                    crossbeam::scope(|scope| {
                        for &(col_idx, model) in slc {
                            guards.push(scope.spawn(move || {
                                (col_idx, model.decision_function(X))
                            }));
                        }
                    });

                    for guard in guards.into_iter() {
                        let (col_idx, res) = guard.join();
                        if res.is_ok() {
                            for (row_idx, &value) in res.unwrap().as_slice().iter().enumerate() {
                                out.set(row_idx, col_idx, value);
                            }
                        } else {
                            return res;
                        }
                    }
                }

                Ok(out)
            }

            fn predict_parallel(&self, X: &$t, num_threads: usize) -> Result<Array, &'static str> {

                let decision = try!(self.decision_function_parallel(X, num_threads));
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
    };
}


macro_rules! impl_multiclass_parallel_supervised {
    ($t:ty) => {
        impl<T: SupervisedModel<$t> + Clone + Sync + Send> ParallelSupervisedModel<$t> for OneVsRestWrapper<T> {
            fn fit_parallel(&mut self, X: &$t, y: &Array, num_threads: usize) -> Result<(), &'static str> {

                let mut ovr = OneVsRest::split(y).collect::<Vec<_>>();

                loop {
                    let ovr_len = ovr.len();
                    let chunk = ovr.drain((if num_threads > ovr_len { 0 } else { ovr_len - num_threads })
                                          ..).collect::<Vec<_>>();

                    if chunk.len() == 0 {
                        break;
                    }

                    let mut guards = Vec::new();
                    
                    crossbeam::scope(|scope| {
                        for (class_label, binary_target) in chunk {
                            let mut model = self.extract_model(class_label);
                            guards.push(scope.spawn(move || {
                                let result = model.fit(X, &binary_target);
                                (class_label, model, result)
                            }));
                        }
                    });

                    for guard in guards.into_iter() {
                        let (class_label, model, result) = guard.join();

                        if result.is_ok() {
                            self.class_labels.push(class_label);
                            self.models.push(model);
                        } else {
                            return result;
                        }
                    }
                }

                Ok(())
            }
        }
    }
}


impl_multiclass_supervised_model!(Array);
impl_multiclass_supervised_model!(SparseRowArray);
impl_multiclass_supervised_model!(SparseColumnArray);

impl_multiclass_parallel_predict!(Array);
impl_multiclass_parallel_predict!(SparseRowArray);
impl_multiclass_parallel_predict!(SparseColumnArray);

impl_multiclass_parallel_supervised!(Array);
impl_multiclass_parallel_supervised!(SparseRowArray);
impl_multiclass_parallel_supervised!(SparseColumnArray);
