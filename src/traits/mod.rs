//! Common rustlearn traits.

use std::cmp::Ordering;

use array::dense::*;


/// Trait describing supervised models.
pub trait SupervisedModel<T> {
    fn fit(&mut self, X: T, y: &Array) -> Result<(), &'static str>;
    fn decision_function(&self, X: T) -> Result<Array, &'static str>;
    fn predict(&self, x: T) -> Result<Array, &'static str> {

        let decision_func = try!(self.decision_function(x));

        Ok(Array::from(decision_func.data()
            .iter()
            .map(|v| {
                match v.partial_cmp(&0.5) {
                    Some(Ordering::Greater) => 1.0,
                    _ => 0.0,
                }
            })
            .collect::<Vec<f32>>()))
    }
}


/// Applies to models capable of making predictions in a parallel fashion.
pub trait ParallelPredict<T> {
    fn decision_function_parallel(&self, X: T, num_threads: usize) -> Result<Array, &'static str>;
    fn predict_parallel(&self, X: T, num_threads: usize) -> Result<Array, &'static str> {

        let decision_func = try!(self.decision_function_parallel(X, num_threads));

        Ok(Array::from(decision_func.data()
            .iter()
            .map(|v| {
                match v.partial_cmp(&0.5) {
                    Some(Ordering::Greater) => 1.0,
                    _ => 0.0,
                }
            })
            .collect::<Vec<f32>>()))
    }
}


/// Applies to models capable of being trained in a parallel fashion.
pub trait ParallelSupervisedModel<T> {
    fn fit_parallel(&mut self, X: T, y: &Array, num_threads: usize) -> Result<(), &'static str>;
}
