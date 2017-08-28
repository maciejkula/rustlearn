//! Accuracy metrics.

use array::prelude::*;

mod ranking;

mod test;

pub use self::ranking::{roc_auc_score, dcg_score, ndcg_score};


/// Measure classifier accuracy
///
/// # Panics
/// Will panic if inputs are of unequal length.
pub fn accuracy_score(y_true: &Array, y_hat: &Array) -> f32 {

    assert!(y_true.rows() == y_hat.rows());

    let mut accuracy = 0.0;

    for (yt, yh) in y_true.data().iter().zip(y_hat.data().iter()) {
        if yt == yh {
            accuracy += 1.0;
        }
    }

    accuracy / (y_true.rows() as f32)
}

/// Measure Regressor Performance
///
/// # Panics
/// Will panic if inputs are of unequal length.
pub fn mean_absolute_error(y_true: &Array, y_hat: &Array) -> f32 {
    assert!(y_true.rows() == y_hat.rows());
    let mut abs_diff = 0.0;
    for (yt, yh) in y_true.data().iter().zip(y_hat.data().iter()) {
        abs_diff += (yt - yh).abs();
    }

    abs_diff / (y_true.rows() as f32)
}

/// Measure Regressor Performance
/// Mean Squared Error
/// # Panics
/// Will panic if inputs are of unequal length.
pub fn mean_squared_error(y_true: &Array, y_hat: &Array) -> f32 {
    assert!(y_true.rows() == y_hat.rows());
    let mut sq_diff = 0.0;
    for (yt, yh) in y_true.data().iter().zip(y_hat.data().iter()) {
        sq_diff += (yt - yh).powf(2.0);
    }

    sq_diff / (y_true.rows() as f32)
}


#[cfg(test)]
mod tests {

    use prelude::*;

    use super::{mean_absolute_error, mean_squared_error};

    #[test]
    fn basic() {
        let y_true = &Array::from(vec![1.0, 1.0, 0.0, 0.0]);

        let mae = mean_absolute_error(y_true, y_true);
        assert!(mae==0.0);
        let mse = mean_squared_error(y_true, y_true);
        assert!(mse==0.0);

    }

}
