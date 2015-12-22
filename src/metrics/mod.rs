//! Accuracy metrics.

use array::prelude::*;

mod ranking;

mod test;

pub use self::ranking::roc_auc_score;


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
