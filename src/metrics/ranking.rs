//! Ranking accuracy metrics.

use std::f32;
use std::cmp::Ordering;

use array::prelude::*;


/// Return (nondecreasing) counts of true positives and false positives.
fn counts_at_score(y_true: &[f32], y_hat: &[f32]) -> (Vec<f32>, Vec<f32>) {

    // vector of pairs (score, label) - the order is switched with respect to function arguments
    let mut pairs: Vec<_> = y_hat.iter().cloned().zip(y_true.iter().cloned()).collect();

    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

    // tp .. true positives, fp .. false positives
    let mut s0 = ::std::f32::NAN;
    let (mut tp, mut fp) = (0.0f32, 0.0f32);
    let (mut tps, mut fps) = (vec![], vec![]);
    for &(s, t) in pairs.iter() {
        if s != s0 {
            tps.push(tp);
            fps.push(fp);
            s0 = s;
        }
        match t == 1.0 {
            false => fp += 1.0,
            true =>  tp += 1.0,
        }
    }
    tps.push(tp);
    fps.push(fp);

    (tps, fps)
}


/// Calculate true positive and false positive rates.
/// Both vectors are nondecreasing.
fn rates_at_score(y_true: &[f32], y_hat: &[f32]) -> (Vec<f32>, Vec<f32>) {

    let (mut true_positive_count, mut false_positive_count) = counts_at_score(y_true, y_hat);

    let true_positives = true_positive_count[true_positive_count.len() - 1];
    let false_positives = false_positive_count[false_positive_count.len() - 1];

    for (tp, fp) in true_positive_count.iter_mut()
        .zip(false_positive_count.iter_mut()) {
        *tp /= true_positives;
        *fp /= false_positives;
    }

    (true_positive_count, false_positive_count)
}


/// Integration using the trapezoidal rule.
fn trapezoidal(x: &[f32], y: &[f32]) -> f32 {

    let mut prev_x = *x.first().unwrap();
    let mut prev_y = *y.first().unwrap();

    let mut integral = 0.0;

    for (&x, &y) in x.iter().skip(1).zip(y.iter().skip(1)) {

        integral += (x - prev_x) * (prev_y + y) / 2.0;

        prev_x = x;
        prev_y = y;
    }

    integral
}


fn check_roc_auc_inputs(y_true: &Array, y_hat: &Array) -> Result<(), &'static str> {

    if y_true.cols() != 1 || y_hat.cols() != 1 {
        return Err("Input array has more than one column.");
    }

    if y_true.rows() != y_hat.rows() {
        return Err("Unequal number of rows");
    }

    if y_true.rows() < 1 {
        return Err("Inputs are empty.");
    }

    let mut pos_present = false;
    let mut neg_present = false;

    for &y in y_true.data() {
        match y {
            0.0 => {
                neg_present = true;
            }
            1.0 => {
                pos_present = true;
            }
            _ => return Err("Invalid labels: target data is not either 0.0 or 1.0"),
        }
    }

    if !pos_present || !neg_present {
        return Err("Both classes must be present.");
    }

    Ok(())
}


/// Compute the ROC AUC score for a binary classification problem.
///
/// # Failures
/// Will fail if inputs are illegal:
///
/// - inputs are of unequal length
/// - both classes are not represented in the input
/// - inputs are empty
pub fn roc_auc_score(y_true: &Array, y_hat: &Array) -> Result<f32, &'static str> {

    try!(check_roc_auc_inputs(y_true, y_hat));

    let (tpr, fpr) = rates_at_score(y_true.data(), y_hat.data());

    Ok(trapezoidal(&fpr, &tpr))
}



#[cfg(test)]
mod tests {

    use prelude::*;

    use super::{counts_at_score, roc_auc_score};

    #[test]
    fn basic() {
        let y_true = vec![1.0, 1.0, 0.0, 0.0];
        let y_hat = vec![0.5, 0.2, 0.3, -1.0];

        let (x, y) = counts_at_score(&y_true, &y_hat);

        let x_expected: Vec<f32> = vec![0.0, 1.0, 1.0, 2.0, 2.0];
        let y_expected: Vec<f32> = vec![0.0, 0.0, 1.0, 1.0, 2.0];

        assert!(allclose(&Array::from(x), &Array::from(x_expected)));
        assert!(allclose(&Array::from(y), &Array::from(y_expected)));

        assert!(close(0.75,
                      roc_auc_score(&Array::from(y_true), &Array::from(y_hat)).unwrap()));
    }

    #[test]
    fn basic_repeated() {
        let y_true = vec![1.0, 1.0, 0.0, 0.0];
        let y_hat = vec![0.5, 0.5, -1.0, 0.5];

        assert!(close(0.75,
                      roc_auc_score(&Array::from(y_true), &Array::from(y_hat)).unwrap()));
    }
}
