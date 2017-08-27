//! Ranking accuracy metrics.

use std::f32;
use std::cmp::{min, Ordering};

use array::prelude::*;


/// Discounted Cumulative Gain
///
/// # Panics
/// Will panic if inputs are of unequal length.
pub fn dcg_score(y_true: &Array, y_hat: &Array, k: i32) -> f32 {
    assert!(y_true.rows() == y_hat.rows());
    let mut pairs: Vec<_> = y_hat.data().iter().enumerate().collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    let mut out: f32 = 0.0;
    let last = min(k as usize, y_true.rows());
    for i in 0..last {
        let (orig_idx, _) = pairs[i];
        let gain = 2f32.powf(y_true.data()[orig_idx]);
        let discount = ((i as f32) + 2.0).log2();
        out += gain / discount;
    }
    out
}

/// Normalized Discounted Cumulative Gain
///
/// # Panics
/// Will panic if inputs are of unequal length.
pub fn ndcg_score(y_true: &Array, y_hat: &Array, k: i32) -> f32 {
    assert!(y_true.rows() == y_hat.rows());
    let best = dcg_score(y_true, y_hat, k);
    let actual = dcg_score(y_true, y_hat, k);
    actual / best
}

/// Return (nondecreasing) counts of true positives and false positives.
fn counts_at_score(y_true: &[f32], y_hat: &[f32]) -> (Vec<f32>, Vec<f32>) {

    // vector of pairs (score, label) - the order is switched with respect to function arguments
    let mut pairs: Vec<_> = y_hat.iter().cloned().zip(y_true.iter().cloned()).collect();

    // Sort by scores in a descending order
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

    let mut score_prev = ::std::f32::NAN;
    // tp .. true positives, fp .. false positives
    let (mut tp, mut fp) = (0.0f32, 0.0f32);
    let (mut tps, mut fps) = (vec![], vec![]);
    for (score, label) in pairs {
        // `tp` and `fp` from the previous iteration are pushed onto the ROC curve only if
        // the `score` changed. This avoids errors due to arbitrary classification of points with
        // identical scores
        if score != score_prev {
            tps.push(tp);
            fps.push(fp);
            score_prev = score;
        }
        tp += label;
        fp += 1.0 - label;
    }
    // Push the final point corresponding to the (1,1) ROC coordinates
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

    use super::{counts_at_score, roc_auc_score, dcg_score, ndcg_score};

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
    fn test_dcg_basic() {
        // From: https://gist.github.com/mblondel/7337391#file-letor_metrics-py-L211
        let winner_1 = &Array::from(vec![5.0, 3.0, 2.0]);
        let winner_2 = &Array::from(vec![4.0, 3.0, 2.0]);
        let loser = &Array::from(vec![2.0, 1.0, 0.0]);
        let score_1 = dcg_score(winner_1, loser, 10);
        let score_2 = dcg_score(winner_2, loser, 10);
        assert!(score_1 > score_2);
        assert!(dcg_score(winner_1, loser, 2) > dcg_score(winner_2, loser, 2));
    }

    #[test]
    fn test_dcg_sample_order() {
        let r1 = &Array::from(vec![5.0, 3.0, 2.0]);
        let r2 = &Array::from(vec![2.0, 1.0, 0.0]);
        let r3 = &Array::from(vec![2.0, 3.0, 5.0]);
        let r4 = &Array::from(vec![0.0, 1.0, 2.0]);
        assert!(dcg_score(r1, r2, 10) == dcg_score(r3, r4, 10));
    }

    #[test]
    fn test_ndcg_ideal() {
        let r1 = &Array::from(vec![5.0, 3.0, 2.0]);
        let r2 = &Array::from(vec![2.0, 1.0, 0.0]);
        let r3 = &Array::from(vec![2.0, 3.0, 5.0]);
        let r4 = &Array::from(vec![0.0, 1.0, 2.0]);

        assert!(close(1.0, ndcg_score(r1, r1, 10)));
        assert!(close(1.0, ndcg_score(r3, r4, 10)));
    }

    #[test]
    fn basic_repeated() {
        let y_true = vec![1.0, 1.0, 0.0, 0.0];
        let y_hat = vec![0.5, 0.5, -1.0, 0.5];

        assert!(close(0.75,
                      roc_auc_score(&Array::from(y_true), &Array::from(y_hat)).unwrap()));
    }
}
