# rustlearn

[![Circle CI](https://circleci.com/gh/maciejkula/rustlearn.svg?style=svg)](https://circleci.com/gh/maciejkula/rustlearn)
[![Crates.io](https://img.shields.io/crates/v/rustlearn.svg)](https://crates.io/crates/rustlearn)

A machine learning package for Rust.

For full usage details, see the [API documentation](https://maciejkula.github.io/rustlearn/doc/rustlearn/).

## Introduction

This crate contains reasonably effective
implementations of a number of common machine learning algorithms.

At the moment, `rustlearn` uses its own basic dense and sparse array types, but I will be happy
to use something more robust once a clear winner in that space emerges.

## Features

### Matrix primitives

- [dense matrices](https://maciejkula.github.io/rustlearn/doc/rustlearn/array/dense/index.html)
- [sparse matrices](https://maciejkula.github.io/rustlearn/doc/rustlearn/array/sparse/index.html)

### Models

- [logistic regression](https://maciejkula.github.io/rustlearn/doc/rustlearn/linear_models/sgdclassifier/index.html) using stochastic gradient descent,
- [support vector machines](https://maciejkula.github.io/rustlearn/doc/rustlearn/svm/libsvm/svc/index.html) using the `libsvm` library,
- [decision trees](https://maciejkula.github.io/rustlearn/doc/rustlearn/trees/decision_tree/index.html) using the CART algorithm,
- [random forests](https://maciejkula.github.io/rustlearn/doc/rustlearn/ensemble/random_forest/index.html) using CART decision trees, and
- [factorization machines](https://maciejkula.github.io/rustlearn/doc/rustlearn/factorization/factorization_machines/index.html).

All the models support fitting and prediction on both dense and sparse data, and the implementations
should be roughly competitive with Python `sklearn` implementations, both in accuracy and performance.

## Cross-validation

- [k-fold cross-validation](https://maciejkula.github.io/rustlearn/doc/rustlearn/cross_validation/cross_validation/index.html)
- [shuffle split](https://maciejkula.github.io/rustlearn/doc/rustlearn/cross_validation/shuffle_split/index.html)

## Metrics

- [accuracy](https://maciejkula.github.io/rustlearn/doc/rustlearn/metrics/fn.accuracy_score.html)
- [ROC AUC score](https://maciejkula.github.io/rustlearn/doc/rustlearn/metrics/ranking/fn.roc_auc_score.html)
- [dcg_score](https://maciejkula.github.io/rustlearn/doc/rustlearn/metrics/ranking/fn.dcg_score.html)
- [ndcg_score](https://maciejkula.github.io/rustlearn/doc/rustlearn/metrics/ranking/fn.ndcg_score.html)
- [mean absolute error](https://maciejkula.github.io/rustlearn/doc/rustlearn/metrics/ranking/fn.mean_absolute_error.html)
- [mean squared error](https://maciejkula.github.io/rustlearn/doc/rustlearn/metrics/ranking/fn.mean_squared_error.html)

## Parallelization

A number of models support both parallel model fitting and prediction.

### Model serialization

Model serialization is supported via `rustc_serialize`. This will probably change to `serde` once compiler plugins land in stable.

## Using `rustlearn`
Usage should be straightforward.

- import the prelude for alll the linear algebra primitives and common traits:

```rust
use rustlearn::prelude::*;
```

- import individual models and utilities from submodules:

```rust
use rustlearn::prelude::*;

use rustlearn::linear_models::sgdclassifier::Hyperparameters;
// more imports
```

## Examples

### Logistic regression

```rust
use rustlearn::prelude::*;
use rustlearn::datasets::iris;
use rustlearn::cross_validation::CrossValidation;
use rustlearn::linear_models::sgdclassifier::Hyperparameters;
use rustlearn::metrics::accuracy_score;


let (X, y) = iris::load_data();

let num_splits = 10;
let num_epochs = 5;

let mut accuracy = 0.0;

for (train_idx, test_idx) in CrossValidation::new(X.rows(), num_splits) {

    let X_train = X.get_rows(&train_idx);
    let y_train = y.get_rows(&train_idx);
    let X_test = X.get_rows(&test_idx);
    let y_test = y.get_rows(&test_idx);

    let mut model = Hyperparameters::new(X.cols())
                                    .learning_rate(0.5)
                                    .l2_penalty(0.0)
                                    .l1_penalty(0.0)
                                    .one_vs_rest();

    for _ in 0..num_epochs {
        model.fit(&X_train, &y_train).unwrap();
    }

    let prediction = model.predict(&X_test).unwrap();
    accuracy += accuracy_score(&y_test, &prediction);
}

accuracy /= num_splits as f32;

```

### Random forest

```rust
use rustlearn::prelude::*;

use rustlearn::ensemble::random_forest::Hyperparameters;
use rustlearn::datasets::iris;
use rustlearn::trees::decision_tree;

let (data, target) = iris::load_data();

let mut tree_params = decision_tree::Hyperparameters::new(data.cols());
tree_params.min_samples_split(10)
    .max_features(4);

let mut model = Hyperparameters::new(tree_params, 10)
    .one_vs_rest();

model.fit(&data, &target).unwrap();

// Optionally serialize and deserialize the model

// let encoded = bincode::rustc_serialize::encode(&model,
//                                                bincode::SizeLimit::Infinite).unwrap();
// let decoded: OneVsRestWrapper<RandomForest> = bincode::rustc_serialize::decode(&encoded).unwrap();

let prediction = model.predict(&data).unwrap();
```

## Contributing
Pull requests are welcome.

To run basic tests, run `cargo test`.

Running `cargo test --features "all_tests" --release` runs all tests, including generated and slow tests.
Running `cargo bench --features bench` (only on the nightly branch) runs benchmarks.
