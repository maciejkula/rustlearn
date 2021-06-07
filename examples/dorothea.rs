//! Example using the DOROTHEA drug discovery dataset.
//! Chemical compounds represented by structural molecular features must be classified as active (binding to thrombin) or inactive.
//! This is one of 5 datasets of the NIPS 2003 feature selection challenge.
//!
//! This dataset is available at [https://archive.ics.uci.edu/ml/datasets/Dorothea](https://archive.ics.uci.edu/ml/datasets/Dorothea).
//!
//! This script downloads the data, transforms it into `rustlearn` matrices, and fits several classification models.
#![allow(non_snake_case)]

use std::io::{Read, Write};
use std::path::Path;
use std::fs::File;

extern crate reqwest;

extern crate rustlearn;

use rustlearn::prelude::*;
use rustlearn::linear_models::sgdclassifier;
use rustlearn::trees::decision_tree;
use rustlearn::ensemble::random_forest;
use rustlearn::metrics;

fn download_data(url: &str) -> reqwest::Result<String> {
    reqwest::blocking::get(url)?.text()
}

fn get_raw_data(url: &str, filename: &str) -> String {

    let path = Path::new(filename);

    let raw_data = match File::open(&path) {
        Err(_) => {
            println!("Downloading data for {}", filename);
            let file_data = download_data(url).unwrap();
            let mut file = File::create(&path).unwrap();
            file.write_all(file_data.as_bytes()).unwrap();

            file_data
        },
        Ok(mut file) => {
            println!("Reading data for {}", filename);
            let mut file_data = String::new();
            file.read_to_string(&mut file_data).unwrap();
            file_data
        }
    };

    raw_data
}


fn build_x_matrix(data: &str) -> SparseRowArray {

    let mut coo = Vec::new();

    for (row, line) in data.lines().enumerate() {
        for col_str in line.split_whitespace() {
            let col = col_str.parse::<usize>().unwrap();
            coo.push((row, col));
        }
    }

    let num_rows = coo.iter().map(|x| x.0).max().unwrap() + 1;
    let num_cols = coo.iter().map(|x| x.1).max().unwrap() + 1;

    let mut array = SparseRowArray::zeros(num_rows, num_cols);

    for &(row, col) in coo.iter() {
        array.set(row, col, 1.0);
    }

    array
}


fn build_y_array(data: &str) -> Array {

    let mut y = Vec::new();
    
    for line in data.lines() {
        for datum_str in line.split_whitespace() {
            let datum = datum_str.parse::<i32>().unwrap();
            y.push(datum);
        }
    }

    Array::from(y.iter().map(|&x| 
                             match x {
                                 -1 => 0.0,
                                 _ => 1.0,
                             }
    ).collect::<Vec<f32>>())
}


fn get_train_data() -> (SparseRowArray, SparseRowArray) {

    let X_train = build_x_matrix(&get_raw_data("https://archive.ics.uci.edu/ml/machine-learning-databases/dorothea/DOROTHEA/dorothea_train.data",
                                               "./examples/dorothea_train.data"));
    let X_test = build_x_matrix(&get_raw_data("https://archive.ics.uci.edu/ml/machine-learning-databases/dorothea/DOROTHEA/dorothea_valid.data",
                                              "./examples/dorothea_valid.data"));

    (X_train, X_test)
}


fn get_target_data() -> (Array, Array) {

    let y_train = build_y_array(&get_raw_data("https://archive.ics.uci.edu/ml/machine-learning-databases/dorothea/DOROTHEA/dorothea_train.labels",
                                              "./examples/dorothea_train.labels"));
    let y_test = build_y_array(&get_raw_data("https://archive.ics.uci.edu/ml/machine-learning-databases/dorothea/dorothea_valid.labels",
                                             "./examples/dorothea_valid.labels"));

    (y_train, y_test)
}


fn run_sgdclassifier(X_train: &SparseRowArray, X_test: &SparseRowArray, y_train: &Array, y_test: &Array) {

    println!("Running SGDClassifier...");
    
    let num_epochs = 10;

    let mut model = sgdclassifier::Hyperparameters::new(X_train.cols())
        .learning_rate(0.5)
        .l2_penalty(0.000001)
        .build();

    for _ in 0..num_epochs {
        model.fit(X_train, y_train).unwrap();
    }

    let predictions = model.predict(X_test).unwrap();
    let accuracy = metrics::accuracy_score(y_test, &predictions);

    println!("SGDClassifier accuracy: {}", accuracy);
}


fn run_decision_tree(X_train: &SparseRowArray, X_test: &SparseRowArray, y_train: &Array, y_test: &Array) {

    println!("Running DecisionTree...");

    let X_train = SparseColumnArray::from(X_train);
    let X_test = SparseColumnArray::from(X_test);

    let mut model = decision_tree::Hyperparameters::new(X_train.cols())
        .build();

    model.fit(&X_train, y_train).unwrap();
    
    let predictions = model.predict(&X_test).unwrap();
    let accuracy = metrics::accuracy_score(y_test, &predictions);

    println!("DecisionTree accuracy: {}", accuracy);
}


fn run_random_forest(X_train: &SparseRowArray, X_test: &SparseRowArray, y_train: &Array, y_test: &Array) {

    println!("Running RandomForest...");

    let num_trees = 10;

    let tree_params = decision_tree::Hyperparameters::new(X_train.cols());
    let mut model = random_forest::Hyperparameters::new(tree_params, num_trees)
        .build();
                                                   
    model.fit(X_train, y_train).unwrap();
    
    let predictions = model.predict(X_test).unwrap();
    let accuracy = metrics::accuracy_score(y_test, &predictions);

    println!("RandomForest accuracy: {}", accuracy);
}


fn main() {

    let (X_train, X_test) = get_train_data();
    let (y_train, y_test) = get_target_data();

    println!("Training data: {} by {} matrix with {} nonzero entries",
             X_train.rows(), X_train.cols(), X_train.nnz());
    println!("Test data: {} by {} matrix with {} nonzero entries",
             X_test.rows(), X_test.cols(), X_test.nnz());

    run_sgdclassifier(&X_train, &X_test, &y_train, &y_test);
    run_decision_tree(&X_train, &X_test, &y_train, &y_test);
    run_random_forest(&X_train, &X_test, &y_train, &y_test);
}
