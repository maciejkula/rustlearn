//! The 20 newsgroups dataset.
//!
//! This is only available when running the full test suite.

 use csv;

use prelude::*;
use feature_extraction::dict_vectorizer::*;

/// Load the newsgroups dataset.
pub fn load_data() -> (SparseRowArray, Array) {

    let mut rdr = csv::Reader::from_file("./test_data/newsgroups/data.csv")
        .unwrap()
        .has_headers(false);

    let mut vectorizer = DictVectorizer::new();
    let mut target = Vec::new();

    for (row, record) in rdr.decode().enumerate() {
        let (y, data): (f32, String) = record.unwrap();
        
        for token in data.split_whitespace() {
            vectorizer.partial_fit(row, token, 1.0);
        }
        
        target.push(y);
    }

    let y = Array::from(target);
    let X = vectorizer.transform();

    (X, y)
}
