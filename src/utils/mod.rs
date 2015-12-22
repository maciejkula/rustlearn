//! Internal utils.
use rand::StdRng;
use rustc_serialize::*;

use prelude::*;

/// Wrapper for making random number generators serializable.
/// Does no actual encoding, and merely creates a new
/// generator on decoding.
#[derive(Clone)]
pub struct EncodableRng {
    pub rng: StdRng,
}


impl EncodableRng {
    pub fn new() -> EncodableRng {
        EncodableRng { rng: StdRng::new().unwrap() }
    }
}


impl Encodable for EncodableRng {
    fn encode<S: Encoder>(&self, _: &mut S) -> Result<(), S::Error> {
        Ok(())
    }
}


impl Decodable for EncodableRng {
    fn decode<D: Decoder>(_: &mut D) -> Result<Self, D::Error> {
        Ok((EncodableRng::new()))
    }
}


pub fn check_valid_labels(y: &Array) -> Result<(), &'static str> {

    if y.cols() != 1 {
        return Err("Target array has more than one column.");
    }

    match y.data().iter().all(|&x| x == 0.0 || x == 1.0) {
        true => Ok(()),
        false => Err("Invalid labels: target data is not either 0.0 or 1.0"),
    }
}


pub fn check_data_dimensionality<T: IndexableMatrix>(model_dim: usize,
                                                     X: &T)
                                                     -> Result<(), &'static str> {
    match X.cols() == model_dim {
        true => Ok(()),
        false => Err("Model input and model dimensionality differ."),
    }
}


pub fn check_matched_dimensions<T: IndexableMatrix>(X: &T, y: &Array) -> Result<(), &'static str> {
    match X.rows() == y.rows() {
        true => Ok(()),
        false => Err("Data matrix and target array do not have the same number of rows"),
    }
}
