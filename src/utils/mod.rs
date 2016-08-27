//! Internal utils.
//!
//! Used mostly for checking the inputs to model fitting routines.
//!
//! Made public to make extending rustlearn easier, but should be treated as semi-public
//! and subject to change.
use rand::StdRng;
use rustc_serialize::*;

use prelude::*;

/// Wrapper for making random number generators serializable.
/// Does no actual encoding, and merely creates a new
/// generator on decoding.
/// This is because rand generators do not expose internal state.
#[derive(Clone)]
pub struct EncodableRng {
    pub rng: StdRng,
}


impl EncodableRng {
    pub fn new() -> EncodableRng {
        EncodableRng { rng: StdRng::new().unwrap() }
    }
}


impl Default for EncodableRng {
    fn default() -> Self {
        EncodableRng::new()
    }
}


impl Encodable for EncodableRng {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        try!(s.emit_struct("EncodableRng", 0, |_| { Ok(()) }));
        Ok(())
    }
}


impl Decodable for EncodableRng {
    fn decode<D: Decoder>(d: &mut D) -> Result<Self, D::Error> {
        try!(d.read_struct("", 0, |_| { Ok(()) }));
        Ok((EncodableRng::new()))
    }
}


/// Check that the input array contains valid binary classification labels.
pub fn check_valid_labels(y: &Array) -> Result<(), &'static str> {

    if y.cols() != 1 {
        return Err("Target array has more than one column.");
    }

    if y.data().iter().all(|&x| x == 0.0 || x == 1.0) {
        Ok(())
    } else {
        Err("Invalid labels: target data is not either 0.0 or 1.0")
    }
}


/// Check compatibility of the model dimensions and the number of columns in X.
pub fn check_data_dimensionality<T: IndexableMatrix>(model_dim: usize,
                                                     X: &T)
                                                     -> Result<(), &'static str> {
    if X.cols() == model_dim {
        Ok(())
    } else {
        Err("Model input and model dimensionality differ.")
    }
}


// Check that X and y have the same number of rows.
pub fn check_matched_dimensions<T: IndexableMatrix>(X: &T, y: &Array) -> Result<(), &'static str> {
    if X.rows() == y.rows() {
        Ok(())
    } else {
        Err("Data matrix and target array do not have the same number of rows")
    }
}


#[cfg(test)]
mod tests {
    use super::EncodableRng;

    use rustc_serialize::json;

    #[test]
    fn test_encodable_rng_serialization() {
        let rng = EncodableRng::new();

        let serialized = json::encode(&rng).unwrap();
        let _: EncodableRng = json::decode(&serialized).unwrap();
    }
}
