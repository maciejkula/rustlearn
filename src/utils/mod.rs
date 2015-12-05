//! Internal utils.
use rand::StdRng;
use rustc_serialize::*;

/// Wrapper for making random number generators serializable.
/// Does no actual encoding, and merely creates a new
/// generator on decoding.
#[derive(Clone)]
pub struct EncodableRng {
    pub rng: StdRng
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
