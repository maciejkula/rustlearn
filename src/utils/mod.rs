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
    fn encode<S: Encoder>(&self, _: &mut S) -> Result<(), S::Error> {
        Ok(())
    }
}


impl Decodable for EncodableRng {
    fn decode<D: Decoder>(_: &mut D) -> Result<Self, D::Error> {
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


pub struct RunningVarianceEstimator {
    num_observations: usize,
    mean: f32,
    variance: f32
}


impl RunningVarianceEstimator {
    pub fn new() -> RunningVarianceEstimator {
        RunningVarianceEstimator { num_observations: 0,
                                   mean: 0.0,
                                   variance: 0.0 }
    }

    pub fn add_observation(&mut self, x: f32) {
        self.num_observations += 1;

        if self.num_observations == 1 {
            self.mean = x;
            self.variance = 0.0;
            return;
        }

        let new_mean = self.mean + (x - self.mean) / self.num_observations as f32;
        let new_var = self.variance + (x - self.mean) * (x - new_mean);

        self.mean = new_mean;
        self.variance = new_var;
    }

    pub fn remove_observation(&mut self, x: f32) {
        if self.num_observations <= 1 {
            self.mean = 0.0;
            self.num_observations -= 1;
            return
        }

        let new_mean = (self.mean * self.num_observations as f32 - x)
            / (self.num_observations as f32 - 1.0);
        let new_var = self.variance - (x - new_mean) * (x - self.mean);

        self.num_observations -= 1;

        self.mean = new_mean;
        self.variance = new_var;
    }

    pub fn get_variance(&self) -> Option<f32> {
        match self.num_observations {
            0 => None,
            1 => Some(0.0),
            _ => Some(self.variance / (self.num_observations as f32 - 1.0))
        }
    }

    pub fn get_mean(&self) -> Option<f32> {
        match self.num_observations {
            0 => None,
            _ => Some(self.mean)
        }
    }
}


#[cfg(test)]
mod tests {

    use super::RunningVarianceEstimator;

    #[test]
    fn test_variance_estimator() {
        let data = vec![0.13023152, -1.09808877, -0.26505087, 0.70939795];

        let mut model = RunningVarianceEstimator::new();

        for datum in &data {
            model.add_observation(*datum);
        }

        assert!(model.get_mean().unwrap() == -0.13087755);
        assert!(model.get_variance().unwrap() == 0.5759136);

        for datum in &data[..2] {
            model.remove_observation(*datum);
            println!("removing {}", *datum);
        }

        assert!(model.get_mean().unwrap() == 0.22217351);
        assert!(model.get_variance().unwrap() == 0.4747753);
    }
}
