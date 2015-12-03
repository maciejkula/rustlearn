//! Cross validation utilities.

pub mod cross_validation;
pub mod shuffle_split;

pub use self::cross_validation::CrossValidation;
pub use self::shuffle_split::ShuffleSplit;
