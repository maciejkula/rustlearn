//! Datasets and dataset loading utilities.

pub mod iris;
pub mod boston;

#[cfg(test)]
#[cfg(any(feature = "all_tests", feature = "bench"))]
pub mod newsgroups;
