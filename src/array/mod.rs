//! Basic matrix-like datastructures.

#[macro_use]
pub mod dense;
pub mod sparse;
pub mod traits;
mod test;

/// Prelude containing basic matrix-like structures and traits.
#[allow(unused_imports)]
pub mod prelude {
    pub use super::dense::*;
    pub use super::sparse::*;
    pub use super::traits::*;
}
