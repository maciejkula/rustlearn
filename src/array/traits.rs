//! Basic traits applying to all types of matrices.

use std::ops::{Range, RangeFrom, RangeTo, RangeFull};


#[derive(RustcEncodable, RustcDecodable)]
#[derive(Clone, Debug)]
pub enum MatrixOrder {
    RowMajor,
    ColumnMajor
}


/// Trait representing a shaped matrix whose entries can be accessed
/// at will using their row and column position.
pub trait IndexableMatrix {

    /// Return the number of rows of the matrix.
    fn rows(&self) -> usize;

    /// Return the number of columns of the matrix.
    fn cols(&self) -> usize;

    /// Get the value of the entry at (`row`, `column`) without bounds checking.
    unsafe fn get_unchecked(&self, row: usize, column: usize) -> f32;

    /// Get a mutable reference to the value of the entry at (`row`, `column`) without bounds checking.
    unsafe fn get_unchecked_mut(&mut self, row: usize, column: usize) -> &mut f32;

    /// Get the value of the entry at (`row`, `column`).
    ///
    /// # Panics
    /// Will panic if the element accessed is out of bounds.
    fn get(&self, row: usize, column: usize) -> f32 {
        assert!(row < self.rows());
        assert!(column < self.cols());

        unsafe {
            self.get_unchecked(row, column)
        }
    }

    /// Get a mutable reference to value of the entry at (`row`, `column`).
    ///
    /// # Panics
    /// Will panic if the element accessed is out of bounds.
    fn get_mut(&mut self, row: usize, column: usize) -> &mut f32 {
        assert!(row < self.rows());
        assert!(column < self.cols());

        unsafe {
            self.get_unchecked_mut(row, column)
        }
    }

    /// Set the value of the entry at (`row`, `column`) to `value`.
    ///
    /// # Panics
    /// Will panic if the element accessed is out of bounds.
    fn set(&mut self, row: usize, column: usize, value: f32) {
        assert!(row < self.rows());
        assert!(column < self.cols());
        
        unsafe {
            self.set_unchecked(row, column, value);
        }
    }

    /// Set the value of the entry at (`row`, `column`) to `value` without bounds checking.
    unsafe fn set_unchecked(&mut self, row: usize, column: usize, value: f32) {
        *self.get_unchecked_mut(row, column) = value;
    }
}


/// Trait representing a matrix that can be iterated over in
/// a row-wise fashion.
pub trait RowIterable {
    type Item: NonzeroIterable;
    type Output: Iterator<Item=Self::Item>;
    /// Iterate over rows of the matrix.
    fn iter_rows(self) -> Self::Output;
    /// View a row of the matrix.
    fn view_row(self, idx: usize) -> Self::Item;
}


/// Trait representing a matrix that can be iterated over in
/// a column-wise fashion.
pub trait ColumnIterable {
    type Item: NonzeroIterable;
    type Output: Iterator<Item=Self::Item>;
    /// Iterate over columns of a the matrix.
    fn iter_columns(self) -> Self::Output;
    /// View a column of the matrix.
    fn view_column(self, idx: usize) -> Self::Item;
}


/// Trait representing an object whose non-zero
/// entries can be iterated over.
pub trait NonzeroIterable {
    type Output: Iterator<Item = (usize, f32)>;
    fn iter_nonzero(self) -> Self::Output;
}


/// Trait representing a matrix whose rows can be selected
/// to create a new matrix containing those rows.
pub trait RowIndex<Rhs> {
    type Output;
    fn get_rows<'a>(&self, index: &'a Rhs) -> Self::Output;
}


impl<T> RowIndex<usize> for T where T: RowIndex<Vec<usize>> {
    type Output = T::Output;
    fn get_rows(&self, index: &usize) -> Self::Output {
        self.get_rows(&vec![*index])
    }
}


impl<T> RowIndex<Range<usize>> for T where T: RowIndex<Vec<usize>> {
    type Output = T::Output;
    fn get_rows(&self, index: &Range<usize>) -> Self::Output {
        self.get_rows(&(index.start..index.end).collect::<Vec<usize>>())
    }
}


impl<T> RowIndex<RangeFrom<usize>> for T where T: RowIndex<Range<usize>> + IndexableMatrix {
    type Output = T::Output;
    fn get_rows(&self, index: &RangeFrom<usize>) -> Self::Output {
        self.get_rows(&(index.start..self.rows()))
    }
}


impl<T> RowIndex<RangeTo<usize>> for T where T: RowIndex<Range<usize>> + IndexableMatrix {
    type Output = T::Output;
    fn get_rows(&self, index: &RangeTo<usize>) -> Self::Output {
        self.get_rows(&(0..index.end))
    }
}


impl<T> RowIndex<RangeFull> for T where T: RowIndex<Range<usize>> + IndexableMatrix {
    type Output = T::Output;
    fn get_rows(&self, _: &RangeFull) -> Self::Output {
        self.get_rows(&(0..self.rows()))
    }
}


/// Elementwise array operations trait.
pub trait ElementwiseArrayOps<Rhs> {
    type Output;
    fn add(&self, rhs: Rhs) -> Self::Output;
    fn add_inplace(&mut self, rhs: Rhs);
    fn sub(&self, rhs: Rhs) -> Self::Output;
    fn sub_inplace(&mut self, rhs: Rhs);
    fn times(&self, rhs: Rhs) -> Self::Output;
    fn times_inplace(&mut self, rhs: Rhs);
    fn div(&self, rhs: Rhs) -> Self::Output;
    fn div_inplace(&mut self, rhs: Rhs);
}

/// A matrix multiplication trait.
pub trait Dot<Rhs> {
    type Output;
    fn dot(&self, rhs: Rhs) -> Self::Output;
}
