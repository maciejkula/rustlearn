//! Basic two-dimensional dense matrix type.
//!
//! # Creation
//! An array of ones or zeros:
//!
//! ```
//! use rustlearn::prelude::*;
//!
//! let zeros = Array::zeros(20, 10);
//! let ones = Array::ones(10, 10);
//! ```
//!
//! From a vector:
//!
//! ```
//! use rustlearn::prelude::*;
//!
//! let mut array = Array::from(vec![0.0, 1.0, 2.0, 3.0]);
//! array.reshape(2, 2);
//! ```
//!
//! From a vector of vectors:
//!
//! ```
//! use rustlearn::prelude::*;
//!
//! let array = Array::from(&vec![vec![0.0, 1.0],
//!                               vec![2.0, 3.0]]);
//! ```
//!
//! # Getting and setting
//!
//! ```
//! use rustlearn::prelude::*;
//!
//! let mut array = Array::zeros(2, 2);
//!
//! array.set(0, 1, 1.0);
//! *array.get_mut(1, 0) = 2.0;
//! unsafe {
//!     array.set_unchecked(1, 1, 3.0);
//! }
//!
//! assert!(allclose(&array,
//!                  &Array::from(&vec![vec![0.0, 1.0],
//!                                     vec![2.0, 3.0]])));
//! ```
//!
//! # Iteration
//! Over raw data:
//!
//! ```
//! use rustlearn::prelude::*;
//!
//! let array = Array::from(&vec![vec![0.0, 1.0],
//!                               vec![2.0, 3.0]]);
//!
//! let sum = array.data().iter().fold(0.0, |sum, val| sum + val);
//!
//! ```
//!
//! Over rows and columns:
//!
//! ```
//! use rustlearn::prelude::*;
//!
//! let array = Array::from(&vec![vec![0.0, 1.0],
//!                               vec![2.0, 3.0]]);
//!
//! let mut sum = 0.0;
//!
//! for row in array.iter_rows() {
//!     for element in row.iter() {
//!         sum += element;
//!     }
//! }
//!
//! sum = 0.0;
//!
//! for col in array.iter_columns() {
//!     for element in col.iter() {
//!         sum += element;
//!     }
//! }
//! ```
//!
//! # Elementwise operations
//!
//! On both `f32` and other `Array`s, with both immutable and in-place variants.
//!
//! ```
//! use rustlearn::prelude::*;
//!
//! let array = Array::from(&vec![vec![0.0, 1.0],
//!                               vec![2.0, 3.0]]);
//!
//! assert!(allclose(&array.add(&array),
//!                  &array.times(2.0)));
//! ```
//!
//! # Matrix multiplication
//!
//! ```
//! use rustlearn::prelude::*;
//!
//! let x = Array::from(vec![1.0, 2.0]);
//! let y = Array::from(vec![3.0, 4.0]);
//!
//! let dot = x.dot(&y.T());
//! ```


use std::iter::Iterator;
use std::ops::Range;

use array::traits::*;


#[derive(Clone, Copy, Debug)]
enum ArrayIteratorAxis {
    Row,
    Column,
}


/// Basic two-dimensional dense matrix type.
#[derive(RustcEncodable, RustcDecodable)]
#[derive(Clone, Debug)]
pub struct Array {
    rows: usize,
    cols: usize,
    order: MatrixOrder,
    data: Vec<f32>,
}


/// A view into a row or column of an existing dense matrix.
#[derive(Clone, Debug)]
pub struct ArrayView<'a> {
    idx: usize,
    axis: ArrayIteratorAxis,
    array: &'a Array,
}


/// Iterator over row or column views of a dense matrix.
pub struct ArrayIterator<'a> {
    stop: usize,
    idx: usize,
    axis: ArrayIteratorAxis,
    array: &'a Array,
}


impl<'a> Iterator for ArrayIterator<'a> {
    type Item = ArrayView<'a>;

    fn next(&mut self) -> Option<ArrayView<'a>> {

        let result = if self.idx < self.stop {
            Some(ArrayView {
                idx: self.idx,
                axis: self.axis,
                array: self.array,
            })
        } else {
            None
        };

        self.idx += 1;

        result
    }
}


impl<'a> RowIterable for &'a Array {
    type Item = ArrayView<'a>;
    type Output = ArrayIterator<'a>;
    fn iter_rows(self) -> ArrayIterator<'a> {
        ArrayIterator {
            stop: self.rows(),
            idx: 0,
            axis: ArrayIteratorAxis::Row,
            array: self,
        }
    }

    fn view_row(self, idx: usize) -> ArrayView<'a> {
        assert!(idx < self.rows);
        ArrayView {
            idx: idx,
            axis: ArrayIteratorAxis::Row,
            array: self,
        }
    }

    fn iter_rows_range(self, range: Range<usize>) -> ArrayIterator<'a> {
        let stop = if range.end > self.rows {
            self.rows
        } else {
            range.end
        };

        ArrayIterator {
            stop: stop,
            idx: range.start,
            axis: ArrayIteratorAxis::Row,
            array: self,
        }
    }
}


impl<'a> ColumnIterable for &'a Array {
    type Item = ArrayView<'a>;
    type Output = ArrayIterator<'a>;
    fn iter_columns(self) -> ArrayIterator<'a> {
        ArrayIterator {
            stop: self.cols(),
            idx: 0,
            axis: ArrayIteratorAxis::Column,
            array: self,
        }
    }

    fn view_column(self, idx: usize) -> ArrayView<'a> {
        assert!(idx < self.cols);
        ArrayView {
            idx: idx,
            axis: ArrayIteratorAxis::Column,
            array: self,
        }
    }

    fn iter_columns_range(self, range: Range<usize>) -> ArrayIterator<'a> {
        let stop = if range.end > self.cols {
            self.cols
        } else {
            range.end
        };

        ArrayIterator {
            stop: stop,
            idx: range.start,
            axis: ArrayIteratorAxis::Column,
            array: self,
        }
    }
}


/// Iterator over entries of a dense matrix view.
pub struct ArrayViewIterator<'a> {
    idx: usize,
    view: &'a ArrayView<'a>,
}


/// Iterator over nonzero entries of a dense matrix view.
pub struct ArrayViewNonzeroIterator<'a> {
    idx: usize,
    view: ArrayView<'a>,
}


impl<'a> Iterator for ArrayViewIterator<'a> {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {

        let result = match self.view.axis {
            ArrayIteratorAxis::Row => {
                if self.idx < self.view.array.cols {
                    unsafe { Some(self.view.array.get_unchecked(self.view.idx, self.idx)) }
                } else {
                    None
                }
            }
            ArrayIteratorAxis::Column => {
                if self.idx < self.view.array.rows {
                    unsafe { Some(self.view.array.get_unchecked(self.idx, self.view.idx)) }
                } else {
                    None
                }
            }
        };

        self.idx += 1;

        result
    }
}


impl<'a> Iterator for ArrayViewNonzeroIterator<'a> {
    type Item = (usize, f32);

    // Todo: actually skip zero entries
    fn next(&mut self) -> Option<(usize, f32)> {

        let result = match self.view.axis {
            ArrayIteratorAxis::Row => {
                if self.idx < self.view.array.cols {
                    unsafe {
                        Some((self.idx,
                              self.view
                            .array
                            .get_unchecked(self.view.idx, self.idx)))
                    }
                } else {
                    None
                }
            }
            ArrayIteratorAxis::Column => {
                if self.idx < self.view.array.cols {
                    unsafe {
                        Some((self.idx,
                              self.view
                            .array
                            .get_unchecked(self.idx, self.view.idx)))
                    }
                } else {
                    None
                }
            }
        };

        self.idx += 1;

        result
    }
}


impl<'a> ArrayView<'a> {
    /// Iterate over elements of the `ArrayView`.
    pub fn iter(&'a self) -> ArrayViewIterator<'a> {
        ArrayViewIterator {
            idx: 0,
            view: self,
        }
    }
}


impl<'a> NonzeroIterable for &'a ArrayView<'a> {
    type Output = ArrayViewNonzeroIterator<'a>;
    fn iter_nonzero(&self) -> ArrayViewNonzeroIterator<'a> {
        ArrayViewNonzeroIterator {
            idx: 0,
            view: (*(*self)).clone(),
        }
    }
}


impl<'a> NonzeroIterable for ArrayView<'a> {
    type Output = ArrayViewNonzeroIterator<'a>;
    fn iter_nonzero(&self) -> ArrayViewNonzeroIterator<'a> {
        ArrayViewNonzeroIterator {
            idx: 0,
            view: self.clone(),
        }
    }
}


impl IndexableMatrix for Array {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    unsafe fn get_unchecked(&self, row: usize, col: usize) -> f32 {

        match self.order {
            MatrixOrder::RowMajor => *self.data.get_unchecked(row * self.cols + col),
            MatrixOrder::ColumnMajor => *self.data.get_unchecked(row + self.rows * col),
        }
    }

    unsafe fn get_unchecked_mut(&mut self, row: usize, col: usize) -> &mut f32 {

        match self.order {
            MatrixOrder::RowMajor => self.data.get_unchecked_mut(row * self.cols + col),
            MatrixOrder::ColumnMajor => self.data.get_unchecked_mut(row + self.rows * col),
        }
    }
}


impl Array {
    /// Create a `rows` by `cols` array of zeros.
    pub fn zeros(rows: usize, cols: usize) -> Array {

        let mut data: Vec<f32> = Vec::with_capacity(rows * cols);

        for _ in 0..(rows * cols) {
            data.push(0.0);
        }

        Array {
            rows: rows,
            cols: cols,
            order: MatrixOrder::RowMajor,
            data: data,
        }
    }

    /// Create a `rows` by `cols` array of ones.
    pub fn ones(rows: usize, cols: usize) -> Array {

        let mut data: Vec<f32> = Vec::with_capacity(rows * cols);

        for _ in 0..(rows * cols) {
            data.push(1.0);
        }

        Array {
            rows: rows,
            cols: cols,
            order: MatrixOrder::RowMajor,
            data: data,
        }
    }

    fn empty(rows: usize, cols: usize) -> Array {
        let data: Vec<f32> = Vec::with_capacity(rows * cols);

        Array {
            rows: rows,
            cols: cols,
            order: MatrixOrder::RowMajor,
            data: data,
        }
    }

    /// Change the shape of the array to `rows` by `cols`.
    ///
    /// # Panics
    /// If the number of elements implied by the new shape
    /// is different from the current number of elements.
    pub fn reshape(&mut self, rows: usize, cols: usize) {

        assert!(rows * cols == self.rows * self.cols);

        self.rows = rows;
        self.cols = cols;
    }

    /// Return the order (row-major or column-major)
    /// of the array.
    pub fn order(&self) -> &MatrixOrder {
        &self.order
    }

    /// Return an immutable reference to the underlying
    /// data buffer of the array.
    ///
    /// The arrangement of the elements in that vector
    /// is dependent on whether this is a row-major
    /// or a column-major array.
    pub fn data(&self) -> &Vec<f32> {
        &self.data
    }

    /// Return an immutable reference to the underlying
    /// data buffer of the array.
    ///
    /// The arrangement of the elements in that vector
    /// is dependent on whether this is a row-major
    /// or a column-major array.
    pub fn as_slice(&self) -> &[f32] {
        &self.data[..]
    }

    /// Return an mutable reference to the underlying
    /// data buffer of the array.
    ///
    /// The arrangement of the elements in that vector
    /// is dependent on whether this is a row-major
    /// or a column-major array.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data[..]
    }

    /// Transpose the matrix.
    pub fn T(mut self) -> Array {
        let (rows, cols) = (self.cols, self.rows);
        self.rows = rows;
        self.cols = cols;

        self.order = match self.order {
            MatrixOrder::RowMajor => MatrixOrder::ColumnMajor,
            MatrixOrder::ColumnMajor => MatrixOrder::RowMajor,
        };

        self
    }

    /// Compute the sum of the entries of the array.
    pub fn sum(&self) -> f32 {
        self.data.iter().fold(0.0, |sum, val| sum + val)
    }

    /// Compute the mean of the array.
    pub fn mean(&self) -> f32 {
        self.sum() / ((self.cols * self.rows) as f32)
    }
}


impl From<Vec<f32>> for Array {
    /// Construct an array from a vector.
    ///
    /// # Panics
    /// This will panic if the input vector is emtpy.
    fn from(data: Vec<f32>) -> Array {

        assert!(data.len() > 0);

        Array {
            rows: data.len(),
            cols: 1,
            order: MatrixOrder::RowMajor,
            data: data,
        }
    }
}


impl<'a> From<&'a Vec<Vec<f32>>> for Array {
    /// Construct an array from a vector of vectors.
    ///
    /// # Panics
    /// This will panic if the input vector is emtpy
    /// or if its rows are of unequal length.
    fn from(input: &Vec<Vec<f32>>) -> Array {

        assert!(input.len() > 0);

        let rows = input.len();
        let cols = input[0].len();

        let mut data: Vec<f32> = Vec::with_capacity(rows * cols);

        for row in input {
            assert!(row.len() == cols);
            for &e in row {
                data.push(e);
            }
        }

        Array {
            rows: rows,
            cols: cols,
            order: MatrixOrder::RowMajor,
            data: data,
        }
    }
}


impl ElementwiseArrayOps<f32> for Array {
    type Output = Array;

    fn add(&self, rhs: f32) -> Array {
        Array {
            rows: self.rows,
            cols: self.cols,
            order: MatrixOrder::RowMajor,
            data: self.data
                .iter()
                .map(|&x| x + rhs)
                .collect::<Vec<f32>>(),
        }
    }

    fn add_inplace(&mut self, rhs: f32) {
        for v in &mut self.data {
            *v += rhs;
        }
    }

    fn sub(&self, rhs: f32) -> Array {
        self.add(-rhs)
    }

    fn sub_inplace(&mut self, rhs: f32) {
        self.add_inplace(-rhs);
    }

    fn times(&self, rhs: f32) -> Array {
        Array {
            rows: self.rows,
            cols: self.cols,
            order: MatrixOrder::RowMajor,
            data: self.data
                .iter()
                .map(|&x| x * rhs)
                .collect::<Vec<f32>>(),
        }
    }

    fn times_inplace(&mut self, rhs: f32) {
        for v in &mut self.data {
            *v *= rhs;
        }
    }

    fn div(&self, rhs: f32) -> Array {
        Array {
            rows: self.rows,
            cols: self.cols,
            order: MatrixOrder::RowMajor,
            data: self.data
                .iter()
                .map(|&x| x / rhs)
                .collect::<Vec<f32>>(),
        }
    }

    fn div_inplace(&mut self, rhs: f32) {
        for v in &mut self.data {
            *v /= rhs;
        }
    }
}

/// Perform elementwise operations between two arrays.
///
/// # Panics
/// Will panic if the two operands are not of the same shape.
impl<'a> ElementwiseArrayOps<&'a Array> for Array {
    type Output = Array;

    fn add(&self, rhs: &'a Array) -> Array {

        assert!(self.rows == rhs.rows && self.cols == rhs.cols);

        let mut data = Vec::with_capacity(self.rows * self.cols);

        unsafe {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    data.push(self.get_unchecked(i, j) + rhs.get_unchecked(i, j));
                }
            }
        }

        Array {
            rows: self.rows,
            cols: self.cols,
            order: MatrixOrder::RowMajor,
            data: data,
        }
    }

    fn add_inplace(&mut self, rhs: &'a Array) {

        assert!(self.rows == rhs.rows && self.cols == rhs.cols);

        unsafe {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let v = self.get_unchecked_mut(i, j);
                    *v += rhs.get_unchecked(i, j);
                }
            }
        }
    }

    fn sub(&self, rhs: &'a Array) -> Array {

        assert!(self.rows == rhs.rows && self.cols == rhs.cols);

        let mut data = Vec::with_capacity(self.rows * self.cols);

        unsafe {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    data.push(self.get_unchecked(i, j) - rhs.get_unchecked(i, j));
                }
            }
        }

        Array {
            rows: self.rows,
            cols: self.cols,
            order: MatrixOrder::RowMajor,
            data: data,
        }
    }

    fn sub_inplace(&mut self, rhs: &'a Array) {

        assert!(self.rows == rhs.rows && self.cols == rhs.cols);

        unsafe {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let v = self.get_unchecked_mut(i, j);
                    *v -= rhs.get_unchecked(i, j);
                }
            }
        }
    }

    fn times(&self, rhs: &'a Array) -> Array {

        assert!(self.rows == rhs.rows && self.cols == rhs.cols);

        let mut data = Vec::with_capacity(self.rows * self.cols);

        unsafe {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    data.push(self.get_unchecked(i, j) * rhs.get_unchecked(i, j));
                }
            }
        }

        Array {
            rows: self.rows,
            cols: self.cols,
            order: MatrixOrder::RowMajor,
            data: data,
        }
    }

    fn times_inplace(&mut self, rhs: &'a Array) {

        assert!(self.rows == rhs.rows && self.cols == rhs.cols);

        unsafe {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let v = self.get_unchecked_mut(i, j);
                    *v *= rhs.get_unchecked(i, j);
                }
            }
        }
    }

    fn div(&self, rhs: &'a Array) -> Array {

        assert!(self.rows == rhs.rows && self.cols == rhs.cols);

        let mut data = Vec::with_capacity(self.rows * self.cols);

        unsafe {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    data.push(self.get_unchecked(i, j) / rhs.get_unchecked(i, j));
                }
            }
        }

        Array {
            rows: self.rows,
            cols: self.cols,
            order: MatrixOrder::RowMajor,
            data: data,
        }
    }

    fn div_inplace(&mut self, rhs: &'a Array) {

        assert!(self.rows == rhs.rows && self.cols == rhs.cols);

        unsafe {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let v = self.get_unchecked_mut(i, j);
                    *v /= rhs.get_unchecked(i, j);
                }
            }
        }
    }
}


impl<'a> Dot<&'a Array> for Array {
    type Output = Array;

    fn dot(&self, rhs: &'a Array) -> Array {

        assert!(self.cols == rhs.rows);

        let mut output = Array::empty(self.rows, rhs.cols);

        unsafe {
            for i in 0..output.rows {
                for j in 0..output.cols {

                    let mut output_entry = 0.0;

                    for k in 0..self.cols {
                        output_entry += self.get_unchecked(i, k) * rhs.get_unchecked(k, j);
                    }

                    *output.get_unchecked_mut(i, j) = output_entry;
                }
            }
        }

        output
    }
}


impl RowIndex<Vec<usize>> for Array {
    type Output = Array;
    fn get_rows(&self, index: &Vec<usize>) -> Array {

        let mut data = Vec::with_capacity(index.len() * self.cols);

        for &row_idx in index {
            for col_idx in 0..self.cols {
                unsafe {
                    data.push(self.get_unchecked(row_idx, col_idx));
                }
            }
        }

        Array {
            rows: index.len(),
            cols: self.cols,
            order: MatrixOrder::RowMajor,
            data: data,
        }
    }
}

/// Determines whether two arrays are sufficiently close to each other.
pub fn allclose(x: &Array, y: &Array) -> bool {

    let atol = 1e-08;
    let rtol = 1e-05;

    if x.rows == y.rows && x.cols == y.cols {
        unsafe {
            for i in 0..x.rows {
                for j in 0..x.cols {
                    let a = x.get_unchecked(i, j);
                    let b = y.get_unchecked(i, j);
                    if !((a - b).abs() < (atol + rtol * b.abs())) {
                        return false;
                    }
                }
            }
            true
        }
    } else {
        false
    }
}


/// Determines whether two floats are sufficiently close to each other.
pub fn close(x: f32, y: f32) -> bool {

    let atol = 1e-08;
    let rtol = 1e-05;

    (x - y).abs() < (atol + rtol * y.abs())
}


#[cfg(test)]
mod tests {

    use bincode;

    use array::traits::*;
    use super::*;


    #[test]
    fn new_from_vec() {
        let mut arr = Array::from(vec![1.0, 2.0, 3.0, 4.0]);
        arr.reshape(2, 2);

        assert!(arr.get(0, 0) == 1.0);
        assert!(arr.get(1, 1) == 4.0);
    }

    #[test]
    fn basic_allclose() {
        let mut arr = Array::from(vec![1.0, 2.0, 3.0, 4.0]);
        arr.reshape(2, 2);

        let mut arr2 = Array::from(vec![1.0, 2.0, 3.0, 4.0]);
        arr2.reshape(2, 2);

        assert!(allclose(&arr, &arr2));
        arr2.reshape(1, 4);
        assert!(!allclose(&arr, &arr2));

        let mut arr3 = Array::from(vec![1.5, 2.0, 3.0, 4.0]);
        arr3.reshape(2, 2);
        assert!(!allclose(&arr, &arr3));

        let mut arr3 = Array::from(vec![1.0, 3.0, 2.0, 4.0]);
        arr3.reshape(2, 2);
        let arr3 = arr3.T();
        assert!(allclose(&arr, &arr3));
    }

    #[test]
    fn basic_matmul() {
        let mut arr = Array::from(vec![1.0, 2.0, 3.0, 4.0]);
        arr.reshape(2, 2);

        let arr2 = arr.clone().T();

        let mut result = Array::from(vec![5.0, 11.0, 11.0, 25.0]);
        result.reshape(2, 2);

        assert!(allclose(&arr.dot(&arr2), &result));
    }

    #[test]
    fn basic_add() {
        let mut arr = Array::from(vec![1.0, 2.0, 3.0, 4.0]);
        arr.reshape(2, 2);

        let mut expected = Array::from(vec![2.0, 3.0, 4.0, 5.0]);
        expected.reshape(2, 2);

        assert!(allclose(&expected, &(arr.add(1.0))));

        arr.add_inplace(1.0);
        assert!(allclose(&expected, &arr));
    }

    #[test]
    fn serialization() {
        let arr = Array::from(&vec![vec![0.0, 1.0], vec![2.0, 3.0]]);

        let encoded = bincode::rustc_serialize::encode(&arr, bincode::SizeLimit::Infinite).unwrap();
        let decoded = bincode::rustc_serialize::decode(&encoded).unwrap();

        assert!(allclose(&arr, &decoded));
    }

    #[test]
    fn scalar_indexing() {

        let mut arr = Array::zeros(2, 3);

        assert!(arr.get(0, 1) == 0.0);

        *arr.get_mut(0, 1) = 3.0;
        *arr.get_mut(0, 1) += 1.0;

        assert!(arr.get(0, 1) == 4.0);
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_scalar_indexing() {

        let arr = Array::zeros(2, 3);

        assert!(arr.get(0, 10) == 0.0);
    }


    #[test]
    fn scalar_fancy_indexing() {

        let mut arr = Array::zeros(2, 3);
        *arr.get_mut(1, 0) = 5.0;

        let res = arr.get_rows(&1);

        assert!(res.rows == 1);
        assert!(res.cols == 3);
        assert!(res.get(0, 0) == 5.0);
    }


    #[test]
    fn vector_fancy_indexing() {

        let mut arr = Array::zeros(2, 3);
        *arr.get_mut(0, 0) = -5.0;
        *arr.get_mut(1, 0) = 5.0;

        let res = arr.get_rows(&vec![0, 1, 0]);

        assert!(res.rows == 3);
        assert!(res.cols == 3);
        assert!(res.get(0, 0) == -5.0);
        assert!(res.get(1, 0) == 5.0);
        assert!(res.get(2, 0) == -5.0);
    }


    #[test]
    fn range_fancy_indexing() {

        let mut arr = Array::zeros(2, 3);
        *arr.get_mut(0, 0) = -5.0;
        *arr.get_mut(1, 0) = 5.0;

        let res = arr.get_rows(&(0..2));

        assert!(res.rows == 2);
        assert!(res.cols == 3);
        assert!(res.get(0, 0) == -5.0);
        assert!(res.get(1, 0) == 5.0);
    }


    #[test]
    fn basic_iteration() {

        let mut arr = Array::from(vec![1.0, 2.0, 3.0, 4.0]);
        arr.reshape(2, 2);

        let mut arr_t = Array::from(vec![1.0, 3.0, 2.0, 4.0]);
        arr_t.reshape(2, 2);

        let arr_t_t = arr_t.T();

        for (i, row) in arr.iter_rows().enumerate() {
            for (j, v) in row.iter().enumerate() {
                assert!(v == arr.get(i, j));
            }
        }

        for (i, row) in arr_t_t.iter_rows().enumerate() {
            for (j, v) in row.iter().enumerate() {
                assert!(v == arr_t_t.get(i, j));
            }
        }

        for (j, col) in arr.iter_columns().enumerate() {
            for (i, v) in col.iter().enumerate() {
                assert!(v == arr.get(i, j));
            }
        }
    }

    use datasets::iris;

    #[test]
    fn range_iteration() {
        let (data, _) = iris::load_data();

        let (start, stop) = (5, 10);

        for (row_num, row) in data.iter_rows_range(start..stop).enumerate() {
            for (col_idx, value) in row.iter_nonzero() {
                assert!(value == data.get(start + row_num, col_idx));
            }

            assert!(row_num < (stop - start));
        }
    }
}
