//! Sparse matrices.
//!
//! Two main sparse matrices are implemented: `SparseRowArray` and `SparseColumnArray`.
//! Both support efficient incremental construction as well as efficient iteration over
//! nonzero entries. Data is stored as indices and data vectors, in a row-wise or column-wise fashion.
//!
//! `SparseRowArray` allows efficient iteration over rows, and `SparseColumnArray` allows efficient
//! iteration over columns.
//!
//! # Examples
//!
//! ## Creating and populating an array
//!
//! ```
//! use rustlearn::prelude::*;
//!
//! let mut array = SparseRowArray::zeros(20, 5);
//!
//! array.set(0, 2, 5.0);
//!
//! println!("Entry at ({}, {}) is {}", 0, 2, array.get(0, 2));
//! ```
//!
//! ## Iterating over an array
//!
//! ```
//! use rustlearn::prelude::*;
//!
//! let array = SparseRowArray::from(&Array::from(&vec![vec![1.0, 2.0],
//!                                                     vec![3.0, 4.0]]));
//!
//! for (row_idx, row) in array.iter_rows().enumerate() {
//!     for (column_idx, value) in row.iter_nonzero() {
//!         println!("Entry at ({}, {}) is {}", row_idx, column_idx, value);
//!     }
//! }
//!
//! ```
use std::iter::Iterator;

use array::dense::*;
use array::traits::*;


/// A sparse matrix with entries arranged row-wise.
#[derive(RustcEncodable, RustcDecodable)]
pub struct SparseRowArray {
    rows: usize,
    cols: usize,
    indices: Vec<Vec<usize>>,
    data: Vec<Vec<f32>>,
}


/// A sparse matrix with entries arranged column-wise.
#[derive(RustcEncodable, RustcDecodable)]
pub struct SparseColumnArray {
    rows: usize,
    cols: usize,
    indices: Vec<Vec<usize>>,
    data: Vec<Vec<f32>>,
}


/// A sparse matrix with entries arranged row-wise.
#[derive(RustcEncodable, RustcDecodable)]
pub struct CompressedSparseRowArray {
    rows: usize,
    cols: usize,
    indices: Vec<usize>,
    indptr: Vec<usize>,
    data: Vec<f32>,
}


/// A view into a row or a column of an existing sparse matrix.
#[derive(Clone, Debug)]
pub struct SparseArrayView<'a> {
    indices: &'a [usize],
    data: &'a [f32],
}


/// Iterator over nonzero entries of a `SparseArrayView`.
pub struct SparseArrayViewIterator<'a> {
    idx: usize,
    view: SparseArrayView<'a>,
}


/// Iterator over row or column views of a sparse matrix.
pub struct SparseArrayIterator<'a> {
    idx: usize,
    dim: usize,
    indices: &'a Vec<Vec<usize>>,
    data: &'a Vec<Vec<f32>>,
}


pub struct CompressedSparseArrayIterator<'a> {
    idx: usize,
    dim: usize,
    indptr: &'a [usize],
    indices: &'a [usize],
    data: &'a [f32]
}


impl IndexableMatrix for SparseRowArray {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    unsafe fn get_unchecked(&self, row: usize, column: usize) -> f32 {
        get(row,
            column,
            &self.indices,
            &self.data,
            MatrixOrder::RowMajor)
    }

    unsafe fn get_unchecked_mut(&mut self, row: usize, column: usize) -> &mut f32 {
        get_mut(row,
                column,
                &mut self.indices,
                &mut self.data,
                MatrixOrder::RowMajor)
    }

    unsafe fn set_unchecked(&mut self, row: usize, column: usize, value: f32) {

        if value != 0.0 {
            *self.get_unchecked_mut(row, column) = value;
        }
    }
}


impl IndexableMatrix for SparseColumnArray {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    unsafe fn get_unchecked(&self, row: usize, column: usize) -> f32 {
        get(row,
            column,
            &self.indices,
            &self.data,
            MatrixOrder::ColumnMajor)
    }

    unsafe fn get_unchecked_mut(&mut self, row: usize, column: usize) -> &mut f32 {
        get_mut(row,
                column,
                &mut self.indices,
                &mut self.data,
                MatrixOrder::ColumnMajor)
    }

    unsafe fn set_unchecked(&mut self, row: usize, column: usize, value: f32) {

        if value != 0.0 {
            *self.get_unchecked_mut(row, column) = value;
        }
    }
}


impl IndexableMatrix for CompressedSparseRowArray {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    unsafe fn get_unchecked(&self, row: usize, column: usize) -> f32 {
        let row_start = *self.indptr.get_unchecked(row);
        let row_stop = *self.indptr.get_unchecked(row + 1);

        let row_indices = &self.indices[row_start..row_stop];

        match row_indices.binary_search(&column) {
            Ok(idx) => *self.data[row_start..row_stop].get_unchecked(idx),
            Err(_) => 0.0,
        }
    }

    unsafe fn get_unchecked_mut(&mut self, row: usize, column: usize) -> &mut f32 {
        let row_start = *self.indptr.get_unchecked(row);
        let row_stop = *self.indptr.get_unchecked(row + 1);

        let (found, index) = {
            let row_indices = &self.indices[row_start..row_stop];
            match row_indices.binary_search(&column) {
                Ok(idx) => (true, row_start + idx),
                Err(idx) => (false, row_start + idx),
            }
        };

        if !found {
            self.indices.insert(index, column);
            self.data.insert(index, 0.0);

            for elem in self.indptr[(row + 1)..].iter_mut() {
                *elem += 1;
            }
        };

        self.data.get_unchecked_mut(index)
    }

    unsafe fn set_unchecked(&mut self, row: usize, column: usize, value: f32) {
        if value != 0.0 {
            *self.get_unchecked_mut(row, column) = value;
        }
    }
}


unsafe fn get(row: usize,
              col: usize,
              array_indices: &[Vec<usize>],
              array_data: &[Vec<f32>],
              order: MatrixOrder)
              -> f32 {

    let (index, indices, data) = match order {
        MatrixOrder::RowMajor => {
            (col, array_indices.get_unchecked(row), array_data.get_unchecked(row))
        }
        MatrixOrder::ColumnMajor => {
            (row, array_indices.get_unchecked(col), array_data.get_unchecked(col))
        }
    };

    match indices.binary_search(&index) {
        Ok(idx) => *data.get_unchecked(idx),
        Err(_) => 0.0,
    }
}


unsafe fn get_mut<'a>(row: usize,
                      col: usize,
                      array_indices: &'a mut Vec<Vec<usize>>,
                      array_data: &'a mut Vec<Vec<f32>>,
                      order: MatrixOrder)
                      -> &'a mut f32 {

    let (index, indices, data) = match order {
        MatrixOrder::RowMajor => {
            (col, array_indices.get_unchecked_mut(row), array_data.get_unchecked_mut(row))
        }
        MatrixOrder::ColumnMajor => {
            (row, array_indices.get_unchecked_mut(col), array_data.get_unchecked_mut(col))
        }
    };

    let result = indices.binary_search(&index);

    match result {
        Ok(idx) => data.get_unchecked_mut(idx),
        Err(idx) => {
            {
                indices.insert(idx, index);
                data.insert(idx, 0.0);
            }
            data.get_unchecked_mut(idx)
        }
    }
}


impl SparseRowArray {
    /// Initialise an empty (`rows` by `cols`) matrix.
    pub fn zeros(rows: usize, cols: usize) -> SparseRowArray {

        let mut indices = Vec::with_capacity(rows);
        let mut data = Vec::with_capacity(rows);

        for _ in 0..rows {
            indices.push(Vec::new());
            data.push(Vec::new());
        }

        SparseRowArray {
            rows: rows,
            cols: cols,
            indices: indices,
            data: data,
        }
    }

    /// Return the number of nonzero entries.
    pub fn nnz(&self) -> usize {
        self.indices.iter().fold(0, |sum, x| sum + x.len())
    }

    pub fn todense(&self) -> Array {

        let mut array = Array::zeros(self.rows, self.cols);

        for (row_idx, (row_indices, row_values)) in self.indices
            .iter()
            .zip(self.data.iter())
            .enumerate() {
            for (&col_idx, &value) in row_indices.iter()
                .zip(row_values.iter()) {
                array.set(row_idx, col_idx, value);
            }
        }

        array
    }
}


impl<'a> From<&'a Array> for SparseRowArray {
    fn from(array: &Array) -> SparseRowArray {

        let mut sparse = SparseRowArray::zeros(array.rows(), array.cols());

        for (row_idx, row) in array.iter_rows().enumerate() {
            for (col_idx, value) in row.iter().enumerate() {
                if value != 0.0 {
                    sparse.set(row_idx, col_idx, value);
                }
            }
        }

        sparse
    }
}


impl CompressedSparseRowArray {
    /// Initialise an empty (`rows` by `cols`) matrix.
    pub fn zeros(rows: usize, cols: usize) -> CompressedSparseRowArray {

        let indptr = vec![0; rows + 1];
        let indices = Vec::new();
        let data = Vec::new();

        CompressedSparseRowArray {
            rows: rows,
            cols: cols,
            indptr: indptr,
            indices: indices,
            data: data,
        }
    }

    /// Return the number of nonzero entries.
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    pub fn todense(&self) -> Array {
        let mut dense = Array::zeros(self.rows, self.cols);

        for (i, row) in self.iter_rows().enumerate() {
            for (j, value) in row.iter_nonzero() {
                dense.set(i, j, value);
            }
        }

        dense
    }
}


impl<'a> From<&'a Array> for CompressedSparseRowArray {
    fn from(array: &Array) -> CompressedSparseRowArray {

        let mut indptr = Vec::with_capacity(array.rows() * array.cols());
        let mut indices = Vec::with_capacity(array.rows() * array.cols());
        let mut data = Vec::with_capacity(array.rows() * array.cols());

        let mut row_ptr = 0;

        for row in array.iter_rows() {
            indptr.push(row_ptr);
            for (col, value) in row.iter_nonzero() {
                if value != 0.0 {
                    row_ptr += 1;
                    indices.push(col);
                    data.push(value);
                }
            }
        }
        indptr.push(row_ptr);

        CompressedSparseRowArray {
            rows: array.rows(),
            cols: array.cols(),
            indptr: indptr,
            indices: indices,
            data: data
        }
    }
}


impl<'a> From<&'a SparseColumnArray> for SparseRowArray {
    fn from(array: &SparseColumnArray) -> SparseRowArray {

        let mut sparse = SparseRowArray::zeros(array.rows(), array.cols());

        for (col_idx, col) in array.iter_columns().enumerate() {
            for (row_idx, value) in col.iter_nonzero() {
                if value != 0.0 {
                    sparse.set(row_idx, col_idx, value);
                }
            }
        }

        sparse
    }
}


impl<'a> RowIterable for &'a SparseRowArray {
    type Item = SparseArrayView<'a>;
    type Output = SparseArrayIterator<'a>;
    fn iter_rows(self) -> SparseArrayIterator<'a> {
        SparseArrayIterator {
            idx: 0,
            dim: self.rows,
            indices: &self.indices,
            data: &self.data,
        }
    }

    fn view_row(self, idx: usize) -> SparseArrayView<'a> {

        SparseArrayView {
            indices: &self.indices[idx],
            data: &self.data[idx],
        }
    }
}


impl<'a> RowIterable for &'a CompressedSparseRowArray {
    type Item = SparseArrayView<'a>;
    type Output = CompressedSparseArrayIterator<'a>;
    fn iter_rows(self) -> CompressedSparseArrayIterator<'a> {
        CompressedSparseArrayIterator {
            idx: 0,
            dim: self.rows,
            indptr: &self.indptr[..],
            indices: &self.indices[..],
            data: &self.data[..],
        }
    }

    fn view_row(self, idx: usize) -> SparseArrayView<'a> {

        let start = self.indptr[idx];
        let stop = self.indptr[idx + 1];

        SparseArrayView {
            indices: &self.indices[start..stop],
            data: &self.data[start..stop],
        }
    }
}


impl SparseColumnArray {
    /// Initialise an empty (`rows` by `cols`) matrix.
    pub fn zeros(rows: usize, cols: usize) -> SparseColumnArray {

        let mut indices = Vec::with_capacity(cols);
        let mut data = Vec::with_capacity(cols);

        for _ in 0..cols {
            indices.push(Vec::new());
            data.push(Vec::new());
        }

        SparseColumnArray {
            rows: rows,
            cols: cols,
            indices: indices,
            data: data,
        }
    }

    /// Return the number of nonzero entries.
    pub fn nnz(&self) -> usize {
        self.indices.iter().fold(0, |sum, x| sum + x.len())
    }

    pub fn todense(&self) -> Array {

        let mut array = Array::zeros(self.rows, self.cols);

        for (col_idx, (col_indices, col_values)) in self.indices
            .iter()
            .zip(self.data.iter())
            .enumerate() {
            for (&row_idx, &value) in col_indices.iter()
                .zip(col_values.iter()) {
                array.set(row_idx, col_idx, value);
            }
        }

        array
    }
}


impl<'a> From<&'a Array> for SparseColumnArray {
    fn from(array: &Array) -> SparseColumnArray {

        let mut sparse = SparseColumnArray::zeros(array.rows(), array.cols());

        for (row_idx, row) in array.iter_rows().enumerate() {
            for (col_idx, value) in row.iter().enumerate() {
                sparse.set(row_idx, col_idx, value);
            }
        }

        sparse
    }
}


impl<'a> From<&'a SparseRowArray> for SparseColumnArray {
    fn from(array: &SparseRowArray) -> SparseColumnArray {

        let mut sparse = SparseColumnArray::zeros(array.rows(), array.cols());

        for (row_idx, row) in array.iter_rows().enumerate() {
            for (col_idx, value) in row.iter_nonzero() {
                sparse.set(row_idx, col_idx, value);
            }
        }

        sparse
    }
}


impl<'a> ColumnIterable for &'a SparseColumnArray {
    type Item = SparseArrayView<'a>;
    type Output = SparseArrayIterator<'a>;
    fn iter_columns(self) -> SparseArrayIterator<'a> {
        SparseArrayIterator {
            idx: 0,
            dim: self.cols,
            indices: &self.indices,
            data: &self.data,
        }
    }
    fn view_column(self, idx: usize) -> SparseArrayView<'a> {
        SparseArrayView {
            indices: &self.indices[idx],
            data: &self.data[idx],
        }
    }
}


impl<'a> NonzeroIterable for SparseArrayView<'a> {
    type Output = SparseArrayViewIterator<'a>;
    fn iter_nonzero(&self) -> SparseArrayViewIterator<'a> {
        SparseArrayViewIterator {
            idx: 0,
            view: self.clone(),
        }
    }
}


impl<'a> SparseArrayView<'a> {
    /// Returns a reference to indices of nonzero entries of the view.
    pub fn indices(&self) -> &[usize] {
        &self.indices[..]
    }

    /// Returns a reference to values of nonzero entries of the view.
    pub fn data(&self) -> &[f32] {
        &self.data[..]
    }

    /// Returns the count of  nonzero entries of the view.
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }
}


impl<'a> Iterator for SparseArrayViewIterator<'a> {
    type Item = (usize, f32);

    fn next(&mut self) -> Option<(usize, f32)> {

        let result = if self.idx < self.view.indices.len() {
            unsafe {
                Some((*self.view.indices.get_unchecked(self.idx),
                      *self.view.data.get_unchecked(self.idx)))
            }
        } else {
            None
        };

        self.idx += 1;

        result
    }
}


impl<'a> Iterator for SparseArrayIterator<'a> {
    type Item = SparseArrayView<'a>;

    fn next(&mut self) -> Option<SparseArrayView<'a>> {

        let result = if self.idx < self.dim {
            Some(SparseArrayView {
                indices: &self.indices[self.idx][..],
                data: &self.data[self.idx][..],
            })
        } else {
            None
        };

        self.idx += 1;

        result
    }
}


impl<'a> Iterator for CompressedSparseArrayIterator<'a> {
    type Item = SparseArrayView<'a>;

    fn next(&mut self) -> Option<SparseArrayView<'a>> {

        let result = if self.idx < self.dim {
            let start = self.indptr[self.idx];
            let stop = self.indptr[self.idx + 1];

            Some(SparseArrayView {
                indices: &self.indices[start..stop],
                data: &self.data[start..stop],
            })
        } else {
            None
        };

        self.idx += 1;

        result
    }
}


impl RowIndex<Vec<usize>> for SparseRowArray {
    type Output = SparseRowArray;
    fn get_rows(&self, index: &Vec<usize>) -> SparseRowArray {

        let mut indices = Vec::with_capacity(index.len());
        let mut data = Vec::with_capacity(index.len());

        for &row_idx in index {
            indices.push(self.indices[row_idx].clone());
            data.push(self.data[row_idx].clone());
        }

        SparseRowArray {
            rows: index.len(),
            cols: self.cols,
            indices: indices,
            data: data,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use array::dense::*;
    use array::traits::*;

    use bincode;

    use rand::{Rng, SeedableRng, StdRng};
    use rand::distributions::{IndependentSample, Range};

    fn generate_dense_array(rows: usize, cols: usize, density: f32, seed: &[usize]) -> Array {
        let mut rng = StdRng::from_seed(seed);

        let mut array = Array::zeros(rows, cols);

        for _ in 0..((density * ((rows * cols) as f32)) as usize) {

            if rng.gen::<f32>() < density {
                let row = Range::new(0, rows).ind_sample(&mut rng);
                let col = Range::new(0, cols).ind_sample(&mut rng);
                let value = Range::new(-1.0, 1.0).ind_sample(&mut rng);

                array.set(row, col, value);
            }
        }

        array
    }

    macro_rules! indexing_test {
        ($test_name:ident, $sparse_type:ident) => {

            #[test]
            fn $test_name() {

                let (rows, cols) = (100, 10);

                let dense = generate_dense_array(rows, cols, 0.5, &[100]);
                let mut sparse = $sparse_type::zeros(rows, cols);

                for row in 0..rows {
                    for col in 0..cols {
                        let value = dense.get(row, col);

                        if value != 0.0 {
                            sparse.set(row, col, value);
                        }
                    }
                }

                for row in 0..rows {
                    for col in 0..cols {
                        assert_eq!(dense.get(row, col),
                                   sparse.get(row, col));
                    }
                }

                assert!(allclose(&dense, &sparse.todense()));
            }
        }
    }

    macro_rules! row_iteration_test {
        ($test_name:ident, $sparse_type:ident) => {

            #[test]
            fn $test_name() {

                let (rows, cols) = (100, 10);

                let dense = generate_dense_array(rows, cols, 0.5, &[100]);
                let mut sparse = $sparse_type::zeros(rows, cols);

                for row in 0..rows {
                    for col in 0..cols {
                        let value = dense.get(row, col);

                        if value != 0.0 {
                            sparse.set(row, col, value);
                        }
                    }
                }

                for (row_idx, row) in sparse.iter_rows().enumerate() {
                    for (col_idx, value) in row.iter_nonzero() {
                        assert_eq!(dense.get(row_idx, col_idx),
                                   value)
                    }
                }

                assert_eq!(sparse.nnz(), dense.data().iter().fold(0, |acc, &x| acc + if x != 0.0 { 1 } else { 0 }));
                assert!(allclose(&dense, &sparse.todense()));
            }
        }
    }

    indexing_test!(test_sparse_row_array_indexing, SparseRowArray);
    indexing_test!(test_sparse_column_array_indexing, SparseColumnArray);
    indexing_test!(test_compressed_sparse_row_array_indexing, CompressedSparseRowArray);

    row_iteration_test!(test_sparse_row_array_iteration, SparseRowArray);
    row_iteration_test!(test_compressed_sparse_row_array_iteration, CompressedSparseRowArray);

    #[test]
    fn row_construction_and_indexing() {

        let dense_arr = Array::from(&vec![vec![0.0, 1.0], vec![2.0, 0.0]]);
        let arr = SparseRowArray::from(&dense_arr);
        let carr = CompressedSparseRowArray::from(&dense_arr);

        assert!(arr.nnz() == 2);
        assert!(allclose(&arr.todense(), &dense_arr));

        assert!(arr.get(0, 0) == 0.0);
        assert!(arr.get(0, 1) == 1.0);
        assert!(arr.get(1, 0) == 2.0);
        assert!(arr.get(1, 1) == 0.0);

        assert!(carr.nnz() == 2);
        // assert!(allclose(&carr.todense(), &dense_arr));

        assert!(carr.get(0, 0) == 0.0);
        assert!(carr.get(0, 1) == 1.0);
        assert!(carr.get(1, 0) == 2.0);
        assert!(carr.get(1, 1) == 0.0);
    }

    #[test]
    fn column_construction_and_indexing() {

        let dense_arr = Array::from(&vec![vec![0.0, 1.0], vec![2.0, 0.0]]);
        let arr = SparseColumnArray::from(&dense_arr);

        assert!(arr.nnz() == 2);
        assert!(allclose(&arr.todense(), &dense_arr));

        assert!(arr.get(0, 0) == 0.0);
        assert!(arr.get(0, 1) == 1.0);
        assert!(arr.get(1, 0) == 2.0);
        assert!(arr.get(1, 1) == 0.0);
    }

    #[test]
    fn row_test_iteration() {

        let dense_arr = Array::from(&vec![vec![0.0, 1.0], vec![2.0, 0.0]]);
        let arr = SparseRowArray::from(&dense_arr);
        let mut target = SparseRowArray::zeros(2, 2);

        for (row_idx, row) in arr.iter_rows().enumerate() {
            for (col_idx, value) in row.iter_nonzero() {
                target.set(row_idx, col_idx, value);
            }
        }

        assert!(allclose(&dense_arr, &target.todense()));
    }

    #[test]
    fn row_test_iteration_compressed() {

        let dense_arr = Array::from(&vec![vec![0.0, 1.0], vec![2.0, 0.0]]);
        let arr = CompressedSparseRowArray::from(&dense_arr);
        let mut target = CompressedSparseRowArray::zeros(2, 2);

        for (row_idx, row) in arr.iter_rows().enumerate() {
            for (col_idx, value) in row.iter_nonzero() {
                target.set(row_idx, col_idx, value);
            }
        }

        assert!(allclose(&dense_arr, &target.todense()));
    }

    #[test]
    fn column_test_iteration() {

        let dense_arr = Array::from(&vec![vec![0.0, 1.0], vec![2.0, 0.0]]);
        let arr = SparseColumnArray::from(&dense_arr);
        let mut target = SparseColumnArray::zeros(2, 2);

        for (col_idx, col) in arr.iter_columns().enumerate() {
            for (row_idx, value) in col.iter_nonzero() {
                target.set(row_idx, col_idx, value);
            }
        }

        assert!(allclose(&dense_arr, &target.todense()));
    }

    #[test]
    fn serialization_sparse_row() {
        let arr = SparseRowArray::from(&Array::from(&vec![vec![0.0, 1.0], vec![2.0, 0.0]]));

        let encoded = bincode::rustc_serialize::encode(&arr, bincode::SizeLimit::Infinite).unwrap();
        let decoded: SparseRowArray = bincode::rustc_serialize::decode(&encoded).unwrap();

        assert!(allclose(&arr.todense(), &decoded.todense()));
    }

    #[test]
    fn serialization_sparse_colum() {
        let arr = SparseColumnArray::from(&Array::from(&vec![vec![0.0, 1.0], vec![2.0, 0.0]]));

        let encoded = bincode::rustc_serialize::encode(&arr, bincode::SizeLimit::Infinite).unwrap();
        let decoded: SparseColumnArray = bincode::rustc_serialize::decode(&encoded).unwrap();

        assert!(allclose(&arr.todense(), &decoded.todense()));
    }

    #[test]
    fn row_index() {
        let dense_arr = Array::from(&vec![vec![0.0, 1.0], vec![2.0, 0.0]]);
        let arr = SparseRowArray::from(&dense_arr);

        assert!(allclose(&arr.get_rows(&0).todense(), &dense_arr.get_rows(&0)));
        assert!(allclose(&arr.get_rows(&vec![1, 0]).todense(),
                         &dense_arr.get_rows(&vec![1, 0])));
        assert!(allclose(&arr.get_rows(&(..)).todense(), &dense_arr.get_rows(&(..))));
    }
}
