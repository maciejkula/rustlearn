import os

import numpy as np


RANDOM_STATE = np.random.RandomState(10)


INDEX_TEST_TEMPLATE = """

    #[test]
    fn test_{test_name}() {{

        let mut arr = Array::from(vec!{arr});
        arr.reshape({rows}, {cols});

        {test}
    }}

"""


MATMUL_TEST_TEMPLATE = """

    #[test]
    fn test_matmul_{test_name}() {{

        let mut arr1 = Array::from(vec!{arr1});
        arr1.reshape({arr1_rows}, {arr1_cols});

        let mut arr2 = Array::from(vec!{arr2});
        arr2.reshape({arr2_rows}, {arr2_cols});

        let mut result = Array::from(vec!{result});
        result.reshape({result_rows}, {result_cols});

        assert!(allclose(&arr1.dot(&arr2), &result));
    }}

"""


MATMUL_TRANSPOSE_TEST_TEMPLATE = """

    #[test]
    fn test_matmul_transpose_{test_name}() {{

        let mut arr1 = Array::from(vec!{arr1});
        arr1.reshape({arr1_rows}, {arr1_cols});

        let mut arr2 = Array::from(vec!{arr2});
        arr2.reshape({arr2_rows}, {arr2_cols});

        let mut result = Array::from(vec!{result});
        result.reshape({result_rows}, {result_cols});

        assert!(allclose(&arr1.dot(&(arr2.T())), &result));
    }}

"""


SCALAR_GET_ROWS_TEST_TEMPLATE = """

    #[test]
    fn test_scalar_get_rows_{test_name}() {{

        let mut arr = Array::from(vec!{arr});
        arr.reshape({arr_rows}, {arr_cols});

        let mut result = Array::from(vec!{result});
        result.reshape({result_rows}, {result_cols});

        let idx: usize = {idx};

        assert!(allclose(&arr.get_rows(&idx), &result));
    }}

"""


SCALAR_GET_ROWS_TRANSPOSE_TEST_TEMPLATE = """

    #[test]
    fn test_scalar_get_rows_transpose_{test_name}() {{

        let mut arr = Array::from(vec!{arr});
        arr.reshape({arr_rows}, {arr_cols});
        arr = arr.T();

        let mut result = Array::from(vec!{result});
        result.reshape({result_rows}, {result_cols});

        let idx: usize = {idx};

        assert!(result.rows() == arr.get_rows(&idx).rows());
        assert!(result.cols() == arr.get_rows(&idx).cols());

        assert!(allclose(&arr.get_rows(&idx), &result));
    }}

"""


VECTOR_GET_ROWS_TEST_TEMPLATE = """

    #[test]
    fn test_vector_get_rows_{test_name}() {{

        let mut arr = Array::from(vec!{arr});
        arr.reshape({arr_rows}, {arr_cols});

        let mut result = Array::from(vec!{result});
        result.reshape({result_rows}, {result_cols});

        let row_indices = vec!{row_indices};

        assert!(allclose(&arr.get_rows(&row_indices), &result));
    }}

"""


SCALAR_ADD_TEMPLATE = """

    #[test]
    fn test_scalar_add_{test_name}() {{

        let mut arr = Array::from(vec!{arr});
        arr.reshape({arr_rows}, {arr_cols});

        let mut result = Array::from(vec!{result});
        result.reshape({result_rows}, {result_cols});

        let rhs = {rhs};

        assert!(allclose(&(arr.add(rhs)), &result));

        arr.add_inplace(rhs);
        assert!(allclose(&(arr), &result));
    }}

"""

ARRAY_ELEM_OP_TEMPLATE = """

    #[test]
    fn test_array_op_{test_name}() {{

        let mut arr1 = Array::from(vec!{arr1});
        arr1.reshape({arr1_rows}, {arr1_cols});

        let mut arr2 = Array::from(vec!{arr2});
        arr2.reshape({arr2_rows}, {arr2_cols});

        let mut result = Array::from(vec!{result});
        result.reshape({result_rows}, {result_cols});

        assert!(allclose(&arr1.{fn}(&arr2), &result));

        arr1.{fn}_inplace(&arr2);
        assert!(allclose(&arr1, &result));
    }}

"""


MODULE_TEMPLATE = """

#[cfg(test)]
#[cfg(feature = "all_tests")]
mod generated_tests {{
    use array::dense::*;
    use array::traits::*;

    {tests}

}}

"""


def _create_matmul_test(name, lop, rop, res):

    for mat in (lop, rop, res):
        assert mat.flags.c_contiguous

    return MATMUL_TEST_TEMPLATE.format(test_name=name,
                                       arr1=lop.flatten().tolist(),
                                       arr1_rows=lop.shape[0],
                                       arr1_cols=lop.shape[1],
                                       arr2=rop.flatten().tolist(),
                                       arr2_rows=rop.shape[0],
                                       arr2_cols=rop.shape[1],
                                       result=res.flatten().tolist(),
                                       result_rows=res.shape[0],
                                       result_cols=res.shape[1])


def _create_matmul_transpose_test(name, lop, rop, res):

    for mat in (lop, rop, res):
        assert mat.flags.c_contiguous

    return MATMUL_TRANSPOSE_TEST_TEMPLATE.format(test_name=name,
                                                 arr1=lop.flatten().tolist(),
                                                 arr1_rows=lop.shape[0],
                                                 arr1_cols=lop.shape[1],
                                                 arr2=rop.flatten().tolist(),
                                                 arr2_rows=rop.shape[0],
                                                 arr2_cols=rop.shape[1],
                                                 result=res.flatten().tolist(),
                                                 result_rows=res.shape[0],
                                                 result_cols=res.shape[1])


def _gen_matmul_tests(number):

    for i in range(number):

        lop_rows = RANDOM_STATE.randint(1, 10)
        lop_cols = RANDOM_STATE.randint(1, 10)

        rop_rows = lop_cols
        rop_cols = RANDOM_STATE.randint(1, 10)

        lop = RANDOM_STATE.random_sample((lop_rows, lop_cols))
        rop = RANDOM_STATE.random_sample((rop_rows, rop_cols))

        yield _create_matmul_test(i, lop, rop, np.dot(lop, rop))


def _gen_matmul_transpose_tests(number):

    for i in range(number):

        lop_rows = RANDOM_STATE.randint(1, 10)
        lop_cols = RANDOM_STATE.randint(1, 10)

        rop_cols = lop_cols
        rop_rows = RANDOM_STATE.randint(1, 10)

        lop = RANDOM_STATE.random_sample((lop_rows, lop_cols))
        rop = RANDOM_STATE.random_sample((rop_rows, rop_cols))

        yield _create_matmul_transpose_test(i, lop, rop, np.dot(lop, rop.T))


def _gen_index_tests(number):

    for test_no in range(number):

        rows = RANDOM_STATE.randint(1, 10)
        cols = RANDOM_STATE.randint(1, 10)

        arr = RANDOM_STATE.random_sample((rows, cols))

        tests = []
        transpose_tests = ['let arr = arr.T();']
        mut_tests = []
        mut_transpose_tests = ['let arr = arr.T();']
        unsafe_tests = []
        unsafe_transpose_tests = ['let arr = arr.T();']

        for i in range(rows):
            for j in range(cols):
                tests.append('assert!(close(arr.get({i}, {j}), {val}));'
                             .format(i=i,
                                     j=j,
                                     val=arr[i, j]))
                transpose_tests.append('assert!(close(arr.get({i}, {j}), {val}));'
                                       .format(i=j,
                                               j=i,
                                               val=arr.T[j, i]))
                mut_tests.append('let mut v = arr.get({i}, {j}); v += 1.0; assert!(close(v, {val}));'
                                 .format(i=i,
                                         j=j,
                                         val=arr[i, j] + 1.0))
                mut_transpose_tests.append('let mut v = arr.get({i}, {j}); '
                                           'v += 1.0; assert!(close(v, {val}));'
                                           .format(i=j,
                                                   j=i,
                                                   val=arr.T[j, i] + 1.0))
                unsafe_tests.append('unsafe {{ assert!(close(arr.get_unchecked({i}, {j}), {val})); }};'
                                    .format(i=i,
                                            j=j,
                                            val=arr[i, j]))
                unsafe_transpose_tests.append('unsafe {{ assert!(close(arr.get_unchecked({i}, {j}), {val})); }};'
                                              .format(i=j,
                                                      j=i,
                                                      val=arr.T[j, i]))

        yield INDEX_TEST_TEMPLATE.format(test_name='index_{}'.format(test_no),
                                         arr=arr.flatten().tolist(),
                                         rows=rows,
                                         cols=cols,
                                         test='\n'.join(tests))
        yield INDEX_TEST_TEMPLATE.format(test_name='index_transpose_{}'.format(test_no),
                                         arr=arr.flatten().tolist(),
                                         rows=rows,
                                         cols=cols,
                                         test='\n'.join(transpose_tests))
        yield INDEX_TEST_TEMPLATE.format(test_name='index_mut_{}'.format(test_no),
                                         arr=arr.flatten().tolist(),
                                         rows=rows,
                                         cols=cols,
                                         test='\n'.join(mut_tests))
        yield INDEX_TEST_TEMPLATE.format(test_name='index_transpose_mut_{}'.format(test_no),
                                         arr=arr.flatten().tolist(),
                                         rows=rows,
                                         cols=cols,
                                         test='\n'.join(mut_transpose_tests))
        yield INDEX_TEST_TEMPLATE.format(test_name='index_unsafe_{}'.format(test_no),
                                         arr=arr.flatten().tolist(),
                                         rows=rows,
                                         cols=cols,
                                         test='\n'.join(unsafe_tests))
        yield INDEX_TEST_TEMPLATE.format(test_name='index_transpose_unsafe_{}'.format(test_no),
                                         arr=arr.flatten().tolist(),
                                         rows=rows,
                                         cols=cols,
                                         test='\n'.join(unsafe_transpose_tests))


def _gen_scalar_get_rows_tests(number):

    for test_no in range(number):

        rows = RANDOM_STATE.randint(1, 10)
        cols = RANDOM_STATE.randint(1, 10)

        arr = RANDOM_STATE.random_sample((rows, cols))
        idx = RANDOM_STATE.randint(0, rows)

        yield SCALAR_GET_ROWS_TEST_TEMPLATE.format(test_name=test_no,
                                                      arr=arr.flatten().tolist(),
                                                      arr_rows=rows,
                                                      arr_cols=cols,
                                                      idx=idx,
                                                      result=arr[idx].flatten().tolist(),
                                                      result_rows=1,
                                                      result_cols=cols)


def _gen_scalar_get_rows_transpose_tests(number):

    for test_no in range(number):

        rows = RANDOM_STATE.randint(1, 10)
        cols = RANDOM_STATE.randint(1, 10)

        arr = RANDOM_STATE.random_sample((rows, cols))
        idx = RANDOM_STATE.randint(0, cols)

        yield (SCALAR_GET_ROWS_TRANSPOSE_TEST_TEMPLATE
               .format(test_name=test_no,
                       arr=arr.flatten().tolist(),
                       arr_rows=rows,
                       arr_cols=cols,
                       idx=idx,
                       result=arr.T[idx].flatten().tolist(),
                       result_rows=1,
                       result_cols=rows))


def _gen_vector_get_rows_transpose_tests(number):

    for test_no in range(number):

        rows = RANDOM_STATE.randint(1, 10)
        cols = RANDOM_STATE.randint(1, 10)

        arr = RANDOM_STATE.random_sample((rows, cols))
        row_idx = RANDOM_STATE.randint(0, rows, size=rows*3)

        yield (VECTOR_GET_ROWS_TEST_TEMPLATE
               .format(test_name=test_no,
                       arr=arr.flatten().tolist(),
                       arr_rows=rows,
                       arr_cols=cols,
                       row_indices=row_idx.tolist(),
                       result=arr[row_idx].flatten().tolist(),
                       result_rows=len(row_idx),
                       result_cols=cols))


def _gen_scalar_add_tests(number):

    for test_no in range(number):

        rows = RANDOM_STATE.randint(1, 10)
        cols = RANDOM_STATE.randint(1, 10)

        arr = RANDOM_STATE.random_sample((rows, cols))
        rhs = RANDOM_STATE.random_sample()
        if RANDOM_STATE.random_sample() < 0.5:
            rhs = rhs * -1

        yield (SCALAR_ADD_TEMPLATE
               .format(test_name=test_no,
                       arr=arr.flatten().tolist(),
                       arr_rows=rows,
                       arr_cols=cols,
                       rhs=rhs,
                       result=(arr + rhs).flatten().tolist(),
                       result_rows=rows,
                       result_cols=cols))


def _gen_array_elem_op_tests(number):

    for i in range(number):

        rows = RANDOM_STATE.randint(1, 10)
        cols = RANDOM_STATE.randint(1, 10)

        lop = RANDOM_STATE.random_sample((rows, cols)).astype(np.float32)
        rop = RANDOM_STATE.random_sample((rows, cols)).astype(np.float32)

        for (op, res) in (('add', lop + rop),
                          ('sub', lop - rop),
                          ('times', lop * rop),
                          ('div', lop / rop)):
            yield (ARRAY_ELEM_OP_TEMPLATE
                   .format(test_name='{}_{}'.format(op, i),
                           fn=op,
                           arr1=lop.flatten().tolist(),
                           arr1_rows=lop.shape[0],
                           arr1_cols=lop.shape[1],
                           arr2=rop.flatten().tolist(),
                           arr2_rows=rop.shape[0],
                           arr2_cols=rop.shape[1],
                           result=res.flatten().tolist(),
                           result_rows=res.shape[0],
                           result_cols=res.shape[1]))


def generate():

    fname = os.path.join(os.path.dirname(__file__),
                         'test.rs')

    tests = (list(_gen_matmul_tests(10))
             + list(_gen_matmul_transpose_tests(10))
             + list(_gen_index_tests(10))
             + list(_gen_scalar_get_rows_tests(10))
             + list(_gen_scalar_get_rows_transpose_tests(10))
             + list(_gen_vector_get_rows_transpose_tests(10))
             + list(_gen_scalar_add_tests(10))
             + list(_gen_array_elem_op_tests(10)))

    module = MODULE_TEMPLATE.format(tests=''.join(tests))

    with open(fname, 'wb') as datafile:
        datafile.write(module)


generate()
