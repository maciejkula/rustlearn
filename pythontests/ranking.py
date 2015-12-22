import os

import numpy as np

from sklearn.metrics import roc_auc_score

from base import Module, Test


class RocAucScoreTest(Test):

    TEMPLATE = """
               #[test]
               fn {name}() {{
                   let y_true = {y_true};
                   let y_hat = {y_hat};
                   let expected = {expected};

                   let computed = roc_auc_score(&y_true, &y_hat).unwrap();

                   if !close(expected, computed) {{
                       println!("Expected: {{}} computed {{}}", expected, computed);
                       assert!(false);
                   }}
               }}

    """


def _generate_test(name, rstate):

    no_points = rstate.randint(5, 20)

    y_true = rstate.randint(0, 2, no_points).astype(np.float32)
    y_hat = rstate.randn(no_points)

    expected = roc_auc_score(y_true, y_hat)

    return RocAucScoreTest(name, {'y_true': y_true,
                                  'y_hat': y_hat,
                                  'expected': expected})


def _generate_repeated_test(name, rstate):

    no_points = rstate.randint(5, 20)

    y_true = rstate.randint(0, 2, no_points).astype(np.float32)
    y_hat = rstate.randint(0, 3, no_points).astype(np.float32)

    expected = roc_auc_score(y_true, y_hat)

    return RocAucScoreTest(name, {'y_true': y_true,
                                  'y_hat': y_hat,
                                  'expected': expected})


def generate_module():

    RANDOM_STATE = np.random.RandomState(1031412)

    module = Module(imports=['metrics::ranking::*'])

    for num in range(5):
        name = 'roc_auc_test_%s' % num
        module.add_test(_generate_test(name, RANDOM_STATE))

    # for num in range(5):
    # name = 'roc_auc_test_repeated_%s' % num
    # module.add_test(_generate_repeated_test(name, RANDOM_STATE))

    return module


def generate():

    fname = os.path.join(os.path.dirname(__file__),
                         '..',
                         'src',
                         'metrics',
                         'test.rs')

    with open(fname, 'wb') as datafile:
        datafile.write(generate_module().render())


generate()
