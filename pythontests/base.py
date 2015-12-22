import os

import numpy as np


def serialize_array(arr):

    template = 'Array::from(&{})'

    if len(arr.shape) == 1:
        return template.format(str([[x] for x in arr]).replace('[', 'vec!['))
    else:
        return template.format(str(arr).replace('[', 'vec!['))


class Module(object):

    TEMPLATE = """

#[cfg(test)]
{flags}
mod generated_tests {{
    {imports}

    {tests}

}}

"""

    def __init__(self, imports=None, flags=None):

        self.imports = (imports or []) + ['prelude::*',
                                          'super::*']
        self.flags = flags or []

        self.tests = []

    def add_test(self, test):

        self.tests.append(test)

    def render_flags(self):

        return '\n'.join(self.flags)

    def render_imports(self):

        return '\n'.join(['use ' + x  + ';' for x in self.imports])

    def render(self):

        return self.TEMPLATE.format(flags=self.render_flags(),
                                    imports=self.render_imports(),
                                    tests='\n'.join([x.render() for x
                                                     in self.tests]))


class Test(object):

    TEMPLATE = """
               #[test]
               fn {name}() {{
                   // Body goes here
               }}

    """

    SERIALIZERS = {np.ndarray: serialize_array}

    def __init__(self, name, args):

        self.name = name
        self.args = args

    def _render_args(self):

        rendered = {}

        for key, value in self.args.items():
            for tpe, fnc in self.SERIALIZERS.items():
                if isinstance(value, tpe):
                    rendered[key] = fnc(value)
                else:
                    rendered[key] = str(value)

        return rendered

    def render(self):

        args = self._render_args()
        args['name'] = self.name

        return self.TEMPLATE.format(**args)
