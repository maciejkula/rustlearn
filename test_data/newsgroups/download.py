import codecs
import os
import re

from sklearn.datasets import fetch_20newsgroups


def _get_target_path():

    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'data.csv')


def get_data():

    path = _get_target_path()

    if os.path.isfile(path):
        return

    data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))

    with codecs.open(path, 'w', encoding='utf-8') as datafile:
        for target, data in zip(data['target'], data['data']):
            datafile.write(u'{}, "{}"\n'.format(target,
                                                re.sub(r'[^0-9a-zA-Z ]+', ' ',
                                                       data.lower())))



if __name__ == '__main__':
    get_data()
