import os
from os.path import join
from nestly import Nest
import numpy as np

nest = Nest()

nest.add(
    'max_leaves',
    [50],
    label_func=lambda c: "maxleaves%d" % c)

nest.add(
    'time',
    [3.2],
    label_func=lambda c: "time%d" % int(c * 10))

nest.add(
    'model_seed',
    range(33,34),
    label_func=lambda c: 'model%d' % c)

nest.add(
    'data_seed',
    lambda c: [c['model_seed']],
    label_func=lambda c: 'data%d' % c)


nest.add(
    'variance',
    [0.005, 0.004, 0.003, 0.002, 0.001],
    label_func=lambda c: 'var%d' % (c * 1000))

nest.build('_output')
