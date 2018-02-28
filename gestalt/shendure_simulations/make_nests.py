import os
from os.path import join
from nestly import Nest
import numpy as np

nest = Nest()

nest.add(
    'min_leaves',
    [40],
    label_func=lambda c: "leaves%d" % c)

nest.add(
    'time',
    [4.5],
    label_func=lambda c: "time%d" % int(c * 10),
)

nest.add(
    'model_seed',
    range(5,8),
    label_func=lambda c: 'model%d' % c)

nest.add(
    'data_seed',
    range(1),
    label_func=lambda c: 'data%d' % c)

nest.add(
    'lasso',
    [0.01, 0.8, 1.6],
    label_func=lambda c: 'lasso%.2f' % c)

nest.add(
    'ridge',
    [1],
    label_func=lambda c: 'ridge%.2f' % c)

nest.build('_output')
