import os
from os.path import join
from nestly import Nest
import numpy as np

nest = Nest()

# Nest for models
model_settings = [
    {'sampling_rate': 0.9, 'time': 1},
]

nest.add(
    'sampling_rate',
    [0.9],
    label_func=lambda c: "rate%d" % int(c * 100))

nest.add(
    'time',
    [1],
    label_func=lambda c: "time%d" % int(c * 10),
)

nest.add(
    'model_seed',
    range(1),
    label_func=lambda c: 'model%d' % c)

nest.add(
    'data_seed',
    range(1),
    label_func=lambda c: 'data%d' % c)

nest.build('_output')
