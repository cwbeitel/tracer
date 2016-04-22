#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click
import numpy as np
import pandas as pd
import random
from six.moves import cPickle as pickle

@click.command()
@click.option('--input', required=True)
@click.option('--output', required=True)
def cli(input, output):
    """Simulate instrument traces."""
    pass

def simulateRead(read_length, steps_per_base):
	"""For now, simulate just generates either 0's or 1's writes the result to files in a directory"""

	keys = ["A","T","C","G"]

	df = pd.DataFrame()
	df[0] = np.zeros(read_length*steps_per_base, dtype=np.float)
	df[1] = np.zeros(read_length*steps_per_base, dtype=np.float)
	df[2] = np.zeros(read_length*steps_per_base, dtype=np.float)
	df[3] = np.zeros(read_length*steps_per_base, dtype=np.float)
	df['correct'] = np.zeros(read_length*steps_per_base, dtype=np.int)

	for j in range(0, read_length):
		key = random.randint(0,3)
		for k in range(0, steps_per_base):
			ind = j*steps_per_base + k
			df.loc[ind, key] = 1
			df.loc[ind, 'correct'] = key

	return df
