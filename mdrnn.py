#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle
import theano
import theano.tensor as T
import numpy as np


def blockify(
        inp, block_size = (1, 1), step_size = (1, 1), direction = (1, 1),
        padding = False):
    input_size = T.shape(inp)
    if padding:
        b0 = T.ceil((input_size[0] - block_size[0]) / step_size[0]) + 1
        b1 = T.ceil((input_size[1] - block_size[1]) / step_size[1]) + 1
    else:
        b0 = T.floor((input_size[0] - block_size[0]) / step_size[0]) + 1
        b1 = T.floor((input_size[1] - block_size[1]) / step_size[1]) + 1
    num_blocks = b0 * b1

    for b in range(num_blocks):
