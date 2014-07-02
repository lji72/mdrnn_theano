#/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle
import numpy as np
import sys
from PIL import Image

if __name__ == '__main__':
    if len(sys.argv) < 4 or (len(sys.argv) - 2) % 2 != 0:
        print 'Usage: prepare_dataset.py img labl [img labl] file.pckl'
        exit(0)

    # First, get image sizes
    im_sizes = []
    for i in range(1, (len(sys.argv) - 1) / 2 + 1):
        im = Image.open(sys.argv[i]).convert('L')
        im_sizes.append(im.size)

    # Compute max width and height
    max_w = reduce(
        lambda acc, x: max(acc, x[0]), im_sizes[1:], im_sizes[0][0])
    max_h = reduce(
        lambda acc, x: max(acc, x[1]), im_sizes[1:], im_sizes[0][1])

    mask = np.zeros((max_h, max_w, len(im_sizes)), dtype = np.int8)
    data = np.zeros((max_h, max_w, len(im_sizes)), dtype = np.float32)

    # Pack all images into a single batch
    j = 0
    for i in range(1, (len(sys.argv) - 1) / 2 + 1):
        im = Image.open(sys.argv[i]).convert('L')
        data[0:im.size[1], 0:im.size[0], j] = np.array(
            im, dtype = np.float32) / 255.0
        mask[0:im.size[1], 0:im.size[0], j] = 1
        j += 1

    labels = []
    for i in range((len(sys.argv) - 1) / 2 + 1, len(sys.argv) - 1):
        labels.append(sys.argv[i])



    dataset = {'data': data, 'mask': mask, 'labels': labels}
    f = open(sys.argv[-1], 'wb')
    cPickle.dump(dataset, f, -1)
    f.close()
