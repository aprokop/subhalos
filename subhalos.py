#!/usr/bin/env python
"""subhalos.py

Usage:
  subhalos.py -i INPUT [-m MINPTS] [-M MINCLUSTER] [-n N] -o OUTPUT
  subhalos.py (-h | --help)

Options:
  -h --help                                 Show this screen.
  -i FILE --input-file=FILE                 Input file
  -m MINPTS --minpts=MINPTS                 minPts [default: 2]
  -M MINCLUSTER --mincluster=MINCLUSTER     Min cluster size [default: 2]
  -n N --max-num-points=N                   Max number of points to read [default: inf]
  -o FILE --output-file=FILE                Output file
"""

from docopt import docopt
import hdbscan
import numpy as np
import struct
import sys


def read_data(filename, max_num_points):
    with open(filename, mode='rb') as f:
        data = f.read()
        (n, dim) = struct.unpack('II', data[:8])
        n = int(min(n, max_num_points))
        dim = int(dim)
        data = data[8:]
        xyz = np.asarray(struct.unpack(str(dim*n) + 'f', data[:dim*n*4]))

    return n, dim, np.reshape(xyz, (n, dim))


def write_data(filename, labels):
    n = len(labels)
    with open(filename, mode='wb') as f:
        f.write(struct.pack('I', n))
        f.write(struct.pack(str(n) + 'i', *labels))


def rescale_data(data):
    assert len(np.shape(data)) == 2 and np.shape(data)[1] == 6, 'Wrong data dimensions'
    n = np.shape(data)[0]
    dim = np.shape(data)[1]

    np.set_printoptions(threshold=sys.maxsize)
    print(data[0:10,:])

    diff = data - np.mean(data, axis=0)
    disp_sq = np.sum(np.square(diff), axis=0) / n
    disp_x = np.sqrt(np.sum(disp_sq[0:3]/3))
    disp_v = np.sqrt(np.sum(disp_sq[3:6]/3))
    disp = [disp_x, disp_x, disp_x, disp_v, disp_v, disp_v]

    print(disp)

    data = data / disp

    print(data[0:10,:])
    np.set_printoptions(threshold=False)

    return data


if __name__ == '__main__':

    # Process input
    options = docopt(__doc__)

    input_file = options['--input-file']
    minpts = int(options['--minpts'])
    mincluster = int(options['--mincluster'])
    max_num_points = float(options['--max-num-points'])
    output_file = options['--output-file']

    print('Reading in data from "' + input_file + '"... ', end='', flush=True)
    n, dim, points = read_data(input_file, max_num_points)
    print('done\nRead in ' + str(n) + ' ' + str(dim) + 'D points')

    if dim == 6:
        rescale_data(points)

    np.set_printoptions(linewidth=100)

    clusterer = hdbscan.HDBSCAN(min_samples=minpts,
                                min_cluster_size=mincluster)
    clusterer.fit(points)

    print(f'dim = {dim}, min_pts = {minpts}, min_cluster = {mincluster}: #clusters {str(len(set(clusterer.labels_)))}')

    write_data(output_file, clusterer.labels_)
