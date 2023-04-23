#!/usr/bin/env python

import numpy as np
import struct


input_file = '/Users/xap/data/datasets/halo_for_subhalos_180K.arborx'
output_file = 'halo_for_subhalos_180K_3D.arborx'
D = 3


def read_data(filename):
    with open(filename, mode='rb') as f:
        data = f.read()
        (n, dim) = struct.unpack('II', data[:8])
        n = int(n)
        dim = int(dim)
        data = data[8:]
        xyz = np.asarray(struct.unpack(str(dim*n) + 'f', data[:dim*n*4]))

    return n, dim, np.reshape(xyz, (n, dim))

print('Reading in data from "' + input_file + '"... ', end='', flush=True)
n, dim, points = read_data(input_file)
print('done\nRead in ' + str(n) + ' ' + str(dim) + 'D points')

if D < dim:
    dim = min(dim, D)
    points = points[:, :dim]
    points = np.reshape(points, n*dim)


fout = open(output_file, mode='wb')
fout.write(struct.pack('II', n, dim))
fout.write(struct.pack(str(dim*n) + 'f', *points))
