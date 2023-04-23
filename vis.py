#!/usr/bin/env python
"""data_vis.py

Usage:
  data_vis.py -i INPUT [-b] [-k K] [-l LABELS] [-m MODE] [-n N] [-o OUTPUT_FILE] [-s SAMPLES] [-t TITLE]
  data_vis.py (-h | --help)

Options:
  -h --help                         Show this screen.
  -b --background                   Plot noise
  -i FILE --input-file=FILE         Input file containing benchmark results in JSON format
  -k K --k-clusters=K               Plot K largest clusters (and plot all others as noise) [default: -1]
  -l LABELS --labels=LABELS         Input file containing results of DBSCAN
  -m MODE --mode=MODE               Mode [default: 2d]
  -n N --max-num-points=N           Max number of points to read [default: inf]
  -o FILE --output-file=FILE        Output file
  -s SAMPLES --samples=SAMPLES      Number of samples to plot [default: 10000]
  -t TITLE --title=TITLE            Plot title
"""
# This import registers the 3D projection, but is otherwise unused.
#  from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np
import math
import random
import re
import struct
from docopt import docopt


def read_data(filename, max_num_points):
    with open(filename, mode='rb') as f:
        data = f.read()
        (n, dim) = struct.unpack('II', data[:8])
        n = int(min(n, max_num_points))
        dim = int(dim)
        data = data[8:]
        xyz = np.asarray(struct.unpack(str(dim*n) + 'f', data[:dim*n*4]))
    return n, dim, xyz

def read_labels(filename, max_num_points):
    with open(filename, mode='rb') as f:
        data = f.read()
        (n,), data = struct.unpack('I', data[:4]), data[4:]
        n = int(min(n, max_num_points))
        labels = np.asarray(struct.unpack(str(n) + 'i', data[:n*4]))
    return n, labels


if __name__ == '__main__':

    noise = -1

    # Process input
    options = docopt(__doc__)

    input_file = options['--input-file']
    k = int(options['--k-clusters'])
    max_num_points = float(options['--max-num-points'])
    samples = int(options['--samples'])
    output_file = options['--output-file']
    plot_title = options['--title']
    plot_noise = options['--background']
    mode = options['--mode']
    labels_file = options['--labels']

    # Input file format
    # - size (4 bytes, integer)
    # - x    (n*4 bytes, float)
    # - y    (n*4 bytes, float)
    print('Reading in data from "' + input_file + '"... ', end='', flush=True)
    n, dim, xyz = read_data(input_file, max_num_points)
    print('done\nRead in ' + str(n) + ' ' + str(dim) + 'D points')

    assert(dim == 2 or dim == 3)
    x = xyz[0::dim]
    y = xyz[1::dim]
    if dim == 3:
        z = xyz[2::dim]
    else:
        z = np.zeros((n, 1))


    if labels_file != None:
        print('Reading in labels from "' + input_file + '"...', end='', flush=True)
        nl, labels = read_labels(labels_file, max_num_points)
        assert nl == n
        print('done')
    else:
        labels = -np.ones(n)
        plot_noise = True
    print('Number of clusters: ' + str(len(set(labels))))

    # Generate random sample
    if samples < n:
        print('Generating sample... ', end='', flush=True)
        rs = random.sample(range(1, n), samples)
        x = x[rs]
        y = y[rs]
        z = z[rs]
        labels = labels[rs]
        print('done')
        print('Number of clusters (sample): ' + str(len(set(labels))))

    # Construct unique labels
    unique_labels = set(labels)
    if k > 0 and k < len(unique_labels):
        unique_labels = np.asarray(list(unique_labels))
        n_unique = len(unique_labels)

        # Find k-largest counts
        counts = np.zeros(n_unique)
        for i in range(n_unique):
            counts[i] = np.count_nonzero(labels == unique_labels[i])
        permute = np.argpartition(-counts, k)

        # Mark all other labels as noise
        labels[permute[k:]] = noise

        unique_labels = set(unique_labels[permute[:k]])
        unique_labels.add(noise)
        print('Number of clusters (plot): ' + str(len(unique_labels)))

    # Black removed and is used for noise instead.
    colors = [plt.cm.tab20(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    #  plt.figure(figsize=(16, 10))
    plt.figure(figsize=(8, 6))
    if mode == '2d':
        ax = plt.subplot(111)
    else:
        ax = plt.subplot(111, projection='3d')

    for cluster_index, color in zip(unique_labels, colors):
        if cluster_index == noise:
            if plot_noise:
                # Black used for noise.
                #  color = [0, 0, 0, 0.3]
                color = [0, 0, 0, 1]
            else:
                continue

        cluster_mask = (labels == cluster_index)

        if mode == '2d':
            ax.scatter(x[cluster_mask], y[cluster_mask],  color=tuple(color), s=8)
        else:
            #  ax.scatter(x[cluster_mask], y[cluster_mask], z[cluster_mask], color=tuple(color), s=10)
            ax.scatter(x[cluster_mask], y[cluster_mask], z[cluster_mask], color=tuple(color), s=2)

    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    plt.xticks(fontsize=24)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    plt.yticks(fontsize=24)
    if mode == '3d':
        ax.zaxis.set_major_locator(plt.MaxNLocator(4))
        ax.zaxis.set_tick_params(labelsize=24)

    if plot_title != None:
        plt.title(plot_title, fontsize=28)
    else:
        plt.title('Point distribution', fontsize=28)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=600)
        #  plt.savefig(output_file, bbox_inches='tight', dpi=300)
    else:
        plt.show()
