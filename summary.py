from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("perf", "", "path of the performance record file")


def main():
    with open(FLAGS.perf, 'r') as ips:
        episode_rwds = list()
        episode_lens = list()
        for line in ips:
            cols = line.strip().split('\t')
            if not line.startswith(">>>") and len(cols) == 2:
                episode_rwds.append(float(cols[0]))
                episode_lens.append(int(cols[1]))

    print("{}\t{}".format(np.mean(episode_rwds), np.mean(episode_lens)))


if __name__=="__main__":
    main()
