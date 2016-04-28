#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from Levenshtein import distance

tf.app.flags.DEFINE_string("decoded", "/tmp", "A file containing decoded traces in DNA sequence form.")
tf.app.flags.DEFINE_string("key", "/tmp", "A file containing correct decodings in DNA sequence form.")
tf.app.flags.DEFINE_string("score_file", "/tmp", "A file to which to write the computed score.")

FLAGS = tf.app.flags.FLAGS

def cli():
    """Evaluate a set of basecalls made on an evaluation dataset."""
    score = 0
    ct = 0
    with open(FLAGS.decoded, "r") as decoded_fh:
        with open(FLAGS.key, "r") as key_fh:
            d, k = decoded_fh.readline().strip(), key_fh.readline().strip()
            while d and k:
                mxlen = max(len(d),len(k))
                num = float(mxlen - distance(d, k))
                score += num/max(len(d),len(k))
                ct +=1
                d, k = decoded_fh.readline().strip(), key_fh.readline().strip()
    with open(FLAGS.score_file, "w") as score_fh:
        score_fh.write("decoded: " + FLAGS.decoded + "\n" + "key: " + FLAGS.key + "\n" + "score: " + str(score/ct) + "\n")

'''
tracer-eval --inputCalls=$TRACERDIR/data/model.txt --inputKey=$TRACERDIR/data/test_data.txt --output=$TRACERDIR/data/qual.txt
'''
