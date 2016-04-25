#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import click
from Levenshtein import distance

@click.command()
@click.option("--translations_dir", help="A directory containing a set of files holding inferred translations to be scored along with the correct translations of these.")
@click.option("--stats_output_path", help="The path to which to write the computed quality statistics.")


tf.app.flags.DEFINE_string("trace_path", "/tmp", "Path to a trace to decode.")


def cli(translations_dir, stats_output_path):
    """Evaluate a set of basecalls made on an evaluation dataset."""
    score = 0
    ct = 0
    # For each file stem in the translations_dir directory,
        # if there is both a .trans.txt and .key.txt file present
            # read in the contents of both into 'trans' and 'key'
            trans = 
            key = 
            score += 2*distance(trans, key)/(len(trans)+len(key))
            ct +=1
    print "the score is " + str(score/ct) + "\n"


'''
tracer-eval --inputCalls=$TRACERDIR/data/model.txt --inputKey=$TRACERDIR/data/test_data.txt --output=$TRACERDIR/data/qual.txt
'''

