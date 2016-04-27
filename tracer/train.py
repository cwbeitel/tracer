#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tracer import data_utils
from tracer import seq2seq_model

from tensorflow.python.platform import flags

'''
# Have to use same model params between training and decoding, would rather have them always
# come from a config script so it can easily be re-used. The config script should be saved with
# all the other model data and loaded automatically.
tracer-train --data_dir=$TRACERDIR/data --train_dir=$TRACERDIR/data/checkpoints
tracer-decode --trace_path=$TRACERDIR/data/dev.ids20000.in --output=$TRACERDIR/data/test
'''

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 10, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("in_vocab_size", 20000, "Source vocabulary size.")
tf.app.flags.DEFINE_integer("out_vocab_size", 8, "Target vocabulary size.")

tf.app.flags.DEFINE_string("in_train", "", "File containing encoded input for training.")
tf.app.flags.DEFINE_string("out_train", "", "File containing encoded output for training.")
tf.app.flags.DEFINE_string("in_dev", "", "File containing encoded input for checkpoint reporting.")
tf.app.flags.DEFINE_string("out_dev", "", "File containing encoded output for checkpoint reporting.")

tf.app.flags.DEFINE_string("data_dir", "/tmp", "Directory containing data to be used to train the model.")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Directory to which to write training checkpoints.")
tf.app.flags.DEFINE_string("data_tag", "train", "Tag to look for in input files.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 50,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_lstm", True,
                            "Whether to use an LSTM (True) or GRU (False).")
tf.app.flags.DEFINE_boolean("debug", False,
                            "Whether to run in debugging mode.")


# For decoding
tf.app.flags.DEFINE_string("trace_path", "/tmp", "Path to a trace to decode.")
tf.app.flags.DEFINE_string("output", "/tmp", "Path to file to which to write the decoded output.")


FLAGS = tf.app.flags.FLAGS

#_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
#_buckets = [(5, 30), (30, 60), (60, 90), (90, 120)] # Read length buckets, so minibatches
# are always working with data of approximately the same size, to avoid wasting computation.
_buckets = [(90,120)]

def cli():
    """Takes pairs of raw PacBio instrument traces together with their DNA sequence and trains a deep LSTM model to be used for making base calls from trace data."""
    f = flags.FLAGS
    f._parse_flags()
    train()

def read_data(source_path, target_path, max_size=None):
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set

def read_flat_data(source_path, target_path, max_size=None):
  data_set = [[] for _ in _buckets]
  with open(source_path, "r") as source_file:
    with open(target_path, "r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 1000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        #print(source_ids)
        #print(target_ids)
        target_ids.append(data_utils.EOS_ID)

        # This bucket thing won't work for the problem of traces because the 
        # length of input and output are so different.
        #for bucket_id, (source_size, target_size) in enumerate(_buckets):
        #  print(bucket_id, source_size, target_size)
        #  if len(source_ids) < source_size and len(target_ids) < target_size:
        #    print("met condition")
        #    data_set[bucket_id].append([source_ids, target_ids])
        #    break
        data_set[0].append([source_ids, target_ids]) #hack
        source, target = source_file.readline(), target_file.readline()
  return data_set

def create_model(session, forward_only, debug=False):
  """Create translation model and initialize or load parameters in session."""
  if debug:
    print("creating model...")
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.in_vocab_size, FLAGS.out_vocab_size, _buckets,
      FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      forward_only=forward_only, use_lstm=FLAGS.use_lstm, debug=FLAGS.debug)
  if debug:
    print("getting checkpoint state")
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if debug:
    print("finished getting checkpoint state")
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model

def train():
  """Train a SMRT sequencing trace to DNA sequence translation model using traces of known sequence."""
  
  # Prepare the input data.
  #in_train, out_train, in_dev, out_dev, _, _ = data_utils.prepare_data(FLAGS.train_path, 
  #  FLAGS.dev_path, FLAGS.data_dir, FLAGS.in_vocab_size, FLAGS.out_vocab_size)

  # If paths to inputs are not provided, assume they are the following (will fail if not present)
  if len(FLAGS.in_train) == 0:
    in_train = os.path.join(FLAGS.data_dir, "train.ids" + str(FLAGS.in_vocab_size) + ".in")
  if len(FLAGS.out_train) == 0:
    out_train = os.path.join(FLAGS.data_dir, "train.ids" + str(FLAGS.out_vocab_size) + ".out")
  if len(FLAGS.in_dev) == 0:
    in_dev = os.path.join(FLAGS.data_dir, "dev.ids" + str(FLAGS.in_vocab_size) + ".in")
  if len(FLAGS.out_dev) == 0:
    out_dev = os.path.join(FLAGS.data_dir, "dev.ids" + str(FLAGS.out_vocab_size) + ".out")

  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    #dev_set = read_data(in_dev, out_dev)
    dev_set = read_flat_data(in_dev, out_dev)
    #train_set = read_data(in_train, out_train, FLAGS.max_train_data_size)
    train_set = read_flat_data(in_train, out_train, FLAGS.max_train_data_size)
    #print(len(dev_set))
    #print(len(train_set))
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        
        # Print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        
        # Run evals on development set and print their perplexity.
        for bucket_id in xrange(len(_buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
          eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()



def decode_cli():
    '''Make basecalls given trace input and a previously trained model.'''

    with tf.Session() as sess:

        print("opened session")

        # Create model and load parameters.
        model = create_model(sess, True, debug=True)
        model.batch_size = 1  # We decode one sentence at a time.

        # Currently this expects traces to be fed in in their encoded form

        print("initialized model")

        with open(FLAGS.output, "w") as decoded_file:
            with open(FLAGS.trace_path, "r") as tracefile:
         
                trace = map(int, tracefile.readline().strip().split(" "))

                while trace:

                    # Which bucket does it belong to?
                    bucket_id = 0
                    #bucket_id = min([b for b in xrange(len(_buckets))
                    #                  if _buckets[b][0] > len(trace)])
                    
                    # Get a 1-element batch to feed the sentence to the model.
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                          {bucket_id: [(trace, [])]}, bucket_id)
                    
                    # Get output logits for the sentence.
                    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                                     target_weights, bucket_id, True)
                    
                    # This is a greedy decoder - outputs are just argmaxes of output_logits.
                    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
                    
                    # If there is an EOS symbol in outputs, cut them at that point.
                    if data_utils.EOS_ID in outputs:
                        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                    
                    # Print out decoded DNA sequence.
                    decoded_file.write("".join([data_utils.decode_base(output) for output in outputs]))
                    #decoded_file.write(" ".join([tf.compat.as_str(rev_target_vocab[output]) for output in outputs]))
                    decoded_file.write("\n")
                    trace = map(int, tracefile.readline().strip().split(" "))

