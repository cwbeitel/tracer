#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.python.platform import flags
from tensorflow.models.rnn.translate import data_utils
import tensorflow as tf
from tracer import train

tf.app.flags.DEFINE_string("trace_path", "/tmp", "Path to a trace to decode.")
tf.app.flags.DEFINE_string("output", "/tmp", "Path to file to which to write the decoded output.")

FLAGS = tf.app.flags.FLAGS

# easier than making a lot of modifications right now...
_buckets = [(80,120)]

def decode_cli():
    '''Make basecalls given trace input and a previously trained model.'''

    with tf.Session() as sess:

        print("opened session")

        # Create model and load parameters.
        model = train.create_model(sess, True, debug=True)
        model.batch_size = 1  # We decode one sentence at a time.

        # Currently this expects traces to be fed in in their encoded form

        print("initialized model")

        with open(FLAGS.output, "w") as decoded_file:
            with open(FLAGS.trace_path, "r") as tracefile:
                
                trace = tracefile.readline()

                while trace:

                    print("indicator")

                    # Which bucket does it belong to?
                    bucket_id = min([b for b in xrange(len(_buckets))
                                   if _buckets[b][0] > len(trace)])
                    
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
                    print outputs
                    decoded_file.write(" ".join([tf.compat.as_str(rev_target_vocab[output]) for output in outputs]))
                    decoded_file.write("> ", end="")
                    decoded_file.write("\n")
                    trace = tracefile.readline()

