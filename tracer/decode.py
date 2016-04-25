#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.python.platform import flags
from tensorflow.models.rnn.translate import data_utils

tf.app.flags.DEFINE_string("trace_path", "/tmp", "Path to a trace to decode.")
tf.app.flags.DEFINE_integer("in_vocab_size", 20000, "Source/input vocabulary size.")
tf.app.flags.DEFINE_integer("out_vocab_size", 8, "Target/output vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Directory containing data to be used to train the model.")

def cli():
    '''Make basecalls given trace input and a previously trained model.'''

  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    source_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.in" % FLAGS.source_vocab_size)
    target_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.out" % FLAGS.target_vocab_size)
    source_vocab, _ = data_utils.initialize_vocabulary(source_vocab_path)
    _, rev_target_vocab = data_utils.initialize_vocabulary(target_vocab_path)

    trace = read_trace(FLAGS.trace_path)

    while trace:
      # Get token-ids for the input sentence.
      token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), source_vocab)
      # Which bucket does it belong to?
      bucket_id = min([b for b in xrange(len(_buckets))
                       if _buckets[b][0] > len(token_ids)])
      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Print out French sentence corresponding to outputs.
      print(" ".join([tf.compat.as_str(rev_target_vocab[output]) for output in outputs])
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()
