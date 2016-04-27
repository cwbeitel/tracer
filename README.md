# tracer

This code implements a deep LSTM neural network for basecalling from raw PacBio single-molecule real-time (SMRT) instrument "traces". In short, in the process of determining the sequence of the DNA in an input sample, the Pacific Biosciences sequencer emits [number] of parallel signals of this sequence, as a four-channel time series (one channel corresponding to each of A, T, C, and G) which must be "called" into a sequence string (e.g. "ATCTGAGTACCATGACATG..."). The single-pass error rate of the PacBio sequencer is currently around 13%. An improvement in the error rate of the platform would be of significant value to users of the platform enabling significant cost reductions and more powerful inquiry.

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/cb01/tracer)

# Installation

## System setup

On Mac OSX,

```bash
brew install homebrew/science/hdf5
```

On Linux,

```bash
sudo apt-get install libhdf5-dev
```

## Environment setup

```bash
virtualenv venv
source venv/bin/activate
```

## TensorFlow setup 
(from the TensorFlow documentation)

### Ubuntu/Linux 64-bit, CPU only:
```bash
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0rc0-cp27-none-linux_x86_64.whl
```

### Ubuntu/Linux 64-bit, GPU enabled. Requires CUDA toolkit 7.5 and CuDNN v4.  For other versions, see "Install from sources" below.
```bash
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0rc0-cp27-none-linux_x86_64.whl
```

### Mac OS X, CPU only:
```bash
$ sudo easy_install --upgrade six
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.8.0rc0-py2-none-any.whl
```

### Running on AWS

We found it was a challenge to configure tensorflow to leverage GPU's on AWS g2.4xlarge instances but included a [script](https://github.com/cb01/tracer/blob/master/ec2.sh) describing how we did it.

## Installation

From the root of the repo:

```bash
make
```

# Training and Usage

## Training

```bash
tracer-train --data_dir=$TRACERDIR/data --train_dir=$TRACERDIR/data/checkpoints --size=256 --num_layers=3 --in_vocab_size=20000
```

## Base calling

```bash
tracer-decode --model=[path to model file] --input=[path to input traces] --output=[path to which to write output]
```

## Evaluation

```bash
tracer-eval --inputCalls=[path to input traces] --inputKey=[path to input traces] --output=[path to which to write output]
```

## Example decodings

As development progresses, we hopefully will see the quality of decodings improve. Here are some of the current rather terrible decodings. Obviously there's a long way to go.

```bash
1 layer, 10 neurons per layer, 5min
decoded: ACAAAAA
correct: TCAGCCGAACGAAGTCGCGATGCAGCCCAGTGGGATGAAACGGTCGATCGGCTCTCTACGCTACTTGAGATTAAAAAGATTTGGTGTGAGGTTGCTCGGTTTAGGTCTAC
```

```bash
1 layer, 10 neurons per layer, 5min
decoded: TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
correct: AATCGGGGAGACCTGCGCTTGTCGGCGCTCGTACACGATTTTTCTTACGAGCATGTTATTCGACGCCAGACATGAAGATTTCGGGATCGCTCGAAGTCTATTCAAAGTGA
```

```bash
3 layers, 256 neurons per layer, ~2h
decoded: TTTTA
correct: TCAGCCGAACGAAGTCGCGATGCAGCCCAGTGGGATGAAACGGTCGATCGGCTCTCTACGCTACTTGAGATTAAAAAGATTTGGTGTGAGGTTGCTCGGTTTAGGTCTAC
```

# Conclusion

With respect to the original goal of improving the basecall error rate beyond the current state of the art single-pass error rate of 13%, this experiment is so far not a success.

## License

tracer is released under the Apache License 2.0. See [LICENSE](https://github.com/cb01/tracer/blob/master/LICENSE). The majority of the code are modifications of the seq2seq example from the Tensor Flow library, which is covered by their [LICENSE](https://github.com/cb01/tracer/blob/master/LICENSE.tflow). If you have any suggestions about how to more appropriately provide attribution on the individual source files, let me know. I'm unsure, for example, whether the original copyright notice should be retained on each file.
