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

### Ubuntu/Linux 64-bit, GPU enabled. Requires CUDA toolkit 7.5 and CuDNN v4.  For
# other versions, see "Install from sources" below.
```bash
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0rc0-cp27-none-linux_x86_64.whl
```

### Mac OS X, CPU only:
```bash
$ sudo easy_install --upgrade six
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.8.0rc0-py2-none-any.whl
```

## Installation

From the root of the repo:

```bash
make
```

# Training and Usage

## Training

```bash
tracer-train --input=[path to input traces] --output=[path to model file] 
```

## Base calling

```bash
tracer-call --model=[path to model file] --input=[path to input traces] --output=[path to which to write output]
```

## Evaluation

```bash
tracer-eval --inputCalls=[path to input traces] --inputKey=[path to input traces] --output=[path to which to write output]
```

# Conclusion

With respect to the original goal of improving the basecall error rate beyond the current state of the art ([remind of value]), this experiment was [a success / so far not a success].

## License

tracer is released under the MIT License. See [LICENSE](https://github.com/cb01/tracer/blob/master/LICENSE).



