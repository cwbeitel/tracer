#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click
import numpy as np
import random
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import os
import math

'''
tracer-sim --output=$TRACERDIR/data/test --tag=train --num_reads=1000
tracer-sim --output=$TRACERDIR/data/test --tag=test --num_reads=100

tracer-sim --output=$TRACERDIR/data --tag=train --num_reads=1000 --read_length_mean=100 --read_length_max_var=10

'''

@click.command()
@click.option('--output', required=True)
@click.option('--tag', default="train")
@click.option('--read_length_mean', default=20000)
@click.option('--read_length_max_var', default=5000)
@click.option('--num_reads', default=10)
@click.option('--steps_per_base', default=10)
@click.option('--noise_max', default=25)
@click.option('--poly_pause_prob', default=0.9)
@click.option('--poly_stall_prob', default=0.2)
@click.option('--poly_max_pause', default=60)
@click.option('--poly_max_stall', default=30)
@click.option('--mean_intensity_adenine', default=200)
#@click.option('--mean_intensity_adenine_2', default=200)
#@click.option('--mean_intensity_adenine_3', default=200)
@click.option('--mean_intensity_thiamine', default=150)
#@click.option('--mean_intensity_thiamine_2', default=150)
#@click.option('--mean_intensity_thiamine_3', default=150)
@click.option('--mean_intensity_guanine', default=110)
#@click.option('--mean_intensity_guanine_2', default=110)
#@click.option('--mean_intensity_guanine_3', default=110)
@click.option('--mean_intensity_cytosine', default=90)
#@click.option('--mean_intensity_cytosine_2', default=90)
#@click.option('--mean_intensity_cytosine_3', default=90)
@click.option('--base_noise_fraction', default=0.6) # The percentage variation to apply to the intensity of a signal from a base incorporation for noise.
@click.option('--make_figs', default=False)
@click.option('--vocab_size', default=80000)
def cli(output, tag, read_length_mean, read_length_max_var, num_reads, 
        steps_per_base, noise_max, poly_pause_prob, poly_stall_prob, 
        poly_max_pause, poly_max_stall, mean_intensity_adenine, 
        mean_intensity_thiamine, mean_intensity_guanine, mean_intensity_cytosine, 
        base_noise_fraction, make_figs, vocab_size):
    """Simulate instrument traces, for development purposes."""
    random.seed(1234)
    noise_max = 0.1

    # So "stalling" vs "pausing" is probably not the right language and this simulator is
    # not going to be at all kinetically accurate. But it will provide well understood input
    # that can be used for development.

    poly_pause_prob = 0.1 # A polymerase pausing after a base incorporation
    poly_stall_prob = 0.1 # A polymerase stalling during base incorporation
    poly_max_pause = 10 # Max duration polymerase can pause
    poly_max_stall = 10 # Max duration polymerase can stall
    mean_intensities = [mean_intensity_adenine, mean_intensity_thiamine, mean_intensity_guanine, mean_intensity_cytosine]
    bases = ["A", "T", "G", "C"]

    # Need to match elsewhere
    #PAD_ID = 0
    #GO_ID = 1
    #EOS_ID = 2
    #UNK_ID = 3
    go_id = 1
    #eos_id = 2
    # Need to shift everything by 4

    if not os.path.exists(output):
        os.mkdir(output)

    raw_trace_fname = os.path.join(output, tag + '.raw.csv')
    raw_seq_fname = os.path.join(output, tag + '.seq.txt')
    encoded_trace_fname = os.path.join(output, tag + '.ids' + str(vocab_size) + '.in')
    encoded_read_fname = os.path.join(output, tag + '.ids' + str(8) + '.out')

    with open(encoded_read_fname, "w") as f_read_encoded:
        with open(encoded_trace_fname, "w") as f_trace_encoded:
            with open(raw_seq_fname, "w") as f_read:
                for i in range(0, num_reads):
                    f_read_encoded.write(str(go_id) + " ")
                    f_trace_encoded.write(str(go_id) + " ")

    #for i in range(0, num_reads):
        #raw_trace_fname = os.path.join(output, tag + '-' + str(i) + '.raw.csv')
        #raw_seq_fname = os.path.join(output, tag + '-' + str(i) + '.seq.txt')
        #encoded_trace_fname = os.path.join(output, tag + '-' + str(i) + '.ids' + str(vocab_size) + '.out')
        #encoded_read_fname = os.path.join(output, tag + '-' + str(i) + '.ids' + str(8) + '.in')
        #with open(encoded_read_fname, "w") as f_read_encoded:
            #f_read_encoded.write("<GO> ")
            #with open(encoded_trace_fname, "w") as f_trace_encoded:
                #f_trace_encoded.write("<GO> ")
                #with open(raw_seq_fname, "w") as f_read:

                    with open(raw_trace_fname, "w") as f:
                        read_length = read_length_mean + random.randint(-1*read_length_max_var, read_length_max_var)
                        for j in range(0, read_length):
                            key = random.randint(0,3)
                            f_read_encoded.write(str(key + 4) + " ")
                            f_read.write(bases[key])
                            mod_val = mean_intensities[key]
                            stall_duration = 0
                            if random.random() < poly_stall_prob:
                                stall_duration = random.randint(0, poly_max_stall)
                            steps = random.randint(1, steps_per_base)
                            for k in range(0, steps + stall_duration):
                                out = gen_noisy_out(noise_max)
                                out[key] += mod_val*(1-base_noise_fraction) + random.random()*mod_val*base_noise_fraction
                                out[key + 4] = 1
                                dump(f, out)
                                encoded = encode(out[:4], vocab_size)
                                f_trace_encoded.write(str(encoded) + " ")
                            if random.random() < poly_pause_prob: # Pause the polymerase after incorporation
                                pause_duration = random.randint(0, poly_max_pause)
                                mod_val = 0
                                for i in range(0, pause_duration):
                                    out = gen_noisy_out(noise_max)
                                    out[key + 4] = 1
                                    dump(f, out)
                                    encoded = encode(out[:4], vocab_size)
                                    f_trace_encoded.write(str(encoded) + " ")
                        f_read.write("\n")
                    #f_trace_encoded.write(str(eos_id) + "\n")
                    #f_read_encoded.write(str(eos_id) + "\n")
                    f_trace_encoded.write("\n")
                    f_read_encoded.write("\n")

        if make_figs:
            plot_read(fname, fname+'.png')

def gen_noisy_out(noise_max):
    out = [random.uniform(0, noise_max),random.uniform(0, noise_max),random.uniform(0, noise_max),random.uniform(0, noise_max),0,0,0,0]
    return out

def dump(fh, out):
    fh.write(str(out[0]) + ',' + str(out[1]) + ',' + str(out[2]) + ',' +str(out[3]) + ',' +str(out[4]) + ',' +str(out[5]) + ',' + str(out[6]) + ',' + str(out[7]) + '\n')

def plot_read(inpath, outpath):
    max_to_plot = 400
    linecount = 0
    a = np.zeros(max_to_plot)
    t = np.zeros(max_to_plot)
    g = np.zeros(max_to_plot)
    c = np.zeros(max_to_plot)

    with open(inpath, 'r') as f:
        for line in f:
            arr = line.split(',')
            a[linecount] = arr[0]
            t[linecount] = arr[1]
            g[linecount] = arr[2]
            c[linecount] = arr[3]
            linecount += 1
            if linecount >= max_to_plot:
                break 

    plt.figure(figsize=(20, 2), dpi=100)
    plt.ylim([0, 400])
    plt.plot(a, 'r', t, 'b', g, 'g', c, 'y')
    plt.savefig(outpath)

def encode(intensities, vocab_size=80000, clip_intensity=400):
    '''Given four intensity values, return an integer encoding, e, of the maximum
    intensity value (max of a_inten.., t_inten.., ...), i, scaled by the ratio of the
    maximum possible intensity, i_max, and the number of bins per fluorophore, b. We
    then perform a shift equal to a multiple of the number of bins per fluorophore
    depending on which of the four were the largest.

                        e = ( i / (i_max/b) ) + (j-1)*b
    '''

    #intensities = [a_intensity, t_intensity, c_intensity, g_intensity]
    #global_max_intensity = 400
    #num_bins = 20000
    vocab_size_per_fluorophore = math.floor((vocab_size - 4)/4)
    max_ind = random.randint(0,3)
    max_intensity = 0
    for i, v in enumerate(intensities):
        #print(v, max_intensity)
        if v > max_intensity:
            max_ind = i
            if v > clip_intensity:
                max_intensity = clip_intensity
            else:
                max_intensity = v

    encoded = math.floor(max_intensity / (clip_intensity / vocab_size_per_fluorophore)) + max_ind*vocab_size_per_fluorophore + 4
     
    return int(encoded)


