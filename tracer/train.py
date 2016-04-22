#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click

@click.command()
@click.option('--input', required=True)
@click.option('--output', required=True)
def cli(input, output):
    """Takes pairs of raw PacBio instrument traces together with their DNA sequence and trains a deep LSTM model to be used for making base calls from trace data."""
    pass

