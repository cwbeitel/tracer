#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click

@click.command()
@click.option('--inputCalls', required=True)
@click.option('--inputKey', required=True)
@click.option('--output', required=True)
def cli(inputCalls, inputKey, output):
    """Evaluate a set of basecalls made on an evaluation dataset."""
    pass
