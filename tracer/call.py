#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click

@click.command()
@click.option('--input', required=True)
@click.option('--output', required=True)
@click.option('--model', required=True)
def cli(input, output, model):
    """Make basecalls given trace input and a previously trained model."""
    pass
