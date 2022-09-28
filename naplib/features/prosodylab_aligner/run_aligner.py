# Copyright (c) 2011-2014 Kyle Gorman and Michael Wagner
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
Function-version of the command-line driver for the prosodylab aligner module
"""

import logging
import os
import sys
import yaml
from os.path import dirname
from os.path import join as join_dirs

from bisect import bisect
from shutil import copyfile
from textgrid import MLF

from .corpus import Corpus
from .aligner import Aligner
from .archive import Archive
from .utilities import splitname, resolve_opts, \
                       ALIGNED, CONFIG, HMMDEFS, MACROS, SCORES

from argparse import ArgumentParser

DICTIONARY = "eng.dict"
MODEL = "eng.zip"

LOGGING_FMT = "%(message)s"

def run_aligner(aligner=False, configuration=False, dictionary=False, samplerate=False,
                epochs=False, read=False, train=False, align=False,
                write=False, verbose=False, extra_verbose=False):
    
    args = locals() # get dictionary of args

    if 'dictionary' not in args:
        filedir_ = dirname(__file__)
        args['dictionary'] = [join_dirs(filedir_, DICTIONARY)]

    # set up logging
    loglevel = logging.WARNING
    if extra_verbose:
        loglevel = logging.DEBUG
    elif verbose:
        loglevel = logging.INFO
    logging.basicConfig(format=LOGGING_FMT, level=loglevel)

    # input: pick one
    if train:
        if read:
            logging.error("Cannot train on persistent model.")
            raise RuntimeError('Cannot train on persistent model.')
        logging.info("Preparing corpus '{}'.".format(args['train']))
        opts = resolve_opts(**args)
        corpus = Corpus(args['train'], opts)
        logging.info("Preparing aligner.")
        aligner = Aligner(opts)
        logging.info("Training aligner on corpus '{}'.".format(args['train']))
        aligner.HTKbook_training_regime(corpus, opts["epochs"],
                                        flatstart=(args['read'] is None))
    else:
        if not read:
            args['read'] = MODEL
        logging.info("Reading aligner from '{}'.".format(args['read']))
        # warn about irrelevant flags
        if configuration:
            logging.warning("Ignoring config flag (-c/--configuration).")
            configuration = None
        if epochs:
            logging.warning("Ignoring epochs flag (-e/--epochs).")
        if samplerate:
            logging.warning("Ignoring samplerate flag (-s/--samplerate).")
            args['samplerate'] = None
        # create archive from -r argument
        archive = Archive(read)
        # read configuration file therefrom, and resolve options with it
        args['configuration'] = os.path.join(archive.dirname, CONFIG)
        opts = resolve_opts(**args)
        # initialize aligner and set it to point to the archive data
        aligner = Aligner(opts)
        aligner.curdir = archive.dirname

    # output: pick one
    if args['align']:
        # check to make sure we're not aligning on the training data
        if (not args['train']) or (os.path.realpath(args['train']) !=
                                os.path.realpath(args['align'])):
            logging.info("Preparing corpus '{}'.".format(args['align']))
            corpus = Corpus(args['align'], opts)
        logging.info("Aligning corpus '{}'.".format(args['align']))
        aligned = os.path.join(args['align'], ALIGNED)
        scores = os.path.join(args['align'], SCORES)
        aligner.align_and_score(corpus, aligned, scores)
        logging.debug("Wrote MLF file to '{}'.".format(aligned))
        logging.debug("Wrote likelihood scores to '{}'.".format(scores))
        logging.info("Writing TextGrids.")
        size = MLF(aligned).write(args['align'])
        if not size:
            logging.error("No paths found!")
            raise RuntimeError('No paths found')
        logging.debug("Wrote {} TextGrids.".format(size))
    elif args['write']:
        # create and populate archive
        (_, basename, _) = splitname(args['write'])
        archive = Archive.empty(basename)
        archive.add(os.path.join(aligner.curdir, HMMDEFS))
        archive.add(os.path.join(aligner.curdir, MACROS))
        # whatever this is, it's not going to work once you move the data
        if "dictionary" in opts:
            del opts["dictionary"]
        with open(os.path.join(archive.dirname, CONFIG), "w") as sink:
            yaml.dump(opts, sink)
        (basename, _) = os.path.splitext(args['write'])
        archive_path = os.path.relpath(archive.dump(basename))
        logging.info("Wrote aligner to '{}'.".format(archive_path))
    # else unreachable

    logging.info("Success!")
