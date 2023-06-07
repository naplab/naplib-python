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

import os
from os.path import dirname
from os.path import join as join_dirs

import yaml
from textgrid import MLF

from . import logger
from .aligner import Aligner
from .archive import Archive
from .corpus import Corpus
from .utilities import ALIGNED, CONFIG, HMMDEFS, MACROS, SCORES, resolve_opts, splitname

DICTIONARY = "eng.dict"
MODEL = "eng.zip"


def run_aligner(aligner=False, configuration=False, dictionary=False, samplerate=False,
                epochs=False, read=False, train=False, align=False, write=False):
    
    args = locals() # get dictionary of args

    if 'dictionary' not in args:
        filedir_ = dirname(__file__)
        args['dictionary'] = [join_dirs(filedir_, DICTIONARY)]

    # input: pick one
    if train:
        if read:
            logger.error("Cannot train on persistent model.")
            raise RuntimeError('Cannot train on persistent model.')
        logger.info(f"Preparing corpus '{args['train']}'.")
        opts = resolve_opts(**args)
        corpus = Corpus(args['train'], opts)
        logger.info("Preparing aligner.")
        aligner = Aligner(opts)
        logger.info(f"Training aligner on corpus '{args['train']}'.")
        aligner.HTKbook_training_regime(corpus, opts["epochs"],
                                        flatstart=args['read'] is None)
    else:
        if not read:
            args['read'] = MODEL
        logger.info(f"Reading aligner from '{args['read']}'.")
        # warn about irrelevant flags
        if configuration:
            logger.warning("Ignoring config flag (-c/--configuration).")
            configuration = None
        if epochs:
            logger.warning("Ignoring epochs flag (-e/--epochs).")
        if samplerate:
            logger.warning("Ignoring samplerate flag (-s/--samplerate).")
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
        if (not args['train']) or (os.path.realpath(args['train']) != os.path.realpath(args['align'])):
            logger.info(f"Preparing corpus '{args['align']}'.")
            corpus = Corpus(args['align'], opts)
        logger.info(f"Aligning corpus '{args['align']}'.")
        aligned = os.path.join(args['align'], ALIGNED)
        scores = os.path.join(args['align'], SCORES)
        aligner.align_and_score(corpus, aligned, scores)
        logger.debug(f"Wrote MLF file to '{aligned}'.")
        logger.debug(f"Wrote likelihood scores to '{scores}'.")
        logger.info("Writing TextGrids.")
        size = MLF(aligned).write(args['align'])
        if not size:
            logger.error("No paths found!")
            raise RuntimeError('No paths found')
        logger.debug(f"Wrote {size} TextGrids.")
    elif args['write']:
        # create and populate archive
        _, basename, _ = splitname(args['write'])
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
        logger.info(f"Wrote aligner to '{archive_path}'.")
    # else unreachable

    logger.info("Success!")

