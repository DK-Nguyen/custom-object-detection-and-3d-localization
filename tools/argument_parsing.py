#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

__author__ = 'Khoa Nguyen -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['get_argument_parser']


def get_argument_parser():
    """
    Creates and returns the ArgumentParser for this project.
    Code after: https://github.com/audio-captioning/dcase-2020-baseline/blob/master/tools/argument_parsing.py

    :return: The argument parser.
    :rtype: argparse.ArgumentParser
    """
    arg_parser = ArgumentParser()
    the_args = [
        # ---------------------------------
        [['--config-file', '-c'],
         {'type': str,
          'default': 'main_settings',
          'help': 'The settings file (without extension).'}],
        # ---------------------------------
        [['--file-dir', '-d'],
         {'type': str,
          'default': 'settings',
          'help': 'Directory that holds the settings file (default: `settings`).'}],
        # ---------------------------------
        [['--file-ext', '-e'],
         {'type': str,
          'default': 'yaml',
          'help': 'Extension of the settings file (default: `yaml`).'}],
        # ---------------------------------
        [['--verbose', '-v'],
         {'default': True,
          'action': 'store_true',
          'help': 'Be verbose flag (default True).'}]]

    [arg_parser.add_argument(*i[0], **i[1]) for i in the_args]

    return arg_parser


if __name__ == '__main__':
    args = get_argument_parser().parse_args()
    file_dir = args.file_dir
    config_file = args.config_file
    file_ext = args.file_ext
    verbose = args.verbose

# EOF
