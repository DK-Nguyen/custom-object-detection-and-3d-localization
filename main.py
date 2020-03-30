#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tools.argument_parsing import get_argument_parser
from tools.file_io import load_yaml_file

__author__ = 'Khoa Nguyen -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['main']


def main():
    args = get_argument_parser().parse_args()
    file_dir = args.file_dir
    print(file_dir)
    config_file = args.config_file
    print(config_file)
    file_ext = args.file_ext
    print(file_ext)
    verbose = args.verbose
    print(verbose)

    print('everything ok')


if __name__ == '__main__':
    main()

