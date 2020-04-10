#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

__all__ = ['argument_parser']


def argument_parser() -> ArgumentParser:
    """
    Creates and returns the ArgumentParser

    :return: The argument parser.
    :rtype: argparse.ArgumentParser
    """
    arg_parser = ArgumentParser()
    main_arguments = [
        # ----------------------------------------------------------------------------------
        [
            ['-d', '--settings_dir'],
            {
                'type': str,
                'default': 'settings',
                'help': 'The directory that contains the setting files, default: "settings".',
            }
        ],
        # ----------------------------------------------------------------------------------
        [
            ['-c', '--main_config'],
            {
                'type': str,
                'default': 'main_settings',
                'help': 'The main setting file name (without extension).',
            }
        ],
        # ----------------------------------------------------------------------------------
        [
            ['-e', '--setting_ext'],
            {
                'type': str,
                'default': 'yaml',
                'help': 'The extension of the setting files, default: "yaml".',
            }
        ],
        # ----------------------------------------------------------------------------------
        [
            ['-v', '--verbose'],
            {
                'default': True,
                'action': 'store_true',
                'help': 'Increase output verbosity, default: True.',
            }
        ],
    ]

    [arg_parser.add_argument(*i[0], **i[1]) for i in main_arguments]

    return arg_parser


if __name__ == '__main__':
    args = argument_parser().parse_args()
    settings_dir = args.settings_dir
    main_config = args.main_config
    setting_ext = args.setting_ext
    verbose = args.verbose

# EOF
