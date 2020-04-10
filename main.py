#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from tools.argument_parsing import argument_parser
from tools.file_io import load_yaml_file
from tools.log_info import init_loggers

__all__ = ['main']


def main():
    args = argument_parser().parse_args()
    settings_dir = args.settings_dir
    main_config = args.main_config
    setting_ext = args.setting_ext
    verbose = args.verbose

    settings = load_yaml_file(Path(settings_dir, f'{main_config}.{setting_ext}'))

    # init_loggers(verbose,
    #              settings=settings['dirs_and_files'])

    if settings['workflow']['dataset_creation']:
        print('datatset creating')

    print(settings)


if __name__ == '__main__':
    main()

    print('end of main')
# EOF

