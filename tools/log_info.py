#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import stdout
from pathlib import Path

from loguru import logger

from typing import Dict

__all__ = ['init_loggers']


def init_loggers(verbose: bool,
                 settings: Dict) -> None:
    """ Initializing the logging process

    :param verbose:
    :type verbose: bool
    :param settings: settings to use
    :type settings: dict
    :return:
    """

    # remove a previously added handler
    logger.remove()

    logging_path = Path(settings['root_dirs']['outputs'],
                        settings['logging']['logger_dir'])

    logging_path.mkdir(parents=True, exist_ok=True)

    log_file_main = f'{settings["logging"]["caption_logger_file"]}'

    log_string = '{level} | [{time}] {name} -- {space}{message}'.format(
        level='{level}',
        time='{time:HH:mm:ss}',
        name='{name}',
        message='{message}',
        space=' ')

    logger.add(
        stdout,
        format=log_string,
        level='INFO'
    )


# if __name__ == '__main__':



# EOF



