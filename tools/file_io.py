#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml
import os
from pathlib import Path
from typing import Dict


__author__ = 'Khoa Nguyen -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['load_yaml_file']


class YAMLLoader(yaml.SafeLoader):
    """
    Custom YAML loader for adding the functionality\
    of including one YAML file inside another.

    Code after: https://stackoverflow.com/a/9577670
    """

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]
        super(YAMLLoader, self).__init__(stream)

    def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(filename, 'r') as f:
            return yaml.load(f, YAMLLoader)


YAMLLoader.add_constructor('!include', YAMLLoader.include)


def load_yaml_file(file_path: Path) \
        -> Dict:
    """
    Reads and returns the contents of a YAML file.
    Code after: https://github.com/audio-captioning/dcase-2020-baseline/blob/master/tools/file_io.py

    :param file_path: Path to the YAML file.
    :type file_path: pathlib.Path
    :return: Contents of the YAML file.
    :rtype: dict
    """
    with file_path.open('r') as f:
        return yaml.load(f, Loader=YAMLLoader)