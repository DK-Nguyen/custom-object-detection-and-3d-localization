"""
Train  neural networks the the synthesized dataset
"""

from pathlib import Path
from omegaconf import DictConfig
import logging

__all__ = ['validation']

PROJECT_PATH = Path(__file__).parents[1]  # get directory 2 levels up
log = logging.getLogger(__name__)  # A logger for this file


def validation(cfg: DictConfig) -> None:
    """
    Transfer learning using pretrained models from detectron2 model zoo.

    :param cfg: the configuration dictionary of dataset_model.
    :type cfg: omegaconf.dictconfig.DictConfig.
    :return: None
    """

    log.info('--- Start Validation ---')

    log.info('--- Validation Done ---')


