"""
Running inference with the trained model.
"""
import logging
from omegaconf import DictConfig

__all__ = ['test']

log = logging.getLogger(__name__)  # A logger for this file


def test(cfg: DictConfig):
    """

    :param cfg:
    :return:
    """
    log.info('Testing')
