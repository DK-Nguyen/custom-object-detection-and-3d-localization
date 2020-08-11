"""
Train  neural networks the the synthesized dataset
"""
import os
from pathlib import Path
from omegaconf import DictConfig
import logging

from tools import register_custom_coco_dataset, visualizing_coco_dataset

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
    val_dataset_dicts, val_dataset_metadata = register_custom_coco_dataset(cfg=cfg,
                                                                           process='val')
    visualizing_coco_dataset(dataset_dicts=val_dataset_dicts,
                             dataset_metadata=val_dataset_metadata,
                             num_ims=cfg.validation.show_images)
    log.info('--- Validation Done ---')


