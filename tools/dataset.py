"""
Process: preparing the dataset for training neural networks
"""

import logging
from omegaconf import DictConfig
from pathlib import Path

from .hard_negative import HardNegativeBackgroundPreparation
from .image_composition import ImageComposition
from .coco_json_utils import CocoJsonCreator

PROJECT_PATH = Path(__file__).parents[1]  # get directory 2 levels up
# A logger for this file
log = logging.getLogger(__name__)


def dataset_creation(cfg: DictConfig) -> None:
    """

    :param cfg: the configuration dictionary
    :type: logging.config.dictConfig
    :return:
    """
    log.info('Dataset creation')
    hard_negative_input_dir = PROJECT_PATH / cfg.hard_negative_background_preparation.input_dir
    hard_negative_output_dir = PROJECT_PATH / cfg.hard_negative_background_preparation.output_dir

    if cfg.hard_negative_background_preparation.option:
        log.info('Preparing hard negative background images')
        hnbg = HardNegativeBackgroundPreparation(input_dir=hard_negative_input_dir,
                                                 output_dir=hard_negative_output_dir,
                                                 output_width=cfg.hard_negative_background_preparation.output_width,
                                                 output_height=cfg.hard_negative_background_preparation.output_height,
                                                 output_type=cfg.hard_negative_background_preparation.output_type)
        hnbg.compose_images(position='middle_bottom')
        log.info('Done preparing hard negative background images')

    if cfg.training_images_preparation.option:
        log.info(f'Preparing training images for {cfg.name} dataset')
        image_comp_training = ImageComposition(cfg.training_images_preparation)
        image_comp_training.main()
        coco_json_creator_training = CocoJsonCreator(cfg.training_images_preparation)
        coco_json_creator_training.main()

