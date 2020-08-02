"""
Process: preparing the dataset for the neural networks
"""

from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import random
import cv2

from omegaconf import DictConfig
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.catalog import Metadata
from detectron2.utils.visualizer import Visualizer

from .hard_negative import HardNegativeBackgroundPreparation
from .image_composition import ImageComposition
from .coco_json_utils import CocoJsonCreator

__all__ = ['dataset_creation',
           'register_custom_coco_dataset',
           'visualizing_coco_dataset']

PROJECT_PATH = Path(__file__).parents[1]  # get directory 2 levels up
log = logging.getLogger(__name__)  # A logger for this file


def dataset_creation(cfg: DictConfig) -> None:
    """
    Creating the dataset using configurations in cfg.dataset.

    :param cfg: the configuration dictionary.
    :type cfg: omegaconf.dictconfig.DictConfig
    :return: None
    """
    log.info('--- Dataset creation ---')
    hard_negative_input_dir: Path = PROJECT_PATH / cfg.hard_negative_background_preparation.input_dir
    hard_negative_output_dir: Path = PROJECT_PATH / cfg.hard_negative_background_preparation.output_dir

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

    if cfg.validation_images_preparation.option:
        log.info(f'Preparing validation images for {cfg.name} dataset')
        image_comp_val = ImageComposition(cfg.validation_images_preparation)
        image_comp_val.main()
        coco_json_creator_val = CocoJsonCreator(cfg.validation_images_preparation)
        coco_json_creator_val.main()

    log.info('--- Dataset creation done ---')


def register_custom_coco_dataset(cfg: DictConfig) \
        -> Tuple[List[Dict], Metadata]:
    """
    Registering the custom dataset in COCO format to detectron2.

    :param cfg: the configuration dictionary of dataset_model.
    :type cfg: omegaconf.dictconfig.DictConfig.
    :return information about images and instances in
             COCO format, together with its metadata.
    :rtype dataset_dicts: List[Dict].
           dataset_metadata: detectron2.data.catalog.Metadata.
    """
    log.info(f'Registering COCO Format dataset for {cfg.name}')
    register_coco_instances(name=cfg.name,
                            metadata={},
                            json_file=PROJECT_PATH/Path(cfg.coco_instances),
                            image_root=PROJECT_PATH/Path(cfg.coco_images))
    dataset_dicts = DatasetCatalog.get(cfg.name)
    dataset_metadata = MetadataCatalog.get(cfg.name)

    return dataset_dicts, dataset_metadata


# noinspection PyUnresolvedReferences
def visualizing_coco_dataset(dataset_dicts: List[Dict],
                             dataset_metadata: Metadata,
                             num_ims: Optional[int] = 3) \
        -> None:
    """
    Visualizing a dataset in COCO format.

    :param dataset_dicts: information about images and instances.
    :type dataset_dicts: List[Dict].
    :param dataset_metadata: contains additional information of the dataset_dicts.
    :type dataset_metadata: detectron2.data.catalog.Metadata.
    :param num_ims: number of images to visualize
    :type num_ims: int (optional, default values: 3)
    :return: None
    """
    for d in random.sample(dataset_dicts, num_ims):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1],
                                metadata=dataset_metadata,
                                scale=1)
        vis = visualizer.draw_dataset_dict(d)
        v = vis.get_image()[:, :, ::-1]
        cv2.imshow('', v)
        cv2.waitKey()
    cv2.destroyAllWindows()
