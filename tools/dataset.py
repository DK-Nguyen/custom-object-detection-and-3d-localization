"""
Process: preparing the dataset for the neural networks
"""

from typing import List, Dict, Tuple
import logging
from pathlib import Path
from types import SimpleNamespace

from omegaconf import DictConfig
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.catalog import Metadata

from .hard_negative import HardNegativeBackgroundPreparation
from .image_composition import ImageComposition
from .coco_json_utils import CocoJsonCreator

__all__ = ['dataset_creation', 'register_custom_coco_dataset']

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
        hnbg: HardNegativeBackgroundPreparation = HardNegativeBackgroundPreparation(
            input_dir=hard_negative_input_dir,
            output_dir=hard_negative_output_dir,
            output_width=cfg.hard_negative_background_preparation.output_width,
            output_height=cfg.hard_negative_background_preparation.output_height,
            output_type=cfg.hard_negative_background_preparation.output_type
        )
        hnbg.compose_images(position='middle_bottom')
        log.info('Done preparing hard negative background images')

    if cfg.training_images_preparation.option:
        log.info(f'Preparing training images for {cfg.name} dataset')
        image_comp_training: ImageComposition = ImageComposition(cfg.training_images_preparation)
        image_comp_training.main()
        coco_json_creator_training: CocoJsonCreator = CocoJsonCreator(cfg.training_images_preparation)
        coco_json_creator_training.main()

    if cfg.validation_images_preparation.option:
        log.info(f'Preparing validation images for {cfg.name} dataset')
        image_comp_val: ImageComposition = ImageComposition(cfg.validation_images_preparation)
        image_comp_val.main()
        coco_json_creator_val: CocoJsonCreator = CocoJsonCreator(cfg.validation_images_preparation)
        coco_json_creator_val.main()

    log.info('--- Dataset creation done ---')


def register_custom_coco_dataset(cfg: DictConfig,
                                 process: str = 'train') \
        -> Tuple[List[Dict], Metadata]:
    """
    Registering the custom dataset in COCO format to detectron2.

    :param cfg: the configuration dictionary of dataset_model.
    :type cfg: omegaconf.dictconfig.DictConfig.
    :param process: value should be 'train' or 'val'
    :type process: str
    :return information about images and instances in
             COCO format, together with its metadata.
    :rtype dataset_dicts: List[Dict].
           dataset_metadata: detectron2.data.catalog.Metadata.
    """
    dataset_dicts: List[Dict] = [{}]
    dataset_metadata: Metadata = Metadata()
    if process == 'val':
        val_dataset: str = cfg.name+"_val"
        log.info(f'Registering COCO Format datasets for {val_dataset}')
        val_images_dir: Path = PROJECT_PATH/cfg.validation.val_dataset_dir/'images'
        val_coco_instances_json: str = str(PROJECT_PATH/cfg.validation.val_dataset_dir/'coco_instances.json')
        register_coco_instances(name=cfg.val_dataset,
                                metadata={},
                                json_file=val_coco_instances_json,
                                image_root=val_images_dir)
        dataset_dicts = DatasetCatalog.get(val_dataset)
        dataset_metadata = MetadataCatalog.get(val_dataset)
    elif process == 'train' or process == 'test':
        train_dataset: str = cfg.name+"_train"
        train_images_dir: Path = PROJECT_PATH/cfg.train.train_dataset_dir/'images'
        train_coco_instances_json: str = str(PROJECT_PATH/cfg.train.train_dataset_dir/'coco_instances.json')
        register_coco_instances(name=train_dataset,
                                metadata={},
                                json_file=train_coco_instances_json,
                                image_root=train_images_dir)
        if process == 'train':
            log.info(f'Registering COCO Format datasets for {train_dataset}')
            dataset_dicts = DatasetCatalog.get(train_dataset)
            dataset_metadata = MetadataCatalog.get(train_dataset)
        elif process == 'test':
            log.info(f'Getting metadata for testing on {cfg.name}')
            dataset_metadata = MetadataCatalog.get(train_dataset)

    return dataset_dicts, dataset_metadata

