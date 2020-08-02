"""
Train  neural networks the the synthesized dataset
"""

from pathlib import Path
from omegaconf import DictConfig
import logging
import os

from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.config.config import CfgNode

from .test import test
from tools import register_custom_coco_dataset, visualizing_coco_dataset

__all__ = ['train']

PROJECT_PATH = Path(__file__).parents[1]  # get directory 2 levels up
log = logging.getLogger(__name__)  # A logger for this file


def _get_model_configs(cfg: DictConfig) -> CfgNode:
    """
    Get the configurations for a model specified in a .yaml file

    :param cfg: the dataset_model configuration
    :type cfg: omegaconf.dictconfig.DictConfig.
    :return trainer: A detectron's trainer with default training logic.
    :rtype trainer:
    """
    log.info(cfg.model.config_url)

    # Setting the hyper-parameters for the model
    model_cfg: CfgNode = get_cfg()
    model_cfg.merge_from_file(model_zoo.get_config_file(cfg.model.config_url))
    model_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg.model.config_url)
    model_cfg.DATASETS.TRAIN = (cfg.name, )
    model_cfg.DATASETS.TEST = ()  # no metrics implemented yet for this dataset
    model_cfg.DATALOADER.NUM_WORKERS = cfg.model.num_workers
    model_cfg.SOLVER.IMS_PER_BATCH = cfg.model.ims_per_batch
    model_cfg.SOLVER.BASE_LR = cfg.model.lr
    model_cfg.SOLVER.MAX_ITER = cfg.model.max_iter
    model_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = cfg.model.batch_size_per_im
    model_cfg.MODEL.ROI_HEADS.NUM_CLASSES = cfg.model.num_classes
    model_cfg.OUTPUT_DIR = os.getcwd()  # using hydra, this will be the dir outputs/date/time

    return model_cfg


def train(cfg: DictConfig) -> None:
    """
    Transfer learning using pretrained models from detectron2 model zoo.

    :param cfg: the configuration dictionary of dataset_model.
    :type cfg: omegaconf.dictconfig.DictConfig.
    :return: None
    """

    log.info('--- Start Training ---')
    dataset_dicts, dataset_metadata = register_custom_coco_dataset(cfg)
    visualizing_coco_dataset(dataset_dicts=dataset_dicts,
                             dataset_metadata=dataset_metadata,
                             num_ims=cfg.show_training_images)
    model_configs: CfgNode = _get_model_configs(cfg)
    trainer: DefaultTrainer = DefaultTrainer(model_configs)
    trainer.train()
    log.info('--- Training Done---')
