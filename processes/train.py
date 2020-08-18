"""
Train  neural networks the the synthesized dataset
"""
import os
from pathlib import Path
from omegaconf import DictConfig
import logging

from detectron2.engine import DefaultTrainer
from detectron2.config.config import CfgNode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from tools import register_custom_coco_dataset, visualizing_coco_dataset, \
                  get_model_configs

__all__ = ['train']

PROJECT_PATH = Path(__file__).parents[1]  # get directory 2 levels up
log = logging.getLogger(__name__)  # A logger for this file


def train(cfg: DictConfig) -> None:
    """
    Transfer learning using models from detectron2 model zoo.

    :param cfg: the configuration dictionary of dataset_model.
    :type cfg: omegaconf.dictconfig.DictConfig.
    :return: None
    """
    log.info(f'--- Start Training with {cfg.name} ---')
    train_dataset_dicts, train_dataset_metadata = register_custom_coco_dataset(cfg=cfg, process='train')
    visualizing_coco_dataset(dataset_dicts=train_dataset_dicts,
                             dataset_metadata=train_dataset_metadata,
                             num_ims=cfg.train.show_images)
    model_cfg: CfgNode = get_model_configs(cfg, process='train')
    trainer: DefaultTrainer = DefaultTrainer(model_cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    log.info('--- Training Done---')
