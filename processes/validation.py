"""

"""
import os
from pathlib import Path
from omegaconf import DictConfig
import logging

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.config.config import CfgNode
from detectron2.engine import DefaultTrainer

from tools import register_custom_coco_dataset, visualizing_coco_dataset, \
    get_model_configs

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
    val_dataset_dicts, val_dataset_metadata = register_custom_coco_dataset(cfg=cfg, process='val')
    visualizing_coco_dataset(dataset_dicts=val_dataset_dicts,
                             dataset_metadata=val_dataset_metadata,
                             num_ims=cfg.validation.show_images)
    model_cfg: CfgNode = get_model_configs(cfg=cfg, process='val')
    evaluator = COCOEvaluator(dataset_name=cfg.name+'_val',
                              cfg=model_cfg,
                              distributed=False,
                              output_dir=os.getcwd())
    val_loader = build_detection_test_loader(cfg=model_cfg,
                                             dataset_name=cfg.name+'_val')
    trainer: DefaultTrainer = DefaultTrainer(model_cfg)
    trainer.resume_or_load(resume=True)
    inference_on_dataset(model=trainer.model,
                         data_loader=val_loader,
                         evaluator=evaluator)
    log.info('--- Validation Done ---')
