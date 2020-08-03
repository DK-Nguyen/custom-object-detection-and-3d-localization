"""
Running inference with the trained model.
"""
from typing import Dict
import logging
from pathlib import Path
import cv2
import os
from numpy import ndarray
from omegaconf import DictConfig

from detectron2.config.config import CfgNode
from detectron2.engine import DefaultPredictor

from tools import get_model_configs, register_custom_coco_dataset, visualizing_predicted_samples

__all__ = ['test']

PROJECT_PATH = Path(__file__).parents[1]  # get directory 2 levels up
log = logging.getLogger(__name__)  # A logger for this file


def test(cfg: DictConfig):
    """

    :param cfg: the configuration dictionary of dataset_model.
    :return:
    """
    log.info(f'--- Start Testing ---')
    _, coco_tree_metadata = register_custom_coco_dataset(cfg=cfg,
                                                         process='test')
    output_dir: Path = Path(os.getcwd())  # using hydra, this will be the dir PROJECT_PATH/outputs/date/time

    if cfg.test.pretrained_weight is not None:
        log.info(f'Loading pretrained weight from {cfg.test.pretrained_weight}')
        model_cfg: CfgNode = get_model_configs(cfg=cfg,
                                               process='test')
        predictor: DefaultPredictor = DefaultPredictor(model_cfg)

        for test_im in Path(PROJECT_PATH/cfg.test.test_dataset_dir).iterdir():
            img: ndarray = cv2.imread(str(test_im))
            outputs: Dict = predictor(img)
            if cfg.test.saving_predicted_ims:
                dir_to_save: Path = output_dir/'predicted_ims'
                dir_to_save.mkdir(parents=False, exist_ok=True)
                path_to_save: str = str(output_dir/'predicted_ims'/test_im.name)
                visualizing_predicted_samples(img=img,
                                              metadata=coco_tree_metadata,
                                              predicted_samples=outputs,
                                              path_to_save=path_to_save)
    else:  # testing automatically after training
        pass
    log.info('--- Testing Done ---')
