"""
Running inference with the trained model on a test (unseen) dataset.
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
    Running inference on an unseen dataset.
    :param cfg: the configuration dictionary of dataset_model.
    :return:
    """
    log.info(f'--- Start Testing ---')
    if cfg.train.option and cfg.test.use_pretrained_weight:
        raise Exception('cfg.train.option is Yes, but cfg.test.use_pretrained_weight is also Yes')
    elif not cfg.train.option and not cfg.test.use_pretrained_weight:
        raise Exception('cfg.train.option is No but cfg.test.use_pretrained_weight is also No')

    _, coco_tree_metadata = register_custom_coco_dataset(cfg=cfg,
                                                         process='test')
    output_dir: Path = Path(os.getcwd()) / 'predicted_ims'  # using hydra, this will be PROJECT_PATH/outputs/date/time
    output_dir.mkdir(parents=False, exist_ok=True)
    log.info(f'Predicted images are saved to {output_dir}')

    model_cfg: CfgNode = get_model_configs(cfg=cfg, process='test')
    predictor: DefaultPredictor = DefaultPredictor(model_cfg)
    log.info(f'Doing inference on {PROJECT_PATH/cfg.test.test_dataset_dir}...')

    for test_im in Path(PROJECT_PATH/cfg.test.test_dataset_dir).iterdir():
        img: ndarray = cv2.imread(str(test_im))
        outputs: Dict = predictor(img)
        path_to_save: str = str(output_dir/test_im.name) if cfg.test.saving_predicted_ims else None
        visualizing_predicted_samples(img=img,
                                      metadata=coco_tree_metadata,
                                      predicted_samples=outputs,
                                      path_to_save=path_to_save,
                                      show_image=cfg.test.show_predicted_ims)
    log.info('--- Testing Done ---')
