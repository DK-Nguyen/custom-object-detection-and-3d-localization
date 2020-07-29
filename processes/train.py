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

from tools import register_custom_coco_dataset, visualizing_coco_dataset

__all__ = ['train']

PROJECT_PATH = Path(__file__).parents[1]  # get directory 2 levels up
log = logging.getLogger(__name__)  # A logger for this file


def _get_pretrained_model(cfg: DictConfig) -> None:
    """


    :param cfg:
    :return:
    """
    log.info(cfg.model.config_url)
    model_cfg: CfgNode = get_cfg()
    print(model_cfg)
    model_cfg.merge_from_file(
        model_zoo.get_config_file(cfg.model.config_url))
    print(model_cfg)


def train(cfg: DictConfig):
    """
    Transfer learning using pretrained models from detectron2 model zoo.

    :param cfg: the configuration dictionary of dataset_model.
    :type cfg: logging.config.dictConfig.
    :return:
    """
    dataset_dicts, dataset_metadata = register_custom_coco_dataset(cfg)
    visualizing_coco_dataset(dataset_dicts=dataset_dicts,
                             dataset_metadata=dataset_metadata,
                             num_ims=3)
    _get_pretrained_model(cfg)


# if __name__ == '__main__':
#     dataset_cfg = get_cfg()
#     dataset_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
#     dataset_cfg.DATASETS.TRAIN = ("balloon_train",)
#     dataset_cfg.DATASETS.TEST = ()
#     dataset_cfg.DATALOADER.NUM_WORKERS = 2
#     dataset_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
#     "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
#     dataset_cfg.SOLVER.IMS_PER_BATCH = 2
#     dataset_cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
#     dataset_cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
#     dataset_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
#     dataset_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
#
#     os.makedirs(dataset_cfg.OUTPUT_DIR, exist_ok=True)
#     trainer = DefaultTrainer(dataset_cfg)
#     trainer.resume_or_load(resume=False)
    # trainer.train(

