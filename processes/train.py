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

from tools import register_custom_coco_dataset, visualizing_coco_dataset

__all__ = ['train']

PROJECT_PATH = Path(__file__).parents[1]  # get directory 2 levels up
log = logging.getLogger(__name__)  # A logger for this file


def _get_pretrained_model():
    pass


def train(cfg: DictConfig):
    """

    :param cfg: the configuration dictionary of dataset_model.
    :type cfg: logging.config.dictConfig.
    :return:
    """
    dataset_dicts, dataset_metadata = register_custom_coco_dataset(cfg)
    visualizing_coco_dataset(dataset_dicts=dataset_dicts,
                             dataset_metadata=dataset_metadata,
                             num_ims=3)


# if __name__ == '__main__':
#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
#     cfg.DATASETS.TRAIN = ("balloon_train",)
#     cfg.DATASETS.TEST = ()
#     cfg.DATALOADER.NUM_WORKERS = 2
#     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
#     "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
#     cfg.SOLVER.IMS_PER_BATCH = 2
#     cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
#     cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
#     cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
#
#     os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#     trainer = DefaultTrainer(cfg)
#     trainer.resume_or_load(resume=False)
    # trainer.train(

