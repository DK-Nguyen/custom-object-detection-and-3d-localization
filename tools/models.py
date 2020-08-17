import os
import logging
from pathlib import Path
from omegaconf import DictConfig

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.config.config import CfgNode

__all__ = ['get_model_configs']

PROJECT_PATH = Path(__file__).parents[1]  # get directory 2 levels up
log = logging.getLogger(__name__)  # A logger for this file


def get_model_configs(cfg: DictConfig,
                      process: str = 'train') -> CfgNode:
    """
    Get the configurations for a model specified in a .yaml file

    :param cfg: the dataset_model configuration.
    :type cfg: omegaconf.dictconfig.DictConfig.
    :param process:
    :return trainer: A detectron's trainer with default training logic.
    :rtype trainer:
    """
    log.info(f'Getting model configurations from {cfg.model.config_url} and {cfg.name}')
    # Setting the hyper-parameters for the model
    model_cfg: CfgNode = get_cfg()
    model_cfg.merge_from_file(model_zoo.get_config_file(cfg.model.config_url))
    model_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg.model.config_url)
    model_cfg.DATALOADER.NUM_WORKERS = cfg.model.num_workers
    model_cfg.SOLVER.IMS_PER_BATCH = cfg.model.ims_per_batch
    model_cfg.SOLVER.BASE_LR = cfg.model.lr
    model_cfg.SOLVER.MAX_ITER = cfg.model.max_iter
    model_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = cfg.model.batch_size_per_im
    model_cfg.MODEL.ROI_HEADS.NUM_CLASSES = cfg.model.num_classes
    model_cfg.OUTPUT_DIR = os.getcwd()  # using hydra, this will be PROJECT_PATH/outputs/date/time

    if process == 'train':
        model_cfg.DATASETS.TRAIN = (cfg.name + '_train', )
        model_cfg.DATASETS.TEST = (cfg.name + '_val', ) if cfg.validation.option else ()
    # elif process == 'val':
    #     model_cfg.DATASETS.TRAIN = (cfg.name + '_train', )
    #     model_cfg.DATASETS.TEST = (cfg.name + '_val', )
    #     model_cfg.MODEL.WEIGHTS = str(PROJECT_PATH / cfg.validation.pretrained_weight)
    elif process == 'test':
        if cfg.test.use_pretrained_weight:
            if cfg.test.pretrained_weight is None:
                raise Exception('cfg.test.use_pretrained_weight is Yes, '
                                'but cfg.test.pretrained_weight is not provided')
            log.info(f'Loading pretrained weight from {PROJECT_PATH / cfg.test.pretrained_weight}')
            model_cfg.MODEL.WEIGHTS = str(PROJECT_PATH / cfg.test.pretrained_weight)
        else:
            weight_path = os.path.join(model_cfg.OUTPUT_DIR, "model_final.pth")
            log.info(f'Loading pretrained weight from {PROJECT_PATH / weight_path}')
            model_cfg.MODEL.WEIGHTS = weight_path
        model_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.test.threshold
        model_cfg.MODEL.DEVICE = cfg.test.testing_device

    return model_cfg
