import os
import hydra
from omegaconf import DictConfig
import logging

from tools import dataset_creation
from processes import train, test, validation

__all__ = ['main']

log: logging.Logger = logging.getLogger(__name__)  # A logger for this file


# TODO: solve the problem of image size in image_composition, try to put bigger trees in the dataset
@hydra.main(config_path="configs/config.yaml")
def main(cfg: DictConfig) -> None:
    log.info(f'Configurations:\n{cfg.pretty()}')
    log.info(f'Output directory: {os.getcwd()}')

    if cfg.workflow.dataset_creation:
        dataset_creation(cfg.dataset)
    if cfg.workflow.dnn_method:
        if cfg.dataset_model.train.option:
            train(cfg.dataset_model)
        if cfg.dataset_model.validation.option:
            validation(cfg.dataset_model)
        if cfg.dataset_model.test.option:
            test(cfg.dataset_model)
    if cfg.workflow.reconstruct_3d:
        pass

    log.info(f'--- Exit program ---')


if __name__ == '__main__':
    main()
