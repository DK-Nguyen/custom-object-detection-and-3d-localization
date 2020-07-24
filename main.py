"""Main file of the project"""
import hydra
from omegaconf import DictConfig
import logging

from tools import dataset_creation

__all__ = ['main']

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(config_path="configs/config.yaml")
def main(cfg: DictConfig) -> None:
    log.info(f'Configurations:\n {cfg.pretty()}')

    # dataset creation
    if cfg.workflow.dataset_creation:
        dataset_creation(cfg.dataset)
    # training
    if cfg.workflow.dnn_training:
        pass
    # testing
    if cfg.workflow.dnn_testing:
        pass
    # 3d reconstruction
    if cfg.workflow.reconstruct_3d:
        pass
    # making demo
    if cfg.workflow.demo:
        pass


if __name__ == '__main__':
    main()
