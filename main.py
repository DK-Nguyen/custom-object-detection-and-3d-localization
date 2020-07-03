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
    if cfg.workflow.dataset_creation:
        dataset_creation(cfg.dataset)


if __name__ == '__main__':
    main()



