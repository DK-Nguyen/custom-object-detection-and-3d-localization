import hydra
from omegaconf import DictConfig
import logging

from tools import dataset_creation
from processes import train, test, validation, reconstruct_3d

__all__ = ['main']

log: logging.Logger = logging.getLogger(__name__)  # A logger for this file


@hydra.main(config_path="configs/config.yaml")
def main(cfg: DictConfig) -> None:
    log.info(f'Configurations:\n{cfg.pretty()}')

    if cfg.workflow.dataset_creation:
        dataset_creation(cfg.dataset)

    if cfg.workflow.reconstruct_3d:
        reconstruct_3d(cfg.reconstruct_3d)

    if cfg.workflow.dnn_method:
        if cfg.dataset_model.train.option:
            train(cfg.dataset_model)
        if cfg.dataset_model.validation.option:
            validation(cfg.dataset_model)
        if cfg.dataset_model.test.option:
            test(cfg.dataset_model)
    log.info(f'--- Exit program ---')


if __name__ == '__main__':
    main()
