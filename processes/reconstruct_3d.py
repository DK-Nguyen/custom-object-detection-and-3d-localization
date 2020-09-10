"""
Doing 3D reconstruction from depth map
"""
from typing import List
from pathlib import Path
import logging
from omegaconf import DictConfig

from tools import constants

__all__ = ['reconstruct_3d']

PROJECT_PATH = constants['project_path']
log = logging.getLogger(__name__)  # A logger for this file


def _check_validity_im_dirs(left_ims_dir: str,
                            right_ims_dir: str,
                            disparity_maps_dir: str)\
        -> None:
    """
    Check the validity of the image directories to do 3d reconstruction
    :return:
    """
    left_images_dir: Path = PROJECT_PATH / left_ims_dir
    right_images_dir: Path = PROJECT_PATH / right_ims_dir
    disp_maps_dir: Path = PROJECT_PATH / disparity_maps_dir
    if not left_images_dir.exists():
        raise Exception(f'The directory that contains left images'
                        f'({left_images_dir}) does not exist')
    if not right_images_dir.exists():
        raise Exception(f'The directory that contains right images'
                        f'({right_images_dir}) does not exist')
    if not disp_maps_dir.exists():
        raise Exception(f'The directory that contains disparity maps'
                        f'({disp_maps_dir}) does not exist')

    # check if the 3 directories have the same number of images
    left_ims_list: List = sorted(left_images_dir.glob('*.png'))
    right_ims_list: List = sorted(right_images_dir.glob('*.png'))
    disparity_maps_list: List = sorted(disp_maps_dir.glob('*.tif'))
    if not len(left_ims_list) == len(right_ims_list) == len(disparity_maps_list):
        raise Exception(f'Left, right, and disparity directory have different '
                        f'number of images.')


def reconstruct_3d(cfg: DictConfig):
    """

    :return:
    """
    log.info(f'--- Start 3d reconstruction ---')
    # log.info(f'Reconstruct configurations:\n{cfg.pretty()}')
    _check_validity_im_dirs(left_ims_dir=cfg.left_ims_dir,
                            right_ims_dir=cfg.right_ims_dir,
                            disparity_maps_dir=cfg.disparity_maps_dir)

    log.info(f'--- Done 3d reconstruction ---')
