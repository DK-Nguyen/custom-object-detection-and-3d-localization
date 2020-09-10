"""
Doing 3D reconstruction from depth map
"""
from typing import List, Tuple, Dict
import logging
from pathlib import Path
from omegaconf import DictConfig
# from matplotlib import pyplot as plt
import numpy as np
from numpy import ndarray

from matlab import mlarray
from matlab.engine import start_matlab
from matlab.engine.matlabengine import MatlabEngine

from tools import constants

__all__ = ['reconstruct_3d']

PROJECT_PATH = constants['project_path']
log = logging.getLogger(__name__)  # A logger for this file


def _check_validity_im_dirs(left_ims_dir: str,
                            right_ims_dir: str,
                            disparity_maps_dir: str)\
        -> Tuple[List, List, List]:
    """
    Check the validity of the image directories to do 3d reconstruction
    :return:
    """
    # check if the directories exist
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

    return left_ims_list, right_ims_list, disparity_maps_list


def _read_and_transform(left_im: mlarray,
                        right_im: mlarray,
                        disparity_map: mlarray):
    pass


# def _do_reconstruction(engine: MatlabEngine,
#                        left_im: mlarray,
#                        right_im: mlarray,
#                        disparity_map: mlarray):
#     undistorted_im, rectified


def reconstruct_3d(cfg: DictConfig):
    """

    :return:
    """
    log.info(f'--- Start 3d reconstruction ---')
    # log.info(f'Reconstruct configurations:\n{cfg.pretty()}')
    left_ims_list, right_ims_list, disparity_maps_list = \
        _check_validity_im_dirs(left_ims_dir=cfg.left_ims_dir,
                                right_ims_dir=cfg.right_ims_dir,
                                disparity_maps_dir=cfg.disparity_maps_dir)
    eng: MatlabEngine = start_matlab()
    stereo_params_dict: Dict = eng.load(str(PROJECT_PATH/cfg.stereo_params))
    stereo_params = stereo_params_dict['stereoParams']

    # experiment with 1 image
    left_im: mlarray = eng.imread(str(left_ims_list[0]))
    left_im_np: ndarray = np.array(left_im._data).reshape(left_im.size[::-1]).T

    right_im: mlarray = eng.imread(str(right_ims_list[0]))
    right_im_np: ndarray = np.array(right_im._data).reshape(right_im.size[::-1]).T

    disp_im: mlarray = eng.read(eng.Tiff(str(disparity_maps_list[0]), 'r'))
    disp_im_np: ndarray = np.array(disp_im._data).reshape(disp_im.size[::-1]).T

    # fig = plt.figure(figsize=(8, 8))
    # fig.add_subplot(1, 3, 1)
    # plt.imshow(left_im_np)
    # fig.add_subplot(1, 3, 2)
    # plt.imshow(right_im_np)
    # fig.add_subplot(1, 3, 3)
    # plt.imshow(disp_im_np)
    # plt.show()

    log.info(f'--- Debugging1 ---')
    undistorted_im, rectified_im = eng.rectifyStereoImages(left_im, right_im, stereo_params)
    xyz_points = eng.reconstructScene(disp_im, stereo_params)
    log.info(f'--- Debugging2 ---')

    a = 1
    log.info(f'--- Done 3d reconstruction ---')
