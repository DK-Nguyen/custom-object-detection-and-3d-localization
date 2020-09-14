"""
Doing 3D reconstruction from depth map
"""
from typing import List, Tuple, Dict
import logging
from pathlib import Path
from omegaconf import DictConfig
from matplotlib import pyplot as plt
import numpy as np
from numpy import ndarray
import cv2
import tifffile as tiff

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
    Check the validity of the image directories to do 3d reconstruction.

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
    left_ims_paths: List = sorted(left_images_dir.glob('*.png'))
    right_ims_paths: List = sorted(right_images_dir.glob('*.png'))
    disparity_maps_paths: List = sorted(disp_maps_dir.glob('*.tif'))
    if not len(left_ims_paths) == len(right_ims_paths) == len(disparity_maps_paths):
        raise Exception(f'Left, right, and disparity directory have different '
                        f'number of images.')

    return left_ims_paths, right_ims_paths, disparity_maps_paths


def _do_reconstruction_matlab(stereo_params_path: str,
                              left_ims_paths: List,
                              right_ims_paths: List,
                              disparity_maps_paths: List) \
        -> None:
    """
    Do reconstruction from stereo images, stereo params, and depth maps
    using matlab engine.

    :param stereo_params_path:
    :param left_ims_paths:
    :param right_ims_paths:
    :param disparity_maps_paths:
    :return:
    """
    eng: MatlabEngine = start_matlab()
    stereo_params_dict: Dict = eng.load(stereo_params_path)
    stereo_params = stereo_params_dict['stereoParams']
    # experiment with 1 image
    left_im: mlarray = eng.imread(str(left_ims_paths[0]))
    left_im_np: ndarray = np.array(left_im._data).reshape(left_im.size[::-1]).T
    right_im: mlarray = eng.imread(str(right_ims_paths[0]))
    right_im_np: ndarray = np.array(right_im._data).reshape(right_im.size[::-1]).T
    disp_im: mlarray = eng.read(eng.Tiff(str(disparity_maps_paths[0]), 'r'))
    disp_im_np: ndarray = np.array(disp_im._data).reshape(disp_im.size[::-1]).T
    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(1, 3, 1)
    plt.imshow(left_im_np)
    fig.add_subplot(1, 3, 2)
    plt.imshow(right_im_np)
    fig.add_subplot(1, 3, 3)
    plt.imshow(disp_im_np)
    plt.show()
    # log.info(f'--- Debugging1 ---')
    # undistorted_im, rectified_im = eng.rectifyStereoImages(left_im, right_im, stereo_params)
    # xyz_points = eng.reconstructScene(disp_im, stereo_params)
    # log.info(f'--- Debugging2 ---')


# TODO: 3D reconstruction (using disparity map, stereo params and stereo images)
def _do_reconstruction_opencv(left_ims_paths: List,
                              right_ims_paths: List,
                              disparity_maps_paths: List,
                              stereo_params_path: str):
    left_im = cv2.imread(str(left_ims_paths[0]))
    right_im = cv2.imread(str(right_ims_paths[0]))
    disparity_map = tiff.imread(str(disparity_maps_paths[0]))

    # cv2.imshow('left im', left_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow('right im', right_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow('disparity map', disparity_map)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # reproject without stereo params provided
    if stereo_params_path is None:
        h, w = left_im.shape[:2]
        focal_length = 0.8 * w
        # Perspective transformation matrix
        Q = np.float32([[1, 0, 0, -w/2.0],
                        [0,-1, 0,  h/2.0],
                        [0, 0, 0, -focal_length],
                        [0, 0, 1, 0]])
        points_3d = cv2.reprojectImageTo3D(disparity_map, Q)
        points_3d_channel0 = points_3d[:, :, 0]
        points_3d_channel1 = points_3d[:, :, 1]
        points_3d_channel2 = points_3d[:, :, 2]

        cv2.imshow('3d points', points_3d_channel0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('3d points', points_3d_channel1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('3d points', points_3d_channel2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def reconstruct_3d(cfg: DictConfig):
    """

    :return:
    """
    log.info(f'--- Start 3d reconstruction ---')
    # log.info(f'Reconstruct configurations:\n{cfg.pretty()}')
    left_ims_paths, right_ims_paths, disparity_maps_paths = \
        _check_validity_im_dirs(left_ims_dir=cfg.left_ims_dir,
                                right_ims_dir=cfg.right_ims_dir,
                                disparity_maps_dir=cfg.disparity_maps_dir)

    stereo_params_path = str(PROJECT_PATH/cfg.stereo_params)

    # _do_reconstruction_matlab(stereo_params_path=stereo_params_path,
    #                           left_ims_paths=left_ims_paths,
    #                           right_ims_paths=right_ims_paths,
    #                           disparity_maps_paths=disparity_maps_paths)

    _do_reconstruction_opencv(stereo_params_path=stereo_params_path,
                              left_ims_paths=left_ims_paths,
                              right_ims_paths=right_ims_paths,
                              disparity_maps_paths=disparity_maps_paths)

    log.info(f'--- Done 3d reconstruction ---')
