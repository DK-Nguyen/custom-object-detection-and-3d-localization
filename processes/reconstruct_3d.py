"""
Doing 3D reconstruction from depth map
"""
from typing import List, Tuple, Dict, Union
import logging
from pathlib import Path
from omegaconf import DictConfig
from matplotlib import pyplot as plt
import numpy as np
from numpy import ndarray
import cv2
import tifffile as tiff
import yaml

from matlab import mlarray
from matlab.engine import start_matlab
from matlab.engine.matlabengine import MatlabEngine

from tools import constants, visualizing_triplets

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
    Do reconstruction from stereo images, stereo params, and depth maps using matlab engine.
    (not working yet)

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


def _reconstruct_no_params(left_im: ndarray,
                           disparity_map: ndarray)\
        -> ndarray:
    """
    Do 3D reconstruction from an image and a corresponding disparity map without stereo parameters.

    :param left_im: the ndarray of the left image
    :type left_im: ndarray of shape (height, width, 3)
    :param disparity_map: the ndarray of the depth map
    :type disparity_map: ndarray of shape (height, width)
    :return points_3d: a 3-channel ndarray representing a 3D surface
    :rtype points_3d: ndarray of shape (height, width, 3)
    """
    h, w = left_im.shape[:2]
    focal_length = 0.8 * w
    # Perspective transformation matrix
    Q: ndarray = np.float32([[1, 0, 0, -w / 2.0],
                             [0, -1, 0, h / 2.0],
                             [0, 0, 0, -focal_length],
                             [0, 0, 1, 0]])
    points_3d: ndarray = cv2.reprojectImageTo3D(disparity_map, Q)
    return points_3d


def _trim_im(im: ndarray,
             desired_height: int,
             desired_width: int)\
        -> ndarray:
    """
    Trimming the image to the desired height and width, assuming that the image
    has bigger size than the desired size.

    :param im: the image in ndarray
    :type im: ndarray with size [height, width]
    :param desired_height:
    :param desired_width:
    :return:
    """
    trimmed_im: ndarray = im
    im_height: int = im.shape[0]
    im_width: int = im.shape[1]

    if im_height != desired_height:
        height_difference: int = im_height - desired_height
        trimmed_im = trimmed_im[height_difference//2: im_height-height_difference//2, :]
    if im_width != desired_width:
        width_difference: int = im_width - desired_width
        trimmed_im = trimmed_im[:, width_difference//2: im_width-width_difference//2]

    return trimmed_im


def _check_sizes(left_im: ndarray,
                 right_im: ndarray,
                 points_3d: ndarray):
    pass


def _reconstruct_stereo_params(stereo_params: Dict,
                               left_im: ndarray,
                               right_im: ndarray,
                               disparity_map: ndarray)\
        -> Tuple[ndarray, ndarray, ndarray]:
    """
    Trimming the left im, right im and disparity map if they are bigger than the size given in the stereo params,
    do 3D points reconstruction, then save all of these into output folder

    :param stereo_params:
    :param left_im:
    :param disparity_map:
    :return:
    """
    camera_matrix1: ndarray = np.array(stereo_params['cameraMatrix1'])
    camera_matrix2: ndarray = np.array(stereo_params['cameraMatrix2'])
    distortion_coeffs1: ndarray = np.array(stereo_params['distCoeffs1'])
    distortion_coeffs2: ndarray = np.array(stereo_params['distCoeffs2'])
    image_size: Tuple = tuple(stereo_params['image_size'])
    rotation_matrix: ndarray = np.array(stereo_params['R'])
    translation_vector: ndarray = np.array(stereo_params['T'])

    trimmed_disp_map: ndarray = _trim_im(im=disparity_map, desired_height=image_size[0], desired_width=image_size[1])
    trimmed_left_im: ndarray = _trim_im(im=left_im, desired_height=image_size[0], desired_width=image_size[1])
    trimmed_right_im: ndarray = _trim_im(im=right_im, desired_height=image_size[0], desired_width=image_size[1])
    # visualizing_triplets(left_im=trimmed_left_im, right_im=trimmed_right_im, disp_map=trimmed_disp_map)

    _, _, _, _, reprojection_matrix, _, _ = cv2.stereoRectify(cameraMatrix1=camera_matrix1,
                                                              distCoeffs1=distortion_coeffs1,
                                                              cameraMatrix2=camera_matrix2,
                                                              distCoeffs2=distortion_coeffs2,
                                                              imageSize=image_size,
                                                              R=rotation_matrix,
                                                              T=translation_vector)

    points_3d: ndarray = cv2.reprojectImageTo3D(disparity=trimmed_disp_map,
                                                Q=reprojection_matrix)

    # TODO: check the images (if they are in the same size and satisfy other conditions)

    # TODO: save the trimmed_left_im, trimmed_right_im, points_3d to disk (so we can run the
    #  neural network and visualize the results later)

    return trimmed_left_im, trimmed_right_im, points_3d


def _do_reconstruction_opencv(left_ims_paths: List,
                              right_ims_paths: List,
                              disparity_maps_paths: List,
                              stereo_params_path: Union[str, None]):
    """

    :param left_ims_paths:
    :param right_ims_paths:
    :param disparity_maps_paths:
    :param stereo_params_path:
    :return:
    """

    left_im: ndarray = cv2.imread(str(left_ims_paths[0]))
    right_im: ndarray = cv2.imread(str(right_ims_paths[0]))
    disparity_map: ndarray = tiff.imread(str(disparity_maps_paths[0]))

    if stereo_params_path is None:
        points_3d: ndarray = _reconstruct_no_params(left_im=left_im,
                                                    disparity_map=disparity_map)
    else:
        stereo_params: Dict = yaml.load(stream=open(stereo_params_path),
                                        Loader=yaml.FullLoader)
        _, _, _ = _reconstruct_stereo_params(stereo_params=stereo_params,
                                                        left_im=left_im,
                                                        right_im=right_im,
                                                        disparity_map=disparity_map)


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

    if cfg.stereo_params is not None:
        stereo_params_path = str(PROJECT_PATH/cfg.stereo_params)
    else:
        stereo_params_path = None

    # _do_reconstruction_matlab(stereo_params_path=stereo_params_path,
    #                           left_ims_paths=left_ims_paths,
    #                           right_ims_paths=right_ims_paths,
    #                           disparity_maps_paths=disparity_maps_paths)

    _do_reconstruction_opencv(left_ims_paths=left_ims_paths,
                              right_ims_paths=right_ims_paths,
                              disparity_maps_paths=disparity_maps_paths,
                              stereo_params_path=stereo_params_path)

    log.info(f'--- Done 3d reconstruction ---')
