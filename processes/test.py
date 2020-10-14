"""
Running inference with the trained model on a test (unseen) dataset.
"""
from typing import Dict, List, Tuple
import logging
from pathlib import Path, PosixPath
import cv2
import os
from numpy import ndarray
from omegaconf import DictConfig
import numpy as np
import math
from tqdm import tqdm
from sklearn import datasets

from detectron2.config.config import CfgNode
from detectron2.engine import DefaultPredictor
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes

from tools import get_model_configs, register_custom_coco_dataset, visualizing_predicted_samples

__all__ = ['test']

PROJECT_PATH = Path(__file__).parents[1]  # get directory 2 levels up
log = logging.getLogger(__name__)  # A logger for this file


def _clean_points_3d(points_3d: ndarray) \
        -> List[Tuple]:
    """
    Remove all the points with inf and -inf values.
    Return the positions (indices in 2D array) of the valid points.
    :param points_3d: the ndarray that contains all 3d points (inf and -inf included)
    :type points_3d: ndarray, shape [height, width, 3]
    :return valid_points_positions: the positions (indices in the image 2D array) of the valid
                                    3d points in the points_3d nd array
    :rtype valid_points_positions: List[Tuple]
    """
    X: ndarray = points_3d[:, :, 0]
    mask: ndarray = ~np.isinf(X)  # should be the same for X, Y, and Z
    _indices_not_inf: Tuple = np.asarray(mask).nonzero()
    valid_points_positions: List[Tuple] = [tuple(a) for a in zip(_indices_not_inf[0], _indices_not_inf[1])]

    return valid_points_positions


def _process_detected_point(position: Tuple,
                            points_3d: ndarray)\
        -> ndarray:
    """
    If the 3d point is invalid (inf or -inf), set its values to be the nearest valid point's values
    based on image point positions.

    :param position: the position of the detected object in (y, x) - (height, width)
    :param points_3d: the ndarray that contains 3d points (X, Y, Z) of all points
    :return:
    """
    point3d: ndarray = points_3d[position] * 1e-4

    if point3d[0] == float("inf") or point3d[0] == float("-inf"):
        valid_points_positions: List[Tuple] = _clean_points_3d(points_3d)
        p: ndarray = np.array(position)
        distances: List = [np.linalg.norm(p - np.array(vp)) for vp in valid_points_positions]
        nearest_valid_point_index = np.argmin(distances)
        nearest_valid_point_pos: Tuple = valid_points_positions[int(nearest_valid_point_index)]
        point3d = points_3d[nearest_valid_point_pos] * 1e-4

    if point3d[0] == float("inf") or point3d[0] == float("-inf"):
        raise Exception("the point is still invalid (contains inf and -inf).")

    return point3d


def _get_detected_points(output_instances: Dict,
                         points_3d: ndarray)\
        -> List[Tuple[Tuple, ndarray]]:
    """
    Get the 3d coordinates of the points at the middle bottom of the detected objects.

    :param output_instances: the dictionary that contains information about outputs from detectron2
                             of the detected objects of an image.
    :type output_instances: Dict[Instances]
    :param points_3d: the matrix that contains the 3d points of the same image.
    :type points_3d: ndarray [height, width, num_channels]
    :return detected_points3d: the list that contains the 3d points of middle bottom points of the detected boxes.
    :rtype detected_points3d: List[Tuple[Tuple, ndarray]]
    the first element of the outside Tuple is the image coordinate of a detected tree
    the second element of the outside Tuple is its corresponding world point (X, Y, Z) in ndarray
    """
    instances: Instances = output_instances['instances']
    fields: Dict = instances.get_fields()
    pred_boxes: Boxes = fields['pred_boxes']  # each box has coord x1, y1, x2, y2
    detected_points3d: List[Tuple[Tuple, ndarray]] = []
    for box in pred_boxes:
        middle_bottom_y: int = math.floor(box[3]) - 1
        middle_bottom_x: int = math.floor((box[0] + box[2])/2)
        middle_bottom: Tuple = (middle_bottom_y, middle_bottom_x)  # middle_bottom is the root of a tree
        point3d_middle_bottom: ndarray = _process_detected_point(position=middle_bottom,
                                                                 points_3d=points_3d)
        detected_points3d.append((middle_bottom, point3d_middle_bottom))

    if len(detected_points3d) != len(pred_boxes):
        raise Exception('the number of detected objects and the number of corresponding'
                        '3d points are different')

    return detected_points3d


def test(cfg: DictConfig):
    """
    Running inference on an unseen dataset.
    :param cfg: the configuration dictionary of dataset_model.
    :return:
    """
    log.info(f'--- Start Testing ---')
    if cfg.train.option and cfg.test.use_pretrained_weight:
        raise Exception('cfg.train.option is Yes, but cfg.test.use_pretrained_weight is also Yes')
    elif not cfg.train.option and not cfg.test.use_pretrained_weight:
        raise Exception('cfg.train.option is No but cfg.test.use_pretrained_weight is also No')

    _, coco_tree_metadata = register_custom_coco_dataset(cfg=cfg,
                                                         process='test')
    output_dir: Path = Path(os.getcwd()) / 'predicted_ims'  # using hydra, this will be PROJECT_PATH/outputs/date/time
    output_dir.mkdir(parents=False, exist_ok=True)

    model_cfg: CfgNode = get_model_configs(cfg=cfg, process='test')
    predictor: DefaultPredictor = DefaultPredictor(model_cfg)
    log.info(f'Doing inference on {PROJECT_PATH/cfg.test.test_dataset_dir}...')

    test_ims_iter: Path.iterdir = Path(PROJECT_PATH/cfg.test.test_dataset_dir).iterdir()
    test_im_paths: List = sorted([f for f in test_ims_iter if f.is_file()])

    points_3d_paths: List = []
    if cfg.test.map_3d_points:
        if cfg.test.map_3d_points is None:
            raise Exception('cfg.map_3d_points is Yes, but no 3d_points_path is provided')
        points_3d_iter: Path.iterdir = Path(PROJECT_PATH/cfg.test.points_3d_path).iterdir()
        points_3d_paths = sorted([f for f in points_3d_iter if f.is_file()])
        if len(points_3d_paths) != len(test_im_paths):
            raise Exception('The number of test images is not the same with the number'
                            'of corresponding 3d points')

    total_iters: int = len(test_im_paths)
    with tqdm(total=total_iters) as progress_bar:
        for index, test_im in enumerate(test_im_paths):
            img: ndarray = cv2.imread(str(test_im))  # [height, width, 3]
            outputs: Dict = predictor(img)

            detected_points3d: List = []
            if cfg.test.map_3d_points:
                points_3d_path: PosixPath = points_3d_paths[index]
                points_3d: ndarray = np.load(str(points_3d_path))
                detected_points3d: List[Tuple[Tuple, ndarray]] = _get_detected_points(output_instances=outputs,
                                                                                      points_3d=points_3d)

            # visualize the result: predicted objects (with 3d coordinates) and save to disk
            path_to_save: str = str(output_dir / test_im.name) if cfg.test.saving_predicted_ims else None
            visualizing_predicted_samples(img=img,
                                          metadata=coco_tree_metadata,
                                          predicted_samples=outputs,
                                          points3d_predicted_samples=detected_points3d,
                                          path_to_save=path_to_save,
                                          show_image=cfg.test.show_predicted_ims)
            progress_bar.update(1)

    log.info(f'Predicted images are saved to {output_dir}')
    log.info('--- Testing Done ---')


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target
    x = 1
