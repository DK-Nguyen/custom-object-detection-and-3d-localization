"""
Contains functions to visualize images and save them to specified directories
"""
from typing import List, Dict, Optional
import random
import cv2
from numpy import ndarray

from detectron2.data.catalog import Metadata
from detectron2.utils.visualizer import Visualizer, VisImage, ColorMode

__all__ = ['visualizing_coco_dataset', 'visualizing_predicted_samples']


def visualizing_coco_dataset(dataset_dicts: List[Dict],
                             dataset_metadata: Metadata,
                             num_ims: Optional[int] = 3)\
        -> None:
    """
    Visualizing a dataset in COCO format.

    :param dataset_dicts: information about images and instances.
    :type dataset_dicts: List[Dict].
    :param dataset_metadata: contains additional information of the dataset_dicts.
    :type dataset_metadata: detectron2.data.catalog.Metadata.
    :param num_ims: number of images to visualize
    :type num_ims: int (optional, default values: 3)
    :return: None
    """
    for d in random.sample(dataset_dicts, num_ims):
        img: ndarray = cv2.imread(d["file_name"])
        visualizer: Visualizer = Visualizer(img[:, :, ::-1],
                                            metadata=dataset_metadata,
                                            scale=1)
        vis: VisImage = visualizer.draw_dataset_dict(d)
        v = vis.get_image()[:, :, ::-1]
        cv2.imshow('', v)
        cv2.waitKey()
    cv2.destroyAllWindows()


def visualizing_predicted_samples(img: ndarray,
                                  metadata: Metadata,
                                  predicted_samples: Dict,
                                  path_to_save: str = None)\
        -> None:
    """
    Visualizing the predicted samples from detectron2's based models

    :param img: the numpy array read from the testing image.
    :type img: numpy.ndarray of shape [width, height, 3].
    :param predicted_samples: the Dict that contains predicted instances.
    :type predicted_samples: Dict.
    :param metadata: the metadata of the predicted samples.
    :type detectron2.data.catalog.Metadata
    :param path_to_save: if value is not None, then save the image to the path
    :type path_to_save: str
    :return: None
    """
    visualizer: Visualizer = Visualizer(img[:, :, ::-1],
                                        metadata=metadata,
                                        scale=1,
                                        instance_mode=ColorMode.IMAGE_BW)
    out: VisImage = visualizer.draw_instance_predictions(predicted_samples["instances"].to("cpu"))
    v: ndarray = out.get_image()[:, :, ::-1]
    cv2.imshow('', v)
    if path_to_save is not None:
        cv2.imwrite(path_to_save, v)
    cv2.waitKey()
    cv2.destroyAllWindows()
