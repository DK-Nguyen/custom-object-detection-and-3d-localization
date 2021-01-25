"""
Contains functions to visualize images and save them to specified directories
"""
from typing import List, Dict, Optional, Tuple
import random
import cv2
import numpy as np
from numpy import ndarray
import PIL
from IPython import display
import matplotlib.colors as mplc

from detectron2.data.catalog import Metadata
from detectron2.utils.visualizer import Visualizer, VisImage, ColorMode
# from detectron2.utils.colormap import _COLORS
from detectron2.structures.instances import Instances

from tools.constants import _COLORS

__all__ = ['visualizing_coco_dataset', 'visualizing_predicted_samples',
           'visualizing_triplets']


def _get_instance_fields(instances: Instances):
    try:
        return instances.pred_boxes, \
           instances.scores, \
           instances.pred_classes, \
           instances.pred_masks
    except:
        return instances.pred_boxes, \
               instances.scores, \
               instances.pred_classes,


def _create_text_labels(classes, scores, class_names):
    """
    Copied from detectron2.utils.visualizer._create_text_labels
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None and class_names is not None and len(class_names) > 0:
        labels = [class_names[i] for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    return labels


def pick_color(rgb=False, maximum=255, color_index=0):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1
        color_index (int): index of the color to pick from _COLORS

    Returns:
        ndarray: a vector of 3 numbers
    """
    # idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[color_index] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


def cv2_imshow(a):
    """A replacement for cv2.imshow() for use in Jupyter notebooks.
    Args:
    a : np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape
      (N, M, 3) is an NxM BGR color image. shape (N, M, 4) is an NxM BGRA color
      image.
    """
    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display.display(PIL.Image.fromarray(a))


def visualizing_coco_dataset(dataset_dicts: List[Dict],
                             dataset_metadata: Metadata,
                             num_ims: Optional[int] = 0)\
        -> None:
    """
    Visualizing a dataset in COCO format, used after registering a coco dataset using
    detectron2.data.datasets.register_coco_instances.

    :param dataset_dicts: information about images and instances.
    :type dataset_dicts: List[Dict].
    :param dataset_metadata: contains additional information of the dataset_dicts.
    :type dataset_metadata: detectron2.data.catalog.Metadata.
    :param num_ims: number of images to visualize
    :type num_ims: int (optional, default values: 0. if value is -1, then show all images)
    :return: None
    """
    ims_to_show: int = len(dataset_dicts) if num_ims == -1 else num_ims
    for d in random.sample(dataset_dicts, ims_to_show):
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
                                  points3d_predicted_samples: List,
                                  path_to_save: str = None,
                                  show_image: bool = True)\
        -> None:
    """
    Visualizing the predicted samples from detectron2's based models

    :param img: the numpy array read from the testing image.
    :type img: numpy.ndarray of shape [width, height, 3].
    :param metadata: the metadata of the predicted samples.
    :type metadata: detectron2.data.catalog.Metadata
    :param predicted_samples: the Dict that contains predicted instances.
    :type predicted_samples: Dict[Instances].
    :param points3d_predicted_samples: the list that contains corresponding 3d points
                                        of the predicted objects
    :type points3d_predicted_samples: List[]
    :param path_to_save: if value is not None, then save the image to the path
    :type path_to_save: str
    :param show_image: if True, then show the images, otherwise do not.
    :type show_image: bool, default value: True
    :return: None
    """
    instances: Instances = predicted_samples['instances'].to("cpu")
    try:
        pred_boxes, scores, pred_classes, pred_masks = _get_instance_fields(instances)
    except:
        pred_boxes, scores, pred_classes = _get_instance_fields(instances)
        pred_masks = None

    circle_radius: int = 5

    visualizer: MyVisualizer = MyVisualizer(img[:, :, ::-1],
                                            metadata=metadata,
                                            scale=1,
                                            instance_mode=ColorMode.SEGMENTATION)
    labels = visualizer.get_labels(classes=pred_classes, scores=scores)

    assigned_colors = [pick_color(rgb=True, maximum=1, color_index=-1) for _ in range(instances.__len__())]

    # out: VisImage = visualizer.draw_instance_predictions(predictions=instances)
    vis_im: VisImage = visualizer.overlay_instances(boxes=pred_boxes,
                                                    masks=pred_masks,
                                                    labels=labels,
                                                    assigned_colors=assigned_colors)

    for i, point in enumerate(points3d_predicted_samples):
        position: Tuple = point[0][::-1]
        point3d: ndarray = point[1]
        text_pos: Tuple = (position[0], position[1]+circle_radius)
        text = f"x:{point3d[0]:.2f}\n" \
               f"y:{point3d[1]:.2f}\n" \
               f"z:{point3d[2]:.2f}"
        vis_im = visualizer.draw_text(text=text, position=text_pos, color="w", bg_color="k")
        vis_im = visualizer.draw_circle(circle_coord=position, color="k", radius=circle_radius)

    v: ndarray = vis_im.get_image()[:, :, ::-1]

    if show_image:
        cv2.imshow('', v)
        cv2.waitKey()
        cv2.destroyAllWindows()
    if path_to_save is not None:
        cv2.imwrite(path_to_save, v)


def visualizing_triplets(left_im: ndarray,
                         right_im: ndarray,
                         disp_map: ndarray)\
        -> None:
    """

    :param left_im:
    :param right_im:
    :param disp_map:
    :return:
    """
    cv2.imshow('', left_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('', right_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('', disp_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class MyVisualizer(Visualizer):
    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        bg_color="b",
        horizontal_alignment="center",
        rotation=0
    ):
        """
        Override the function detectron2.utils.visualizer.Visualizer.draw_text()
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            bg_color: the background color of the text
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW

        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family="sans-serif",
            bbox={"facecolor": bg_color, "alpha": 0.4, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output

    def get_labels(self, classes, scores):
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        return labels

