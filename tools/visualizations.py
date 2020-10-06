"""
Contains functions to visualize images and save them to specified directories
"""
from typing import List, Dict, Optional
import random
import cv2
import numpy as np
from numpy import ndarray
import PIL
from IPython import display

from detectron2.data.catalog import Metadata
from detectron2.utils.visualizer import Visualizer, VisImage, ColorMode, \
                                        GenericMask, _SMALL_OBJECT_AREA_THRESH, \
                                        _create_text_labels
from detectron2.utils.colormap import random_color, _COLORS
from matplotlib import colors

__all__ = ['visualizing_coco_dataset', 'visualizing_predicted_samples',
           'visualizing_triplets']


def my_random_color(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
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
    visualizer: Visualizer = Visualizer(img[:, :, ::-1],
                                        metadata=metadata,
                                        scale=1,
                                        instance_mode=ColorMode.IMAGE_BW)
    out: VisImage = visualizer.draw_instance_predictions(predicted_samples["instances"].to("cpu"))
    # my_colors = [random_color(rgb=True, maximum=1) for _ in range(3)]
    # out = visualizer.overlay_instances(assigned_colors=my_colors)
    v: ndarray = out.get_image()[:, :, ::-1]

    # a = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)

    for i, point in enumerate(points3d_predicted_samples):
        pos = point[0][::-1]
        point3d = point[1]
        v = cv2.circle(img=v, center=pos, radius=10, color=(255, 255, 255), thickness=(-1))
        text = f"x:{point3d[0]:.2f}, y:{point3d[1]:.2f}, z:{point3d[2]:.2f}"
        v = cv2.putText(img=v, text=text, org=pos,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65,
                        color=(0, 0, 255), thickness=1)

    if show_image:
        cv2.imshow('', v)
        cv2.waitKey()
        cv2.destroyAllWindows()
        # cv2_imshow(v)
    if path_to_save is not None:
        cv2.imwrite(path_to_save, v)


def visualizing_triplets(left_im: ndarray,
                         right_im: ndarray,
                         disp_map:ndarray)\
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
    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            # self.output.img = self._create_grayscale_image(
            #     (predictions.pred_masks.any(dim=0) > 0).numpy()
            # )
            alpha = 0.3

        _, assigned_colors = self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output, assigned_colors

    def overlay_instances(
        self,
        *,
        boxes=None,
        labels=None,
        masks=None,
        keypoints=None,
        assigned_colors=None,
        alpha=0.5
    ):
        """
        Args:
            boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
                or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
                or a :class:`RotatedBoxes`,
                or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
                for the N objects in a single image,
            labels (list[str]): the text to be displayed for each instance.
            masks (masks-like object): Supported types are:

                * :class:`detectron2.structures.PolygonMasks`,
                  :class:`detectron2.structures.BitMasks`.
                * list[list[ndarray]]: contains the segmentation masks for all objects in one image.
                  The first level of the list corresponds to individual instances. The second
                  level to all the polygon that compose the instance, and the third level
                  to the polygon coordinates. The third level should have the format of
                  [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
                * list[ndarray]: each ndarray is a binary mask of shape (H, W).
                * list[dict]: each dict is a COCO-style RLE.
            keypoints (Keypoint or array like): an array-like object of shape (N, K, 3),
                where the N is the number of instances and K is the number of keypoints.
                The last dimension corresponds to (x, y, visibility or score).
            assigned_colors (list[matplotlib.colors]): a list of colors, where each color
                corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
                for full list of formats that the colors are accepted in.

        Returns:
            output (VisImage): image object with visualizations.
        """
        num_instances = None
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)
        if masks is not None:
            masks = self._convert_masks(masks)
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)
        if keypoints is not None:
            if num_instances:
                assert len(keypoints) == num_instances
            else:
                num_instances = len(keypoints)
            keypoints = self._convert_keypoints(keypoints)
        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
        if num_instances == 0:
            return self.output
        if boxes is not None and boxes.shape[1] == 5:
            return self.overlay_rotated_instances(
                boxes=boxes, labels=labels, assigned_colors=assigned_colors
            )

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
            keypoints = keypoints[sorted_idxs] if keypoints is not None else None

        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                self.draw_box(boxes[i], edge_color=color)

            if masks is not None:
                for segment in masks[i].polygons:
                    self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)

            if labels is not None:
                # first get a box
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                    horiz_align = "left"
                elif masks is not None:
                    x0, y0, x1, y1 = masks[i].bbox()

                    # draw text in the center (defined by median) when box is not drawn
                    # median is less sensitive to outliers.
                    text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                    horiz_align = "center"
                else:
                    continue  # drawing the box confidence for keypoints isn't very useful.
                # for small objects, draw text at the side to avoid occlusion
                instance_area = (y1 - y0) * (x1 - x0)
                if (
                    instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                    or y1 - y0 < 40 * self.output.scale
                ):
                    if y1 >= self.output.height - 5:
                        text_pos = (x1, y0)
                    else:
                        text_pos = (x0, y1)

                height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
                lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                font_size = (
                    np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                    * 0.5
                    * self._default_font_size
                )
                self.draw_text(
                    labels[i],
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                )

        # draw keypoints
        if keypoints is not None:
            for keypoints_per_instance in keypoints:
                self.draw_and_connect_keypoints(keypoints_per_instance)

        return self.output, assigned_colors
