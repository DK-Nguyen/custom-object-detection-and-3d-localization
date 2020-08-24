"""
Process: preparing the dataset for the neural networks
"""
from typing import List, Dict, Tuple
import logging
import os
from pathlib import Path
import json
import numpy as np
from PIL import ImageDraw, Image

from omegaconf import DictConfig
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.catalog import Metadata

from .hard_negative import HardNegativeBackgroundPreparation
from .image_composition import ImageComposition
from .coco_json_utils import CocoJsonCreator


__all__ = ['dataset_creation', 'register_custom_coco_dataset']

PROJECT_PATH = Path(__file__).parents[1]  # get directory 2 levels up
log = logging.getLogger(__name__)  # A logger for this file


def dataset_creation(cfg: DictConfig) -> None:
    """
    Creating the dataset using configurations in cfg.dataset.

    :param cfg: the configuration dictionary.
    :type cfg: omegaconf.dictconfig.DictConfig
    :return: None
    """
    log.info('--- Dataset creation ---')
    hard_negative_input_dir: Path = PROJECT_PATH / cfg.hard_negative_background_preparation.input_dir
    hard_negative_output_dir: Path = PROJECT_PATH / cfg.hard_negative_background_preparation.output_dir

    if cfg.hard_negative_background_preparation.option:
        log.info('Preparing hard negative background images')
        hnbg: HardNegativeBackgroundPreparation = HardNegativeBackgroundPreparation(
            input_dir=hard_negative_input_dir,
            output_dir=hard_negative_output_dir,
            output_width=cfg.hard_negative_background_preparation.output_width,
            output_height=cfg.hard_negative_background_preparation.output_height,
            output_type=cfg.hard_negative_background_preparation.output_type
        )
        hnbg.compose_images(position=cfg.hard_negative_background_preparation.position)
        log.info('Done preparing hard negative background images')

    if cfg.training_images_preparation.option:
        log.info(f'Preparing training images for {cfg.name} dataset')
        image_comp_training: ImageComposition = ImageComposition(cfg.training_images_preparation)
        image_comp_training.main()
        coco_json_creator_training: CocoJsonCreator = CocoJsonCreator(cfg.training_images_preparation)
        coco_json_creator_training.main()

    if cfg.validation_images_preparation.option:
        val_cfg: DictConfig = cfg.validation_images_preparation
        log.info(f'Preparing validation images for {cfg.name} dataset')
        val_images_dir: Path = PROJECT_PATH / val_cfg.val_images_dir
        if not val_images_dir.exists():
            raise Exception('Validation images directory does not exist')
        labelme_annotation_dir: Path = PROJECT_PATH / val_cfg.labelme_annotation_dir
        if val_cfg.labelme_annotating:
            pass
            os.system(f'labelme {val_images_dir} --output {labelme_annotation_dir} '
                      f'--nodata --autosave --logger-level debug')
        if val_cfg.converting_labelme_to_coco:
            coco_json_path: str = str(PROJECT_PATH / val_cfg.coco_json_path)
            log.info(f'Converting labelme annotations from {labelme_annotation_dir} \n'
                     f'to COCO data format in JSON, output to file {coco_json_path}.')
            labelme_json_paths: List = list(labelme_annotation_dir.glob('*.json'))
            _Labelme2coco(labelme_json=labelme_json_paths,
                          save_json_path=coco_json_path)

    log.info('--- Dataset creation done ---')


def register_custom_coco_dataset(cfg: DictConfig,
                                 process: str = 'train') \
        -> Tuple[List[Dict], Metadata]:
    """
    Registering the custom dataset in COCO format to detectron2.

    :param cfg: the configuration dictionary of dataset_model.
    :type cfg: omegaconf.dictconfig.DictConfig.
    :param process: value should be 'train', 'val', or 'test'
    :type process: str
    :return information about images and instances in
             COCO format, together with its metadata.
    :rtype dataset_dicts: List[Dict].
           dataset_metadata: detectron2.data.catalog.Metadata.
    """
    if process not in ['train', 'test', 'val']:
        raise Exception(f"process is {process}, but it must be either 'train', 'test', or 'val'")
    dataset_dicts: List[Dict] = [{}]
    dataset_metadata: Metadata = Metadata()

    train_dataset: str = cfg.name + "_train"
    train_images_dir: Path = PROJECT_PATH / cfg.train.train_dataset_dir / 'images'
    train_coco_instances_json: str = str(PROJECT_PATH / cfg.train.train_dataset_dir / 'coco_instances.json')
    try:
        log.info(f'Registering {train_dataset} as a COCO-format dataset')
        register_coco_instances(name=train_dataset,
                                metadata={},
                                json_file=train_coco_instances_json,
                                image_root=train_images_dir)
    except AssertionError:  # if the dataset is already registered, do nothing
        pass

    if process == 'train':
        dataset_dicts = DatasetCatalog.get(train_dataset)
        dataset_metadata = MetadataCatalog.get(train_dataset)
    elif process == 'test':
        log.info(f'Getting metadata for testing on {cfg.name}')
        dataset_metadata = MetadataCatalog.get(train_dataset)
    elif process == 'val':
        val_dataset: str = cfg.name+"_val"
        val_images_dir: Path = PROJECT_PATH/cfg.validation.val_dataset_dir/'images'
        val_coco_instances_json: str = str(PROJECT_PATH/cfg.validation.val_dataset_dir/'coco_instances.json')
        log.info(f'Registering {val_dataset} as a COCO-format dataset')
        register_coco_instances(name=val_dataset,
                                metadata={},
                                json_file=val_coco_instances_json,
                                image_root=val_images_dir)
        dataset_dicts = DatasetCatalog.get(val_dataset)
        dataset_metadata = MetadataCatalog.get(val_dataset)

    return dataset_dicts, dataset_metadata


class _Labelme2coco(object):
    """
    Convert labelme annotation files to COCO dataset format.
    Code based on https://github.com/Tony607/labelme2coco/blob/master/labelme2coco.py
    Tutorial: https://www.dlology.com/blog/how-to-create-custom-coco-data-set-for-instance-segmentation/
    """
    def __init__(self,
                 labelme_json: List = None,
                 save_json_path: str = None)\
            -> None:
        """
        :param labelme_json: the list of all labelme json files
        :param save_json_path: the path to save new json
        """
        self.labelme_json = labelme_json if labelme_json is not None else []
        if save_json_path is None:
            raise Exception('The path to save json output in COCO format '
                            'for the validation dataset is not defined.')
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0

        self.save_json()

    def data_transfer(self):
        for num, json_file in enumerate(self.labelme_json):
            with open(json_file, "r") as fp:
                data = json.load(fp)
                self.images.append(self.image(data, num))
                for shapes in data["shapes"]:
                    label = shapes["label"].split("_")
                    if label not in self.label:
                        self.label.append(label)
                    points = shapes["points"]
                    self.annotations.append(self.annotation(points, label, num))
                    self.annID += 1

        # Sort all text labels so they are in the same order across data splits.
        self.label.sort()
        for label in self.label:
            self.categories.append(self.category(label))
        for annotation in self.annotations:
            annotation["category_id"] = self.getcatid(annotation["category_id"])

    def image(self, data, num):
        image = {}
        image["height"] = data["imageHeight"]
        image["width"] = data["imageWidth"]
        image["id"] = num
        image["file_name"] = data["imagePath"].split("/")[-1]

        self.height = data["imageHeight"]
        self.width = data["imageWidth"]

        return image

    def category(self, label):
        category = {"supercategory": label[0],
                    "id": len(self.categories),
                    "name": label[0]}
        return category

    def annotation(self, points, label, num):
        annotation = {}
        contour = np.array(points)
        x = contour[:, 0]
        y = contour[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        annotation["segmentation"] = [list(np.asarray(points).flatten())]
        annotation["iscrowd"] = 0
        annotation["area"] = area
        annotation["image_id"] = num

        annotation["bbox"] = list(map(float, self.getbbox(points)))

        annotation["category_id"] = label[0]  # self.getcatid(label)
        annotation["id"] = self.annID
        return annotation

    def getcatid(self, label):
        for category in self.categories:
            if label == category["name"]:
                return category["id"]
        # log.info("label: {} not in categories: {}.".format(label, self.categories))
        exit()
        return -1

    def getbbox(self, points):
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):

        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]

        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        return [
            left_top_c,
            left_top_r,
            right_bottom_c - left_top_c,
            right_bottom_r - left_top_r,
        ]

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {"images": self.images,
                     "categories": self.categories,
                     "annotations": self.annotations}
        return data_coco

    def save_json(self):
        # log.info(f'Save COCO instances converted from labelme format to {self.save_json_path}')
        self.data_transfer()
        self.data_coco = self.data2coco()
        json.dump(self.data_coco, open(self.save_json_path, "w"), indent=4)
