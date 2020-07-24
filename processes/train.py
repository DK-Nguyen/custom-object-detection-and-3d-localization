"""
Train  neural networks the the synthesized dataset
"""
import random
from pathlib import Path
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
from detectron2.data.datasets import register_coco_instances
import logging

PROJECT_PATH = Path(__file__).parents[1]  # get directory 2 levels up
log = logging.getLogger(__name__)  # A logger for this file

# log.info(f'Registering COCO Format dataset for {cfg.training_images_preparation.name}')
# coco_instances_path = PROJECT_PATH / cfg.training_images_preparation.output_dir / 'coco_instances.json'
# coco_image_dir = PROJECT_PATH / cfg.training_images_preparation.output_dir / 'images'
# log.info(f'{cfg.training_images_preparation.name}')
# register_coco_instances(name=cfg.training_images_preparation.name,
#                         metadata={},
#                         json_file=coco_instances_path,
#                         image_root=coco_image_dir)


# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=fruits_nuts_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     cv2_imshow(vis.get_image()[:, :, ::-1])

if __name__ == '__main__':
    # fruits_nuts_metadata = MetadataCatalog.get("coco_tree_train")
    log.info(f'Registering COCO Format dataset for coc_tree_train')
    coco_instances_path = PROJECT_PATH / 'datasets/coco_like/training' / 'coco_instances.json'
    coco_image_dir = PROJECT_PATH / 'datasets/coco_like/training' / 'images'

    register_coco_instances(name='coco_tree_train',
                            metadata={},
                            json_file=coco_instances_path,
                            image_root=coco_image_dir)

    dataset_dicts = DatasetCatalog.get("coco_tree_train")
    print(dataset_dicts)
