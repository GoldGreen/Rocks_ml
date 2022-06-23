import pickle
import os
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
import torch
from utils import *
import torch
from trainer import Trainer

setup_logger()

config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
checkpoint_url = config_file_path

output_dir = "output/object_segmentation"
num_classes = 1

#train_device = "cuda" if torch.cuda.is_available() else "cpu"
train_device = "cpu"

train_dataset_name = "rock_train"
train_images_path = "dataset/train"
train_annotation_path = "dataset/train_annotation.json"

test_dataset_name = "rock_test"
test_images_path = "dataset/test"
test_annotation_path = "dataset/test_annotation.json"

cfg_save_path = "IS_rocks_cfg.pickle"

register_coco_instances(name=train_dataset_name, metadata={},
                        json_file=train_annotation_path, image_root=train_images_path)
register_coco_instances(name=test_dataset_name, metadata={},
                        json_file=test_annotation_path, image_root=test_images_path)

#plot_samples(train_dataset_name, 15)
#plot_samples(test_dataset_name, 5)


def main():
    torch.cuda.empty_cache()
    cfg = get_train_cfg(config_file_path, checkpoint_url, train_dataset_name,
                        test_dataset_name, num_classes, train_device, output_dir)

    with open(cfg_save_path, 'wb') as f:
        pickle.dump(cfg, f)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=True)

    trainer.train()


if __name__ == '__main__':
    main()
