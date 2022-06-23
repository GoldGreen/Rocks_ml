from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import GenericMask
from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2.utils.visualizer import ColorMode
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor


def plot_samples(dataset_name, n=1):
    dataset = DatasetCatalog.get(dataset_name)
    dataset_metadata = MetadataCatalog.get(dataset_name)

    for s in random.sample(dataset, n):
        img = cv2.imread(s["file_name"])
        v = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, scale=0.5)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize=(15, 20))
        plt.imshow(v.get_image())
        plt.show()


def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device

    cfg.OUTPUT_DIR = output_dir

    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.MAX_ITER = 60000
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.0001

    cfg.SOLVER.MOMENTUM = 0.9

    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0

    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (7000,)

    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1500
    cfg.SOLVER.WARMUP_ITERS = 1500
    cfg.SOLVER.WARMUP_METHOD = "linear"

    cfg.SOLVER.CHECKPOINT_PERIOD = 4000
    cfg.TEST.EVAL_PERIOD = 2000

    return cfg


def on_image(image_path, predictor):
    img = cv2.imread(image_path)
    outputs = predictor(img)
    predictions = outputs["instances"].to("cpu")

    v = Visualizer(img[:, :, ::-1], {}, 0.5, ColorMode.SEGMENTATION)
    v = v.draw_instance_predictions(
        outputs["instances"].to("cpu"))
    plt.figure(figsize=(14, 10))
    plt.imshow(v.get_image())
    plt.show()
