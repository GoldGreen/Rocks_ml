from ast import NodeTransformer
import random
import cv2
from detectron2.data.transforms.augmentation import Augmentation
from detectron2.data.transforms.transform import Transform, NoOpTransform
import numpy as np
import glob
import pycocotools.mask as mask_util
from detectron2.structures import PolygonMasks
import itertools


class RandomBackground(Augmentation):
    def __init__(self, background_folder, annotations):
        super().__init__()
        background_paths = [file for file in glob.glob(background_folder)]
        polygones = list(
            map(lambda x: list(map(np.array, x["segmentation"])), annotations))
        self._init(locals())

    def get_transform(self, image):
        return RandomBackgroundTransform(cv2.imread(random.choice(self.background_paths)), self.polygones)


class RandomBackgroundTransform(Transform):
    def __init__(self, background, polygones):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, interp: str = None) -> np.ndarray:
        shape = img.shape
        polygones = list(itertools.chain(*self.polygones))
        mask = self.polygons_to_bitmask(polygones, shape[0], shape[1])
        res = self.background.copy()
        res = cv2.resize(res, (shape[1], shape[0]))
        res[mask] = img[mask]
        return res

    def polygons_to_bitmask(self, polygons, height, width):
        rles = mask_util.frPyObjects(polygons, height, width)
        rle = mask_util.merge(rles)
        return mask_util.decode(rle).astype(np.bool)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation

    def inverse(self) -> Transform:
        return NoOpTransform()
