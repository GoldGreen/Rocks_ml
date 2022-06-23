from time import time
from detectron2.engine import DefaultPredictor

import os
import pickle
from utils import *


cfg_save_path = "IS_rocks_cfg.pickle"
with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)

#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0043999.pth")
cfg.MODEL.WEIGHTS = 'dataset/old_weights/model_final.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.TEST.DETECTIONS_PER_IMAGE = 400

predictor = DefaultPredictor(cfg)

on_image('dataset/Eval/me.jpg', predictor)
