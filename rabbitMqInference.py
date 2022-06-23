from time import time
from detectron2.engine import DefaultPredictor

import os
import json
import pickle

from sklearn.metrics import get_scorer
from utils import *

import pika
import cv2
import numpy as np
from utils import *

cfg_save_path = "IS_rocks_cfg.pickle"
with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)

#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0013999.pth")
cfg.MODEL.WEIGHTS = 'dataset/old_weights/model_final.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

queueName = 'rocks'

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))

channel = connection.channel()
channel.queue_declare(queue=queueName)


class DetectionResultDto(object):
    def __init__(self, detections):
        self.detections = detections


class DetectionDto(object):
    def __init__(self, prediction_class, score, bbox, polygon):
        self.prediction_class = prediction_class
        self.score = score
        self.bbox = bbox
        self.polygon = polygon


def get_bboxes(pred_bbox):
    return list(
        map(lambda x: list(map(lambda z: z.item(), x)), pred_bbox.tensor.detach().numpy()))


def get_polygones(pred_masks, height, width):
    masks = np.asarray(pred_masks)
    masks = [GenericMask(x, height,
                         width) for x in masks]

    return list(
        map(lambda x: list(map(list, x.polygons)), masks))


def get_classes(pred_classes):
    return pred_classes.tolist()


def get_scores(pred_scores):
    return pred_scores.tolist()


def on_request(ch, method, props, body):
    jpg = np.frombuffer(body, dtype=np.uint8)
    mat = cv2.imdecode(jpg, cv2.IMREAD_COLOR)

    outputs = predictor(mat)
    predictions = outputs["instances"].to("cpu")

    bboxes = get_bboxes(predictions.pred_boxes)
    classes = get_classes(predictions.pred_classes)
    scores = get_scores(predictions.scores)
    polygons = get_polygones(predictions.pred_masks,
                             mat.shape[0], mat.shape[1])

    detections = [DetectionDto(cls, score, bbox, polygon)
                  for (cls, score, bbox, polygon)

                  in zip(classes, scores, bboxes, polygons)]

    detectonResult = DetectionResultDto(detections)

    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(
                         correlation_id=props.correlation_id),
                     body=json.dumps(detectonResult, default=lambda o: o.__dict__))

    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=queueName, on_message_callback=on_request)

print(" [x] Awaiting RPC requests")
channel.start_consuming()
