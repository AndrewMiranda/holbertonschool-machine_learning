#!/usr/bin/env python3
"""
this is the yolo class
"""
import tensorflow.keras as K
import numpy as np


class Yolo():
    """
    The Yolo class
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """the constructor
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            class_names = [line.strip() for line in f]
        self.class_names = class_names
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def _sigmoid(self, x):
        """
        sigmoid function
        """
        return 1. / (1. + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """outputs
        """
        boxes = [output[..., :4] for output in outputs]
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for i in range(len(outputs)):
            grid_height, grid_width, anchor_boxes, _ = outputs[i].shape

            tx = outputs[i][..., 0]
            ty = outputs[i][..., 1]
            tw = outputs[i][..., 2]
            th = outputs[i][..., 3]

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            cx = np.tile(np.arange(0, grid_width), grid_height)
            cx = cx.reshape(grid_width, grid_width, 1)

            cy = np.tile(np.arange(0, grid_width), grid_height)
            cy = cy.reshape(grid_height, grid_height).T
            cy = cy.reshape(grid_height, grid_height, 1)

            bx = self._sigmoid(tx) + cx
            by = self._sigmoid(ty) + cy
            bw = pw * np.exp(tw)
            bh = ph * np.exp(th)

            bx /= grid_width
            by /= grid_height

            input_width = self.model.input.shape[1].value
            input_height = self.model.input.shape[2].value
            bw /= input_width
            bh /= input_height

            x1 = (bx - bw / 2)
            y1 = (by - bh / 2)
            x2 = (bx + bw / 2)
            y2 = (by + bh / 2)

            boxes[i][..., 0] = x1 * image_width
            boxes[i][..., 1] = y1 * image_height
            boxes[i][..., 2] = x2 * image_width
            boxes[i][..., 3] = y2 * image_height

            box_confidence = self._sigmoid(outputs[i][:, :, :, 4])
            box_confidences.append(box_confidence.reshape(grid_height,
                                                          grid_width,
                                                          anchor_boxes,
                                                          1))

            box_class_prob = self._sigmoid(outputs[i][:, :, :, 5:])
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """filter all boxes
        """
        box_scores = []
        for box_confidence, box_class_prob in zip(box_confidences,
                                                  box_class_probs):
            box_scores.append(box_confidence * box_class_prob)

        box_class = [score.argmax(axis=-1) for score in box_scores]
        box_class_list = [box.reshape(-1) for box in box_class]
        box_class_concat = np.concatenate(box_class_list, axis=-1)

        box_class_scores = [score.max(axis=-1) for score in box_scores]
        box_score_list = [box.reshape(-1) for box in box_class_scores]
        box_scores_concat = np.concatenate(box_score_list, axis=-1)

        boxes_list = [box.reshape(-1, 4) for box in boxes]
        boxes_concat = np.concatenate(boxes_list, axis=0)

        filtering_mask = np.where(box_scores_concat >= self.class_t)

        boxes_ = boxes_concat[filtering_mask]
        box_classes = box_class_concat[filtering_mask]
        box_scores = box_scores_concat[filtering_mask]

        return (boxes_, box_classes, box_scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """suppression
        """
        tmp_boxes = []
        tmp_classes = []
        tmp_scores = []

        for clase in np.unique(box_classes):
            indexes = np.where(box_classes == clase)
            boxes_ofclas = filtered_boxes[indexes]
            classes_ofclas = box_classes[indexes]
            scores_ofclas = box_scores[indexes]

            x1 = boxes_ofclas[:, 0]
            y1 = boxes_ofclas[:, 1]
            x2 = boxes_ofclas[:, 2]
            y2 = boxes_ofclas[:, 3]

            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            order = scores_ofclas.argsort()[::-1]

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)
                inter = w * h
                ovr = inter / (areas[i] + areas[order[1:]] - inter)

                inds = np.where(ovr <= self.nms_t)[0]
                order = order[inds + 1]

            tmp_boxes.append(boxes_ofclas[keep])
            tmp_classes.append(classes_ofclas[keep])
            tmp_scores.append(scores_ofclas[keep])

        boxes_predic = np.concatenate(tmp_boxes, axis=0)
        classes_predic = np.concatenate(tmp_classes, axis=0)
        scores_predic = np.concatenate(tmp_scores, axis=0)

        return boxes_predic, classes_predic, scores_predic
