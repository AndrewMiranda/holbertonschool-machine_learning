#!/usr/bin/env python3
"""
this is the yolo class
"""
import tensorflow.keras as K


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
