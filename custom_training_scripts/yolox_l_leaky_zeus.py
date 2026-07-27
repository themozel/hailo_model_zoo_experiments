#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.act='lrelu'
        
        # Dataset configuration
        self.data_dir = "/data/GERALD/"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.test_ann = "instances_test2017.json"
        # Percept dataset images are 1280x736 (16:9) — use this pair instead if switching datasets
        # self.input_size = (736, 992)  # (height, width); matches native 4:3 image aspect ratio, both dims multiples of 32
        # self.test_size = (736, 992)
        # GERALD dataset images are 1920x1080 (16:9, some 1280x720) — use this pair instead if switching datasets
        self.input_size = (640, 1152)  # (height, width); matches 16:9 aspect ratio, both dims multiples of 32
        self.test_size = (640, 1152)
        self.num_classes = 60  # Set this to the number of classes in your dataset
        self.data_num_workers = 4

        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 200
