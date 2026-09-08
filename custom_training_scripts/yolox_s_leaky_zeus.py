#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.act = 'lrelu'
        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 200
        # Effective peak LR = basic_lr_per_img * CLI batch size (`-b`). Historical GERALD-cropped
        # runs used inconsistent batch sizes across sizes (s=48, m=96, l=96, x=64), giving
        # different effective LRs (0.0075/0.015/0.015/0.01) despite an identical 200-epoch
        # budget — a likely cause of yolox-x underperforming yolox-l despite being the larger
        # model (see inference_results/yolox-vs-damoyolo-GERALD-cropped/comparison.md). A first
        # fix pinned this to `-b 64` (peak LR 0.01) for all four sizes, matching DAMO-YOLO's own
        # GERALD-cropped configs at the time. Now re-pinned to `-b 48` (peak LR 0.0075) instead —
        # matching the full-frame GERALD training recipe (yolox_{s,m,l,x}_leaky_zeus and
        # damoyolo_tinynasL{20_T,25_S,35_M} all trained at batch=48 there, see
        # data/GERALD/{YOLOX,DAMOYOLO}_outputs/*/train_log.txt) — so a cropped-vs-non-cropped
        # accuracy comparison isolates cropping as the only variable, not a different LR/batch
        # recipe. DAMO-YOLO's GERALD-cropped configs were re-pinned to batch_size=48 to match.
        self.basic_lr_per_img = 0.01 / 64.0

        # ### GERALD dataset configuration
        # self.data_dir = "/data/GERALD/"
        # self.train_ann = "instances_train2017.json"
        # self.val_ann = "instances_val2017.json"
        # self.test_ann = "instances_test2017.json"
        # # GERALD dataset images are 1920x1080 (16:9, some 1280x720) — use this pair instead if switching datasets
        # self.input_size = (640, 1152)  # (height, width); matches 16:9 aspect ratio, both dims multiples of 32
        # self.test_size = (640, 1152)
        # self.num_classes = 60  # Set this to the number of classes in your dataset
        # self.data_num_workers = 4
        
        # ### Percept dataset configuration
        # self.data_dir = "/data/percept/"
        # self.train_ann = "instances_train2017.json"
        # self.val_ann = "instances_val2017.json"
        # self.test_ann = "instances_test2017.json"
        # # Percept dataset images are 1280x736 (16:9) — use this pair instead if switching datasets
        # self.input_size = (736, 992)  # (height, width); matches native 4:3 image aspect ratio, both dims multiples of 32
        # self.test_size = (736, 992)
        # self.num_classes = 34  # Set this to the number of classes in your dataset
        # self.data_num_workers = 4
        
        # ### Percept-cropped dataset configuration
        # self.data_dir = "/data/percept-cropped/"
        # self.train_ann = "instances_train2017.json"
        # self.val_ann = "instances_val2017.json"
        # self.test_ann = "instances_test2017.json"
        # # Percept dataset images are 1280x736 (16:9) — use this pair instead if switching datasets
        # self.input_size = (640, 640)  # (height, width); matches native 4:3 image aspect ratio, both dims multiples of 32
        # self.test_size = (640, 640)
        # self.num_classes = 34  # Set this to the number of classes in your dataset
        # self.data_num_workers = 14
        
        ### GERALD dataset configuration
        self.data_dir = "/data/GERALD-cropped/"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.test_ann = "instances_test2017.json"
        # GERALD dataset images are 1920x1080 (16:9, some 1280x720) — use this pair instead if switching datasets
        self.input_size = (704, 704)  # (height, width); matches 16:9 aspect ratio, both dims multiples of 32
        self.test_size = (704, 704)
        self.num_classes = 60  # Set this to the number of classes in your dataset
        self.data_num_workers = 6


