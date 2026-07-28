#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.act = 'lrelu'
        self.num_classes = 60
        self.test_size = (640, 1152)
        self.input_size = (640, 1152)
