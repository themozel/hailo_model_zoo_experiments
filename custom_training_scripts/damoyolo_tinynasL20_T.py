#!/usr/bin/env python3

import os

from damo.config import Config as MyConfig


class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.miscs.exp_name = os.path.split(
            os.path.realpath(__file__))[1].split('.')[0]
        self.miscs.eval_interval_epochs = 10
        self.miscs.ckpt_interval_epochs = 10
        # optimizer
        # batch_size=48, base_lr_per_img=0.01/64 -> effective peak LR 0.0075, deliberately matched
        # to both YOLOX's yolox_{s,m,l,x}_leaky_zeus.py (`-b 48`) here AND to the full-frame
        # GERALD training recipe (yolox_{s,m,l,x}_leaky_zeus and damoyolo_tinynasL{20_T,25_S,35_M}
        # all trained at batch=48 there too, see data/GERALD/{YOLOX,DAMOYOLO}_outputs/*/train_log.txt)
        # — so a cropped-vs-non-cropped accuracy comparison isolates cropping as the only
        # variable. (Previously batch_size=64/peak LR 0.01, matched only within GERALD-cropped —
        # see inference_results/yolox-vs-damoyolo-GERALD-cropped/comparison.md.)
        self.train.batch_size = 48
        self.train.base_lr_per_img = 0.01 / 64
        self.train.min_lr_ratio = 0.05
        self.train.weight_decay = 5e-4
        self.train.momentum = 0.9
        self.train.no_aug_epochs = 16
        self.train.warmup_epochs = 5
        self.train.total_epochs = 200
        self.train.finetune_path = '/workspace/DAMO-YOLO/damoyolo_tinynasL20_T_418.pth'

        # augment
        # GERALD
        # self.train.augment.transform.image_max_range = (640, 1152)
        # self.train.augment.mosaic_mixup.mosaic_size = (640, 1152)   # <-- add this; default (640,640) is square
        # Percept
        # self.train.augment.transform.image_max_range = (736, 992)
        # self.train.augment.mosaic_mixup.mosaic_size = (736, 992)   # <-- add this; default (640,640) is square
        # # Percept-cropped
        # self.train.augment.transform.image_max_range = (640, 640)
        # self.train.augment.mosaic_mixup.mosaic_size = (640, 640)   # <-- add this; default (640,640) is square
        # GERALD-cropped
        # self.train.augment.transform.image_max_range = (704, 704)
        # self.train.augment.mosaic_mixup.mosaic_size = (704, 704)   # <-- add this; default (640,640) is square
        # Zeus-cropped
        self.train.augment.transform.image_max_range = (704, 704)
        self.train.augment.mosaic_mixup.mosaic_size = (704, 704)   # <-- add this; default (640,640) is square
        self.train.augment.mosaic_mixup.mixup_prob = 1.0
        self.train.augment.mosaic_mixup.degrees = 10.0
        self.train.augment.mosaic_mixup.translate = 0.1
        self.train.augment.mosaic_mixup.shear = 2.0
        self.train.augment.mosaic_mixup.mosaic_scale = (0.1, 2.0)

        # GERALD
        # self.dataset.train_ann = ('gerald_coco_train', )
        # self.dataset.val_ann = ('gerald_coco_val', )
        # Percept
        # self.dataset.train_ann = ('percept_coco_train', )
        # self.dataset.val_ann = ('percept_coco_val', )
        # # Percept-cropped
        # self.dataset.train_ann = ('percept_cropped_coco_train', )
        # self.dataset.val_ann = ('percept_cropped_coco_val', )
        # GERALD-cropped
        # self.dataset.train_ann = ('gerald_cropped_coco_train', )
        # self.dataset.val_ann = ('gerald_cropped_coco_val', )
        # Zeus-cropped
        self.dataset.train_ann = ('zeus_cropped_coco_train', )
        self.dataset.val_ann = ('zeus_cropped_coco_val', )

        # GERALD-cropped, >=129-instance class subset (31 of 60 classes; rare classes like
        # ICE/LZB/Lf_2 were getting near-zero/undefined AP, see
        # inference_results/yolox-vs-damoyolo-GERALD-cropped/comparison.md). Needs
        # gerald_cropped_ge129_coco_{train,val,test} registered in paths_catalog.py —
        # see DAMOYOLO_KNOWN_ISSUES.md.
        # self.dataset.train_ann = ('gerald_cropped_ge129_coco_train', )
        # self.dataset.val_ann = ('gerald_cropped_ge129_coco_val', )

        # backbone
        structure = self.read_structure(
            './damo/base_models/backbones/nas_backbones/tinynas_L20_k1kx.txt')
        TinyNAS = {
            'name': 'TinyNAS_res',
            'net_structure_str': structure,
            'out_indices': (2, 4, 5),
            'with_spp': True,
            'use_focus': True,
            'act': 'relu',
            'reparam': True,
        }

        self.model.backbone = TinyNAS

        GiraffeNeckV2 = {
            'name': 'GiraffeNeckV2',
            'depth': 1.0,
            'hidden_ratio': 1.0,
            'in_channels': [96, 192, 384],
            'out_channels': [64, 128, 256],
            'act': 'relu',
            'spp': False,
            'block_name': 'BasicBlock_3x3_Reverse',
        }

        self.model.neck = GiraffeNeckV2

        ZeroHead = {
            'name': 'ZeroHead',
            # GERALD / GERALD-cropped
            # 'num_classes': 60,
            # GERALD-cropped, >=129-instance class subset
            # 'num_classes': 31,
            # # Percept
            # 'num_classes': 33,
            # Zeus-cropped
            'num_classes': 33,
            'in_channels': [64, 128, 256],
            'stacked_convs': 0,
            'reg_max': 16,
            'act': 'silu',
            'nms_conf_thre': 0.05,
            'nms_iou_thre': 0.7
        }
        self.model.head = ZeroHead

        # GERALD (60 classes)
        # self.dataset.class_names = ['El_6', 'Hectometer_Sign', 'Hp_0_HV', 'Hp_0_Ks', 'Hp_0_Sh', 'Hp_1', 'Hp_2', 'ICE', 'Ks_1', 'Ks_2', 'LZB', 'Lf_2', 'Lf_3', 'Lf_6', 'Lf_7', 'Mast_Sign_WRW', 'Mast_Sign_WYWYW', 'Mast_Sign_Y_Triangle', 'Ne_1', 'Ne_2', 'Ne_3_1', 'Ne_3_2', 'Ne_3_3', 'Ne_3_4', 'Ne_3_5', 'Ne_4', 'Ne_5', 'Ne_6', 'Ne_7a', 'Ne_7b', 'Platform_Display', 'Platform_Text_Sign', 'Platform_Track_Sign', 'Platform_Warn_Sign', 'Ra_10', 'Ride_Indicator_1', 'Ride_Indicator_Off', 'Sh_0', 'Sh_1', 'Sh_2', 'Sign_Back', 'Signal_Back', 'Signal_Identifier_Sign', 'Signal_Invalid', 'Signal_Off', 'So_20_Left', 'So_20_Right', 'Traffic_Light', 'Traffic_Sign', 'Vr_0', 'Vr_1', 'Vr_2', 'Wn_1', 'Wn_2', 'Zs_2', 'Zs_2v', 'Zs_3', 'Zs_3v', 'Zs_6', 'Zs_Off']

        # Zeus-cropped (33 classes)
        self.dataset.class_names = ['sig_stop', 'sig_stop_occupied', 'sig_free_straight', 'sig_free_left', 'sig_free_right', 'sig_switch_straight_locked', 'sig_switch_left_locked', 'sig_switch_right_locked', 'sig_switch_straight_free', 'sig_switch_left_free', 'sig_switch_right_free', 'sig_switch_faulty_1', 'sig_switch_faulty_2', 'sig_switch_faulty_3', 'sig_aux_arrow_right', 'sig_aux_arrow_left', 'sig_aux_arrow_right_diagonal', 'sig_aux_arrow_left_diagonal', 'sig_aux_arrow_straight', 'sig_aux_tram_num_arrow_straight', 'sig_aux_tram_num', 'sig_aux_tram_arrow_straight', 'sig_aux_tram_arrow_right', 'sig_aux_tram_arrow_left', 'sig_aux_tram_arrow_right_diagonal', 'sig_aux_tram_arrow_left_diagonal', 'sig_aux_tram', 'sig_aux_bus_num_arrow', 'sig_aux_bus_arrow_left', 'sig_aux_bus_arrow_right', 'sig_aux_bus', 'sig_aux_right_forward_arrow', 'overexposed']

        # GERALD-cropped, >=129-instance class subset (31 classes, remapped to contiguous
        # category IDs 1-31 in instances_{train,val,test}2017_ge129.json)
        # self.dataset.class_names = ['Hectometer_Sign', 'Hp_0_HV', 'Hp_0_Ks', 'Hp_0_Sh', 'Hp_1', 'Hp_2', 'Ks_1', 'Ks_2', 'Lf_6', 'Lf_7', 'Mast_Sign_WRW', 'Mast_Sign_Y_Triangle', 'Ne_2', 'Ne_3_1', 'Ne_3_2', 'Ne_3_3', 'Ne_5', 'Platform_Display', 'Platform_Text_Sign', 'Platform_Track_Sign', 'Platform_Warn_Sign', 'Sign_Back', 'Signal_Back', 'Signal_Identifier_Sign', 'Signal_Off', 'Traffic_Sign', 'Vr_0', 'Vr_1', 'Vr_2', 'Zs_3', 'Zs_Off']

        # Percept (33 classes)
        # self.dataset.class_names = ['obs_animal', 'obs_bus', 'obs_car', 'obs_cyclist', 'obs_motorcyclist', 'obs_person', 'obs_stroller', 'obs_tram', 'sig_free_left', 'sig_free_right', 'sig_free_straight', 'sig_max_curve_speed', 'sig_max_speed_12', 'sig_max_speed_18', 'sig_max_speed_24', 'sig_max_speed_30', 'sig_max_speed_36', 'sig_max_speed_42', 'sig_max_speed_48', 'sig_max_speed_60', 'sig_stop', 'sig_switch_left_free', 'sig_switch_left_locked', 'sig_switch_right_free', 'sig_switch_right_locked', 'sig_switch_straight_free', 'sig_switch_straight_locked', 'sw_cross_left', 'sw_cross_right', 'sw_forward_left', 'sw_forward_right', 'sw_reverse_left', 'sw_reverse_right', 'sw_unknown']
