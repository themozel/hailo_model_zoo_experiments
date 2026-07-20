#!/usr/bin/env python3
"""
Custom YOLOX experiment configuration for <class_count>-class signal detection.
"""
import os
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        
        # Model configuration
        self.depth = 0.33  # YOLOX-S depth multiplier
        self.width = 0.50  # YOLOX-S width multiplier
        self.num_classes = <class_count>  # Set this to the number of classes in your dataset
        
        # Dataset configuration
        self.data_dir = "/data/images_coco"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.test_ann = "instances_test2017.json"
        self.num_workers = 4
        
        # Training configuration
        self.max_epoch = 300
        self.eval_interval = 1
        self.input_size = (640, 640)
        
        # Learning rate and optimization
        self.warmup_epochs = 5
        self.basic_lr_per_img = 0.01 / 64.0
        
        # Batch size (adjust if OOM)
        self.batch_size = 8

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        """Return dataloaders for COCO."""
        from yolox.data import (
            COCODataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
        )

        def worker_init_fn(worker_id):
            import random
            import numpy as np
            random.seed(worker_id + self.seed)
            np.random.seed(worker_id + self.seed)

        local_rank = getattr(self, "local_rank", 0)

        train_data = COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=0.5,
                hsv_prob=0.5
            ) if not no_aug else None,
            cache=cache_img,
        )

        self.dataset = train_data

        if is_distributed:
            batch_sampler = YoloBatchSampler(
                InfiniteSampler(len(train_data), rank=local_rank, world_size=self.world_size),
                batch_size=batch_size,
                drop_last=False,
            )
        else:
            batch_sampler = YoloBatchSampler(
                InfiniteSampler(len(train_data), shuffle=True, seed=self.seed),
                batch_size=batch_size,
                drop_last=False,
            )

        dataloader_kwargs = {"num_workers": self.num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        dataloader_kwargs["worker_init_fn"] = worker_init_fn

        train_loader = DataLoader(train_data, **dataloader_kwargs)
        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        """Return evaluation dataloader."""
        from yolox.data import COCODataset, DataLoader

        local_rank = getattr(self, "local_rank", 0)

        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            img_size=self.input_size,
            preproc=None,
        )

        sampler = None
        if is_distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
            batch_size = batch_size

        val_loader = DataLoader(
            valdataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return val_loader
