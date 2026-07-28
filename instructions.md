# YOLOX Training — Build & Run

Expected folder layout (this file's directory):

```
train/
├── Dockerfile_ME_Server
├── custom_training_scripts/
│   └── yolox_{s,m,l,x}_leaky_zeus[_export].py
└── instructions.md
```

## 1. Build the image

Build from inside this folder so `custom_training_scripts/` is part of the
build context — the Dockerfile copies those exp files into the image:

```bash
cd /path/to/train
sudo docker build -f Dockerfile_ME_Server --build-arg timezone=`cat /etc/timezone` -t yolox:v0 .
```

## 2. Run the container

```bash
sudo docker run --name yolox_training -it --gpus all --ipc=host \
  -v /path/to/host/data:/data \
  yolox:v0
```

`/path/to/host/data` should contain one subdirectory per dataset, each in
COCO format (`annotations/instances_{train,val,test}2017.json`, `images/`,
`labels/`, `classes.txt`). Mounting the parent directory once covers all of
them under `/data/<dataset_name>` inside the container.

If `yolox_training` already exists: `sudo docker rm -f yolox_training` first.

The exp files from `custom_training_scripts/` are already in the image at
`exps/default/` (copied in during build — see §1), so nothing else needs to
be copied in after the container starts.

## 3. Set the dataset in the exp file

Inside the container, edit `exps/default/yolox_{s,m,l,x}_leaky_zeus.py`:

```python
        self.data_dir = "/data/<dataset_name>/"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.test_ann = "instances_test2017.json"
        self.input_size = (<height>, <width>)   # both dims multiples of 32
        self.test_size = (<height>, <width>)
        self.num_classes = <num_classes>
        self.data_num_workers = 4
        self.warmup_epochs = 5
        self.max_epoch = 200
```

`num_classes` and `input_size`/`test_size` must match whichever dataset
`data_dir` points at — check its `classes.txt` and native image resolution
before training. Mixing up values left over from a previous dataset will
silently train/evaluate against the wrong class count.

This edit only affects the running container. If you want it to persist into
future containers, edit the matching file in `custom_training_scripts/` on
the host and rebuild the image (§1).

## 4. Train

```bash
cd /data/<dataset_name>
python /workspace/YOLOX/tools/train.py -f /workspace/YOLOX/exps/default/yolox_s_leaky_zeus.py \
  -d 1 -b 64 --fp16 -c /workspace/YOLOX/yolox_s.pth
```

Swap `s` for `m`/`l`/`x` (exp file and `-c yolox_{m,l,x}.pth` both change).
Run from inside `/data/...`, not `/workspace/YOLOX` — checkpoints land in
`./YOLOX_outputs/`, relative to the working directory, so this keeps them on
the host bind mount instead of the container's writable layer.
