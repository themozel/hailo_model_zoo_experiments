# YOLOX Retraining & Hailo Export — Reproduction Notes

This documents the actual working setup for retraining YOLOX on the GERALD /
Percept signal-detection datasets and exporting to Hailo `.hef`. It reflects
what is verified to work on this machine now, not the exploratory dead ends
that came before it.

## 1. Paths

- Repo root: `/home/amo/zeus-training/hailo_model_zoo`
- Datasets (COCO-format), both under one directory so a single bind mount covers them:
  - `/home/amo/zeus-training/master_thesis/data/GERALD` — 60 classes
  - `/home/amo/zeus-training/master_thesis/data/percept` — 33 classes
- Each dataset dir has: `annotations/instances_{train,val,test}2017.json`, `images/`, `labels/`, `classes.txt`, `data.yaml`.

## 2. Build the Docker image

The Dockerfile committed at `training/yolox/Dockerfile` is still the
**unmodified upstream Hailo file** (base `nvcr.io/nvidia/pytorch:21.10-py3`).
That is not what actually produced the `yolox:v0` image in use — the image
history shows it was built from a modified Dockerfile with these changes on
top of the stock one:

1. Base image: `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel` (needed for the
   RTX PRO 6000 Blackwell GPU on this machine — the stock `nvcr.io` 21.10 base
   is too old to support it).
2. `apt-get install` list extended with `libgl1 libglib2.0-0` (avoids
   OpenCV's `ImportError: libGL.so.1: cannot open shared object file`).
3. Before `pip install -r requirements.txt` on the cloned `hailo-ai/YOLOX`
   repo:
   - `pip install --upgrade pip`
   - `sed -i '/^torch/d' requirements.txt`
   - `sed -i '/^onnx/d' requirements.txt`
   - `sed -i '/^onnxruntime/d' requirements.txt`
   - `sed -i 's/^numpy==.*/numpy<2/' requirements.txt`
4. After the requirements install: `pip install cython==0.29.24 'protobuf<3.21'`
   (avoids `TypeError: Descriptors cannot be created directly` from a
   tensorboard/protobuf version clash).
5. `pip install -e . --no-build-isolation` (build isolation must be off so
   YOLOX's `setup.py` sees the base image's preinstalled torch).

**Known gap:** `training/yolox/Dockerfile` in git needs these changes applied
before a fresh `docker build` will reproduce the current environment. Until
that's done, don't trust `docker build` from the committed file alone.

Build command (once the Dockerfile above is in place):

```bash
cd /home/amo/zeus-training/hailo_model_zoo/training/yolox
sudo docker build --build-arg timezone=`cat /etc/timezone` -t yolox:v0 .
```

## 3. Run the container

One bind mount covers both datasets:

```bash
sudo docker run --name yolox_training -it --gpus all --ipc=host \
  -v /home/amo/zeus-training/master_thesis/data:/data \
  yolox:v0
```

Inside the container this gives `/data/GERALD` and `/data/percept`.

If a container named `yolox_training` already exists:

```bash
sudo docker rm -f yolox_training
# then re-run the docker run command above
```

## 4. Verify GPU inside the container

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

Expected: `PyTorch 2.7.1+cu128`, `CUDA: True`, GPU name contains `RTX PRO 6000 Blackwell`.

## 5. One-time repo patch (already applied in the current image/container)

`yolox/data/datasets/coco.py` loads images from `<data_dir>/<split_name>/` by
default. This dataset layout instead uses a flat `images/` dir, so:

```bash
sed -i 's|img_file = os.path.join(self.data_dir, self.name, file_name)|img_file = os.path.join(self.data_dir, "images", file_name)|' \
  /workspace/YOLOX/yolox/data/datasets/coco.py
```

Verify: `grep -A3 "def load_image" /workspace/YOLOX/yolox/data/datasets/coco.py`

If you rebuild from a fresh container this needs to be re-applied (or baked
into the Dockerfile as a `RUN sed -i ...` step).

## 6. Create/edit the experiment file

Per model size, `exps/default/yolox_{s,m,l,x}_leaky_zeus.py` is a copy of the
matching stock `yolox_{s,m,l,x}_leaky.py` template with a dataset config block
appended. Host-side backups of these live in
`hailo_model_zoo/custom_training_scripts/`. Pattern (from the `s` model):

```python
        # Dataset configuration
        self.data_dir = "/data/GERALD/"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.test_ann = "instances_test2017.json"
        # GERALD dataset images are 1920x1080 (16:9, some 1280x720)
        self.input_size = (640, 1152)  # (height, width), both dims multiples of 32
        self.test_size = (640, 1152)
        self.num_classes = 60
        self.data_num_workers = 4

        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 200
```

**Switching dataset**: GERALD is 60 classes at 1920x1080 (use `input_size =
(640, 1152)`); Percept is **33 classes** at 1280x736 (use `input_size = (736,
992)`). `num_classes` and `input_size`/`test_size` must change together — an
exp file pointed at `/data/percept/` with `num_classes = 60` left over from a
GERALD run is wrong (percept only has 33 classes in `classes.txt`) and will
silently train/evaluate against the wrong class count.

Verify: `cat exps/default/yolox_s_leaky_zeus.py`

## 7. Train

```bash
python tools/train.py -f exps/default/yolox_s_leaky_zeus.py -d 1 -b 64 --fp16 -c yolox_s.pth
```

Swap `s` for `m`/`l`/`x` (exp file and `-c yolox_{m,l,x}.pth` both change;
`yolox_{s,m,l,x}.pth` pretrained weights are already in `/workspace/YOLOX/`
from the Docker build).

## 8. What a successful start looks like

1. Trainer argument print with the matching `experiment_name`.
2. Config table shows the `num_classes`, `data_dir`, `train_ann`, `val_ann` you expect.
3. Model summary printed.
4. Pretrained checkpoint loaded (class-head shape warnings are expected —
   going from 80 COCO classes to 60/33 custom classes).
5. COCO annotations loaded and indexed, prefetcher initialized.

## 9. Persisting checkpoints across container restarts (important)

`tools/train.py` writes checkpoints to `./YOLOX_outputs/<exp_name>/`,
**relative to the current working directory**, not to `/data`. If you launch
training from `/workspace/YOLOX` (the container's default `WORKDIR`), outputs
land in the container's writable layer and are **lost if the container is
removed**.

To keep checkpoints on the host through the `/data` bind mount, `cd` into the
mounted dataset dir first and reference the exp file by full path:

```bash
cd /data/GERALD
python /workspace/YOLOX/tools/train.py -f /workspace/YOLOX/exps/default/yolox_s_leaky_zeus.py \
  -d 1 -b 64 --fp16 -c /workspace/YOLOX/yolox_s.pth
```

This is how the completed GERALD runs (`/data/GERALD/YOLOX_outputs/yolox_{s,m,l,x}_leaky_zeus/`,
each with `best_ckpt.pth` and tfevents) ended up persisted on the host.

## 10. Evaluate a checkpoint

```bash
python tools/eval.py -f exps/default/yolox_s_leaky_zeus.py \
  -c /data/GERALD/YOLOX_outputs/yolox_s_leaky_zeus/best_ckpt.pth \
  -b 32 -d 1 --test
```

Point `-f`/`-c` at whichever exp file and checkpoint match the model/dataset
you're evaluating.

## 11. Export to ONNX and compile to HEF

Export uses a separate, minimal exp file per size
(`exps/default/yolox_{s,m,l,x}_leaky_zeus_export.py` — same
depth/width/activation/num_classes as the training exp file, but a fixed
export `input_size`/`test_size`, no dataset config):

```bash
python tools/export_onnx.py --output-name yolox_s_leaky_zeus.onnx \
  -f exps/default/yolox_s_leaky_zeus_export.py \
  -c /data/GERALD/YOLOX_outputs/yolox_s_leaky_zeus/best_ckpt.pth
```

Compiling the ONNX to `.hef` for Hailo-8/8L needs the proprietary Hailo
Dataflow Compiler and the `hailomz` CLI from a `hailo_model_zoo` checkout on
the `hailo8-migration` branch (the mainline branch only targets
Hailo-10H/15H/15L). The network/`.alls`/NMS-config files for all four sizes
already exist under `hailo_model_zoo/cfg/`.

**Full step-by-step status, the exact `hailomz compile` command, and the
open questions around NMS layer names for the `m`/`x` configs are documented
in `training/YOLOX_outputs/hef_compiled/STATUS.md` under `master_thesis/data/GERALD/`
— read that instead of duplicating it here, since it's the live record of
what's done vs. still blocked for this step.**

## 12. Known gaps

- `training/yolox/Dockerfile` in git doesn't match the image actually in use (see §2).
- The `yolox_infer_eval.py` script previously used for batch inference +
  metrics (`results_yolox_test_{s,m}/metrics_report.json` under
  `master_thesis/inference_results/`) is no longer present anywhere on this
  machine. Locate/restore it before repeating that step — `tools/eval.py`
  (§10) still works for COCO-style mAP, but per-class/latency reporting used
  that missing script.

## 13. General notes

- `sudo` prompts can silently pause commands in non-interactive contexts —
  if a command looks stuck, check for a password prompt.
- Reusing a container name causes Docker error 125; remove the old one first (§3).
- Class-head mismatch warnings on checkpoint load are expected whenever
  `num_classes` differs from the pretrained weights' 80.
