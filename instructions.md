# YOLOX Retraining Reproduction Instructions

This file documents exactly what was done, in order, so you can repeat the same process.

## 1. Prepare and verify paths

1. Use this repository root:
   - `/home/themozel/Projects/master_thesis/hailo_model_zoo`
2. Use this dataset path (already converted to COCO format):
   - `/home/themozel/Projects/master_thesis/signal_detection/data/PERCEPT/images_coco`
3. Confirm the dataset has:
   - `annotations/instances_train2017.json`
   - `annotations/instances_val2017.json`
   - `annotations/instances_test2017.json`
   - `train2017/`, `val2017/`, `test2017/`

## 2. Build Docker image (with Python 3.11 compatibility fixes)

The final working Dockerfile is at:
- `training/yolox/Dockerfile`

Important changes that were required:

1. Base image changed to:
   - `pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel`
2. During YOLOX requirements install, these pins were removed/adjusted:
   - Remove `torch` line from YOLOX `requirements.txt`
   - Remove `onnx` line
   - Remove `onnxruntime` line
   - Replace pinned `numpy==1.21.2` with unpinned `numpy`

Exact build command used:

```bash
cd /home/themozel/Projects/master_thesis/hailo_model_zoo/training/yolox
sudo docker build --build-arg timezone=`cat /etc/timezone` -t yolox:v0 .
```

Expected successful end:

```text
Successfully built <image_id>
Successfully tagged yolox:v0
```

## 3. Run container with GPU + mounted dataset

Command used:

```bash
sudo docker run --name "zeus_training" -it --gpus all --ipc=host \
  -v /home/amo/zeus-training/master_thesis/data/GERALD:/data/GERALD \
  yolox:v0
```

If container name already exists, remove and rerun:

```bash
sudo docker rm yolox_training
sudo docker run --name "yolox_training" -it --gpus all --ipc=host \
  -v /home/amo/zeus-training/master_thesis/data/GERALD:/data/GERALD \
  yolox:v0
```

## 4. Verify GPU inside container

Inside container:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

Expected result:
- PyTorch 2.7.1+cu118
- CUDA True
- GPU name contains RTX 4070 SUPER

### Import torchvision missing in /workspace/YOLOX/yolox/utils/boxes.py 

## 5. Create custom YOLOX experiment file

Inside container from `/workspace/YOLOX`:

```bash
cp exps/default/yolox_s_leaky.py exps/default/yolox_s_signal.py
```

The copied file was short/minimal and then these lines were appended:

```python
        # Custom signal detection dataset config
        self.num_classes = 4
        self.data_dir = '/data/images_coco'
        self.train_ann = 'instances_train2017.json'
        self.val_ann = 'instances_val2017.json'
        self.test_ann = 'instances_test2017.json'
```

How this was appended:

```bash
cat >> exps/default/yolox_s_signal.py << 'EOF'
        # Custom signal detection dataset config
        self.num_classes = 4
        self.data_dir = '/data/GERALD/split_dataset'
        self.train_ann = 'instances_train2017.json'
        self.val_ann = 'instances_val2017.json'
        self.test_ann = 'instances_test2017.json'
EOF
```

Verify config content:

```bash
cat exps/default/yolox_s_signal.py
```

Make the coco.py file read from the data folder (yolo) instead of train2017, ...:
```bash
sed -i 's|img_file = os.path.join(self.data_dir, self.name, file_name)|img_file = os.path.join(self.data_dir, "images", file_name)|' /workspace/YOLOX/yolox/data/datasets/coco.py
```
Verify the change:
```bash
grep -A3 "def load_image" /workspace/YOLOX/yolox/data/datasets/coco.py
```

## 6. Fix runtime environment inside container (required)

Several runtime issues appeared and were fixed in this order.

### 6.1 Install YOLOX package editable

```bash
pip install -e . --no-build-isolation
```

### 6.2 Fix missing system libraries for OpenCV

Error seen:
- `ImportError: libGL.so.1: cannot open shared object file`

Fix:

```bash
apt-get update && apt-get install -y libgl1 libglib2.0-0
```

### 6.3 Fix protobuf/tensorboard incompatibility

Error seen:
- `TypeError: Descriptors cannot be created directly`

Fix:

```bash
pip install "protobuf<3.21"
```

### 6.4 Fix NumPy/TensorBoard incompatibility

Error seen:
- `AttributeError: module 'numpy' has no attribute 'bool8'`

Fix:

```bash
pip install "numpy<2"
```

### 6.5 Fix OpenCV ABI mismatch after NumPy changes

Error seen earlier:
- `numpy.core.multiarray failed to import`

Fix used:

```bash
pip install --upgrade opencv-python
```

Note: This produced a YOLOX requirement warning (opencv pinned in setup), but training proceeded successfully afterward.

## 7. Start training

Inside container:

```bash
python tools/train.py -f exps/default/yolox_s_signal.py -d 1 -b 8 -c yolox_s.pth
```

## 8. What successful start looked like

The run reached these key milestones:

1. Trainer argument print with `experiment_name='yolox_s_signal'`
2. Config table shows:
   - `num_classes = 4`
   - `data_dir = '/data/images_coco'`
   - `train_ann = 'instances_train2017.json'`
   - `val_ann = 'instances_val2017.json'`
3. Model summary printed
4. Pretrained checkpoint loaded (with expected class-head shape warnings)
5. COCO annotations loaded and indexed
6. Prefetcher initialized

At that point training was running.

## 9. Important notes from the actual run

1. The class-head mismatch warnings when loading `yolox_s.pth` are expected when changing from 80 classes to 4 classes.
2. Sudo prompts can pause commands in non-interactive contexts. If a command appears stuck, check whether a password prompt is waiting.
3. Reusing a container name causes Docker conflict error 125. Remove old container first.

## 10. Optional: commands to continue later

If you stop and want to continue with a fresh container:

```bash
sudo docker rm -f yolox_training 2>/dev/null || true
sudo docker run --name "yolox_training" -it --gpus all --ipc=host \
  -v /home/themozel/Projects/master_thesis/signal_detection/data/PERCEPT/images_coco:/data/images_coco \
  yolox:v0
```

Then re-run sections 5, 6, and 7 inside the container.

## 11. Inference

1. From outside of the container copy the yolox_infer_eval.py to the container
```bash
docker cp /home/ubuntu2404/mee119/hailo_model_zoo/yolox_infer_eval.py yolox_training:/workspace/YOLOX/exps/default/yolox_infer_eval.py
```

2. Run the inference script:
```bash
python exps/default/yolox_infer_eval.py   \
   --exp-file exps/default/yolox_s_signal.py   \
   --ckpt YOLOX_outputs/yolox_s_signal/best_ckpt.pth   \
   --input-dir /data/images_coco/test2017   \
   --ann-file /data/images_coco/annotations/instances_test2017.json   \
   --output-dir /data/images_coco/results_yolox_test   \
   --conf 0.25   \
   --nms 0.65   \
   --tsize 640   \
   --device cuda
```