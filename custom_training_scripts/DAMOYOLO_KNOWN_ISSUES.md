# DAMO-YOLO: environment setup, training, and HEF conversion

## 1. Container setup (reapply on every new `damoyolo_training2`-style container)

`/workspace/DAMO-YOLO` is cloned fresh at container build time (not
bind-mounted), so these patches don't persist across containers.

```bash
# --local-rank argparse fix (torch >=2.x injects --local-rank, repo only accepts --local_rank)
sed -i "s/parser.add_argument('--local_rank', type=int, default=0)/parser.add_argument('--local_rank', '--local-rank', type=int, default=0)/" \
    /workspace/DAMO-YOLO/tools/train.py

# torchvision affine() fix (resample/fillcolor removed in torchvision >=0.17)
sed -i "s/resample=2,/interpolation=transforms.InterpolationMode.BILINEAR,/; s/fillcolor=tuple(\[int(i) for i in pixel_mean\]))/fill=tuple([int(i) for i in pixel_mean]))/" \
    /workspace/DAMO-YOLO/damo/augmentations/box_level_augs/geometric_augs.py

# pycocotools np.float fix (removed in numpy >=1.24)
sed -i 's/dtype=np\.float)/dtype=float)/g' \
    /opt/conda/lib/python3.11/site-packages/pycocotools/cocoeval.py

# ONNX export fix (torch.onnx._export removed in torch 2.x)
sed -i 's/torch\.onnx\._export(/torch.onnx.export(/' \
    /workspace/DAMO-YOLO/tools/converter.py
```

Add custom dataset entries to
`/workspace/DAMO-YOLO/damo/config/paths_catalog.py`, in `DatasetCatalog.DATASETS`,
right before the closing `}` (after `coco_2017_test_dev`):

```python
        'gerald_coco_train': {
            'img_dir': '/data/GERALD/images',
            'ann_file': '/data/GERALD/annotations/instances_train2017.json'
        },
        'gerald_coco_val': {
            'img_dir': '/data/GERALD/images',
            'ann_file': '/data/GERALD/annotations/instances_val2017.json'
        },
        'percept_coco_train': {
            'img_dir': '/data/percept/images',
            'ann_file': '/data/percept/annotations/instances_train2017.json'
        },
        'percept_coco_val': {
            'img_dir': '/data/percept/images',
            'ann_file': '/data/percept/annotations/instances_val2017.json'
        },
```

(`/data/GERALD`, `/data/percept` = bind-mounted
`/home/amo/zeus-training/master_thesis/data/{GERALD,percept}`.)

## 2. Exp configs (already applied to all 3: `damoyolo_tinynasL20_T.py`,
`damoyolo_tinynasL25_S.py`, `damoyolo_tinynasL35_M.py`)

- `train.batch_size = 32` (single-GPU; was 256, tuned for 8 GPUs)
- `train.total_epochs = 200`
- `dataset.train_ann`/`val_ann` = `gerald_coco_{train,val}` (Percept
  equivalents commented directly below — toggle both together)
- `model.head['num_classes'] = 60`, `dataset.class_names` = GERALD's 60
  classes (Percept's 33 commented below — toggle together with the above)

After any host-side edit to a config file, sync it into the running container:

```bash
docker cp /home/amo/zeus-training/hailo_model_zoo/custom_training_scripts/<config>.py \
    damoyolo_training2:/workspace/DAMO-YOLO/configs/<config>.py
```

## 3. Train

```bash
python -m torch.distributed.launch --nproc_per_node=1 \
    tools/train.py -f configs/<config>.py
```

## 4. Export to ONNX

```bash
python tools/converter.py -f configs/<config>.py \
    -c workdirs/<config>/latest_ckpt.pth \
    --batch_size 1 --img_size 640
```

## 5. Compile to HEF

`conda activate hailo_dfc` (or use the `hailomz` shell function in
`~/.bashrc`, which runs it from any active conda env).

Custom yamls already exist with the correct node names baked in:
`custom_training_scripts/damoyolo_tinynasL20_T_gerald.yaml`,
`..._L25_S_gerald.yaml`, `..._L35_M_gerald.yaml`
(`parser.nodes`: start `null`, end
`[/head/Softmax, /head/Sigmoid, /head/Softmax_1, /head/Sigmoid_1, /head/Softmax_2, /head/Sigmoid_2]`;
`evaluation.classes: 60`).

```bash
hailomz compile \
    --ckpt <variant>.onnx \
    --calib-path /home/amo/zeus-training/master_thesis/data/GERALD/images/train \
    --yaml custom_training_scripts/<variant>_gerald.yaml \
    --hw-arch hailo8
```

`damoyolo_tinynasL25_S` needs one extra flag — it's the only variant with a
hardware-tuned placement script (`cfg/alls/hailo8/base/damoyolo_tinynasL25_S.alls`),
whose hand-picked layer names don't match our retrained model and crash the
allocator. Its **generic** alls also has a problem: unlike the other two
variants' generic alls files, it sets `performance_param(optimization_level=max)`,
which makes the compiler search for the best possible placement instead of a
merely-valid one — took 2.5+ hours and counting before being killed, vs. ~25min
for the other two variants. Use a trimmed alls script instead, checked in at
`custom_training_scripts/damoyolo_tinynasL25_S_gerald.alls`
(just `post_quantization_optimization(finetune, policy=disabled)`, no
placement hints, no `optimization_level=max`):

```bash
hailomz compile \
    --ckpt damoyolo_tinynasL25_S.onnx \
    --calib-path /home/amo/zeus-training/master_thesis/data/GERALD/images/train \
    --yaml custom_training_scripts/damoyolo_tinynasL25_S_gerald.yaml \
    --hw-arch hailo8 \
    --model-script custom_training_scripts/damoyolo_tinynasL25_S_gerald.alls
```

**Note:** the compiled HEF's raw output is pre-decode (per-scale `Softmax`
box-regression distribution `[1,4,H,W,17]` + `Sigmoid` class scores
`[1,61,H,W]`, 3 scales). Decoding to real boxes (softmax-weighted-sum ×
stride → `distance2bbox`, slice sigmoid to 60 classes, then NMS) has to be
done in software after inference — not handled by this yaml's
`postprocessing` block as-is.

### 5a. Always set explicit `resources_param` caps — don't rely on the default/auto-retry path

This host has no GPU available to the `hailo_dfc` toolchain (model
optimization always falls back to `optimization_level=0` and the allocator
runs CPU-only), and only ~23GB RAM total. Left at its defaults, the allocator
first tries a 60% utilization target, times out after ~10-12min if that
fails, then automatically retries at 100% utilization (`max_control_utilization`
etc. all at 100%) — and at 100%, the "Trying parallel splits" search step
spawns multiple worker subprocesses exploring cluster placements concurrently,
which for the larger variants (`damoyolo_tinynasL35_M` especially, at the real
640x1152 training resolution rather than the stock 640x640) can blow past
23GB and get OOM-killed outright (`dmesg`: `Out of memory: Killed process
... (hailomz)`). This happened twice in a row for `damoyolo_tinynasL35_M`
before switching approach.

**Fix:** always add an explicit `resources_param(...)` line to the model
script instead of letting it default-then-retry. Start around
`max_utilization=0.85` (worked fine for `damoyolo_tinynasL25_S`, checked in at
`custom_training_scripts/damoyolo_tinynasL25_S_gerald_loose.alls`); if that
still OOMs for a bigger variant, go tighter —
`custom_training_scripts/damoyolo_tinynasL35_M_gerald_tight.alls` uses
`max_utilization=0.65, max_compute_utilization=0.65,
max_compute_16bit_utilization=0.65, max_memory_utilization=0.55,
max_input_aligner_utilization=0.65, max_apu_utilization=0.65` and finished
the partition search in ~13 minutes (154 iterations) with memory to spare,
vs. hanging/OOMing for 2+ hours at the 85%/100% caps. Tighter caps mean a
smaller search space (and a possibly-less-optimal-but-still-valid partition),
not a failure — don't be afraid to go tight on a memory-constrained host.

**If a compile has already gotten through model optimization/quantization
(the "Quantization-Aware Fine-Tuning" stage, which can itself take 1-2h on
this host and is NOT the part that OOMs) and only fails/hangs at the
allocation stage, don't restart from `--ckpt` and redo the fine-tuning** —
the HAR saved right after optimization (`<network_name>.hef`'s sibling
`<network_name>.har` in the cwd `hailomz` was run from) already has the
quantized model baked in. Resume straight into allocation/compile with a new
(tighter) model script instead:

```bash
hailomz compile \
    --har <network_name>.har \
    --yaml custom_training_scripts/<variant>_gerald.yaml \
    --hw-arch hailo8 \
    --model-script custom_training_scripts/<variant>_gerald_tight.alls
```

This skips straight to "Loading network parameters" / "Starting Hailo
allocation and compilation flow", saving the ~1-2h optimization stage
entirely on a retry.

## 6. Writing a custom yaml for another model

Copy `hailo_model_zoo/cfg/networks/<model>.yaml` to
`custom_training_scripts/<model>_gerald.yaml` and set:

- `evaluation.classes: <n>`
- `parser.nodes: [null, [<end nodes>]]` — find the raw, per-scale,
  **pre-decode** node names (not the final decoded/concatenated output) by
  loading the ONNX and looking for the per-scale activation ops scoped under
  the head:
  ```python
  import onnx
  m = onnx.load('model.onnx')
  for n in m.graph.node:
      if n.op_type in ('Softmax', 'Sigmoid') and '/head/' in n.name:
          print(n.name, n.op_type, list(n.output))
  ```
  For `ZeroHead`-based models (all 3 DAMO-YOLO variants here) this is one
  `Softmax`+`Sigmoid` pair per FPN scale. A different head architecture will
  have different op types/names for its raw per-scale outputs — same idea
  (find outputs computed once per scale, before any cross-scale concat).
- Keep `network.network_name` unchanged from the stock yaml **unless** no
  hardware-specific alls exists for this model at `cfg/alls/hailo8/base/<model>.alls`
  — if one does exist, it hardcodes `network_group([<stock_name>])`, and
  renaming breaks it.

Validate with `hailomz parse` (fast, no calibration needed):

```bash
hailomz parse --ckpt model.onnx --yaml custom_training_scripts/<model>_gerald.yaml
```

A clean pass with no "retrying parsing" message confirms the end nodes are
correct at the graph-translation level — **but this alone is not enough.**
Only a full `hailomz compile` (with real calibration images) confirms the
graph is actually mappable onto Hailo hardware; parse-clean node names can
still fail at compile time in two ways seen here:
- Decoded/concatenated end nodes instead of raw per-scale ones →
  `Slice`/`Concat` kernel-validation errors during allocation.
- A hardware-tuned alls script referencing layer names from the stock
  checkpoint → `do not exist in the HN` or an internal `map::at` crash during
  multi-context partitioning. Fix: `--model-script
  cfg/alls/generic/<model>.alls` to bypass the hand-tuned placement.
