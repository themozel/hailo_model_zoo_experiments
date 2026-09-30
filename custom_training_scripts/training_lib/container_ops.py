"""docker exec/cp plumbing for the two long-lived training containers.

See ../DAMOYOLO_KNOWN_ISSUES.md and ../../instructions.md for the manual
workflow this automates: /workspace/{YOLOX,DAMO-YOLO} are baked into each
image (not bind-mounted), so exp/config files have to be docker cp'd in, and
DAMO-YOLO additionally needs new datasets registered in its
paths_catalog.py before they can be referenced from a config.
"""
from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from .dataset_prep import DatasetInfo

CONTAINER_NAMES = {"yolox": "yolox_training", "damoyolo": "damoyolo_training2"}
CONTAINER_CONFIG_DIR = {
    "yolox": "/workspace/YOLOX/exps/default",
    "damoyolo": "/workspace/DAMO-YOLO/configs",
}
TRAIN_ENTRYPOINT = {
    "yolox": "python /workspace/YOLOX/tools/train.py",
    "damoyolo": (
        "python -m torch.distributed.launch --nproc_per_node=1 "
        "/workspace/DAMO-YOLO/tools/train.py"
    ),
}
PATHS_CATALOG_FILE = "/workspace/DAMO-YOLO/damo/config/paths_catalog.py"


class ContainerError(RuntimeError):
    pass


def _run(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def container_running(name: str) -> bool:
    result = _run(["docker", "inspect", "-f", "{{.State.Running}}", name])
    return result.returncode == 0 and result.stdout.strip() == "true"


def docker_cp(src: Path, container: str, dest: str) -> None:
    result = _run(["docker", "cp", str(src), f"{container}:{dest}"])
    if result.returncode != 0:
        raise ContainerError(f"docker cp {src} -> {container}:{dest} failed:\n{result.stderr}")


def exec_python(container: str, code: str, timeout: int | None = None) -> subprocess.CompletedProcess:
    return _run(["docker", "exec", "-i", container, "python3", "-c", code], timeout=timeout)


def _argv_to_shell(argv: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)


def exec_stream(container: str, workdir: str, argv: list[str]) -> int:
    """Run argv inside the container, streaming stdout/stderr live. Returns exit code."""
    cmd = ["docker", "exec", "-w", workdir, container, "bash", "-lc", _argv_to_shell(argv)]
    return subprocess.run(cmd).returncode


def exec_capture(
    container: str, workdir: str, argv: list[str], timeout: int
) -> subprocess.CompletedProcess:
    cmd = ["docker", "exec", "-w", workdir, container, "bash", "-lc", _argv_to_shell(argv)]
    return _run(cmd, timeout=timeout)


def ensure_damoyolo_dataset_registered(dataset_info: DatasetInfo) -> None:
    container = CONTAINER_NAMES["damoyolo"]
    name = dataset_info.name
    train_key = f"{name}_coco_train"

    check = exec_python(
        container,
        f"print({train_key!r} in open({PATHS_CATALOG_FILE!r}).read())",
        timeout=30,
    )
    if check.returncode != 0:
        raise ContainerError(f"could not read paths_catalog.py in {container}:\n{check.stderr}")
    if check.stdout.strip() == "True":
        return

    entries = "".join(
        f"        {name + '_coco_' + split!r}: "
        f"{{'img_dir': {dataset_info.container_path + '/images'!r}, "
        f"'ann_file': {dataset_info.container_path + f'/annotations/instances_{split}2017.json'!r}}},\n"
        for split in ("train", "val", "test")
    )
    patch_code = f'''
path = {PATHS_CATALOG_FILE!r}
text = open(path).read()
marker = "DATASETS = {{"
idx = text.index(marker) + len(marker)
entries = {entries!r}
open(path, "w").write(text[:idx] + "\\n" + entries + text[idx:])
print("patched")
'''
    result = exec_python(container, patch_code, timeout=30)
    if result.returncode != 0 or "patched" not in result.stdout:
        raise ContainerError(
            f"failed to register {name} in paths_catalog.py inside {container}:\n{result.stderr}"
        )
    print(f"[container] registered {name}_coco_{{train,val,test}} in {container}'s paths_catalog.py")


def discover_damoyolo_workers_opt(dataset_info: DatasetInfo) -> str | None:
    """Best-effort search for a dataloader-worker-count attribute on DAMO-YOLO's Config.

    DAMO-YOLO's source only exists inside the container (cloned fresh at image
    build, see DAMOYOLO_KNOWN_ISSUES.md), so the exact attribute path can't be
    hardcoded from the host — it's discovered live instead. Returns a dotted
    path suitable for `config.merge(opts)` (e.g. "train.num_workers"), or None
    if nothing was found (worker count is then left at the framework default).
    """
    container = CONTAINER_NAMES["damoyolo"]
    code = '''
import json
from damo.config.base import Config

found = []

def walk(obj, prefix, depth=0):
    if depth > 3:
        return
    for k in dir(obj):
        if k.startswith("_"):
            continue
        try:
            v = getattr(obj, k)
        except Exception:
            continue
        if callable(v):
            continue
        if "work" in k.lower() and isinstance(v, int):
            found.append(prefix + k)
        elif hasattr(v, "__dict__"):
            walk(v, prefix + k + ".", depth + 1)

walk(Config(), "")
print(json.dumps(found))
'''
    result = exec_python(container, code, timeout=60)
    if result.returncode != 0:
        return None
    try:
        found = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return None
    return found[0] if found else None


def _build_worker_probe_script(images_dir: str, batch_size: int, height: int, width: int) -> str:
    return f'''
import json, os, time

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

IMG_DIR = {images_dir!r}
BATCH = {batch_size}
SIZE_HW = ({height}, {width})

paths = [
    os.path.join(IMG_DIR, f) for f in sorted(os.listdir(IMG_DIR))
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"))
]
if not paths:
    print(json.dumps({{"error": f"no images found under {{IMG_DIR}}"}}))
    raise SystemExit(0)


class Probe(Dataset):
    def __len__(self):
        return len(paths)

    def __getitem__(self, idx):
        with Image.open(paths[idx % len(paths)]) as im:
            im = im.convert("RGB").resize((SIZE_HW[1], SIZE_HW[0]))
        arr = np.asarray(im, dtype="float32").transpose(2, 0, 1).copy()
        return torch.from_numpy(arr)


cpu_count = os.cpu_count() or 1
candidates = sorted({{0, 2, 4, max(cpu_count - 1, 1), cpu_count}})
candidates = [c for c in candidates if 0 <= c <= cpu_count]

shm_bytes = None
try:
    st = os.statvfs("/dev/shm")
    shm_bytes = st.f_frsize * st.f_blocks
except OSError:
    pass

results = []
for workers in candidates:
    try:
        loader = DataLoader(Probe(), batch_size=BATCH, num_workers=workers, shuffle=True)
        it = iter(loader)
        n_batches = min(15, max(1, len(paths) // max(BATCH, 1)))
        start = time.time()
        for _ in range(n_batches):
            next(it)
        elapsed = time.time() - start
        del it, loader
        results.append({{"workers": workers, "ok": True, "batches_per_sec": n_batches / elapsed}})
    except Exception as exc:
        results.append({{"workers": workers, "ok": False, "error": f"{{type(exc).__name__}}: {{exc}}"}})

print(json.dumps({{"cpu_count": cpu_count, "shm_bytes": shm_bytes, "results": results}}))
'''


def probe_workers(family: str, dataset_info: DatasetInfo, batch_size: int) -> dict:
    """CPU-only DataLoader worker-count sweep, run inside the target container.

    Builds a minimal-but-real Dataset (JPEG/PNG decode + resize, no CUDA
    anywhere) against the dataset's actual train images at the real batch
    size, so the timing/crash behavior reflects the real I/O and CPU-decode
    cost without depending on either framework's internal Dataset class
    (DAMO-YOLO's isn't inspectable from the host at all — see module docstring).
    """
    container = CONTAINER_NAMES[family]
    images_dir = f"{dataset_info.container_path}/images/train"
    height, width = dataset_info.input_size
    script = _build_worker_probe_script(images_dir, batch_size, height, width)
    result = exec_python(container, script, timeout=600)
    if result.returncode != 0:
        raise ContainerError(f"worker probe crashed in {container}:\n{result.stderr}")
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise ContainerError(
            f"worker probe produced no parseable output:\n{result.stdout}\n{result.stderr}"
        ) from exc


def pick_best_worker_count(probe_result: dict) -> tuple[int, str]:
    if "error" in probe_result:
        return 0, f"probe could not run ({probe_result['error']}); falling back to workers=0"
    ok = [r for r in probe_result["results"] if r["ok"]]
    if not ok:
        shm = probe_result.get("shm_bytes")
        shm_msg = f" — /dev/shm is only {shm / 1e6:.0f}MB, likely the cause" if shm else ""
        return 0, f"all multi-worker candidates failed{shm_msg}; falling back to workers=0"
    best = max(ok, key=lambda r: r["batches_per_sec"])
    return best["workers"], f"picked workers={best['workers']} ({best['batches_per_sec']:.1f} batches/s)"
