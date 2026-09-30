#!/usr/bin/env python3
"""One-shot training launcher for the DAMO-YOLO / YOLOX custom training containers.

    run_training.py --model yolox_s --dataset zeus-cropped --batch-size 48 --epochs 200

Given a model + dataset + batch size + epoch count, this:
  1. validates the dataset under --data-root/<dataset> and splits/reformats it
     into COCO format if needed (see training_lib/dataset_prep.py for exactly
     what layouts it understands),
  2. generates the right exp/config file for the model+dataset combo and
     copies it into the right container (yolox_training / damoyolo_training2),
  3. runs a CPU-only DataLoader worker-count sweep to pick --workers,
  4. runs a short real (GPU) smoke run to confirm --batch-size actually fits,
     aborting with a clear message instead of launching a long run that OOMs,
  5. launches the real run, streaming its output.

Assumes both containers already exist and are running (see
../instructions.md and ../DAMOYOLO_KNOWN_ISSUES.md for how they were built),
with the dataset's host parent directory bind-mounted at /data inside each.

Two things this script assumes but can't verify without touching a live
container (flagged here so a first real run is easy to sanity-check):
  - DAMO-YOLO's `config.merge(opts)` accepts dotted keys ("train.total_epochs
    200"), matching the dotted attribute style already used throughout
    damo/config — same convention as YOLOX's flat `exp.merge(opts)`.
  - Checkpoints land under the /data/<dataset> bind mount rather than the
    container's writable layer because training runs with that as the
    working directory (confirmed for YOLOX in ../instructions.md; DAMO-YOLO's
    own `miscs.output_dir` should have the same effect but wasn't checked
    against its actual `workdirs/` output logic).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training_lib import container_ops, dataset_prep, exp_templates

DEFAULT_DATA_ROOT = Path("/home/amo/zeus-training/master_thesis/data")
GENERATED_DIR = Path(__file__).resolve().parent / "generated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, choices=sorted(exp_templates.MODEL_FAMILY))
    parser.add_argument("--dataset", required=True, help="name of a subdir under --data-root")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--split-ratio",
        default="0.7,0.2,0.1",
        help="train,val,test — only used if the dataset needs an initial split",
    )
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument(
        "--force-reformat",
        action="store_true",
        help="regenerate COCO annotations / the exp-config file even if they look up to date",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="do dataset prep and generate the config, then stop before touching docker",
    )
    return parser.parse_args()


def parse_split_ratio(raw: str) -> tuple[float, float, float]:
    parts = [float(x) for x in raw.split(",")]
    if len(parts) != 3 or abs(sum(parts) - 1.0) > 1e-6:
        raise SystemExit(f"--split-ratio must be three numbers summing to 1.0, got {raw!r}")
    return parts[0], parts[1], parts[2]


def build_train_argv(
    family: str,
    config_path_in_container: str,
    batch_size: int,
    epochs: int,
    model_key: str,
    workers_opt: list[str],
) -> list[str]:
    if family == "yolox":
        ckpt = exp_templates.YOLOX_MODELS[model_key]["ckpt"]
        return [
            *container_ops.TRAIN_ENTRYPOINT["yolox"].split(),
            "-f", config_path_in_container,
            "-d", "1",
            "-b", str(batch_size),
            "-c", f"/workspace/YOLOX/{ckpt}",
            *workers_opt,
            "max_epoch", str(epochs),
        ]
    return [
        *container_ops.TRAIN_ENTRYPOINT["damoyolo"].split(),
        "-f", config_path_in_container,
        *workers_opt,
        "train.total_epochs", str(epochs),
    ]


def main() -> None:
    args = parse_args()
    split_ratio = parse_split_ratio(args.split_ratio)
    family = exp_templates.MODEL_FAMILY[args.model]
    container = container_ops.CONTAINER_NAMES[family]

    print(
        f"[plan] model={args.model} ({family}) dataset={args.dataset} "
        f"batch_size={args.batch_size} epochs={args.epochs} container={container}"
    )

    try:
        info = dataset_prep.ensure_dataset(
            args.data_root,
            args.dataset,
            split_ratio=split_ratio,
            force_reformat=args.force_reformat,
        )
    except dataset_prep.DatasetError as exc:
        raise SystemExit(f"[dataset] ERROR: {exc}")

    print(
        f"[dataset] {info.name}: {info.num_classes} classes, input_size={info.input_size}, "
        f"container_path={info.container_path}"
    )

    if family == "yolox":
        config_text = exp_templates.render_yolox_exp(args.model, info, args.epochs)
    else:
        config_text = exp_templates.render_damoyolo_config(
            args.model, info, args.batch_size, args.epochs
        )

    GENERATED_DIR.mkdir(exist_ok=True)
    config_name = f"{args.model}__{info.name}.py"
    config_path = GENERATED_DIR / config_name
    if args.force_reformat or not config_path.is_file() or config_path.read_text() != config_text:
        config_path.write_text(config_text)
        print(f"[config] wrote {config_path}")
    else:
        print(f"[config] {config_path} already up to date")

    if args.dry_run:
        print("[dry-run] stopping before any docker calls")
        return

    if not container_ops.container_running(container):
        raise SystemExit(f"[container] {container!r} is not running — start it first.")

    if family == "damoyolo":
        container_ops.ensure_damoyolo_dataset_registered(info)

    container_dir = container_ops.CONTAINER_CONFIG_DIR[family]
    container_config_path = f"{container_dir}/{config_name}"
    container_ops.docker_cp(config_path, container, container_config_path)
    print(f"[container] copied {config_name} into {container}:{container_dir}/")

    print("[probe] running CPU-only DataLoader worker sweep...")
    probe_result = container_ops.probe_workers(family, info, args.batch_size)
    workers, reason = container_ops.pick_best_worker_count(probe_result)
    print(f"[probe] {reason}")

    workers_opt: list[str] = []
    if family == "yolox":
        workers_opt = ["data_num_workers", str(workers)]
    else:
        workers_key = container_ops.discover_damoyolo_workers_opt(info)
        if workers_key:
            workers_opt = [workers_key, str(workers)]
            print(f"[probe] applying via config key '{workers_key}'")
        else:
            print(
                "[probe] could not auto-detect a worker-count config key for DAMO-YOLO — "
                "leaving it at the framework default"
            )

    workdir = info.container_path

    print(
        f"[smoke] running {args.smoke_epochs} epoch(s) on the real GPU entrypoint to check "
        f"batch_size={args.batch_size} fits..."
    )
    smoke_argv = build_train_argv(
        family, container_config_path, args.batch_size, args.smoke_epochs, args.model, workers_opt
    )
    smoke = container_ops.exec_capture(container, workdir, smoke_argv, timeout=1800)
    if smoke.returncode != 0:
        combined = smoke.stdout + smoke.stderr
        tail = "\n".join(combined.splitlines()[-40:])
        print(
            f"[smoke] FAILED (exit {smoke.returncode}) — batch_size={args.batch_size} does not "
            f"work for {args.model}/{info.name} on this setup. Last output:\n{tail}"
        )
        if "out of memory" in combined.lower():
            print("[smoke] looks like a CUDA out-of-memory error — try a smaller --batch-size.")
        raise SystemExit(1)
    print("[smoke] OK")

    print(f"[train] launching the full {args.epochs}-epoch run in {container} (streaming output)...")
    full_argv = build_train_argv(
        family, container_config_path, args.batch_size, args.epochs, args.model, workers_opt
    )
    raise SystemExit(container_ops.exec_stream(container, workdir, full_argv))


if __name__ == "__main__":
    main()
