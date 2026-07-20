#!/usr/bin/env python3
"""Run YOLOv5 inference and COCO evaluation with visualization outputs.

What this script does:
- Loads YOLOv5 weights using a local YOLOv5 repository checkout.
- Runs inference on images referenced by a COCO annotation file.
- Saves rendered detections to `drawn_images`.
- Writes COCO-format predictions (`predictions_coco.json`).
- Computes and saves summary metrics plus per-class AP (`metrics_report.json`).

Typical usage:
        python yolov5_infer_eval.py \
                --yolov5-root /path/to/yolov5 \
                --weights /path/to/best.pt \
                --input-dir /data/GERALD/test2017 \
                --ann-file /data/GERALD/annotations/instances_test2017.json \
                --output-dir /data/GERALD/results_yolov5_test \
                --conf 0.001 --nms 0.65 --imgsz 640 --device cuda

Notes:
- `--weights` is required and should match the dataset/class setup.
- If no predictions are generated, the script raises a runtime error with
    guidance to inspect weights, thresholds, and input paths.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLOv5 inference on a dataset folder and compute COCO metrics.",
        epilog=(
            "Example:\n"
            "  python yolov5_infer_eval.py "
            "--yolov5-root /path/to/yolov5 "
            "--weights /path/to/best.pt "
            "--input-dir /data/GERALD/test2017 "
            "--ann-file /data/GERALD/annotations/instances_test2017.json "
            "--output-dir /data/GERALD/results_yolov5_test --device cuda"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--yolov5-root",
        type=str,
        default="/home/themozel/Projects/master_thesis/signal_detection/yolov5",
        help="Path to the YOLOv5 repository root.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to YOLOv5 weights, e.g. best.pt",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/home/themozel/Projects/master_thesis/signal_detection/data/PERCEPT/images_coco/test2017",
        help="Directory with test images.",
    )
    parser.add_argument(
        "--ann-file",
        type=str,
        default="/home/themozel/Projects/master_thesis/signal_detection/data/PERCEPT/images_coco/annotations/instances_test2017.json",
        help="COCO annotation json for evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./inference_results/yolov5_test2017",
        help="Output directory for drawn images and metric files.",
    )
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for evaluation.")
    parser.add_argument("--nms", type=float, default=0.65, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run on.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 for inference when on CUDA.")
    return parser.parse_args()


def setup_yolov5_imports(yolov5_root):
    root = Path(yolov5_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from models.common import DetectMultiBackend
    from utils.dataloaders import letterbox
    from utils.general import non_max_suppression, scale_boxes
    from utils.torch_utils import select_device

    return DetectMultiBackend, letterbox, non_max_suppression, scale_boxes, select_device


def draw_detections(img, dets_xyxy, scores, class_ids, class_names):
    canvas = img.copy()
    for box, score, cls_id in zip(dets_xyxy, scores, class_ids):
        x1, y1, x2, y2 = [int(v) for v in box]
        cls_id = int(cls_id)
        color = (0, 255, 0)
        label_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"cls_{cls_id}"
        label = f"{label_name}: {score:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def coco_eval(ann_file, preds_json_path, class_names):
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(preds_json_path)

    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats.tolist()
    metric_names = [
        "AP@[0.50:0.95]",
        "AP@0.50",
        "AP@0.75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR@1",
        "AR@10",
        "AR@100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    metrics = dict(zip(metric_names, stats))

    precisions = evaluator.eval["precision"]
    cat_ids = coco_gt.getCatIds()
    cats = coco_gt.loadCats(cat_ids)

    per_class_ap = {}
    for k, cat in enumerate(cats):
        precision_k = precisions[:, :, k, 0, -1]
        precision_k = precision_k[precision_k > -1]
        ap = float(np.mean(precision_k)) if precision_k.size else float("nan")
        name = cat.get("name", class_names[k] if k < len(class_names) else str(cat.get("id", k)))
        per_class_ap[name] = ap

    return metrics, per_class_ap


def main():
    args = parse_args()

    DetectMultiBackend, letterbox, non_max_suppression, scale_boxes, select_device = setup_yolov5_imports(
        args.yolov5_root
    )

    device = select_device("0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    drawn_dir = output_dir / "drawn_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    drawn_dir.mkdir(parents=True, exist_ok=True)

    coco_gt = COCO(args.ann_file)
    categories = sorted(coco_gt.loadCats(coco_gt.getCatIds()), key=lambda c: c["id"])
    class_names = [c["name"] for c in categories]
    category_id_by_cls_index = [c["id"] for c in categories]

    model = DetectMultiBackend(args.weights, device=device, dnn=False, fp16=args.fp16)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = (args.imgsz, args.imgsz)

    predictions = []
    processed = 0

    for entry in coco_gt.dataset.get("images", []):
        image_id = entry["id"]
        file_name = entry["file_name"]
        img_path = Path(args.input_dir) / Path(file_name).name
        if not img_path.exists():
            continue

        im0 = cv2.imread(str(img_path))
        if im0 is None:
            continue

        im = letterbox(im0, imgsz, stride=stride, auto=pt)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)

        im_tensor = torch.from_numpy(im).to(model.device)
        im_tensor = im_tensor.half() if model.fp16 else im_tensor.float()
        im_tensor /= 255.0
        if len(im_tensor.shape) == 3:
            im_tensor = im_tensor[None]

        with torch.no_grad():
            pred = model(im_tensor)
            pred = non_max_suppression(pred, args.conf, args.nms, max_det=1000)

        vis_boxes = []
        vis_scores = []
        vis_classes = []

        det = pred[0]
        if det is not None and len(det):
            det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im0.shape).round()
            det = det.cpu().numpy()

            for row in det:
                x1, y1, x2, y2, score, cls_idx = row[:6]
                cls_idx = int(cls_idx)
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= 0 or h <= 0:
                    continue
                if cls_idx < 0 or cls_idx >= len(category_id_by_cls_index):
                    continue

                predictions.append(
                    {
                        "image_id": int(image_id),
                        "category_id": int(category_id_by_cls_index[cls_idx]),
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(score),
                    }
                )

                vis_boxes.append([x1, y1, x2, y2])
                vis_scores.append(float(score))
                vis_classes.append(cls_idx)

        drawn = draw_detections(im0, vis_boxes, vis_scores, vis_classes, class_names)
        cv2.imwrite(str(drawn_dir / Path(file_name).name), drawn)

        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed} images...")

    if not predictions:
        raise RuntimeError("No predictions were generated. Check weights/thresholds/path.")

    preds_json = output_dir / "predictions_coco.json"
    with open(preds_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    metrics, per_class_ap = coco_eval(args.ann_file, str(preds_json), class_names)

    report = {
        "summary_metrics": metrics,
        "per_class_ap": per_class_ap,
        "num_images_processed": processed,
        "num_detections": len(predictions),
        "weights": args.weights,
        "input_dir": args.input_dir,
        "ann_file": args.ann_file,
    }

    report_path = output_dir / "metrics_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nDone.")
    print(f"Drawn images: {drawn_dir}")
    print(f"Predictions JSON: {preds_json}")
    print(f"Metrics report: {report_path}")


if __name__ == "__main__":
    main()
