#!/usr/bin/env python3
"""Run YOLOX inference, COCO evaluation, and confusion-matrix reporting.

What this script does:
- Loads a YOLOX experiment and checkpoint.
- Runs inference on images referenced by a COCO annotation file.
- Saves visualized detections to `drawn_images`.
- Exports COCO predictions JSON and computes COCO metrics.
- Builds and saves a confusion matrix (JSON, CSV, PNG).

Typical usage:
        python exps/default/yolox_infer_eval.py \
                --exp-file exps/default/yolox_s_signal.py \
                --ckpt YOLOX_outputs/yolox_s_signal/best_ckpt.pth \
                --input-dir /data/GERALD/test2017 \
                --ann-file /data/GERALD/annotations/instances_test2017.json \
                --output-dir /data/GERALD/results_yolox_test \
                --conf 0.25 --nms 0.65 --tsize 640 --device cuda

Notes:
- `--confusion-iou` controls matching between predictions and ground truth for
    confusion matrix computation.
- If no predictions are produced, the script raises a runtime error with
    diagnostics (processed images, path/checkpoint mismatch hints).
"""

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLOX inference on a dataset folder, then compute COCO metrics and confusion matrix artifacts.",
        epilog=(
            "Example:\n"
            "  python exps/default/yolox_infer_eval.py "
            "--exp-file exps/default/yolox_s_signal.py "
            "--ckpt YOLOX_outputs/yolox_s_signal/best_ckpt.pth "
            "--input-dir /data/GERALD/test2017 "
            "--ann-file /data/GERALD/annotations/instances_test2017.json "
            "--output-dir /data/GERALD/results_yolox_test --device cuda"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--exp-file",
        type=str,
        required=True,
        help="Path to experiment file, e.g. yolox_s_signal_detection.py",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to model checkpoint, e.g. best_ckpt.pth",
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
        default="./inference_results/yolox_test2017",
        help="Output directory for drawn images and metric files.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--nms", type=float, default=0.65, help="NMS threshold.")
    parser.add_argument("--tsize", type=int, default=640, help="Input test size.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run on.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 for inference when on CUDA.")
    parser.add_argument(
        "--confusion-iou",
        type=float,
        default=0.5,
        help="IoU threshold used to match predictions to ground truth for the confusion matrix.",
    )
    return parser.parse_args()


def load_model(exp_file, ckpt_path, device, fp16, tsize):
    exp = get_exp(exp_file, None)
    exp.test_size = (tsize, tsize)
    model = exp.get_model()
    model.eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    load_result = model.load_state_dict(state_dict, strict=False)

    if hasattr(model, "head") and hasattr(model.head, "decode_in_inference"):
        model.head.decode_in_inference = True

    model.to(device)
    if fp16 and device == "cuda":
        model.half()
    return exp, model, load_result


def resolve_image_path(input_dir, file_name):
    candidates = [
        input_dir / file_name,
        input_dir / Path(file_name).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def preprocess_image(img, test_size):
    h, w = img.shape[:2]
    transform = ValTransform(legacy=False)
    img_processed, _ = transform(img, None, test_size)
    img_processed = torch.from_numpy(img_processed).unsqueeze(0).float()
    ratio = min(test_size[0] / h, test_size[1] / w)
    return img_processed, ratio


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

    # Per-class AP from precision tensor: [TxRxKxAxM]
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


def bbox_iou_xywh(box_a, box_b):
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def build_confusion_matrix(coco_gt, predictions, category_id_by_cls_index, class_names, iou_threshold):
    background_label = "background"
    labels = list(class_names) + [background_label]
    background_index = len(class_names)
    category_to_index = {category_id: idx for idx, category_id in enumerate(category_id_by_cls_index)}

    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    predictions_by_image = {}
    for pred in predictions:
        predictions_by_image.setdefault(int(pred["image_id"]), []).append(pred)

    for image_entry in coco_gt.dataset.get("images", []):
        image_id = int(image_entry["id"])
        gt_annotations = [
            ann
            for ann in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[image_id]))
            if not ann.get("iscrowd", 0)
        ]
        gt_annotations = [ann for ann in gt_annotations if ann.get("category_id") in category_to_index]
        pred_annotations = sorted(predictions_by_image.get(image_id, []), key=lambda pred: pred["score"], reverse=True)

        matched_gt_indices = set()
        for pred in pred_annotations:
            pred_cls = category_to_index.get(int(pred["category_id"]))
            if pred_cls is None:
                continue

            best_gt_idx = -1
            best_iou = 0.0
            for gt_idx, gt_ann in enumerate(gt_annotations):
                if gt_idx in matched_gt_indices:
                    continue
                iou = bbox_iou_xywh(pred["bbox"], gt_ann["bbox"])
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx >= 0:
                matched_gt_indices.add(best_gt_idx)
                gt_cls = category_to_index[int(gt_annotations[best_gt_idx]["category_id"])]
                confusion[gt_cls, pred_cls] += 1
            else:
                confusion[background_index, pred_cls] += 1

        for gt_idx, gt_ann in enumerate(gt_annotations):
            if gt_idx in matched_gt_indices:
                continue
            gt_cls = category_to_index[int(gt_ann["category_id"])]
            confusion[gt_cls, background_index] += 1

    return confusion, labels


def render_confusion_matrix_image(confusion, labels, output_path):
    cell_size = 72
    left_margin = 180
    top_margin = 180
    right_margin = 40
    bottom_margin = 40
    n = len(labels)
    width = left_margin + n * cell_size + right_margin
    height = top_margin + n * cell_size + bottom_margin

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    matrix = confusion.astype(np.float32)
    max_value = float(matrix.max()) if matrix.size else 0.0
    norm = np.zeros_like(matrix, dtype=np.uint8)
    if max_value > 0:
        norm = np.round((matrix / max_value) * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)

    for row in range(n):
        for col in range(n):
            y1 = top_margin + row * cell_size
            y2 = y1 + cell_size
            x1 = left_margin + col * cell_size
            x2 = x1 + cell_size
            canvas[y1:y2, x1:x2] = heatmap[row, col]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (220, 220, 220), 1)
            value = str(int(confusion[row, col]))
            text_size = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
            text_x = x1 + (cell_size - text_size[0]) // 2
            text_y = y1 + (cell_size + text_size[1]) // 2
            cv2.putText(canvas, value, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)

    for idx, label in enumerate(labels):
        label_x = left_margin + idx * cell_size + 8
        cv2.putText(canvas, label[:10], (label_x, top_margin - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        label_y = top_margin + idx * cell_size + cell_size // 2
        cv2.putText(canvas, label[:16], (12, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(canvas, "Predicted", (left_margin, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Ground Truth", (12, top_margin - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas)


def save_confusion_matrix(output_dir, confusion, labels, iou_threshold):
    confusion_json_path = output_dir / "confusion_matrix.json"
    confusion_csv_path = output_dir / "confusion_matrix.csv"
    confusion_png_path = output_dir / "confusion_matrix.png"

    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, row_sums, out=np.zeros_like(confusion, dtype=np.float64), where=row_sums > 0)

    with open(confusion_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "labels": labels,
                "iou_threshold": iou_threshold,
                "matrix": confusion.tolist(),
                "normalized_rows": normalized.tolist(),
            },
            f,
            indent=2,
        )

    with open(confusion_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ground_truth\\predicted", *labels])
        for label, row in zip(labels, confusion.tolist()):
            writer.writerow([label, *row])

    render_confusion_matrix_image(confusion, labels, confusion_png_path)
    return confusion_json_path, confusion_csv_path, confusion_png_path


def main():
    args = parse_args()
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    drawn_dir = output_dir / "drawn_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    drawn_dir.mkdir(parents=True, exist_ok=True)

    coco_gt = COCO(args.ann_file)
    categories = sorted(coco_gt.loadCats(coco_gt.getCatIds()), key=lambda c: c["id"])
    class_names = [c["name"] for c in categories]
    category_id_by_cls_index = [c["id"] for c in categories]

    exp, model, load_result = load_model(args.exp_file, args.ckpt, device, args.fp16, args.tsize)
    num_classes = exp.num_classes
    test_size = (args.tsize, args.tsize)

    missing_keys = list(getattr(load_result, "missing_keys", []))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
    if missing_keys or unexpected_keys:
        print(
            "Checkpoint load mismatch: "
            f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
        )
        if missing_keys:
            print(f"  sample missing keys: {missing_keys[:5]}")
        if unexpected_keys:
            print(f"  sample unexpected keys: {unexpected_keys[:5]}")

    image_entries = coco_gt.dataset.get("images", [])
    if not image_entries:
        raise RuntimeError("No images found in annotation file.")

    predictions = []
    processed = 0
    images_found = 0
    images_with_detections = 0
    raw_output_logged = False

    for entry in image_entries:
        image_id = entry["id"]
        file_name = entry["file_name"]
        img_path = resolve_image_path(input_dir, file_name)

        if img_path is None:
            continue

        images_found += 1

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        tensor_img, ratio = preprocess_image(img, test_size)
        tensor_img = tensor_img.to(device)
        if args.fp16 and device == "cuda":
            tensor_img = tensor_img.half()

        with torch.no_grad():
            outputs = model(tensor_img)
            if not raw_output_logged:
                if isinstance(outputs, (list, tuple)):
                    shape_info = [tuple(out.shape) for out in outputs if hasattr(out, "shape")]
                else:
                    shape_info = tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs).__name__
                print(f"First raw output shape: {shape_info}")
                raw_output_logged = True
            outputs = postprocess(outputs, num_classes, args.conf, args.nms)

        dets = outputs[0]
        vis_boxes = []
        vis_scores = []
        vis_classes = []

        if dets is not None and len(dets) > 0:
            dets = dets.cpu().numpy()
            bboxes = dets[:, :4] / ratio
            obj_conf = dets[:, 4]
            cls_conf = dets[:, 5]
            cls_idx = dets[:, 6].astype(int)
            scores = obj_conf * cls_conf

            for box, score, cidx in zip(bboxes, scores, cls_idx):
                x1, y1, x2, y2 = box.tolist()
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= 0 or h <= 0:
                    continue

                if cidx < 0 or cidx >= len(category_id_by_cls_index):
                    continue

                predictions.append(
                    {
                        "image_id": int(image_id),
                        "category_id": int(category_id_by_cls_index[cidx]),
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(score),
                    }
                )

                vis_boxes.append([x1, y1, x2, y2])
                vis_scores.append(float(score))
                vis_classes.append(int(cidx))

            if vis_boxes:
                images_with_detections += 1

        drawn = draw_detections(img, vis_boxes, vis_scores, vis_classes, class_names)
        cv2.imwrite(str(drawn_dir / Path(file_name).name), drawn)

        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed} images...")

    if not predictions:
        raise RuntimeError(
            "No predictions were generated. "
            f"processed={processed}, images_found={images_found}, images_with_detections={images_with_detections}, "
            f"checkpoint_missing_keys={len(missing_keys)}, checkpoint_unexpected_keys={len(unexpected_keys)}. "
            "Check checkpoint compatibility, thresholds, and input paths."
        )

    preds_json = output_dir / "predictions_coco.json"
    with open(preds_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    metrics, per_class_ap = coco_eval(args.ann_file, str(preds_json), class_names)
    confusion, confusion_labels = build_confusion_matrix(
        coco_gt,
        predictions,
        category_id_by_cls_index,
        class_names,
        args.confusion_iou,
    )
    confusion_json_path, confusion_csv_path, confusion_png_path = save_confusion_matrix(
        output_dir,
        confusion,
        confusion_labels,
        args.confusion_iou,
    )

    report = {
        "summary_metrics": metrics,
        "per_class_ap": per_class_ap,
        "confusion_matrix": {
            "labels": confusion_labels,
            "iou_threshold": args.confusion_iou,
            "json_path": str(confusion_json_path),
            "csv_path": str(confusion_csv_path),
            "image_path": str(confusion_png_path),
        },
        "num_images_processed": processed,
        "num_detections": len(predictions),
        "exp_file": args.exp_file,
        "ckpt": args.ckpt,
        "input_dir": args.input_dir,
        "ann_file": args.ann_file,
    }

    report_path = output_dir / "metrics_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nDone.")
    print(f"Drawn images: {drawn_dir}")
    print(f"Predictions JSON: {preds_json}")
    print(f"Confusion matrix JSON: {confusion_json_path}")
    print(f"Confusion matrix CSV: {confusion_csv_path}")
    print(f"Confusion matrix image: {confusion_png_path}")
    print(f"Metrics report: {report_path}")


if __name__ == "__main__":
    main()
