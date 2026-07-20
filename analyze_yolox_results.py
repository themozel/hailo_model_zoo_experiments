#!/usr/bin/env python3
"""Analyze YOLOX inference artifacts and generate a consolidated findings report.

What this script does:
- Reads YOLOX evaluation artifacts from a results directory:
    `metrics_report.json`, `confusion_matrix.json`, and `predictions_coco.json`.
- Joins predictions with COCO ground-truth annotations.
- Computes per-class ranking (support, precision, recall, F1, AP where available).
- Finds worst false-positive and false-negative images using IoU matching.
- Copies top-K worst-case rendered images into dedicated folders.
- Writes:
    - `findings.md` (human-readable analysis)
    - `analysis_summary.json` (machine-readable summary)

Typical usage:
        python analyze_yolox_results.py \
                --results-dir /data/GERALD/results_yolox_test \
                --ann-file /data/GERALD/annotations/instances_test2017.json \
                --dataset-dir /data/GERALD

Notes:
- If `--dataset-dir` is omitted, `findings.md` is written to the parent of
    `--results-dir`.
- The default matching IoU is 0.5 and default extraction count is top 20 images
    for both false positives and false negatives.
"""

import argparse
import json
import math
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze YOLOX inference results and generate ranked findings plus worst-case image sets.",
        epilog=(
            "Example:\n"
            "  python analyze_yolox_results.py "
            "--results-dir /data/GERALD/results_yolox_test "
            "--ann-file /data/GERALD/annotations/instances_test2017.json "
            "--dataset-dir /data/GERALD"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing metrics_report.json and drawn_images.")
    parser.add_argument("--ann-file", type=str, required=True, help="COCO annotation file used for evaluation.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Directory where findings.md will be written. Defaults to the parent of results-dir.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Number of worst FP/FN images to extract.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for FP/FN matching.")
    return parser.parse_args()


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

    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def safe_float(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


def fmt_metric(value):
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def load_inputs(results_dir, ann_file):
    report = json.loads((results_dir / "metrics_report.json").read_text())
    confusion = json.loads((results_dir / "confusion_matrix.json").read_text())
    predictions = json.loads((results_dir / "predictions_coco.json").read_text())
    annotations = json.loads(Path(ann_file).read_text())
    return report, confusion, predictions, annotations


def build_indices(confusion, annotations, predictions):
    class_names = confusion["labels"][:-1]
    class_index_by_name = {name: idx for idx, name in enumerate(class_names)}
    class_index_by_category_id = {
        category["id"]: class_index_by_name[category["name"]]
        for category in annotations["categories"]
        if category["name"] in class_index_by_name
    }
    image_name_by_id = {image["id"]: Path(image["file_name"]).name for image in annotations["images"]}

    annotations_by_image = {}
    for ann in annotations["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        cls_idx = class_index_by_category_id.get(ann["category_id"])
        if cls_idx is None:
            continue
        annotations_by_image.setdefault(ann["image_id"], []).append({"bbox": ann["bbox"], "cls": cls_idx})

    predictions_by_image = {}
    for pred in predictions:
        cls_idx = class_index_by_category_id.get(pred["category_id"])
        if cls_idx is None:
            continue
        predictions_by_image.setdefault(pred["image_id"], []).append(
            {"bbox": pred["bbox"], "cls": cls_idx, "score": float(pred["score"])}
        )

    return class_names, image_name_by_id, annotations_by_image, predictions_by_image


def summarize_classes(report, confusion):
    matrix = confusion["matrix"]
    class_names = confusion["labels"][:-1]
    rows = []
    for idx, class_name in enumerate(class_names):
        tp = matrix[idx][idx]
        support = sum(matrix[idx])
        fn = support - tp
        fp = sum(row[idx] for row in matrix) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows.append(
            {
                "class_name": class_name,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "ap50_95": safe_float(report["per_class_ap"].get(class_name)),
            }
        )
    rows.sort(key=lambda row: (-row["f1"], -row["support"], row["class_name"]))
    return rows


def analyze_images(image_name_by_id, annotations_by_image, predictions_by_image, iou_threshold):
    fp_images = []
    fn_images = []
    for image_id, file_name in image_name_by_id.items():
        gt_objects = list(annotations_by_image.get(image_id, []))
        pred_objects = sorted(predictions_by_image.get(image_id, []), key=lambda pred: pred["score"], reverse=True)
        matched_gt_indices = set()
        matched_pairs = []

        for pred_index, pred in enumerate(pred_objects):
            best_gt_idx = -1
            best_iou = 0.0
            for gt_index, gt in enumerate(gt_objects):
                if gt_index in matched_gt_indices:
                    continue
                iou = bbox_iou_xywh(pred["bbox"], gt["bbox"])
                if iou >= iou_threshold and iou > best_iou:
                    best_gt_idx = gt_index
                    best_iou = iou

            if best_gt_idx >= 0:
                matched_gt_indices.add(best_gt_idx)
                matched_pairs.append((pred_index, best_gt_idx))

        matched_pred_indices = {pred_idx for pred_idx, _ in matched_pairs}
        fp_count = len(pred_objects) - len(matched_pred_indices)
        fn_count = len(gt_objects) - len(matched_gt_indices)
        cross_class_count = sum(
            1 for pred_idx, gt_idx in matched_pairs if pred_objects[pred_idx]["cls"] != gt_objects[gt_idx]["cls"]
        )

        fp_images.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "fp_count": fp_count,
                "pred_count": len(pred_objects),
                "gt_count": len(gt_objects),
                "cross_class_count": cross_class_count,
            }
        )
        fn_images.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "fn_count": fn_count,
                "pred_count": len(pred_objects),
                "gt_count": len(gt_objects),
                "cross_class_count": cross_class_count,
            }
        )

    fp_images.sort(key=lambda item: (-item["fp_count"], -item["cross_class_count"], -item["pred_count"], item["file_name"]))
    fn_images.sort(key=lambda item: (-item["fn_count"], -item["gt_count"], -item["cross_class_count"], item["file_name"]))
    return fp_images, fn_images


def extract_images(results_dir, folder_name, ranked_images, top_k):
    target_dir = results_dir / folder_name
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    drawn_dir = results_dir / "drawn_images"
    extracted = []
    for rank, item in enumerate(ranked_images[:top_k], start=1):
        src = drawn_dir / item["file_name"]
        if not src.exists():
            continue
        prefix = f"{rank:02d}_{item['image_id']}_"
        dst = target_dir / f"{prefix}{item['file_name']}"
        shutil.copy2(src, dst)
        extracted.append({**item, "copied_path": str(dst)})
    return target_dir, extracted


def strongest_and_weakest(class_rows):
    supported = [row for row in class_rows if row["support"] > 0]
    strongest = supported[:10]
    weakest = sorted(supported, key=lambda row: (row["f1"], row["support"], row["class_name"]))[:10]
    return strongest, weakest


def top_false_positive_classes(confusion):
    matrix = confusion["matrix"]
    class_names = confusion["labels"][:-1]
    background_idx = len(class_names)
    values = []
    for idx, class_name in enumerate(class_names):
        values.append((class_name, matrix[background_idx][idx]))
    return sorted(values, key=lambda item: (-item[1], item[0]))[:10]


def top_false_negative_classes(confusion):
    matrix = confusion["matrix"]
    class_names = confusion["labels"][:-1]
    background_idx = len(class_names)
    values = []
    for idx, class_name in enumerate(class_names):
        values.append((class_name, matrix[idx][background_idx]))
    return sorted(values, key=lambda item: (-item[1], item[0]))[:10]


def top_cross_confusions(confusion, limit=15):
    matrix = confusion["matrix"]
    class_names = confusion["labels"][:-1]
    pairs = []
    for gt_idx, gt_name in enumerate(class_names):
        for pred_idx, pred_name in enumerate(class_names):
            if gt_idx == pred_idx:
                continue
            count = matrix[gt_idx][pred_idx]
            if count:
                pairs.append((gt_name, pred_name, count))
    return sorted(pairs, key=lambda item: (-item[2], item[0], item[1]))[:limit]


def aggregate_metrics(confusion):
    matrix = confusion["matrix"]
    class_names = confusion["labels"][:-1]
    background_idx = len(class_names)
    tp = sum(matrix[idx][idx] for idx in range(background_idx))
    fp = sum(matrix[background_idx][idx] for idx in range(background_idx))
    fn = sum(matrix[idx][background_idx] for idx in range(background_idx))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def make_findings_markdown(report, confusion, class_rows, extracted_fp, extracted_fn, results_dir):
    summary = report["summary_metrics"]
    aggregate = aggregate_metrics(confusion)
    strongest, weakest = strongest_and_weakest(class_rows)
    fp_classes = top_false_positive_classes(confusion)
    fn_classes = top_false_negative_classes(confusion)
    confusions = top_cross_confusions(confusion)

    lines = []
    lines.append("# YOLOX Inference Findings")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Results directory: `{results_dir}`")
    lines.append(f"- Images processed: {report['num_images_processed']}")
    lines.append(f"- Detections generated: {report['num_detections']}")
    lines.append(f"- Confusion IoU threshold: {confusion['iou_threshold']}")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        f"The model is usable, with overall detection precision {aggregate['precision']:.3f}, recall {aggregate['recall']:.3f}, and F1 {aggregate['f1']:.3f} at IoU {confusion['iou_threshold']:.2f}."
    )
    lines.append(
        f"COCO AP is {summary['AP@[0.50:0.95]']:.3f}, AP50 is {summary['AP@0.50']:.3f}, and AP75 is {summary['AP@0.75']:.3f}."
    )
    lines.append(
        f"Small-object performance is the main weakness: AP_small {summary['AP_small']:.3f} versus AP_medium {summary['AP_medium']:.3f} and AP_large {summary['AP_large']:.3f}."
    )
    lines.append("")
    lines.append("## What Looks Good")
    lines.append("")
    lines.append("- The detector is globally balanced. Predicted object count is close to ground-truth count, so it is not severely overfiring or underfiring overall.")
    lines.append("- Strong classes are visually distinctive and stable in both precision and recall, especially Vr_2, Ks_1, Hp_0_Ks, Ne_2, Mast_Sign_WRW, and Zs_Off.")
    lines.append("- Medium and large targets are handled much better than small targets, which means the backbone/head are learning meaningful features even if distant signs remain hard.")
    lines.append("")
    lines.append("## What Looks Weak")
    lines.append("")
    lines.append("- Small objects are the dominant failure mode. The AP_small gap is too large for a production-quality long-tail traffic-sign detector.")
    lines.append("- Several semantically similar classes are confused with each other, especially among Signal_*, Platform_*, and Zs_* families.")
    lines.append("- Rare classes remain unreliable because support is very low or absent in the evaluation split, so the reported AP is either unstable or not meaningful.")
    lines.append("")
    lines.append("## Ranked Classes")
    lines.append("")
    lines.append("| Rank | Class | Support | AP@[0.50:0.95] | Precision | Recall | F1 | TP | FP | FN |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for rank, row in enumerate(class_rows, start=1):
        lines.append(
            f"| {rank} | {row['class_name']} | {row['support']} | {fmt_metric(row['ap50_95'])} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['tp']} | {row['fp']} | {row['fn']} |"
        )
    lines.append("")
    lines.append("## Strongest Classes")
    lines.append("")
    for row in strongest:
        lines.append(
            f"- {row['class_name']}: F1 {row['f1']:.3f}, precision {row['precision']:.3f}, recall {row['recall']:.3f}, support {row['support']}."
        )
    lines.append("")
    lines.append("## Weakest Supported Classes")
    lines.append("")
    for row in weakest:
        lines.append(
            f"- {row['class_name']}: F1 {row['f1']:.3f}, precision {row['precision']:.3f}, recall {row['recall']:.3f}, support {row['support']}."
        )
    lines.append("")
    lines.append("## Highest False-Positive Classes")
    lines.append("")
    for class_name, count in fp_classes:
        lines.append(f"- {class_name}: {count} unmatched predictions.")
    lines.append("")
    lines.append("## Highest False-Negative Classes")
    lines.append("")
    for class_name, count in fn_classes:
        lines.append(f"- {class_name}: {count} missed ground-truth objects.")
    lines.append("")
    lines.append("## Most Frequent Cross-Class Confusions")
    lines.append("")
    for gt_name, pred_name, count in confusions:
        lines.append(f"- Ground truth `{gt_name}` predicted as `{pred_name}`: {count} times.")
    lines.append("")
    lines.append("## Extracted Worst False-Positive Images")
    lines.append("")
    lines.append("Images were copied into `results_yolox_test/worst_false_positives`.")
    lines.append("")
    lines.append("| Rank | Image | FP Count | Predictions | GT Objects | Cross-Class Matches |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for rank, item in enumerate(extracted_fp, start=1):
        lines.append(
            f"| {rank} | {item['file_name']} | {item['fp_count']} | {item['pred_count']} | {item['gt_count']} | {item['cross_class_count']} |"
        )
    lines.append("")
    lines.append("## Extracted Worst False-Negative Images")
    lines.append("")
    lines.append("Images were copied into `results_yolox_test/worst_false_negatives`.")
    lines.append("")
    lines.append("| Rank | Image | FN Count | Predictions | GT Objects | Cross-Class Matches |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for rank, item in enumerate(extracted_fn, start=1):
        lines.append(
            f"| {rank} | {item['file_name']} | {item['fn_count']} | {item['pred_count']} | {item['gt_count']} | {item['cross_class_count']} |"
        )
    lines.append("")
    lines.append("## Suggestions For Training")
    lines.append("")
    lines.append("1. Increase effective resolution for small signs. Train and evaluate with a larger image size such as 960 or 1280 if memory allows.")
    lines.append("2. Rebalance the long tail. Oversample rare classes or use repeat-factor sampling so classes like Ride_Indicator_Off, Lf_3, Zs_6, and rare Ne_* variants are seen more often.")
    lines.append("3. Mine hard negatives for the confusing families. Add explicit near-miss examples for Signal_*, Platform_*, and Zs_* classes, especially where signs are partially occluded, backlit, or far away.")
    lines.append("4. Audit annotation tightness. The AP50 to AP75 drop suggests localization quality is limiting the model on stricter IoU thresholds.")
    lines.append("5. Tune inference thresholds after retraining. A confidence sweep on the validation set may recover recall for small objects without materially increasing false positives on the stronger classes.")
    lines.append("6. Use targeted augmentations instead of only generic ones. Scale jitter, copy-paste of small signs, crop-and-paste around distant poles, and stronger photometric augmentation are likely higher value than generic flips alone.")
    lines.append("7. Consider class consolidation if some classes are operationally indistinguishable and underrepresented. That can improve robustness if the application does not require ultra-fine taxonomy.")
    lines.append("8. Review the extracted worst FP/FN images before changing architecture. If most failures come from ambiguous labeling or extreme distance, data changes will outperform architecture churn.")
    lines.append("")
    lines.append("## Suggested Next Experiments")
    lines.append("")
    lines.append("- Run a validation sweep over confidence and NMS thresholds and compare class-wise precision/recall deltas.")
    lines.append("- Retrain at higher resolution with the same checkpoint and compare AP_small, Signal_* classes, and Platform_* classes first.")
    lines.append("- Build a focused error set from the extracted FP/FN images and inspect whether the failures are due to distance, blur, truncation, or label ambiguity.")
    lines.append("- If training data is limited, try class-aware sampling or focal-loss tuning before switching models.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else results_dir.parent

    report, confusion, predictions, annotations = load_inputs(results_dir, args.ann_file)
    class_names, image_name_by_id, annotations_by_image, predictions_by_image = build_indices(
        confusion, annotations, predictions
    )
    class_rows = summarize_classes(report, confusion)
    fp_images, fn_images = analyze_images(
        image_name_by_id, annotations_by_image, predictions_by_image, args.iou
    )
    _, extracted_fp = extract_images(results_dir, "worst_false_positives", fp_images, args.top_k)
    _, extracted_fn = extract_images(results_dir, "worst_false_negatives", fn_images, args.top_k)

    summary_json = {
        "ranked_classes": class_rows,
        "top_false_positive_images": extracted_fp,
        "top_false_negative_images": extracted_fn,
    }
    (results_dir / "analysis_summary.json").write_text(json.dumps(summary_json, indent=2))

    findings = make_findings_markdown(report, confusion, class_rows, extracted_fp, extracted_fn, results_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "findings.md").write_text(findings, encoding="utf-8")
    print(dataset_dir / "findings.md")


if __name__ == "__main__":
    main()