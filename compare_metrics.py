#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


PRIMARY_KEYS = [
    "AP@[0.50:0.95]",
    "AP@0.50",
    "AP@0.75",
    "AR@100",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two COCO metrics_report.json files.")
    parser.add_argument("--name-a", type=str, default="model_a")
    parser.add_argument("--report-a", type=str, required=True)
    parser.add_argument("--name-b", type=str, default="model_b")
    parser.add_argument("--report-b", type=str, required=True)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def load_report(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def winner(a, b):
    if a > b:
        return "A"
    if b > a:
        return "B"
    return "tie"


def main():
    args = parse_args()
    report_a = load_report(args.report_a)
    report_b = load_report(args.report_b)

    metrics_a = report_a.get("summary_metrics", {})
    metrics_b = report_b.get("summary_metrics", {})

    print(f"Comparing {args.name_a} vs {args.name_b}\n")
    print(f"{'Metric':<18} {'A':>10} {'B':>10} {'Winner':>10}")
    print("-" * 52)

    comparison = {
        "name_a": args.name_a,
        "name_b": args.name_b,
        "summary": {},
        "primary_winner": None,
    }

    for key in PRIMARY_KEYS:
        a_val = float(metrics_a.get(key, float("nan")))
        b_val = float(metrics_b.get(key, float("nan")))
        win = winner(a_val, b_val)
        label = args.name_a if win == "A" else args.name_b if win == "B" else "tie"
        print(f"{key:<18} {a_val:>10.4f} {b_val:>10.4f} {label:>10}")
        comparison["summary"][key] = {
            args.name_a: a_val,
            args.name_b: b_val,
            "winner": label,
        }

    primary_a = float(metrics_a.get("AP@[0.50:0.95]", float("nan")))
    primary_b = float(metrics_b.get("AP@[0.50:0.95]", float("nan")))
    primary = winner(primary_a, primary_b)
    comparison["primary_winner"] = (
        args.name_a if primary == "A" else args.name_b if primary == "B" else "tie"
    )

    print("\nPrimary decision metric: AP@[0.50:0.95]")
    print(f"Winner: {comparison['primary_winner']}")

    per_class_a = report_a.get("per_class_ap", {})
    per_class_b = report_b.get("per_class_ap", {})
    class_names = sorted(set(per_class_a) | set(per_class_b))
    if class_names:
        print("\nPer-class AP")
        print(f"{'Class':<24} {'A':>10} {'B':>10} {'Winner':>10}")
        print("-" * 60)
        comparison["per_class_ap"] = {}
        for class_name in class_names:
            a_val = float(per_class_a.get(class_name, float("nan")))
            b_val = float(per_class_b.get(class_name, float("nan")))
            win = winner(a_val, b_val)
            label = args.name_a if win == "A" else args.name_b if win == "B" else "tie"
            print(f"{class_name:<24} {a_val:>10.4f} {b_val:>10.4f} {label:>10}")
            comparison["per_class_ap"][class_name] = {
                args.name_a: a_val,
                args.name_b: b_val,
                "winner": label,
            }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nSaved comparison JSON to {output_path}")


if __name__ == "__main__":
    main()
