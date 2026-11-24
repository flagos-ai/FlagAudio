"""Read scheduler output and convert it into tabular operator results."""

import json
from pathlib import Path


HEADERS = (
    "operator", "gpu", "accuracy_status", "passed", "failed", "skipped",
    "performance_status", "dtype", "shape", "latency_base_ms",
    "latency_ms", "speedup", "error",
)


def rows_from_summary(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("operators"), list):
        raise ValueError("expected scheduler summary.json with an operators list")
    rows = []
    for item in data["operators"]:
        accuracy = item.get("accuracy") or {}
        performance = item.get("performance") or {}
        counts = accuracy.get("counts") or {}
        common = {
            "operator": item["operator"], "gpu": item.get("gpu", ""),
            "accuracy_status": accuracy.get("status", ""),
            "passed": counts.get("passed", ""), "failed": counts.get("failed", ""),
            "skipped": counts.get("skipped", ""),
            "performance_status": performance.get("status", ""),
        }
        for measurement in performance.get("measurements") or [{}]:
            rows.append({
                **common,
                "dtype": measurement.get("dtype", ""),
                "shape": str(measurement.get("shape_detail", "")),
                "latency_base_ms": measurement.get("latency_base", ""),
                "latency_ms": measurement.get("latency", ""),
                "speedup": measurement.get("speedup", ""),
                "error": measurement.get("error_msg", "") or "",
            })
    return rows
