#!/usr/bin/env python3
"""Run FlagAudio accuracy and performance tests selected from conf/operators.yaml."""

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
STAGES = {"alpha", "beta", "stable", "removed"}
ID_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")


def load_catalog(path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("ops"), list):
        raise ValueError(f"{path}: expected a top-level ops list")
    seen = set()
    for index, op in enumerate(data["ops"], 1):
        if not isinstance(op, dict):
            raise ValueError(f"{path}: ops[{index}] must be a mapping")
        name = op.get("id")
        if not isinstance(name, str) or not ID_PATTERN.fullmatch(name) or name in seen:
            raise ValueError(f"{path}: invalid or duplicate operator id {name!r}")
        seen.add(name)
        if not isinstance(op.get("description"), str) or not op["description"].strip():
            raise ValueError(f"{path}: {name} needs a description")
        for field in ("for", "labels", "kind"):
            values = op.get(field)
            if not isinstance(values, list) or not values or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(f"{path}: {name}.{field} must be a nonempty string list")
        stages = op.get("stages")
        if not isinstance(stages, list) or not stages or any(
            not isinstance(stage, dict)
            or len(stage) != 1
            or next(iter(stage)) not in STAGES
            or not isinstance(next(iter(stage.values())), str)
            or not next(iter(stage.values())).strip()
            for stage in stages
        ):
            raise ValueError(f"{path}: {name}.stages must contain stage/version mappings")
    return data["ops"]


def select_ops(catalog, args):
    by_id = {op["id"]: op for op in catalog}
    if args.ops or args.op_list_file:
        if args.ops:
            names = [name.strip() for name in args.ops.split(",")]
        else:
            names = [line.split("#", 1)[0].strip() for line in
                     args.op_list_file.read_text(encoding="utf-8").splitlines()]
        names = [name for name in names if name]
        unknown = sorted(set(names) - by_id.keys())
        if unknown:
            raise ValueError(f"unknown operator IDs: {', '.join(unknown)}")
        names = list(dict.fromkeys(names))
    else:
        requested = {stage.strip() for stage in args.stages.split(",")}
        if not requested or requested - STAGES - {"all"}:
            raise ValueError(f"invalid --stages: {args.stages}")
        if "all" in requested:
            requested = STAGES - {"removed"}
        names = [op["id"] for op in catalog
                 if next(iter(op["stages"][-1])) in requested]
    if args.start:
        names = [name for name in names if name >= args.start]
    if not names:
        raise ValueError("no operators selected")
    return names


def accuracy_result(path, code):
    try:
        cases = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except json.JSONDecodeError:
        cases = {}
    if not isinstance(cases, dict):
        cases = {}
    counts = {key: 0 for key in ("passed", "failed", "skipped")}
    for case in cases.values():
        if isinstance(case, dict) and case.get("result") in counts:
            counts[case["result"]] += 1
    status = "PASS" if code == 0 and counts["passed"] > 0 else "FAIL"
    return {"status": status, "exit_code": code, "counts": counts,
            "case_count": len(cases), "result_file": str(path)}


def performance_result(path, code, name):
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except json.JSONDecodeError:
        data = {}
    entry = data.get(name, {}) if isinstance(data, dict) else {}
    details = entry.get("details", []) if isinstance(entry, dict) else []
    if not isinstance(details, list):
        details = []
    measurements = []
    for detail in details:
        if not isinstance(detail, dict):
            continue
        for metric in detail.get("result", []):
            if isinstance(metric, dict):
                measurements.append({"dtype": detail.get("dtype"), **metric})
    status = "PASS" if code == 0 and measurements else "FAIL"
    return {"status": status, "exit_code": code, "case_count": len(details),
            "measurement_count": len(measurements), "measurements": measurements,
            "result_file": str(path)}


def run_phase(root, name, phase, gpu, op_dir, args):
    report = op_dir / ("accuracy_result.json" if phase == "accuracy" else "performance_result.json")
    log = op_dir / ("accuracy.log" if phase == "accuracy" else "performance.log")
    # pytest's existing hooks merge with an existing report, so discard only this
    # phase's stale report before a rerun.
    report.unlink(missing_ok=True)
    command = [sys.executable, "-m", "pytest", "tests/" if phase == "accuracy" else "benchmark/",
               "-m", name, "-q", "--record", "json", "--output", str(report)]
    reference = args.accuracy_reference
    if reference == "auto":
        reference = "cpu" if os.environ.get("BLAS_VENDOR") == "iluvatar" else "device"
    if args.device_reference:
        reference = "device"
    if phase == "accuracy" and reference == "cpu":
        command.extend(["--ref", "cpu"])
    if phase == "performance":
        command.extend(["--level", args.level])
        if args.warmup is not None:
            command.extend(["--warmup", str(args.warmup)])
        if args.iterations is not None:
            command.extend(["--iter", str(args.iterations)])
    env = os.environ.copy()
    env[args.device_env] = gpu
    with log.open("w", encoding="utf-8") as stream:
        code = subprocess.run(command, cwd=root, env=env, stdout=stream,
                              stderr=subprocess.STDOUT, check=False).returncode
    result = (accuracy_result(report, code) if phase == "accuracy"
              else performance_result(report, code, name))
    result["log_file"] = str(log)
    return result


def write_summary(path, results):
    payload = {"operators": [results[name] for name in sorted(results)]}
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flagaudio", type=Path, default=ROOT, help="FlagAudio root directory")
    parser.add_argument("--catalog", type=Path, help="operator inventory (default: ROOT/conf/operators.yaml)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ops", help="comma-separated operator IDs")
    group.add_argument("--op-list-file", "--op-list", type=Path, help="one operator ID per line")
    parser.add_argument("--stages", default="all", help="alpha,beta,stable,removed or all")
    parser.add_argument("--start", help="start at this operator ID")
    parser.add_argument("--gpus", default="0", help="comma-separated GPU IDs")
    parser.add_argument("--device-env", default="CUDA_VISIBLE_DEVICES",
                        help="device visibility environment variable (default: CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--output-dir", "--results-dir", type=Path)
    parser.add_argument("--accuracy-only", action="store_true")
    parser.add_argument("--performance-only", action="store_true")
    parser.add_argument("--accuracy-reference", choices=["auto", "cpu", "device"], default="auto",
                        help="accuracy reference device (auto: CPU on Iluvatar, device elsewhere)")
    parser.add_argument("--device-reference", action="store_true", help="alias for --accuracy-reference device")
    parser.add_argument("--level", choices=["core", "comprehensive"], default="core")
    parser.add_argument("--warmup-ms", "--warmup", dest="warmup", metavar="MS", type=int,
                        help="warmup duration in milliseconds in kernel mode")
    parser.add_argument("--measurement-ms", "--iterations", dest="iterations", metavar="MS", type=int,
                        help="measurement duration in milliseconds in kernel mode")
    parser.add_argument("--list", action="store_true", help="print selected operators and exit")
    args = parser.parse_args(argv)
    try:
        if args.accuracy_only and args.performance_only:
            raise ValueError("choose at most one of --accuracy-only and --performance-only")
        root = args.flagaudio.resolve()
        catalog = load_catalog(args.catalog or root / "conf/operators.yaml")
        names = select_ops(catalog, args)
        if args.list:
            print("\n".join(names))
            return 0
        gpus = [gpu.strip() for gpu in args.gpus.split(",")]
        if not gpus or any(not gpu.isdigit() for gpu in gpus) or len(set(gpus)) != len(gpus):
            raise ValueError("--gpus must be distinct comma-separated GPU IDs")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", args.device_env):
            raise ValueError("--device-env must be an environment variable name")
        if (args.warmup is not None and args.warmup < 0) or (args.iterations is not None and args.iterations < 1):
            raise ValueError("warmup must be nonnegative and measurement duration positive")
        output = (args.output_dir or root / f"results_{dt.datetime.now():%Y%m%d_%H%M%S}").resolve()
        output.mkdir(parents=True, exist_ok=True)
        results = {}
        lock = threading.Lock()
        work = {gpu: [] for gpu in gpus}
        for index, name in enumerate(names):
            work[gpus[index % len(gpus)]].append(name)

        def worker(gpu):
            for name in work[gpu]:
                op_dir = output / name
                op_dir.mkdir(exist_ok=True)
                item = {"operator": name, "gpu": gpu}
                for phase in ("accuracy", "performance"):
                    if (phase == "accuracy" and args.performance_only) or (phase == "performance" and args.accuracy_only):
                        continue
                    item[phase] = run_phase(root, name, phase, gpu, op_dir, args)
                    print(f"[{gpu}] {name} {phase}: {item[phase]['status']}", flush=True)
                with lock:
                    results[name] = item
                    write_summary(output / "summary.json", results)

        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            list(executor.map(worker, gpus))
        print(f"Results: {output}")
        return 1 if any(phase.get("status") != "PASS" for item in results.values()
                        for phase in (item.get("accuracy"), item.get("performance")) if phase) else 0
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
