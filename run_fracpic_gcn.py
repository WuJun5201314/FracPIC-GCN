#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run the 12 FracPIC-GCN configurations with five seeds per configuration.

Completed configurations are saved in state.json and skipped on restart.
The runner writes one log per attempt and updates results.csv and results.txt
after each completed configuration. Training arguments omitted here use the
defaults in train.py.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

METHOD_NAME = "FracPIC-GCN"
SCRIPT_VERSION = "fracpic-gcn-v3"
RUNS_PER_TASK = 5

# The 12 primary dataset/noise settings from Table 2.
# Each tuple:
# (dataset, noise, alpha, c, extra train.py arguments)
EXPERIMENTS: List[Tuple[str, str, float, float, Tuple[str, ...]]] = [
    (
        "cs", "uniform", 0.17, 1.262,
        (
            "--ptb_rate", "0.2",
            "--label_rate", "0.01",
            "--K", "50",
            "--local_conflict_weight", "0.9",
            "--warmup_epochs", "25",
            "--scale1", "1.0",
            "--p_u", "0.9",
            "--pseudo_type", "mix",
            "--temp", "1.0",
            "--runs", "5",
        ),
    ),
    (
        "cs", "pair", 0.34, 1.0,
        (
            "--ptb_rate", "0.2",
            "--label_rate", "0.01",
            "--K", "50",
            "--local_conflict_weight", "0.9",
            "--warmup_epochs", "25",
            "--scale1", "1.0",
            "--p_u", "0.9",
            "--pseudo_type", "mix",
            "--temp", "1.0",
            "--runs", "5",
        ),
    ),
    (
        "photo", "uniform", 0.3, 2.5,
        (
            "--ptb_rate", "0.2",
            "--label_rate", "0.01",
            "--K", "100",
            "--local_conflict_weight", "0.8",
            "--warmup_epochs", "15",
            "--temp", "1.0",
            "--runs", "5",
        ),
    ),
    (
        "photo", "pair", 0.406, 1.188,
        (
            "--ptb_rate", "0.2",
            "--label_rate", "0.01",
            "--K", "100",
            "--local_conflict_weight", "0.8",
            "--warmup_epochs", "15",
            "--temp", "1.0",
            "--runs", "5",
        ),
    ),
    (
        "cora", "uniform", 0.72, 0.75,
        (
            "--ptb_rate", "0.2",
            "--label_rate", "0.05",
            "--K", "50",
            "--local_conflict_weight", "0.9",
            "--warmup_epochs", "30",
            "--scale1", "1.0",
            "--temp", "0.01",
            "--runs", "5",
        ),
    ),
    (
        "cora", "pair", 0.702, 1.284,
        (
            "--ptb_rate", "0.2",
            "--label_rate", "0.05",
            "--K", "50",
            "--local_conflict_weight", "0.9",
            "--warmup_epochs", "30",
            "--scale1", "1.0",
            "--temp", "0.01",
            "--runs", "5",
        ),
    ),
    (
        "pubmed", "uniform", 0.808, 1.0,
        (
            "--ptb_rate", "0.2",
            "--label_rate", "0.01",
            "--K", "75",
            "--local_conflict_weight", "0.8",
            "--warmup_epochs", "30",
            "--scale1", "1.0",
            "--temp", "0.5",
            "--runs", "5",
        ),
    ),
    (
        "pubmed", "pair", 0.72, 0.5,
        (
            "--ptb_rate", "0.2",
            "--label_rate", "0.01",
            "--K", "75",
            "--local_conflict_weight", "0.8",
            "--warmup_epochs", "30",
            "--scale1", "1.0",
            "--temp", "0.5",
            "--runs", "5",
        ),
    ),
    (
        "dblp", "uniform", 0.6, 0.952,
        (
            "--ptb_rate", "0.2",
            "--label_rate", "0.01",
            "--K", "50",
            "--local_conflict_weight", "0.9",
            "--warmup_epochs", "30",
            "--scale1", "1.0",
            "--temp", "0.01",
            "--runs", "5",
        ),
    ),
    (
        "dblp", "pair", 0.721, 1.0,
        (
            "--ptb_rate", "0.2",
            "--label_rate", "0.01",
            "--K", "50",
            "--local_conflict_weight", "0.9",
            "--warmup_epochs", "30",
            "--scale1", "1.0",
            "--temp", "0.01",
            "--runs", "5",
        ),
    ),
    (
        "citeseer", "uniform", 0.2, 0.3,
        (
            "--ptb_rate", "0.2",
            "--label_rate", "0.05",
            "--K", "50",
            "--local_conflict_weight", "0.8",
            "--warmup_epochs", "15",
            "--scale1", "1.0",
            "--pseudo_type", "mix",
            "--temp", "0.1",
            "--runs", "5",
        ),
    ),
    (
        "citeseer", "pair", 0.17, 0.751,
        (
            "--ptb_rate", "0.2",
            "--label_rate", "0.05",
            "--K", "50",
            "--local_conflict_weight", "0.8",
            "--warmup_epochs", "15",
            "--scale1", "1.0",
            "--pseudo_type", "mix",
            "--temp", "0.1",
            "--runs", "5",
        ),
    ),
]

RESULT_PREFIX = re.escape(f"{METHOD_NAME} | ")
FINAL_RE = re.compile(
    rf"{RESULT_PREFIX}test_acc:\s*\[mean:([0-9eE+\-.]+)\]\[std:([0-9eE+\-.]+)\]"
)
SEED_ACC_RE = re.compile(
    rf"{RESULT_PREFIX}Test accuracy \(seed \d+\):\s*([0-9eE+\-.]+)"
)


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: dict) -> None:
    atomic_write_text(
        path,
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def task_key(experiment_index: int, dataset: str, noise: str) -> str:
    return f"exp_{experiment_index:02d}_{dataset}_{noise}"


def extra_args_to_dict(extra_args: Sequence[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    i = 0
    while i < len(extra_args):
        key = extra_args[i]
        if not key.startswith("--") or i + 1 >= len(extra_args):
            raise ValueError(f"Malformed experiment args near: {extra_args[i:]}")
        result[key[2:]] = extra_args[i + 1]
        i += 2
    return result


def build_plan_payload(train_py_sha256: str) -> dict:
    return {
        "script_version": SCRIPT_VERSION,
        "runs_per_task": RUNS_PER_TASK,
        "train_py_sha256": train_py_sha256,
        "experiments": [
            {
                "experiment_index": idx,
                "dataset": dataset,
                "noise": noise,
                "alpha": alpha,
                "c": c,
                "extra_args": list(extra_args),
            }
            for idx, (dataset, noise, alpha, c, extra_args)
            in enumerate(EXPERIMENTS, 1)
        ],
    }


def plan_signature(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class RunnerLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str = "") -> None:
        line = f"[{now_text()}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def acquire_lock(lock_path: Path) -> None:
    if lock_path.exists():
        try:
            info = json.loads(lock_path.read_text(encoding="utf-8"))
            old_pid = int(info.get("pid", -1))
        except Exception:
            old_pid = -1

        if old_pid > 0 and process_alive(old_pid):
            raise SystemExit(
                f"Another runner appears active (PID={old_pid}). "
                f"Lock: {lock_path}"
            )

        # Stale lock from an interrupted/dead process.
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    payload = {"pid": os.getpid(), "created_at": now_text()}
    fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(fd, (json.dumps(payload) + "\n").encode("utf-8"))
    finally:
        os.close(fd)


def release_lock(lock_path: Path) -> None:
    try:
        if not lock_path.exists():
            return
        info = json.loads(lock_path.read_text(encoding="utf-8"))
        if int(info.get("pid", -1)) == os.getpid():
            lock_path.unlink()
    except Exception:
        pass


def build_child_env(gpu: str, omp_threads: int) -> Dict[str, str]:
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = str(omp_threads)
    env["MKL_NUM_THREADS"] = str(omp_threads)
    return env


def run_preflight(
    python_exe: str,
    project_root: Path,
    env: Dict[str, str],
    logger: RunnerLogger,
) -> None:
    train_py = project_root / "train.py"
    if not train_py.is_file():
        raise SystemExit(f"train.py not found: {train_py}")

    if os.path.sep in python_exe and not os.path.isfile(python_exe):
        raise SystemExit(f"Python interpreter not found: {python_exe}")

    train_text = train_py.read_text(encoding="utf-8", errors="replace")
    required_tokens = [
        "--dataset", "--noise", "--alpha", "--c", "--ptb_rate",
        "--label_rate", "--K", "--local_conflict_weight",
        "--warmup_epochs", "--temp", "--runs",
    ]
    missing = [token for token in required_tokens if token not in train_text]
    if missing:
        raise SystemExit(
            "train.py is missing required FracPIC-GCN experiment flags: "
            + ", ".join(missing)
        )

    code = (
        "import json,sys,torch,torch_geometric;"
        "print(json.dumps({"
        "'python':sys.version.split()[0],"
        "'executable':sys.executable,"
        "'torch':torch.__version__,"
        "'torch_cuda':torch.version.cuda,"
        "'pyg':torch_geometric.__version__,"
        "'cuda_available':torch.cuda.is_available(),"
        "'cuda_count':torch.cuda.device_count(),"
        "'gpu_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None"
        "}))"
    )
    cp = subprocess.run(
        [python_exe, "-c", code],
        cwd=str(project_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if cp.returncode != 0:
        raise SystemExit("Environment preflight failed:\n" + cp.stdout)

    try:
        info = json.loads(cp.stdout.strip().splitlines()[-1])
    except Exception:
        raise SystemExit("Could not parse environment preflight output:\n" + cp.stdout)

    logger.log(
        "Environment: "
        f"Python={info.get('python')} "
        f"Torch={info.get('torch')} "
        f"TorchCUDA={info.get('torch_cuda')} "
        f"PyG={info.get('pyg')} "
        f"CUDA={info.get('cuda_available')} "
        f"VisibleGPUCount={info.get('cuda_count')} "
        f"GPU={info.get('gpu_name')}"
    )

    if not info.get("cuda_available"):
        raise SystemExit(
            "CUDA preflight failed. The runner refuses to fall back to CPU. "
            f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}"
        )

    logger.log(
        "GPU mapping: physical/requested GPU "
        f"{env.get('CUDA_VISIBLE_DEVICES')} is exposed to train.py as logical cuda:0."
    )

    # Report expected PPR presence without modifying anything.
    missing_ppr = []
    for dataset, noise, _alpha, _c, extra_args in EXPERIMENTS:
        params = extra_args_to_dict(extra_args)
        ppr = (
            project_root
            / "data"
            / f"{dataset}_{noise}_{params['ptb_rate']}_{params['label_rate']}_ppr.pt"
        )
        if not ppr.is_file():
            missing_ppr.append(str(ppr.relative_to(project_root)))

    if missing_ppr:
        logger.log(
            "PPR preflight WARNING: missing files detected; train.py may regenerate them: "
            + ", ".join(missing_ppr)
        )
    else:
        logger.log("PPR preflight: all 12 expected PPR files are present.")


def build_command(
    python_exe: str,
    dataset: str,
    noise: str,
    alpha: float,
    c: float,
    extra_args: Sequence[str],
) -> List[str]:
    return [
        python_exe,
        "-u",
        "train.py",
        "--dataset", dataset,
        "--noise", noise,
        "--alpha", str(alpha),
        "--c", str(c),
        *extra_args,
    ]


def next_attempt_number(task_dir: Path) -> int:
    maximum = 0
    for p in task_dir.glob("attempt_*.log"):
        m = re.fullmatch(r"attempt_(\d+)\.log", p.name)
        if m:
            maximum = max(maximum, int(m.group(1)))
    return maximum + 1


def parse_result(output_text: str) -> Tuple[float, float, List[float]]:
    final_matches = FINAL_RE.findall(output_text)
    if not final_matches:
        raise ValueError(
            "Could not find final 'test_acc: [mean:...][std:...]' line."
        )

    mean_s, std_s = final_matches[-1]
    seed_accs = [float(x) for x in SEED_ACC_RE.findall(output_text)]

    if len(seed_accs) != RUNS_PER_TASK:
        raise ValueError(
            f"Expected {RUNS_PER_TASK} per-seed {METHOD_NAME} test accuracies, "
            f"found {len(seed_accs)}."
        )

    return float(mean_s), float(std_s), seed_accs


def write_results_csv(output_dir: Path, state: dict) -> None:
    path = output_dir / "results.csv"
    rows = sorted(
        state.get("completed", {}).values(),
        key=lambda record: record["experiment_index"],
    )
    fields = [
        "experiment_index",
        "method",
        "dataset",
        "noise",
        "alpha",
        "c",
        "mean",
        "std",
        "seed_test_accs",
        "attempts",
        "duration_seconds",
        "completed_at",
        "log_file",
        "command",
    ]

    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fields}
            out["method"] = METHOD_NAME
            out["seed_test_accs"] = json.dumps(
                row.get("seed_test_accs", []), ensure_ascii=False
            )
            writer.writerow(out)
    os.replace(tmp, path)


def write_results_txt(output_dir: Path, state: dict) -> None:
    completed = state["completed"]
    lines = [
        f"{METHOD_NAME} experiment results",
        f"Completed configurations: {len(completed)}/{len(EXPERIMENTS)}",
        f"Seeds per configuration: {RUNS_PER_TASK}",
        "",
    ]

    for index, (dataset, noise, alpha, c, _extra_args) in enumerate(EXPERIMENTS, 1):
        label = f"[{index:02d}] {dataset}/{noise} alpha={alpha} c={c}"
        record = completed.get(task_key(index, dataset, noise))
        if record is None:
            lines.append(f"{label} PENDING")
        else:
            lines.append(
                f"{label} mean={record['mean']:.6f} std={record['std']:.6f} "
                f"seed_test_accs={record['seed_test_accs']}"
            )

    atomic_write_text(output_dir / "results.txt", "\n".join(lines) + "\n")


def write_all_results(output_dir: Path, state: dict) -> None:
    write_results_csv(output_dir, state)
    write_results_txt(output_dir, state)


def print_plan(logger: RunnerLogger) -> None:
    logger.log(
        f"Plan: {len(EXPERIMENTS)} configurations, one train.py invocation per configuration; "
        f"each task uses --runs {RUNS_PER_TASK}."
    )
    for idx, (dataset, noise, alpha, c, extra) in enumerate(EXPERIMENTS, 1):
        logger.log(
            f"EXP {idx:02d}: dataset={dataset} noise={noise} "
            f"alpha={alpha} c={c} {' '.join(extra)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the 12 fixed FracPIC-GCN experiments with resume."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="FracPIC-GCN project directory containing train.py (default: current directory).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to execute train.py (default: this interpreter).",
    )
    parser.add_argument(
        "--gpu",
        default="1",
        help="Physical CUDA GPU index placed in CUDA_VISIBLE_DEVICES (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        default="fracpic_gcn_results",
        help="Result directory, relative to project-root unless absolute.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Retries after a failure. 0 means unlimited immediate retries (default: 0).",
    )
    parser.add_argument(
        "--omp-threads",
        type=int,
        default=4,
        help="OMP_NUM_THREADS and MKL_NUM_THREADS for child processes (default: 4).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preflight and print the fixed plan without launching train.py.",
    )
    args = parser.parse_args()

    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    if args.omp_threads <= 0:
        parser.error("--omp-threads must be > 0")

    # Do not call Path.resolve() on the Python interpreter; resolving a venv
    # symlink can accidentally point at a base Python and lose venv packages.
    python_exe = (
        os.path.abspath(os.path.expanduser(args.python))
        if os.path.sep in args.python
        else args.python
    )
    project_root = Path(os.path.abspath(os.path.expanduser(args.project_root)))
    output_dir_arg = Path(os.path.expanduser(args.output_dir))
    output_dir = (
        output_dir_arg
        if output_dir_arg.is_absolute()
        else project_root / output_dir_arg
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = RunnerLogger(output_dir / "runner.log")
    lock_path = output_dir / ".runner.lock"
    acquire_lock(lock_path)
    atexit.register(release_lock, lock_path)

    env = build_child_env(str(args.gpu), args.omp_threads)

    logger.log("=" * 88)
    logger.log(f"Method        : {METHOD_NAME}")
    logger.log(f"Runner version: {SCRIPT_VERSION}")
    logger.log(f"Project root : {project_root}")
    logger.log(f"Python       : {python_exe}")
    logger.log(f"GPU request  : {args.gpu}")
    logger.log(f"Output dir   : {output_dir}")
    logger.log(f"Retry policy : {'UNLIMITED' if args.max_retries == 0 else args.max_retries}")
    logger.log("Retry delay  : 0 seconds (immediate)")
    logger.log(
        f"Threads      : OMP_NUM_THREADS={args.omp_threads}, "
        f"MKL_NUM_THREADS={args.omp_threads}"
    )

    train_py = project_root / "train.py"
    train_hash = sha256_file(train_py)
    payload = build_plan_payload(train_hash)
    signature = plan_signature(payload)

    run_preflight(python_exe, project_root, env, logger)
    print_plan(logger)

    if args.dry_run:
        logger.log("DRY RUN complete; no training process was started.")
        return 0

    state_path = output_dir / "state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SystemExit(f"Cannot read checkpoint {state_path}: {exc}")

        old_sig = state.get("signature")
        if old_sig != signature:
            raise SystemExit(
                "Checkpoint signature mismatch. The experiment plan or train.py "
                "has changed since this output directory was created. "
                "Use a new --output-dir rather than mixing results."
            )
        logger.log(
            f"Checkpoint loaded: {len(state.get('completed', {}))}/"
            f"{len(EXPERIMENTS)} tasks already completed."
        )
        if state.get("active"):
            logger.log(
                "Previous run stopped during task: "
                + json.dumps(state["active"], ensure_ascii=False)
                + ". That incomplete task will be rerun."
            )
    else:
        state = {
            "signature": signature,
            "plan": payload,
            "created_at": now_text(),
            "updated_at": now_text(),
            "completed": {},
            "active": None,
        }
        atomic_write_json(state_path, state)
        logger.log("New checkpoint created.")

    completed = state.setdefault("completed", {})
    total_tasks = len(EXPERIMENTS)

    for exp_idx, (dataset, noise, alpha, c, extra_args) in enumerate(EXPERIMENTS, 1):
        key = task_key(exp_idx, dataset, noise)

        if key in completed:
            rec = completed[key]
            logger.log(
                f"SKIP completed [{len(completed)}/{total_tasks}] "
                f"exp={exp_idx:02d} "
                f"{dataset}/{noise} alpha={alpha} c={c} "
                f"mean={rec.get('mean')} std={rec.get('std')}"
            )
            continue

        task_dir = output_dir / "logs" / key
        task_dir.mkdir(parents=True, exist_ok=True)
        attempt = next_attempt_number(task_dir)

        command = build_command(
            python_exe, dataset, noise, alpha, c, extra_args
        )
        command_text = shlex.join(command)
        task_start = time.time()

        while True:
            attempt_log = task_dir / f"attempt_{attempt:03d}.log"

            state["active"] = {
                "task_key": key,
                "experiment_index": exp_idx,
                "dataset": dataset,
                "noise": noise,
                "alpha": alpha,
                "c": c,
                "attempt": attempt,
                "command": command_text,
                "log_file": str(attempt_log.relative_to(output_dir)),
                "started_at": now_text(),
            }
            state["updated_at"] = now_text()
            atomic_write_json(state_path, state)

            logger.log("=" * 88)
            logger.log(
                f"START exp={exp_idx:02d}/{total_tasks} "
                f"dataset={dataset} noise={noise} alpha={alpha} c={c} "
                f"attempt={attempt}"
            )
            logger.log(f"COMMAND: {command_text}")
            logger.log(f"LOG: {attempt_log}")

            attempt_started = time.time()
            collected_lines: List[str] = []
            return_code: Optional[int] = None

            try:
                with attempt_log.open("w", encoding="utf-8", buffering=1) as log_f:
                    log_f.write(f"# started_at={now_text()}\n")
                    log_f.write(f"# command={command_text}\n")
                    log_f.write(
                        f"# CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}\n"
                    )
                    log_f.flush()

                    proc = subprocess.Popen(
                        command,
                        cwd=str(project_root),
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        errors="replace",
                    )

                    assert proc.stdout is not None
                    for line in proc.stdout:
                        print(line, end="", flush=True)
                        log_f.write(line)
                        collected_lines.append(line)

                    return_code = proc.wait()

            except KeyboardInterrupt:
                logger.log(
                    "KeyboardInterrupt received. The current train.py task "
                    "is incomplete and will be rerun when this runner is started again."
                )
                try:
                    if "proc" in locals() and proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                except Exception:
                    pass
                state["updated_at"] = now_text()
                atomic_write_json(state_path, state)
                return 130

            attempt_duration = time.time() - attempt_started
            output_text = "".join(collected_lines)

            if return_code == 0:
                try:
                    mean_value, std_value, seed_accs = parse_result(output_text)
                except Exception as exc:
                    logger.log(
                        f"ATTEMPT PARSE FAILURE after {attempt_duration:.1f}s: {exc}"
                    )
                else:
                    total_duration = time.time() - task_start
                    record = {
                        "task_key": key,
                        "experiment_index": exp_idx,
                        "dataset": dataset,
                        "noise": noise,
                        "alpha": alpha,
                        "c": c,
                        "mean": mean_value,
                        "std": std_value,
                        "seed_test_accs": seed_accs,
                        "attempts": attempt,
                        "duration_seconds": round(total_duration, 3),
                        "completed_at": now_text(),
                        "log_file": str(attempt_log.relative_to(output_dir)),
                        "command": command_text,
                    }
                    completed[key] = record
                    state["active"] = None
                    state["updated_at"] = now_text()
                    atomic_write_json(state_path, state)
                    write_all_results(output_dir, state)

                    logger.log(
                        f"SUCCESS [{len(completed)}/{total_tasks}] "
                        f"exp={exp_idx:02d} "
                        f"{dataset}/{noise} alpha={alpha} c={c} "
                        f"mean={mean_value:.6f} std={std_value:.6f} "
                        f"seed_accs={seed_accs} attempts={attempt}"
                    )
                    break
            else:
                logger.log(
                    f"ATTEMPT FAILED: return_code={return_code} "
                    f"duration={attempt_duration:.1f}s"
                )

            # Failure or result-parse failure: immediate retry, no sleep.
            retries_used = attempt - 1
            if args.max_retries > 0 and retries_used >= args.max_retries:
                logger.log(
                    f"Retry limit reached for {key}; checkpoint preserved."
                )
                state["updated_at"] = now_text()
                atomic_write_json(state_path, state)
                return 1

            attempt += 1
            logger.log(
                f"IMMEDIATE RETRY: {key}, next attempt={attempt}, delay=0s"
            )

    state["active"] = None
    state["updated_at"] = now_text()
    atomic_write_json(state_path, state)
    write_all_results(output_dir, state)

    logger.log("=" * 88)
    logger.log(
        f"ALL DONE: {len(completed)}/{total_tasks} train.py tasks completed."
    )
    logger.log(f"Results CSV: {output_dir / 'results.csv'}")
    logger.log(f"Results summary: {output_dir / 'results.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
