"""Run resumable, unattended RoboTwin RGB-D collection across available GPUs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from validate_robotwin_rgbd import scan_task


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TASK_FILE = SCRIPT_DIR / "rgbd_tasks_10.txt"
DEFAULT_CONFIG_TEMPLATE = SCRIPT_DIR / "configs" / "demo_clean_depth.yml"
TASK_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_tasks(path: Path) -> list[str]:
    tasks: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        task = raw_line.split("#", 1)[0].strip()
        if not task:
            continue
        if not TASK_NAME_RE.fullmatch(task):
            raise ValueError(f"invalid task name in {path}: {task!r}")
        if task in tasks:
            raise ValueError(f"duplicate task name in {path}: {task}")
        tasks.append(task)
    if not tasks:
        raise ValueError(f"task file is empty: {path}")
    return tasks


def parse_gpu_list(raw_value: str | None) -> list[str]:
    if raw_value:
        devices = [value.strip() for value in raw_value.split(",") if value.strip()]
    else:
        visible = os.environ.get("ROBOTWIN_COLLECTION_GPUS") or os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        )
        if visible:
            devices = [value.strip() for value in visible.split(",") if value.strip()]
        else:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            )
            devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not devices:
        raise RuntimeError("no NVIDIA GPUs detected; pass --gpus explicitly")
    if len(set(devices)) != len(devices):
        raise ValueError(f"GPU list contains duplicates: {devices}")
    return devices


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def stage_config(
    template_path: Path,
    target_path: Path,
    output_root: Path,
    episodes_per_task: int,
) -> None:
    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"config template must contain a YAML mapping: {template_path}")
    config["episode_num"] = episodes_per_task
    config["save_path"] = str(output_root)
    data_type = config.setdefault("data_type", {})
    camera = config.setdefault("camera", {})
    data_type.update({"rgb": True, "depth": True, "qpos": True, "endpose": True})
    # Preserve all three RGB views. The RoboTwin HDF5 patch keeps depth only
    # for the fixed head camera.
    camera.update({"collect_head_camera": True, "collect_wrist_camera": True})
    config.update({"render_freq": 0, "collect_data": True})

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_path.with_suffix(target_path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(temporary, target_path)


def prepare_environment(base: dict[str, str], gpu: str) -> dict[str, str]:
    environment = base.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("PYTHONWARNINGS", "ignore::UserWarning")
    return environment


def run_update_path(robotwin_path: Path, environment: dict[str, str]) -> None:
    update_path = robotwin_path / "script" / ".update_path.sh"
    if not update_path.is_file():
        return
    result = subprocess.run(
        ["bash", str(update_path)],
        cwd=robotwin_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"RoboTwin path update failed ({update_path}):\n{result.stdout}\n{result.stderr}"
        )


def render_smoke_test(
    robotwin_path: Path,
    python: Path,
    gpus: list[str],
    log_dir: Path,
    environment: dict[str, str],
    test_script: Path,
) -> None:
    if not test_script.is_file():
        raise FileNotFoundError(f"RoboTwin render probe is missing: {test_script}")
    log_dir.mkdir(parents=True, exist_ok=True)

    def probe(slot_and_gpu: tuple[int, str]) -> tuple[int, str, bool, str]:
        slot, gpu = slot_and_gpu
        gpu_safe = re.sub(r"[^A-Za-z0-9_.-]", "_", gpu)
        log_path = log_dir / f"render_smoke_slot_{slot:02d}_gpu_{gpu_safe}.log"
        outputs: list[str] = []
        for attempt in range(1, 4):
            try:
                result = subprocess.run(
                    [str(python), "-u", str(test_script)],
                    cwd=robotwin_path,
                    env=prepare_environment(environment, gpu),
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                output = result.stdout + result.stderr
                outputs.append(f"=== attempt {attempt}/3 ===\n{output}")
                if result.returncode == 0 and "Render Well" in output:
                    log_path.write_text("\n".join(outputs), encoding="utf-8")
                    return slot, gpu, True, str(log_path)
            except subprocess.TimeoutExpired as error:
                output = (error.stdout or "") + (error.stderr or "")
                outputs.append(f"=== attempt {attempt}/3 timed out ===\n{output}")
            if attempt < 3:
                time.sleep(5)
        log_path.write_text("\n".join(outputs), encoding="utf-8")
        return slot, gpu, False, str(log_path)

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        results = list(executor.map(probe, enumerate(gpus)))
    failed = [
        (slot, gpu, log) for slot, gpu, success, log in results if not success
    ]
    if failed:
        details = ", ".join(
            f"slot {slot} GPU {gpu}: {log}" for slot, gpu, log in failed
        )
        raise RuntimeError(f"concurrent headless rendering smoke test failed: {details}")


def latest_activity_ns(paths: list[Path]) -> int:
    latest = 0
    for path in paths:
        try:
            latest = max(latest, path.stat().st_mtime_ns)
        except FileNotFoundError:
            continue
    return latest


def terminate_process(process: subprocess.Popen[Any], grace_seconds: float = 30.0) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGKILL)
    else:
        process.kill()
    process.wait(timeout=10)


def quarantine_file(path: Path, quarantine_dir: Path, reason: str) -> Path:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = quarantine_dir / f"{path.stem}_{timestamp}_{reason}{path.suffix}"
    suffix = 1
    while candidate.exists():
        candidate = quarantine_dir / (
            f"{path.stem}_{timestamp}_{reason}_{suffix}{path.suffix}"
        )
        suffix += 1
    os.replace(path, candidate)
    return candidate


def quarantine_invalid_episodes(
    report: dict[str, Any], quarantine_root: Path, task: str
) -> list[str]:
    moved: list[str] = []
    for episode in report["episodes"]:
        path = Path(episode["path"])
        if not episode["valid"] and path.is_file():
            destination = quarantine_file(path, quarantine_root / task, "invalid")
            moved.append(f"{path} -> {destination}")
    return moved


@dataclass
class AttemptResult:
    task: str
    gpu: str
    attempt: int
    return_code: int
    reason: str
    log_path: str
    started_at: str
    finished_at: str
    elapsed_seconds: float
    quarantined: list[str] = field(default_factory=list)


@dataclass
class TaskState:
    task: str
    status: str = "pending"
    attempts: int = 0
    gpu: str = ""
    valid_episodes: int = 0
    target_episodes: int = 0
    total_frames: int = 0
    last_error: str = ""
    log_path: str = ""
    started_at: str = ""
    finished_at: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class RuntimeSettings:
    robotwin_path: Path
    collector_path: Path
    python: Path
    output_root: Path
    config_name: str
    embodiment_dir: str
    episodes_per_task: int
    log_dir: Path
    quarantine_dir: Path
    attempt_timeout_seconds: float
    stall_timeout_seconds: float
    poll_seconds: float
    environment: dict[str, str]
    stop_event: threading.Event


def run_attempt(
    task: str,
    gpu: str,
    attempt: int,
    settings: RuntimeSettings,
) -> AttemptResult:
    report_before = scan_task(
        settings.output_root,
        task,
        settings.config_name,
        settings.episodes_per_task,
        settings.embodiment_dir,
    )
    task_dir = Path(report_before["task_dir"])
    data_dir = Path(report_before["data_dir"])
    quarantined = quarantine_invalid_episodes(
        report_before, settings.quarantine_dir, task
    )
    before_mtime = {
        path: path.stat().st_mtime_ns
        for path in data_dir.glob("episode*.hdf5")
        if path.is_file()
    }

    log_path = settings.log_dir / f"{task}_attempt_{attempt:02d}_gpu_{gpu}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(settings.python),
        "-u",
        str(settings.collector_path),
        task,
        settings.config_name,
    ]
    started_at = utc_now()
    started_monotonic = time.monotonic()
    environment = prepare_environment(settings.environment, gpu)

    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n[{started_at}] command: {' '.join(command)}\n")
        log.write(f"[{started_at}] CUDA_VISIBLE_DEVICES={gpu}\n")
        process = subprocess.Popen(
            command,
            cwd=settings.robotwin_path,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=(os.name == "posix"),
        )
        watched_paths = [log_path, task_dir / "seed.txt", data_dir]
        last_signature = latest_activity_ns(watched_paths)
        last_activity = time.monotonic()
        reason = "process-exit"
        while process.poll() is None:
            if settings.stop_event.wait(settings.poll_seconds):
                reason = "scheduler-stopped"
                terminate_process(process)
                break
            now = time.monotonic()
            signature = latest_activity_ns(watched_paths)
            if signature != last_signature:
                last_signature = signature
                last_activity = now
            if now - started_monotonic > settings.attempt_timeout_seconds:
                reason = "attempt-timeout"
                log.write(f"[{utc_now()}] terminating after attempt timeout\n")
                terminate_process(process)
                break
            if now - last_activity > settings.stall_timeout_seconds:
                reason = "stalled"
                log.write(f"[{utc_now()}] terminating after no observable progress\n")
                terminate_process(process)
                break
        return_code = process.wait()

    report_after = scan_task(
        settings.output_root,
        task,
        settings.config_name,
        settings.episodes_per_task,
        settings.embodiment_dir,
    )
    data_dir = Path(report_after["data_dir"])
    if not report_after["complete"] and (return_code != 0 or reason != "process-exit"):
        # RoboTwin writes HDF5 before its final task-success assertion. On a failed
        # attempt, conservatively quarantine only the newest changed episode. This
        # may recollect one valid episode, but cannot count an unsuccessful replay.
        changed: list[tuple[int, Path]] = []
        for path in data_dir.glob("episode*.hdf5"):
            match = re.fullmatch(r"episode_?(\d+)\.hdf5", path.name)
            if not match:
                continue
            if before_mtime.get(path) == path.stat().st_mtime_ns:
                continue
            changed.append((int(match.group(1)), path))
        if changed:
            _, suspect = max(changed)
            destination = quarantine_file(
                suspect, settings.quarantine_dir / task, "attempt_failed"
            )
            quarantined.append(f"{suspect} -> {destination}")

    elapsed = time.monotonic() - started_monotonic
    return AttemptResult(
        task=task,
        gpu=gpu,
        attempt=attempt,
        return_code=return_code,
        reason=reason,
        log_path=str(log_path),
        started_at=started_at,
        finished_at=utc_now(),
        elapsed_seconds=round(elapsed, 1),
        quarantined=quarantined,
    )


def build_summary(
    states: dict[str, TaskState],
    output_root: Path,
    config_name: str,
    episodes_per_task: int,
    gpus: list[str],
) -> dict[str, Any]:
    ordered_states = list(states.values())
    return {
        "updated_at": utc_now(),
        "output_root": str(output_root),
        "config_name": config_name,
        "gpus": gpus,
        "episodes_per_task": episodes_per_task,
        "tasks_total": len(ordered_states),
        "tasks_complete": sum(state.status == "complete" for state in ordered_states),
        "tasks_failed": sum(state.status == "failed" for state in ordered_states),
        "valid_episodes": sum(state.valid_episodes for state in ordered_states),
        "target_episodes": len(ordered_states) * episodes_per_task,
        "total_frames": sum(state.total_frames for state in ordered_states),
        "complete": all(state.status == "complete" for state in ordered_states),
        "tasks": [asdict(state) for state in ordered_states],
    }


def refresh_state(
    state: TaskState,
    output_root: Path,
    config_name: str,
    episodes_per_task: int,
    embodiment_dir: str,
) -> dict[str, Any]:
    report = scan_task(
        output_root,
        state.task,
        config_name,
        episodes_per_task,
        embodiment_dir,
    )
    state.valid_episodes = report["valid_episodes"]
    state.target_episodes = episodes_per_task
    state.total_frames = report["total_frames"]
    return report


def print_progress(summary: dict[str, Any]) -> None:
    task_parts = []
    for state in summary["tasks"]:
        task_parts.append(
            f"{state['task']}={state['valid_episodes']}/{state['target_episodes']}"
            f"({state['status']})"
        )
    print(
        f"[{summary['updated_at']}] RGB-D progress "
        f"{summary['valid_episodes']}/{summary['target_episodes']} episodes; "
        + ", ".join(task_parts),
        flush=True,
    )


def resolve_robotwin_layout(robotwin_path: Path) -> tuple[Path, Path]:
    layouts = (
        (
            robotwin_path / "scripts" / "collect_data.py",
            robotwin_path / "env_cfg" / "task_config",
        ),
        (
            robotwin_path / "script" / "collect_data.py",
            robotwin_path / "task_config",
        ),
    )
    for collector_path, config_dir in layouts:
        if collector_path.is_file() and config_dir.is_dir():
            return collector_path, config_dir
    checked = ", ".join(str(collector) for collector, _ in layouts)
    raise FileNotFoundError(
        f"no supported RoboTwin collector found; checked: {checked}"
    )


def validate_preconditions(
    args: argparse.Namespace, tasks: list[str], collector_path: Path
) -> None:
    if not args.robotwin_path.is_dir():
        raise FileNotFoundError(f"RoboTwin repository not found: {args.robotwin_path}")
    if not collector_path.is_file():
        raise FileNotFoundError(f"RoboTwin collector not found: {collector_path}")
    if not args.python.is_file():
        raise FileNotFoundError(f"RoboTwin Python not found: {args.python}")
    if not args.config_template.is_file():
        raise FileNotFoundError(f"RGB-D config template not found: {args.config_template}")
    if is_under(args.robotwin_path, Path("/mnt")):
        raise ValueError("copy RoboTwin source out of /mnt before collection")
    if is_under(args.output_root, Path("/mnt")):
        raise ValueError("collection output must be outside /mnt")
    if args.episodes_per_task <= 0:
        raise ValueError("--episodes-per-task must be positive")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    for task in tasks:
        task_module = args.robotwin_path / "envs" / f"{task}.py"
        task_package = args.robotwin_path / "envs" / task / "__init__.py"
        if not task_module.is_file() and not task_package.is_file():
            raise FileNotFoundError(f"RoboTwin task implementation not found: {task}")


def check_disk_space(path: Path, min_free_gb: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / (1024**3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"only {free_gb:.1f} GiB free at {path}; require {min_free_gb:.1f} GiB "
            "(--min-free-gb overrides the guard)"
        )
    print(f"[INFO] output disk free: {free_gb:.1f} GiB", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robotwin-path",
        type=Path,
        default=Path(os.environ.get("ROBOTWIN_PATH", "/root/RoboTwin")),
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(os.environ.get("ROBOTWIN_PYTHON", sys.executable)),
        help="Python executable from the working RoboTwin conda environment",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("ROBOTWIN_RGBD_ROOT", "/root/robotwin_rgbd_raw")),
    )
    parser.add_argument("--task-file", type=Path, default=DEFAULT_TASK_FILE)
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--config-template", type=Path, default=DEFAULT_CONFIG_TEMPLATE)
    parser.add_argument("--config-name", default="demo_clean_depth_turbovla")
    parser.add_argument("--embodiment-dir", default="aloha_agilex")
    parser.add_argument("--gpus", help="comma-separated physical GPU indices or UUIDs")
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=int(os.environ.get("ROBOTWIN_WORKERS_PER_GPU", "1")),
        help="concurrent RoboTwin collection processes per physical GPU",
    )
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--attempt-timeout-hours", type=float, default=8.0)
    parser.add_argument("--stall-timeout-minutes", type=float, default=60.0)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--retry-delay-seconds", type=float, default=30.0)
    parser.add_argument("--min-free-gb", type=float, default=50.0)
    parser.add_argument("--skip-render-smoke", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.robotwin_path = args.robotwin_path.expanduser().resolve()
    # Keep virtual-environment interpreter symlinks intact. Resolving this path can
    # bypass the venv prefix and make child processes lose its site-packages.
    args.python = args.python.expanduser().absolute()
    args.output_root = args.output_root.expanduser().resolve()
    args.task_file = args.task_file.expanduser().resolve()
    args.config_template = args.config_template.expanduser().resolve()
    tasks = read_tasks(args.task_file)
    physical_gpus = parse_gpu_list(args.gpus)
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be at least 1")
    gpus = [
        gpu
        for gpu in physical_gpus
        for _worker_index in range(args.workers_per_gpu)
    ]
    collector_path, config_dir = resolve_robotwin_layout(args.robotwin_path)

    states = {
        task: TaskState(task=task, target_episodes=args.episodes_per_task)
        for task in tasks
    }
    for state in states.values():
        report = refresh_state(
            state,
            args.output_root,
            args.config_name,
            args.episodes_per_task,
            args.embodiment_dir,
        )
        if report["complete"]:
            state.status = "complete"

    control_dir = args.output_root / "_autocollect"
    status_path = control_dir / "status.json"
    log_dir = control_dir / "logs"
    quarantine_dir = control_dir / "quarantine"

    initial_summary = build_summary(
        states, args.output_root, args.config_name, args.episodes_per_task, gpus
    )
    if args.status_only:
        print(json.dumps(initial_summary, ensure_ascii=False, indent=2))
        return 0 if initial_summary["complete"] else 1

    validate_preconditions(args, tasks, collector_path)
    check_disk_space(args.output_root, args.min_free_gb)
    stage_config(
        args.config_template,
        config_dir / f"{args.config_name}.yml",
        args.output_root,
        args.episodes_per_task,
    )
    print(
        f"[INFO] tasks={len(tasks)}, episodes_per_task={args.episodes_per_task}, "
        f"target={len(tasks) * args.episodes_per_task}, "
        f"physical_gpus={','.join(physical_gpus)}, "
        f"workers_per_gpu={args.workers_per_gpu}, worker_slots={len(gpus)}",
        flush=True,
    )
    print(f"[INFO] output={args.output_root}", flush=True)
    print(f"[INFO] collector={collector_path}", flush=True)
    print(f"[INFO] status={status_path}", flush=True)

    environment = os.environ.copy()
    run_update_path(args.robotwin_path, environment)
    if not args.skip_render_smoke and not args.dry_run:
        print("[INFO] running concurrent headless rendering smoke test", flush=True)
        render_smoke_test(
            args.robotwin_path,
            args.python,
            gpus,
            log_dir,
            environment,
            collector_path.parent / "test_render.py",
        )

    for state in states.values():
        refresh_state(
            state,
            args.output_root,
            args.config_name,
            args.episodes_per_task,
            args.embodiment_dir,
        )
    summary = build_summary(
        states, args.output_root, args.config_name, args.episodes_per_task, gpus
    )
    atomic_write_json(status_path, summary)
    print_progress(summary)
    if args.dry_run or summary["complete"]:
        return 0

    stop_event = threading.Event()
    settings = RuntimeSettings(
        robotwin_path=args.robotwin_path,
        collector_path=collector_path,
        python=args.python,
        output_root=args.output_root,
        config_name=args.config_name,
        embodiment_dir=args.embodiment_dir,
        episodes_per_task=args.episodes_per_task,
        log_dir=log_dir,
        quarantine_dir=quarantine_dir,
        attempt_timeout_seconds=args.attempt_timeout_hours * 3600,
        stall_timeout_seconds=args.stall_timeout_minutes * 60,
        poll_seconds=args.poll_seconds,
        environment=environment,
        stop_event=stop_event,
    )

    pending = [task for task, state in states.items() if state.status != "complete"]
    available_gpus = list(gpus)
    active: dict[Future[AttemptResult], tuple[str, str]] = {}
    last_progress_print = 0.0

    previous_signal_handlers: dict[signal.Signals, Any] = {}

    def request_stop(signum: int, _frame: Any) -> None:
        stop_event.set()
        raise KeyboardInterrupt(f"received signal {signum}")

    for stop_signal in (signal.SIGINT, signal.SIGTERM):
        previous_signal_handlers[stop_signal] = signal.getsignal(stop_signal)
        signal.signal(stop_signal, request_stop)

    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            while pending or active:
                while pending and available_gpus:
                    task = pending.pop(0)
                    state = states[task]
                    if state.attempts >= args.max_attempts:
                        state.status = "failed"
                        state.finished_at = utc_now()
                        continue
                    gpu = available_gpus.pop(0)
                    state.attempts += 1
                    state.status = "running"
                    state.gpu = gpu
                    state.started_at = state.started_at or utc_now()
                    print(
                        f"[INFO] starting task={task} attempt={state.attempts}/"
                        f"{args.max_attempts} gpu={gpu}",
                        flush=True,
                    )
                    future = executor.submit(
                        run_attempt, task, gpu, state.attempts, settings
                    )
                    active[future] = (task, gpu)

                done, _ = wait(
                    active,
                    timeout=max(args.poll_seconds, 1.0),
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    task, gpu = active.pop(future)
                    available_gpus.append(gpu)
                    state = states[task]
                    try:
                        result = future.result()
                        state.history.append(asdict(result))
                        state.log_path = result.log_path
                        report = refresh_state(
                            state,
                            args.output_root,
                            args.config_name,
                            args.episodes_per_task,
                            args.embodiment_dir,
                        )
                        if report["complete"]:
                            state.status = "complete"
                            state.finished_at = utc_now()
                            state.last_error = ""
                            print(
                                f"[INFO] completed task={task}: "
                                f"{state.valid_episodes}/{state.target_episodes} episodes",
                                flush=True,
                            )
                        else:
                            state.last_error = (
                                f"exit={result.return_code}, reason={result.reason}, "
                                f"progress={state.valid_episodes}/{state.target_episodes}"
                            )
                            if state.attempts < args.max_attempts:
                                state.status = "retrying"
                                print(
                                    f"[WARN] task={task} will retry: {state.last_error}; "
                                    f"log={result.log_path}",
                                    flush=True,
                                )
                                if stop_event.wait(args.retry_delay_seconds):
                                    raise KeyboardInterrupt
                                pending.append(task)
                            else:
                                state.status = "failed"
                                state.finished_at = utc_now()
                                print(
                                    f"[ERROR] task={task} exhausted retries: "
                                    f"{state.last_error}",
                                    flush=True,
                                )
                    except Exception as error:
                        state.last_error = f"scheduler error: {error}"
                        if state.attempts < args.max_attempts:
                            state.status = "retrying"
                            pending.append(task)
                        else:
                            state.status = "failed"
                            state.finished_at = utc_now()
                        print(f"[ERROR] task={task}: {state.last_error}", flush=True)

                now = time.monotonic()
                if done or now - last_progress_print >= 60:
                    for state in states.values():
                        refresh_state(
                            state,
                            args.output_root,
                            args.config_name,
                            args.episodes_per_task,
                            args.embodiment_dir,
                        )
                    summary = build_summary(
                        states,
                        args.output_root,
                        args.config_name,
                        args.episodes_per_task,
                        gpus,
                    )
                    atomic_write_json(status_path, summary)
                    print_progress(summary)
                    last_progress_print = now
    except KeyboardInterrupt:
        print("[WARN] stopping active collection processes", flush=True)
        stop_event.set()
        for state in states.values():
            if state.status == "running":
                state.status = "interrupted"
        return_code = 130
    else:
        return_code = 0
    finally:
        stop_event.set()
        for stop_signal, previous_handler in previous_signal_handlers.items():
            signal.signal(stop_signal, previous_handler)
        for state in states.values():
            refresh_state(
                state,
                args.output_root,
                args.config_name,
                args.episodes_per_task,
                args.embodiment_dir,
            )
            if state.valid_episodes == state.target_episodes:
                state.status = "complete"
                state.finished_at = state.finished_at or utc_now()
        summary = build_summary(
            states, args.output_root, args.config_name, args.episodes_per_task, gpus
        )
        atomic_write_json(status_path, summary)
        print_progress(summary)

    if summary["complete"]:
        print(
            f"[INFO] collection complete: {summary['valid_episodes']} synchronized "
            f"RGB-D episodes, {summary['total_frames']} frames",
            flush=True,
        )
        return 0
    failed_tasks = [state.task for state in states.values() if state.status == "failed"]
    if failed_tasks:
        print(f"[ERROR] incomplete tasks: {', '.join(failed_tasks)}", flush=True)
    return return_code or 1


if __name__ == "__main__":
    raise SystemExit(main())
