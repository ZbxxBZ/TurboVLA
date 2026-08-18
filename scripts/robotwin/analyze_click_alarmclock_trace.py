#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from statistics import fmean

def load_trace(path: Path):
    with path.open("r", encoding="utf-8") as trace_file:
        records = [json.loads(line) for line in trace_file if line.strip()]
    if not records:
        raise ValueError(f"trace is empty: {path}")
    return records


def analyze(records):
    active_arm = records[0]["before"]["active_arm"]
    ee_key = f"{active_arm}_ee_pose"
    joint_slice = slice(0, 6) if active_arm == "left" else slice(7, 13)
    gripper_index = 6 if active_arm == "left" else 13

    ee_positions = [record["after"][ee_key][:3] for record in records]
    targets = [record["after"]["target_position"] for record in records]
    actions = [record["action"] for record in records]
    states = [record["state"] for record in records]

    offsets = [
        [position[index] - target[index] for index in range(3)]
        for position, target in zip(ee_positions, targets)
    ]
    xy_distance = [math.hypot(offset[0], offset[1]) for offset in offsets]
    xyz_distance = [math.dist(position, target) for position, target in zip(ee_positions, targets)]
    z_error = [abs(offset[2]) for offset in offsets]
    active_actions = [action[joint_slice] for action in actions]
    action_delta_l2 = [
        math.dist(previous, current)
        for previous, current in zip(active_actions, active_actions[1:])
    ]
    ee_step_distance = [
        math.dist(previous, current)
        for previous, current in zip(ee_positions, ee_positions[1:])
    ]
    tracking_error_l2 = [
        math.dist(action[joint_slice], next_state[joint_slice])
        for action, next_state in zip(actions, states[1:])
    ]
    closed_indices = [
        index for index, action in enumerate(actions) if action[gripper_index] < 0.5
    ]

    return {
        "steps": len(records),
        "active_arm": active_arm,
        "success": any(record["stage_success"] for record in records),
        "initial_ee_z_m": float(ee_positions[0][2]),
        "minimum_ee_z_m": float(min(position[2] for position in ee_positions)),
        "final_ee_z_m": float(ee_positions[-1][2]),
        "max_downward_displacement_m": float(
            ee_positions[0][2] - min(position[2] for position in ee_positions)
        ),
        "min_xy_distance_to_button_m": float(min(xy_distance)),
        "min_abs_z_error_to_button_m": float(min(z_error)),
        "min_3d_distance_to_button_m": float(min(xyz_distance)),
        "steps_within_5cm_xy": sum(distance < 0.05 for distance in xy_distance),
        "steps_within_task_xyz_tolerance": sum(
            xy < 0.03 and z < 0.03 for xy, z in zip(xy_distance, z_error)
        ),
        "mean_active_joint_action_delta_l2": (
            float(fmean(action_delta_l2)) if action_delta_l2 else 0.0
        ),
        "max_active_joint_action_delta_l2": (
            float(max(action_delta_l2)) if action_delta_l2 else 0.0
        ),
        "mean_ee_step_distance_m": (
            float(fmean(ee_step_distance)) if ee_step_distance else 0.0
        ),
        "max_ee_step_distance_m": (
            float(max(ee_step_distance)) if ee_step_distance else 0.0
        ),
        "mean_joint_tracking_error_l2": (
            float(fmean(tracking_error_l2)) if tracking_error_l2 else 0.0
        ),
        "first_gripper_close_step": (
            int(records[closed_indices[0]]["step"]) if len(closed_indices) else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Gate label and JSONL path; may be repeated.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = {}
    for item in args.trace:
        label, separator, raw_path = item.partition("=")
        if not separator:
            parser.error(f"invalid --trace value: {item!r}")
        results[label] = analyze(load_trace(Path(raw_path)))

    rendered = json.dumps(results, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
