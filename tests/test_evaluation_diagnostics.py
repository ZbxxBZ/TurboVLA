from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.robotwin.evaluation.model2robotwin_interface import (
    _append_action_trace,
    _get_action_trace_context,
)
from scripts.robotwin.analyze_click_alarmclock_trace import analyze
from third_party.starvla_runtime.deployment.model_server.server_policy import (
    apply_depth_gate_override,
)


class _Pose:
    p = np.array([0.1, -0.2, 0.3])


class _Alarm:
    def get_pose(self):
        return _Pose()

    def get_contact_point(self, _index):
        return np.array([0.11, -0.19, 0.34, 1.0])


class _Robot:
    def get_left_ee_pose(self):
        return np.arange(7)

    def get_right_ee_pose(self):
        return np.arange(7) + 10


class _Fusion:
    def __init__(self):
        self.value = None

    def set_gate_override(self, value):
        self.value = value

    def effective_gate(self):
        return torch.full((4,), self.value)


def test_action_trace_context_and_jsonl(tmp_path, monkeypatch):
    env = SimpleNamespace(robot=_Robot(), alarm=_Alarm(), stage_success_tag=True)
    context = _get_action_trace_context(env)
    assert context["active_arm"] == "right"
    assert context["target_position"] == [0.11, -0.19, 0.34]

    trace_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("ROBOTWIN_ACTION_TRACE_PATH", str(trace_path))
    _append_action_trace(env, 3, np.zeros(14), np.ones(14), context, context)

    assert '"step":3' in trace_path.read_text(encoding="utf-8")
    assert '"stage_success":true' in trace_path.read_text(encoding="utf-8")


def test_apply_depth_gate_override():
    fusion = _Fusion()
    vla = SimpleNamespace(model=SimpleNamespace(depth_fusion=fusion))
    assert apply_depth_gate_override(vla, "0.16") == pytest.approx(0.16)
    assert fusion.value == pytest.approx(0.16)


def test_apply_depth_gate_override_rejects_incompatible_model():
    with pytest.raises(RuntimeError, match="no compatible depth fusion"):
        apply_depth_gate_override(SimpleNamespace(), "0.08")


def test_analyze_click_alarmclock_trace_metrics():
    records = []
    for step, z, gripper in [(0, 0.40, 1.0), (1, 0.35, 0.0), (2, 0.32, 0.0)]:
        action = np.zeros(14)
        action[7:13] = step * 0.1
        action[13] = gripper
        context = {
            "active_arm": "right",
            "target_position": [0.1, 0.2, 0.30],
            "left_ee_pose": [0.0] * 7,
            "right_ee_pose": [0.1, 0.2, z, 1.0, 0.0, 0.0, 0.0],
        }
        records.append(
            {
                "step": step,
                "state": np.zeros(14).tolist(),
                "action": action.tolist(),
                "before": context,
                "after": context,
                "stage_success": step == 2,
            }
        )

    metrics = analyze(records)
    assert metrics["active_arm"] == "right"
    assert metrics["success"] is True
    assert metrics["max_downward_displacement_m"] == pytest.approx(0.08)
    assert metrics["min_3d_distance_to_button_m"] == pytest.approx(0.02)
    assert metrics["first_gripper_close_step"] == 1
