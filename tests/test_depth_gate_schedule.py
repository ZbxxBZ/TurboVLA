from third_party.starvla_runtime.starVLA.training.train_robotwin_clean_act_pi05_recipe import (
    linear_gate_warmup_value,
)


def test_linear_gate_warmup_reaches_end_without_transition_jump():
    assert linear_gate_warmup_value(0, 5, 0.02, 0.08) == 0.02
    assert linear_gate_warmup_value(2, 5, 0.02, 0.08) == 0.05
    assert linear_gate_warmup_value(4, 5, 0.02, 0.08) == 0.08
    assert linear_gate_warmup_value(5, 5, 0.02, 0.08) is None
    assert linear_gate_warmup_value(0, 0, 0.02, 0.08) is None
