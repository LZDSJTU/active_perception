from pathlib import Path
import pytest
from active_perception.config import load_config, motion_envelope_clearance
from active_perception.policy import Decision, legal_actions, validate_decision
from active_perception.qwen import parse_ball_inspection, parse_decision
from active_perception.state import KnowledgeState
from active_perception.visualization import knowledge_rows

CONFIGS = sorted(Path("configs/final_showcase").glob("*.yaml"))

@pytest.mark.parametrize("path", CONFIGS)
def test_all_experiment_configs_are_safe(path):
    config = load_config(path)
    assert motion_envelope_clearance(config)["minimum_center_distance_m"] >= .1
    assert len(config.target_ids) == 1
    assert config.target_ids <= config.movable_ids & config.pressable_ids

def test_policy_enforces_property_order():
    state = KnowledgeState(["A"], ["movable", "pressable", "contains_target"])
    assert legal_actions(state) == ["push(A)"]
    state.update("A", {"movable": True})
    assert legal_actions(state) == ["press(A)"]
    state.update("A", {"pressable": True})
    assert legal_actions(state) == ["lift_box(A)"]
    state.update("A", {"contains_target": True})
    assert legal_actions(state) == ["stop(A)"]
    assert validate_decision(Decision("stop", "A"), state)[0]

def test_negative_property_skips_candidate():
    state = KnowledgeState(["A", "B"], ["movable", "pressable", "contains_target"])
    state.update("A", {"movable": False})
    assert legal_actions(state) == ["push(B)"]

def test_qwen_output_parsers_are_strict():
    assert parse_decision('{"action":"lift_box","target":"C"}').target == "C"
    assert parse_ball_inspection(
        '{"contains_green_ball":true,"confidence":"high","reason":"visible"}'
    )["value"] is True
    with pytest.raises(ValueError):
        parse_ball_inspection('{"contains_green_ball":"yes"}')

def test_video_labels_use_human_readable_words():
    state = KnowledgeState(["A"], ["movable", "pressable", "contains_target"])
    assert knowledge_rows(state, ["A"]) == ["A: movable=? clickable=? ball=?"]
