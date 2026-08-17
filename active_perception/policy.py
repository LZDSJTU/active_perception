"""Deterministic action order for the three observable properties."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    action: str
    target: str
    thinking: str = ""
    reason: str = ""


def legal_actions(knowledge):
    options = []
    for target, facts in knowledge.known.items():
        if facts["movable"] is None:
            options.append(f"push({target})")
        elif not facts["movable"]:
            continue
        elif facts["pressable"] is None:
            options.append(f"press({target})")
        elif not facts["pressable"]:
            continue
        elif facts["contains_target"] is None:
            options.append(f"lift_box({target})")
        elif facts["contains_target"]:
            options.append(f"stop({target})")
    return options


def validate_decision(decision, knowledge):
    expression = f"{decision.action}({decision.target})"
    options = legal_actions(knowledge)
    return (expression in options,
            "valid" if expression in options else
            f"{expression} is not one of: {', '.join(options)}")


def decision_from_expression(expression, reason="runtime safety fallback"):
    action, rest = expression.split("(", 1)
    return Decision(action, rest[:-1], "模型连续输出非法动作", reason)
