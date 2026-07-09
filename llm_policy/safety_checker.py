from __future__ import annotations

import ast
from dataclasses import dataclass


def clip_steering(angle: float, limit: float = 0.45) -> float:
    return max(-limit, min(limit, float(angle)))


def safe_following_speed(distance_m: float, current_speed_mps: float) -> float:
    if distance_m < 7.0:
        return 3.0
    if distance_m < 13.0:
        return min(current_speed_mps, 5.5)
    return min(current_speed_mps, 8.0)


ALLOWED_CALLS = {
    "abs",
    "bool",
    "execute_primitive",
    "float",
    "int",
    "mark_task_complete",
    "max",
    "min",
    "observe_driving_state",
    "observe_execution_feedback",
    "range",
    "task_finished",
}

ALLOWED_METHOD_CALLS = {
    "as_float",
}

DISALLOWED_NODES = (
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.With,
    ast.AsyncWith,
)


@dataclass(frozen=True)
class PolicyValidationResult:
    ok: bool
    errors: tuple[str, ...] = ()


def validate_generated_policy_code(code: str) -> PolicyValidationResult:
    """Validate that policy code only orchestrates exposed driving APIs."""
    errors: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return PolicyValidationResult(False, (f"syntax error: {exc.msg}",))

    function_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    policy_defs = [node for node in function_defs if node.name == "policy"]
    if len(policy_defs) != 1:
        errors.append("define exactly one function named policy()")
    if len(function_defs) != 1:
        errors.append("do not define helper functions")
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            errors.append("top-level code is not allowed outside policy()")
            break

    execute_primitive_calls = 0
    observe_state_calls = 0
    feedback_calls = 0
    for node in ast.walk(tree):
        if isinstance(node, DISALLOWED_NODES):
            errors.append(f"{type(node).__name__} is not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            errors.append("dunder attribute access is not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            errors.append("dunder names are not allowed")
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name == "execute_primitive":
                execute_primitive_calls += 1
            if name == "observe_driving_state":
                observe_state_calls += 1
            if name == "observe_execution_feedback":
                feedback_calls += 1
            if isinstance(node.func, ast.Name):
                if node.func.id not in ALLOWED_CALLS:
                    errors.append(f"call to {node.func.id}() is not allowed")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr not in ALLOWED_METHOD_CALLS:
                    errors.append(f"method call .{node.func.attr}() is not allowed")
            else:
                errors.append("dynamic calls are not allowed")

    if execute_primitive_calls == 0:
        errors.append("policy must call execute_primitive(...) at least once")
    if observe_state_calls == 0:
        errors.append("policy must read observe_driving_state()")
    if feedback_calls == 0:
        errors.append("policy must read observe_execution_feedback()")

    return PolicyValidationResult(not errors, tuple(dict.fromkeys(errors)))


def _call_name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return "<dynamic>"
