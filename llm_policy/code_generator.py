from __future__ import annotations

import subprocess
from pathlib import Path


DEFAULT_STAGE1_POLICY = r'''
def policy():
    while not task_finished():
        state = observe_driving_state()
        feedback = observe_execution_feedback()

        if state.perception_confidence.value < 0.45:
            slow_down()
            return_center(duration=0.3)
            continue

        if feedback.event == "front_vehicle_too_close":
            slow_down()
            return_center(duration=0.4)
            continue

        if feedback.event == "turn_too_early":
            return_center(duration=0.4)
            hold_center()
            continue

        if feedback.event == "turn_too_late":
            direction = state.route_command.value
            if direction in ("turn_left", "turn_right"):
                execute_turn(direction)
            else:
                hold_center()
            continue

        if feedback.event in ("under_steer", "over_steer", "lane_departure_risk"):
            repeat_with_stronger_steering()
            continue

        if state.front_vehicle_exists.value and state.front_vehicle_distance_m.value is not None:
            if state.front_vehicle_distance_m.value < 13.0:
                follow_front_vehicle()
                continue

        if state.route_command.value in ("turn_left", "turn_right"):
            if state.distance_to_maneuver_m.value is not None and state.distance_to_maneuver_m.value < 9.5:
                execute_turn(state.route_command.value)
            else:
                prepare_turn(state.route_command.value)
            continue

        offset = state.lane_center_offset_m.value
        heading = state.heading_error_rad.value
        curvature = state.lane_curvature.value

        if offset > 0.35 and heading < -0.10:
            hold_center()
            continue

        if offset < -0.35 and heading > 0.10:
            hold_center()
            continue

        if offset > 0.45 and heading > 0.04:
            hard_left()
            continue

        if offset < -0.45 and heading < -0.04:
            hard_right()
            continue

        if offset > 0.90:
            hard_left()
            continue

        if offset < -0.90:
            hard_right()
            continue

        if offset > 0.35:
            medium_left()
            continue

        if offset < -0.35:
            medium_right()
            continue

        if curvature > 0.015:
            medium_right()
            continue

        if curvature > 0.006:
            soft_right()
            continue

        if curvature < -0.015:
            medium_left()
            continue

        if curvature < -0.006:
            soft_left()
            continue

        if heading > 0.18:
            medium_left()
            continue

        if heading < -0.18:
            medium_right()
            continue

        if offset > 0.15 or heading > 0.06:
            soft_left()
            continue

        if offset < -0.15 or heading < -0.06:
            soft_right()
            continue

        hold_center()
'''

NO_FEEDBACK_POLICY = r'''
def policy():
    while not task_finished():
        state = observe_driving_state()

        if state.front_vehicle_exists.value and state.front_vehicle_distance_m.value is not None:
            if state.front_vehicle_distance_m.value < 13.0:
                follow_front_vehicle()
                continue

        if state.route_command.value in ("turn_left", "turn_right"):
            if state.distance_to_maneuver_m.value is not None and state.distance_to_maneuver_m.value < 9.5:
                execute_turn(state.route_command.value)
            else:
                prepare_turn(state.route_command.value)
            continue

        offset = state.lane_center_offset_m.value
        heading = state.heading_error_rad.value
        curvature = state.lane_curvature.value

        if offset > 0.35 and heading < -0.10:
            hold_center()
            continue

        if offset < -0.35 and heading > 0.10:
            hold_center()
            continue

        if offset > 0.45 and heading > 0.04:
            hard_left()
            continue

        if offset < -0.45 and heading < -0.04:
            hard_right()
            continue

        if offset > 0.90:
            hard_left()
            continue

        if offset < -0.90:
            hard_right()
            continue

        if offset > 0.35:
            medium_left()
            continue

        if offset < -0.35:
            medium_right()
            continue

        if curvature > 0.015:
            medium_right()
            continue

        if curvature > 0.006:
            soft_right()
            continue

        if curvature < -0.015:
            medium_left()
            continue

        if curvature < -0.006:
            soft_left()
            continue

        if heading > 0.18:
            medium_left()
            continue

        if heading < -0.18:
            medium_right()
            continue

        if offset > 0.15 or heading > 0.06:
            soft_left()
            continue

        if offset < -0.15 or heading < -0.06:
            soft_right()
            continue

        hold_center()
'''


RULE_POLICY = r'''
def policy():
    while not task_finished():
        state = observe_driving_state()
        if state.front_vehicle_exists.value and state.front_vehicle_distance_m.value is not None and state.front_vehicle_distance_m.value < 12.0:
            follow_front_vehicle()
        elif state.route_command.value in ("turn_left", "turn_right") and state.distance_to_maneuver_m.value is not None and state.distance_to_maneuver_m.value < 8.0:
            execute_turn(state.route_command.value)
        elif abs(state.lane_center_offset_m.value) > 0.3:
            recover_from_offset()
        else:
            keep_lane_center()
'''


class PolicyCodeGenerator:
    """Codex-backed generator with deterministic fallback for repeatable demos."""

    def __init__(self, use_codex: bool = False, codex_cmd: str = "codex"):
        self.use_codex = use_codex
        self.codex_cmd = codex_cmd

    def generate(self, task_description: str, prompt_path: Path | None = None, mode: str = "llm_feedback") -> str:
        if mode == "rule":
            return RULE_POLICY
        if mode == "llm_no_feedback":
            return NO_FEEDBACK_POLICY
        if not self.use_codex:
            return DEFAULT_STAGE1_POLICY
        prompt = self._build_prompt(task_description, prompt_path)
        try:
            result = subprocess.run(
                [self.codex_cmd, "-c", 'model_provider="axonhub"', "exec", prompt],
                check=True,
                text=True,
                capture_output=True,
                timeout=120,
            )
        except Exception:
            return DEFAULT_STAGE1_POLICY
        code = self._extract_code(result.stdout)
        return code or DEFAULT_STAGE1_POLICY

    def _build_prompt(self, task_description: str, prompt_path: Path | None) -> str:
        base = prompt_path.read_text() if prompt_path and prompt_path.exists() else ""
        return (
            base
            + "\nGenerate only Python code defining def policy(): for this task:\n"
            + task_description
        )

    def _extract_code(self, text: str) -> str:
        if "```python" in text:
            return text.split("```python", 1)[1].split("```", 1)[0].strip()
        if "def policy" in text:
            return text[text.index("def policy") :].strip()
        return ""
