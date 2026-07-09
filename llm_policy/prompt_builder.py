from __future__ import annotations

from dataclasses import dataclass


DRIVING_POLICY_SYSTEM_PROMPT = """You generate Python policy code for a closed-loop driving-skill orchestrator.

Write exactly one executable Python script defining def policy():.
Do not include markdown, prose, imports, file IO, networking, class definitions, or external package calls.
The code will run in a restricted runtime. Only the APIs described below are available.
The policy must call steering skills through execute_primitive(...); never implement low-level continuous wheel control.
Treat the API reference as capability descriptions. Do not assume hidden implementations or simulator internals.
"""


API_REFERENCE = """--- API Reference ---

observe_driving_state() -> state
    Returns structured driving state. Fields are Estimated objects with .value and .as_float(default).
    Important fields:
      state.distance_to_maneuver_m.as_float(default)
      state.lane_center_offset_m.as_float(default)
      state.heading_error_rad.as_float(default)
      state.speed_mps.as_float(default)
      state.perception_confidence.value

observe_execution_feedback() -> feedback
    Returns feedback from the previous attempt or previous skill.
    Important fields:
      feedback.event
    Possible events:
      "none", "turn_too_early", "turn_too_late", "lane_departure_risk"

task_finished() -> bool
    Returns True if the runtime should stop.

mark_task_complete() -> None
    Call after the maneuver sequence is complete.

execute_primitive(name, target_angle_rad, duration_s, speed_mps, stage="...", trigger="...") -> None
    Execute one replaceable steering skill primitive.
    This is the only action API. It will call the current SteeringSkill implementation
    and can later be replaced by an imitation-learned robot steering skill.
    Allowed primitive names:
      "hold_center", "hard_left", "hold_left", "return_center",
      "medium_left", "medium_right", "hold_center_stop"
    target_angle_rad range: [-0.45, 0.45].
    Negative target angle means left steering; positive means right steering.
"""


POLICY_REQUIREMENTS = """--- Policy Requirements ---

Generate a feedback-aware fixed-point driving maneuver policy for the given task.
The task starts with distance_to_maneuver_m around 8.0 meters.
For this benchmark, trigger_distance_m MUST be less than 8.0.
Use initial trigger_distance_m in [4.5, 6.0].
If feedback.event == "turn_too_early", reduce trigger_distance_m into [3.7, 4.2].
Do not reduce trigger_distance_m below 3.7 after turn_too_early; model-estimated marker
distance can become non-monotonic near the marker, so too-low thresholds can miss the
trigger window entirely.
This is important: the policy must spend at least several loop iterations calling
hold_center while distance_to_maneuver_m > trigger_distance_m before the first maneuver primitive.

The policy should:
  1. Initialize tunable parameters:
       trigger_distance_m, turn_angle_rad, turn_duration_s, hold_duration_s, return_duration_s.
       Also initialize fixed numeric speed targets such as approach_speed_mps and maneuver_speed_mps.
  2. Read observe_execution_feedback().
  3. If feedback.event == "turn_too_early":
       reduce trigger_distance_m, reduce absolute steering angle, shorten hold, lengthen return-center.
  4. If feedback.event == "turn_too_late":
       increase trigger_distance_m and slightly increase absolute steering angle.
  5. In a while not task_finished() loop:
       read observe_driving_state()
       wait with hold_center while distance_to_maneuver_m is greater than trigger_distance_m
       then execute the task-specific primitive sequence
       call mark_task_complete()

Task-specific primitive guidance:
  - For a right-angle left turn: use hard_left or medium_left, hold_left, return_center, hold_center.
  - For a fixed-point lane change left: use medium_left, medium_right counter-steer, return_center, hold_center.

Keep the code compact and robust. Use only numeric literals and the APIs above.
Use fixed numeric speed targets in execute_primitive(...), usually 2.2 to 4.0 m/s.
Do not pass the current observed state.speed_mps back as the speed_mps command; that can keep the car near zero speed.
Do not set trigger_distance_m >= 8.0. Do not immediately turn at the first loop iteration unless distance_to_maneuver_m <= trigger_distance_m.
Do not call helper functions that are not listed in the API reference.
Do not define helper functions or classes.
"""


@dataclass(frozen=True)
class PolicyPromptInput:
    task_description: str
    feedback_event: str = "none"
    previous_failure: str = "none"
    validation_error: str = "none"


def build_policy_messages(prompt_input: PolicyPromptInput) -> list[dict[str, str]]:
    user_prompt = f"""{API_REFERENCE}

{POLICY_REQUIREMENTS}

--- Task ---
{prompt_input.task_description}

--- Current Feedback ---
feedback.event = {prompt_input.feedback_event!r}
previous_failure = {prompt_input.previous_failure!r}
previous_validation_error = {prompt_input.validation_error!r}

Return only Python code. The first line must be: def policy():
"""
    return [
        {"role": "system", "content": DRIVING_POLICY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
