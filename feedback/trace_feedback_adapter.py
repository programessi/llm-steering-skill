from __future__ import annotations

from skills.feedback import ExecutionFeedback


MANEUVER_PRIMITIVES = {
    "hard_left",
    "medium_left",
    "soft_left",
    "hold_left",
    "hard_right",
    "medium_right",
    "soft_right",
    "hold_right",
}


def infer_execution_feedback(
    trace: list[dict],
    raw_metrics: dict,
    policy_runtime_valid_code: bool = True,
    early_distance_m: float = 4.4,
    late_distance_m: float = 2.5,
    max_lane_error_m: float = 1.8,
) -> ExecutionFeedback:
    """Infer high-level retry feedback from trace/metrics.

    This is the Stage-1 bridge from simulator execution evidence to the compact
    ExecutionFeedback event consumed by generated policy code.
    """
    max_lane_error = float(raw_metrics.get("max_lane_center_offset_m") or 0.0)
    if not policy_runtime_valid_code:
        return ExecutionFeedback(
            event="lane_departure_risk",
            last_skill_success=False,
            lane_error_after_skill_m=max_lane_error,
        )
    if bool(raw_metrics.get("lane_departure")):
        return ExecutionFeedback(
            event="lane_departure_risk",
            last_skill_success=False,
            lane_error_after_skill_m=max_lane_error,
        )

    first_turn = first_maneuver_row(trace)
    if first_turn is None:
        return ExecutionFeedback(event="turn_too_late", last_skill_success=False)

    turn_distance = float_or_none(first_turn.get("runtime_distance_to_maneuver_m"))
    lane_error = float_or_none(first_turn.get("lane_center_offset_m"))
    heading_error = float_or_none(first_turn.get("heading_error_rad"))
    target_angle = float_or_none(first_turn.get("planned_target_angle_rad"))
    primitive = str(first_turn.get("planned_primitive") or first_turn.get("steering_primitive") or "")

    if turn_distance is not None and turn_distance > early_distance_m:
        event = "turn_too_early"
        success = False
    elif turn_distance is not None and turn_distance < late_distance_m:
        event = "turn_too_late"
        success = False
    elif max_lane_error > max_lane_error_m:
        event = "lane_departure_risk"
        success = False
    else:
        event = "none"
        success = True

    return ExecutionFeedback(
        last_skill_name=primitive or None,
        last_skill_success=success,
        target_angle=target_angle,
        actual_steering_angle=float_or_none(first_turn.get("steering_angle_rad")),
        lane_error_after_skill_m=lane_error,
        heading_error_after_skill_rad=heading_error,
        event=event,
    )


def feedback_input_text(feedback: ExecutionFeedback) -> str:
    event = feedback.event or "none"
    if event == "turn_too_early":
        return "turn_too_early: delay trigger, reduce peak steering, return center later"
    if event == "turn_too_late":
        return "turn_too_late: trigger earlier and slightly increase peak steering"
    if event == "lane_departure_risk":
        return "lane_departure_risk: reduce peak steering, slow down, and stabilize before retrying"
    return "none"


def apply_auto_feedback_to_trace(trace: list[dict], feedback: ExecutionFeedback) -> None:
    event = feedback.event or "none"
    for row in trace:
        row["auto_execution_feedback_event"] = event
        row["execution_feedback"] = event


def first_maneuver_row(trace: list[dict]) -> dict | None:
    for row in trace:
        primitive = str(row.get("planned_primitive") or row.get("steering_primitive") or "")
        if primitive in MANEUVER_PRIMITIVES:
            return row
    return None


def float_or_none(value) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
