from __future__ import annotations

from robot_steering.robot_steering_skill import RobotSteeringSkill
from robot_steering.stub_skill import StubRobotSteeringSkill
from robot_steering.valve_skill import KinematicValveSteeringSkill, ManiSkillValveSteeringSkill


def make_robot_steering_skill(
    backend: str = "stub",
    *,
    dt_s: float = 0.05,
    max_abs_angle_rad: float = 0.45,
) -> RobotSteeringSkill:
    if backend == "stub":
        return StubRobotSteeringSkill(dt_s=dt_s, response_rate=2.0)
    if backend == "kinematic_valve":
        return KinematicValveSteeringSkill(dt_s=dt_s, response_rate=2.0, max_abs_angle_rad=max_abs_angle_rad)
    if backend == "maniskill_valve":
        return ManiSkillValveSteeringSkill(dt_s=dt_s)
    raise ValueError(f"unknown robot steering backend: {backend}")


def robot_steering_backend_name(backend: str) -> str:
    if backend == "stub":
        return "StubRobotSteeringSkill"
    if backend == "kinematic_valve":
        return "KinematicValveSteeringSkill"
    if backend == "maniskill_valve":
        return "ManiSkillValveSteeringSkill"
    return backend
