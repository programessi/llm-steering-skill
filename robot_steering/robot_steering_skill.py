from __future__ import annotations

from typing import Protocol

from robot_steering.schema import SteeringSkillCommand, SteeringTrajectorySample


class RobotSteeringSkill(Protocol):
    """Stage-2 replacement boundary for OracleSteeringSkill.

    Implementations can be a real robot controller, an imitation-learned
    policy, or a bench stub. The LLM policy layer should only depend on this
    command-level contract.
    """

    def execute(self, command: SteeringSkillCommand) -> SteeringTrajectorySample:
        ...

    def steer_to(self, target_angle_rad: float, duration_s: float) -> SteeringTrajectorySample:
        ...

    def hold(self, angle_rad: float, duration_s: float) -> SteeringTrajectorySample:
        ...

    def return_center(self, duration_s: float) -> SteeringTrajectorySample:
        ...

