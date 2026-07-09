from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from robot_steering.schema import (
    RobotAction,
    SteeringObservation,
    SteeringSkillCommand,
    SteeringTrajectorySample,
)


@dataclass
class StubRobotSteeringSkill:
    """Deterministic bench stub for the robot steering skill interface."""

    dt_s: float = 0.05
    response_rate: float = 5.0
    angle_tolerance_rad: float = 0.035
    action_type: str = "ee_delta_pose"

    def __post_init__(self) -> None:
        self.current_angle_rad = 0.0
        self._episode_counter = itertools.count(1)

    def execute(self, command: SteeringSkillCommand) -> SteeringTrajectorySample:
        episode_id = f"stub_{next(self._episode_counter):04d}"
        self.current_angle_rad = command.current_angle_rad
        steps = max(1, int(math.ceil(command.duration_s / self.dt_s)))
        observations: list[SteeringObservation] = []
        actions: list[RobotAction] = []
        target = command.target_angle_rad
        for idx in range(steps):
            timestamp = (idx + 1) * self.dt_s
            error = target - self.current_angle_rad
            max_step = None
            if command.max_speed_rad_s is not None:
                max_step = abs(command.max_speed_rad_s) * self.dt_s
            delta = error * min(1.0, self.response_rate * self.dt_s)
            if max_step is not None:
                delta = max(-max_step, min(max_step, delta))
            self.current_angle_rad += delta
            observations.append(
                SteeringObservation(
                    timestamp_s=timestamp,
                    wheel_angle_rad=self.current_angle_rad,
                    target_angle_rad=target,
                    proprio={"stub_angle_error_rad": target - self.current_angle_rad},
                )
            )
            actions.append(
                RobotAction(
                    timestamp_s=timestamp,
                    action_type=self.action_type,
                    values=(delta, 0.0, 0.0, 0.0, 0.0, 0.0),
                )
            )
        hold_steps = max(0, int(math.ceil(command.hold_s / self.dt_s)))
        for idx in range(hold_steps):
            timestamp = command.duration_s + (idx + 1) * self.dt_s
            observations.append(
                SteeringObservation(
                    timestamp_s=timestamp,
                    wheel_angle_rad=self.current_angle_rad,
                    target_angle_rad=target,
                    proprio={"stub_angle_error_rad": target - self.current_angle_rad},
                )
            )
            actions.append(
                RobotAction(
                    timestamp_s=timestamp,
                    action_type=self.action_type,
                    values=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                )
            )
        final_error = target - self.current_angle_rad
        return SteeringTrajectorySample(
            episode_id=episode_id,
            command=command,
            observations=observations,
            actions=actions,
            success=abs(final_error) <= self.angle_tolerance_rad,
            final_angle_error_rad=final_error,
            metadata={
                "skill_impl": "StubRobotSteeringSkill",
                "dt_s": self.dt_s,
                "response_rate": self.response_rate,
                "angle_tolerance_rad": self.angle_tolerance_rad,
            },
        )

    def steer_to(self, target_angle_rad: float, duration_s: float) -> SteeringTrajectorySample:
        return self.execute(
            SteeringSkillCommand(
                skill_name="steer_to",
                current_angle_rad=self.current_angle_rad,
                target_angle_rad=target_angle_rad,
                duration_s=duration_s,
            )
        )

    def hold(self, angle_rad: float, duration_s: float) -> SteeringTrajectorySample:
        return self.execute(
            SteeringSkillCommand(
                skill_name="hold",
                current_angle_rad=self.current_angle_rad,
                target_angle_rad=angle_rad,
                duration_s=duration_s,
            )
        )

    def return_center(self, duration_s: float) -> SteeringTrajectorySample:
        return self.execute(
            SteeringSkillCommand(
                skill_name="return_center",
                current_angle_rad=self.current_angle_rad,
                target_angle_rad=0.0,
                duration_s=duration_s,
            )
        )

