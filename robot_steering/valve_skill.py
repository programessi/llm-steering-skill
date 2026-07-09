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
class KinematicValveSteeringSkill:
    """Valve-as-steering proxy without robot physics.

    This is the first bridge step before using a real ManiSkill robot task. It
    exposes the same RobotSteeringSkill command contract, simulates a valve
    joint moving toward the requested steering angle, then reports the measured
    valve joint angle as the actual steering angle.
    """

    dt_s: float = 0.05
    response_rate: float = 2.0
    angle_tolerance_rad: float = 0.04
    max_abs_angle_rad: float = 0.45
    action_type: str = "valve_joint_delta"

    def __post_init__(self) -> None:
        self.valve_angle_rad = 0.0
        self._episode_counter = itertools.count(1)

    def execute(self, command: SteeringSkillCommand) -> SteeringTrajectorySample:
        episode_id = f"valve_kinematic_{next(self._episode_counter):04d}"
        self.valve_angle_rad = _clip(command.current_angle_rad, -self.max_abs_angle_rad, self.max_abs_angle_rad)
        target = _clip(command.target_angle_rad, -self.max_abs_angle_rad, self.max_abs_angle_rad)
        steps = max(1, int(math.ceil(command.duration_s / self.dt_s)))
        observations: list[SteeringObservation] = []
        actions: list[RobotAction] = []

        for idx in range(steps):
            timestamp = (idx + 1) * self.dt_s
            error = target - self.valve_angle_rad
            delta = error * min(1.0, self.response_rate * self.dt_s)
            if command.max_speed_rad_s is not None:
                max_step = abs(command.max_speed_rad_s) * self.dt_s
                delta = _clip(delta, -max_step, max_step)
            self.valve_angle_rad = _clip(
                self.valve_angle_rad + delta,
                -self.max_abs_angle_rad,
                self.max_abs_angle_rad,
            )
            observations.append(
                SteeringObservation(
                    timestamp_s=timestamp,
                    wheel_angle_rad=self.valve_angle_rad,
                    target_angle_rad=target,
                    proprio={
                        "valve_joint_qpos_rad": self.valve_angle_rad,
                        "valve_angle_error_rad": target - self.valve_angle_rad,
                    },
                )
            )
            actions.append(
                RobotAction(
                    timestamp_s=timestamp,
                    action_type=self.action_type,
                    values=(delta,),
                )
            )

        final_error = target - self.valve_angle_rad
        return SteeringTrajectorySample(
            episode_id=episode_id,
            command=command,
            observations=observations,
            actions=actions,
            success=abs(final_error) <= self.angle_tolerance_rad,
            final_angle_error_rad=final_error,
            metadata={
                "skill_impl": "KinematicValveSteeringSkill",
                "valve_backend": "kinematic_valve",
                "dt_s": self.dt_s,
                "response_rate": self.response_rate,
                "angle_tolerance_rad": self.angle_tolerance_rad,
                "max_abs_angle_rad": self.max_abs_angle_rad,
            },
        )

    def steer_to(self, target_angle_rad: float, duration_s: float) -> SteeringTrajectorySample:
        return self.execute(
            SteeringSkillCommand(
                skill_name="steer_to",
                current_angle_rad=self.valve_angle_rad,
                target_angle_rad=target_angle_rad,
                duration_s=duration_s,
            )
        )

    def hold(self, angle_rad: float, duration_s: float) -> SteeringTrajectorySample:
        return self.execute(
            SteeringSkillCommand(
                skill_name="hold",
                current_angle_rad=self.valve_angle_rad,
                target_angle_rad=angle_rad,
                duration_s=duration_s,
            )
        )

    def return_center(self, duration_s: float) -> SteeringTrajectorySample:
        return self.execute(
            SteeringSkillCommand(
                skill_name="return_center",
                current_angle_rad=self.valve_angle_rad,
                target_angle_rad=0.0,
                duration_s=duration_s,
            )
        )


@dataclass
class ManiSkillValveSteeringSkill:
    """ManiSkill valve-as-steering proxy.

    This is still not a learned robot policy. It uses ManiSkill's real valve
    articulation as the steering proxy: command targets are converted to valve
    joint qpos updates, then the measured qpos is returned as the actual wheel
    angle consumed by CARLA.
    """

    dt_s: float = 0.05
    env_id: str = "RotateValveLevel0-v1"
    response_rate: float = 2.0
    angle_tolerance_rad: float = 0.04
    max_abs_angle_rad: float = 0.45

    def __post_init__(self) -> None:
        try:
            import gymnasium as gym
            import mani_skill.envs  # noqa: F401
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ManiSkill valve backend is not available yet. Install ManiSkill3 "
                "under third_party/maniskill3 and use --robot-skill-backend "
                "kinematic_valve for the current bridge smoke."
            ) from exc

        self.torch = torch
        self.env = gym.make(self.env_id, num_envs=1, obs_mode="state", render_mode=None)
        self.env.reset(seed=0)
        self.base_env = self.env.unwrapped
        self.valve_angle_rad = 0.0
        self._episode_counter = itertools.count(1)
        self._set_valve_qpos(self.valve_angle_rad)

    def execute(self, command: SteeringSkillCommand) -> SteeringTrajectorySample:
        episode_id = f"maniskill_valve_{next(self._episode_counter):04d}"
        self.valve_angle_rad = _clip(command.current_angle_rad, -self.max_abs_angle_rad, self.max_abs_angle_rad)
        self._set_valve_qpos(self.valve_angle_rad)
        target = _clip(command.target_angle_rad, -self.max_abs_angle_rad, self.max_abs_angle_rad)
        steps = max(1, int(math.ceil(command.duration_s / self.dt_s)))
        observations: list[SteeringObservation] = []
        actions: list[RobotAction] = []

        for idx in range(steps):
            timestamp = (idx + 1) * self.dt_s
            error = target - self.valve_angle_rad
            delta = error * min(1.0, self.response_rate * self.dt_s)
            if command.max_speed_rad_s is not None:
                max_step = abs(command.max_speed_rad_s) * self.dt_s
                delta = _clip(delta, -max_step, max_step)
            commanded_qpos = _clip(
                self.valve_angle_rad + delta,
                -self.max_abs_angle_rad,
                self.max_abs_angle_rad,
            )
            previous_angle = self.valve_angle_rad
            self._set_valve_qpos(commanded_qpos)
            self._step_zero_action()
            self.valve_angle_rad = self._read_valve_qpos()
            observations.append(
                SteeringObservation(
                    timestamp_s=timestamp,
                    wheel_angle_rad=self.valve_angle_rad,
                    target_angle_rad=target,
                    proprio={
                        "valve_joint_qpos_rad": self.valve_angle_rad,
                        "valve_angle_error_rad": target - self.valve_angle_rad,
                    },
                )
            )
            actions.append(
                RobotAction(
                    timestamp_s=timestamp,
                    action_type="maniskill_valve_qpos_delta",
                    values=(self.valve_angle_rad - previous_angle,),
                )
            )

        final_error = target - self.valve_angle_rad
        return SteeringTrajectorySample(
            episode_id=episode_id,
            command=command,
            observations=observations,
            actions=actions,
            success=abs(final_error) <= self.angle_tolerance_rad,
            final_angle_error_rad=final_error,
            metadata={
                "skill_impl": "ManiSkillValveSteeringSkill",
                "valve_backend": "maniskill_valve",
                "env_id": self.env_id,
                "dt_s": self.dt_s,
                "response_rate": self.response_rate,
                "angle_tolerance_rad": self.angle_tolerance_rad,
                "max_abs_angle_rad": self.max_abs_angle_rad,
                "direct_qpos_bridge": True,
            },
        )

    def steer_to(self, target_angle_rad: float, duration_s: float) -> SteeringTrajectorySample:
        return self.execute(
            SteeringSkillCommand("steer_to", 0.0, target_angle_rad, duration_s)
        )

    def hold(self, angle_rad: float, duration_s: float) -> SteeringTrajectorySample:
        return self.execute(
            SteeringSkillCommand("hold", angle_rad, angle_rad, duration_s)
        )

    def return_center(self, duration_s: float) -> SteeringTrajectorySample:
        return self.execute(
            SteeringSkillCommand("return_center", 0.0, 0.0, duration_s)
        )

    def close(self) -> None:
        close = getattr(getattr(self, "env", None), "close", None)
        if callable(close):
            close()

    def _set_valve_qpos(self, angle_rad: float) -> None:
        tensor = self.torch.tensor([[float(angle_rad)]], device=self.base_env.device)
        self.base_env.valve.set_qpos(tensor)

    def _read_valve_qpos(self) -> float:
        return float(self.base_env.valve.qpos.detach().cpu().numpy()[0][0])

    def _step_zero_action(self) -> None:
        action = self.torch.zeros(self.env.action_space.shape, dtype=self.torch.float32)
        self.env.step(action)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
