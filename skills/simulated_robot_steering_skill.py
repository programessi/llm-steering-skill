from __future__ import annotations

from dataclasses import dataclass, field

from envs.simulator_adapter import SimulatorAdapter
from robot_steering.robot_steering_skill import RobotSteeringSkill
from robot_steering.schema import SteeringSkillCommand, SteeringTrajectorySample
from robot_steering.stub_skill import StubRobotSteeringSkill
from skills.feedback import SkillResult


@dataclass
class SimulatedRobotSteeringSkill:
    """SteeringSkill adapter backed by a simulated robot wheel skill.

    The generated LLM policy still calls the same SteeringSkill methods as it
    does for OracleSteeringSkill. The difference is that this class first asks a
    robot-side skill to produce the actual wheel-angle trajectory, then applies
    that measured trajectory to the vehicle simulator.
    """

    env: SimulatorAdapter
    dt: float = 0.05
    default_speed_mps: float = 8.0
    robot_skill: RobotSteeringSkill | None = None
    max_speed_rad_s: float | None = 2.2
    target_speed_mps: float = field(init=False)
    trace_primitive_name: str | None = field(default=None, init=False)
    last_robot_sample: SteeringTrajectorySample | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.target_speed_mps = self.default_speed_mps
        if self.robot_skill is None:
            self.robot_skill = StubRobotSteeringSkill(dt_s=self.dt, response_rate=2.0)

    def set_target_speed(self, speed_mps: float) -> None:
        self.target_speed_mps = max(0.0, min(float(speed_mps), 12.0))

    def set_trace_primitive(self, primitive_name: str | None) -> None:
        self.trace_primitive_name = primitive_name

    def steer_to(self, target_angle: float, duration: float, *, ramp_s: float | None = None) -> SkillResult:
        return self._run("steer_to", target_angle, duration if ramp_s is None else ramp_s)

    def hold(self, angle: float, duration: float, *, hold_s: float | None = None) -> SkillResult:
        if hold_s is not None:
            duration = hold_s
        return self._run("hold_steering", angle, duration)

    def return_center(self, duration: float, *, ramp_s: float | None = None) -> SkillResult:
        return self._run("return_center", 0.0, duration if ramp_s is None else ramp_s)

    def execute_turn(self, direction: str, duration: float = 0.9) -> SkillResult:
        target = -0.34 if direction in {"turn_left", "left"} else 0.34
        return self._run("execute_turn", target, duration)

    def _run(self, name: str, target_angle: float, duration: float) -> SkillResult:
        if self.robot_skill is None:
            raise RuntimeError("robot_skill was not initialized")
        start_idx = len(self.env.trace) if hasattr(self.env, "trace") else 0
        command = SteeringSkillCommand(
            skill_name=name,
            current_angle_rad=self.env.get_observation().steering_angle_rad,
            target_angle_rad=float(target_angle),
            duration_s=max(float(duration), self.dt),
            max_speed_rad_s=self.max_speed_rad_s,
        )
        sample = self.robot_skill.execute(command)
        self.last_robot_sample = sample
        primitive_name = self.trace_primitive_name or name
        if hasattr(self.env, "set_trace_context"):
            self.env.set_trace_context(name, primitive_name, target_angle)
        try:
            for obs in sample.observations:
                if self.env.task_finished():
                    break
                self.env.step(obs.wheel_angle_rad, self.target_speed_mps)
            self._annotate_robot_trace(start_idx, sample)
        finally:
            if hasattr(self.env, "clear_trace_context"):
                self.env.clear_trace_context()
        vehicle_obs = self.env.get_observation()
        event = self.env.evaluate_after_skill(name, target_angle)
        success = bool(sample.success and event in {"none", "turn_too_early", "turn_too_late"})
        return SkillResult(
            name=name,
            success=success,
            target_angle=target_angle,
            final_angle=vehicle_obs.steering_angle_rad,
            duration=sample.observations[-1].timestamp_s if sample.observations else command.duration_s,
            event=event,
        )

    def _annotate_robot_trace(self, start_idx: int, sample: SteeringTrajectorySample) -> None:
        trace = getattr(self.env, "trace", None)
        if not isinstance(trace, list):
            return
        observations = sample.observations
        actions = sample.actions
        rows = trace[start_idx:]
        for idx, row in enumerate(rows):
            if row.get("event") == "reset":
                continue
            obs_idx = min(idx, len(observations) - 1) if observations else -1
            action_idx = min(idx, len(actions) - 1) if actions else -1
            row["steering_skill_impl"] = "SimulatedRobotSteeringSkill"
            row["robot_skill_impl"] = str(sample.metadata.get("skill_impl", type(self.robot_skill).__name__))
            row["robot_backend"] = sample.metadata.get("valve_backend") or sample.metadata.get("robot_backend")
            row["robot_steering_episode_id"] = sample.episode_id
            row["robot_command_skill_name"] = sample.command.skill_name
            row["robot_command_target_angle_rad"] = sample.command.target_angle_rad
            row["robot_command_duration_s"] = sample.command.duration_s
            row["robot_final_angle_error_rad"] = sample.final_angle_error_rad
            row["robot_skill_success"] = sample.success
            if obs_idx >= 0:
                obs = observations[obs_idx]
                row["robot_observed_wheel_angle_rad"] = obs.wheel_angle_rad
                row["robot_observed_target_angle_rad"] = obs.target_angle_rad
                row["valve_joint_qpos_rad"] = obs.proprio.get("valve_joint_qpos_rad")
                row["valve_angle_error_rad"] = obs.proprio.get("valve_angle_error_rad")
            if action_idx >= 0:
                action = actions[action_idx]
                row["robot_action_type"] = action.action_type
                row["robot_action_delta_angle_rad"] = action.values[0] if action.values else 0.0
