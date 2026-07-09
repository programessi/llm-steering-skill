from __future__ import annotations

import math

import numpy as np

from envs.simulator_adapter import SimObservation, TaskConfig


class MetaDriveAdapter:
    """Minimal MetaDrive-backed SimulatorAdapter.

    This adapter intentionally starts with oracle lane state from MetaDrive. It
    verifies that the same SteeringSkill interface can control a real driving
    simulator. Front-camera perception can be added by enabling MetaDrive camera
    sensors and returning real RGB frames from get_observation().
    """

    def __init__(
        self,
        config: dict | None = None,
        max_steering_rad: float = 0.45,
        render_topdown: bool = False,
        render_size: tuple[int, int] = (960, 540),
    ):
        from metadrive import MetaDriveEnv

        default = {
            "use_render": False,
            "num_scenarios": 1,
            "start_seed": 0,
            "traffic_density": 0.0,
            "log_level": 50,
        }
        if config:
            default.update(config)
        self.env = MetaDriveEnv(default)
        self.max_steering_rad = max_steering_rad
        self.render_topdown = render_topdown
        self.render_size = render_size
        self.task = TaskConfig(name="metadrive_smoke", horizon_steps=120)
        self.step_count = 0
        self.time_s = 0.0
        self.dt = 0.1
        self.last_reward = 0.0
        self.last_done = False
        self.last_info = {}
        self.trace: list[dict[str, float | int | bool | str | None]] = []
        self.latest_rgb: np.ndarray | None = None
        self.render_frames: list[np.ndarray] = []

    def reset(self, task: TaskConfig) -> SimObservation:
        self.task = task
        self.step_count = 0
        self.time_s = 0.0
        self.last_reward = 0.0
        self.last_done = False
        self.last_info = {}
        try:
            self.env.reset()
        except ValueError:
            self.env.reset()
        self.trace = []
        self.render_frames = []
        self._append_trace("reset")
        return self.get_observation()

    def step(self, steering_target_rad: float, target_speed_mps: float | None = None) -> SimObservation:
        steering_norm = float(np.clip(steering_target_rad / self.max_steering_rad, -1.0, 1.0))
        speed = self._speed_mps()
        if target_speed_mps is None:
            throttle = 0.25
        else:
            throttle = float(np.clip((target_speed_mps - speed) / 8.0, -1.0, 1.0))
        ret = self.env.step([steering_norm, throttle])
        if len(ret) == 5:
            _, reward, terminated, truncated, info = ret
            done = bool(terminated or truncated)
        else:
            _, reward, done, info = ret
        self.last_reward = float(reward)
        self.last_done = bool(done)
        self.last_info = info or {}
        self.step_count += 1
        self.time_s += self.dt
        self._append_trace("step")
        return self.get_observation()

    def get_observation(self) -> SimObservation:
        rgb = np.zeros((360, 640, 3), dtype=np.uint8)
        values = self.get_oracle_state_values()
        if self.render_topdown:
            rgb = self.capture_render_frame()
        return SimObservation(
            rgb=rgb,
            depth_m=None,
            speed_mps=float(values["speed_mps"] or 0.0),
            steering_angle_rad=float(values["steering_angle_rad"] or 0.0),
            timestamp=self.time_s,
        )

    def capture_render_frame(self) -> np.ndarray:
        if not self.render_topdown:
            if self.latest_rgb is None:
                self.latest_rgb = np.zeros((360, 640, 3), dtype=np.uint8)
            return self.latest_rgb
        width, height = self.render_size
        frame = self.env.render(
            mode="topdown",
            target_vehicle_heading_up=True,
            draw_target_vehicle_trajectory=True,
            screen_size=(width, height),
            film_size=(width * 2, height * 2),
        )
        # MetaDrive topdown returns RGB uint8.
        self.latest_rgb = np.ascontiguousarray(frame)
        return self.latest_rgb

    def get_oracle_state_values(self) -> dict[str, float | bool | str | None]:
        vehicle = self.env.vehicle
        lane = vehicle.lane
        longitudinal, lateral = lane.local_coordinates(vehicle.position)
        lane_heading = lane.heading_theta_at(longitudinal)
        heading = float(vehicle.heading_theta)
        heading_error = _wrap_angle(heading - lane_heading)
        steering = 0.0
        try:
            steering = float(vehicle.last_current_action[-1][0]) * self.max_steering_rad
        except Exception:
            steering = 0.0
        return {
            "speed_mps": self._speed_mps(),
            "steering_angle_rad": steering,
            "lane_center_offset_m": float(lateral),
            "heading_error_rad": float(heading_error),
            "lane_curvature": 0.0,
            "drivable_area_confidence": 1.0,
            "front_vehicle_exists": False,
            "front_vehicle_distance_m": None,
            "front_vehicle_relative_speed_mps": None,
            "route_command": "keep_lane",
            "distance_to_maneuver_m": None,
            "perception_confidence": 1.0,
        }

    def evaluate_after_skill(self, skill_name: str, target_angle: float | None) -> str:
        values = self.get_oracle_state_values()
        if self.last_info.get("crash", False):
            return "lane_departure_risk"
        if abs(float(values["lane_center_offset_m"] or 0.0)) > 1.8:
            return "lane_departure_risk"
        return "none"

    def task_finished(self) -> bool:
        return self.last_done or self.step_count >= self.task.horizon_steps

    def metrics(self) -> dict[str, float | int | bool | str]:
        offsets = [abs(float(row["lane_center_offset_m"])) for row in self.trace if row["event"] != "reset"]
        max_offset = max(offsets) if offsets else 0.0
        lane_departure = max_offset >= 1.8
        success = not self.last_info.get("crash", False) and not lane_departure
        return {
            "task": self.task.name,
            "steps": self.step_count,
            "success": bool(success),
            "collision": bool(self.last_info.get("crash_vehicle", False)),
            "lane_departure": bool(lane_departure),
            "mean_lane_center_offset_m": float(np.mean(offsets)) if offsets else 0.0,
            "max_lane_center_offset_m": float(max_offset),
            "front_vehicle_min_distance_m": -1.0,
            "turn_started_distance_m": -1.0,
        }

    def close(self) -> None:
        self.env.close()

    def _speed_mps(self) -> float:
        return float(getattr(self.env.vehicle, "speed", 0.0))

    def _append_trace(self, event: str) -> None:
        values = self.get_oracle_state_values()
        self.trace.append(
            {
                "step": self.step_count,
                "time_s": self.time_s,
                "event": event,
                "speed_mps": values["speed_mps"],
                "steering_angle_rad": values["steering_angle_rad"],
                "lane_center_offset_m": values["lane_center_offset_m"],
                "heading_error_rad": values["heading_error_rad"],
                "lane_curvature": values["lane_curvature"],
                "front_vehicle_distance_m": None,
                "distance_to_maneuver_m": None,
            }
        )
        if self.render_topdown:
            self.render_frames.append(self.capture_render_frame().copy())


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle
