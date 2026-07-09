from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .simulator_adapter import SimObservation, TaskConfig


@dataclass
class KinematicDrivingSimulator:
    """Small front-camera driving fallback used when CARLA/MetaDrive are absent.

    It is not a physics simulator. It exists to validate the stage-1 software
    contract: sensor image -> perception -> DrivingState -> policy -> skill.
    """

    dt: float = 0.1
    width: int = 640
    height: int = 360
    lane_width_m: float = 3.6
    max_steering_rad: float = 0.45
    steering_time_constant: float = 0.25
    steering_gain: float = 0.85
    task: TaskConfig = field(default_factory=lambda: TaskConfig(name="lane_keeping"))

    step_count: int = 0
    time_s: float = 0.0
    speed_mps: float = 8.0
    target_speed_mps: float = 8.0
    steering_angle_rad: float = 0.0
    lane_center_offset_m: float = 0.0
    heading_error_rad: float = 0.0
    road_curvature: float = 0.0
    front_vehicle_distance_m: float | None = None
    front_vehicle_relative_speed_mps: float | None = None
    distance_to_maneuver_m: float | None = None
    collision: bool = False
    lane_departure: bool = False
    turn_started_distance_m: float | None = None
    trace_skill_name: str | None = None
    trace_steering_primitive: str | None = None
    trace_target_angle_rad: float | None = None
    trace: list[dict[str, float | int | bool | str | None]] = field(default_factory=list)

    def reset(self, task: TaskConfig) -> SimObservation:
        self.task = task
        self.step_count = 0
        self.time_s = 0.0
        self.speed_mps = 8.0
        self.target_speed_mps = 8.0
        self.steering_angle_rad = 0.0
        self.lane_center_offset_m = task.initial_offset_m
        self.heading_error_rad = 0.0
        self.road_curvature = task.road_curvature
        self.front_vehicle_distance_m = task.front_vehicle_distance_m if task.has_front_vehicle else None
        self.front_vehicle_relative_speed_mps = None
        self.distance_to_maneuver_m = task.distance_to_maneuver_m
        self.collision = False
        self.lane_departure = False
        self.turn_started_distance_m = None
        self.trace_skill_name = None
        self.trace_steering_primitive = None
        self.trace_target_angle_rad = None
        self.trace = []
        obs = self.get_observation()
        self._append_trace("reset")
        return obs

    def set_trace_context(
        self,
        skill_name: str | None,
        steering_primitive: str | None,
        target_angle_rad: float | None,
    ) -> None:
        self.trace_skill_name = skill_name
        self.trace_steering_primitive = steering_primitive
        self.trace_target_angle_rad = target_angle_rad

    def clear_trace_context(self) -> None:
        self.trace_skill_name = None
        self.trace_steering_primitive = None
        self.trace_target_angle_rad = None

    def step(self, steering_target_rad: float, target_speed_mps: float | None = None) -> SimObservation:
        if target_speed_mps is not None:
            self.target_speed_mps = float(np.clip(target_speed_mps, 0.0, 12.0))

        target = float(np.clip(steering_target_rad, -self.max_steering_rad, self.max_steering_rad))
        if self.task.execution_noise:
            target += float(np.random.normal(0.0, self.task.execution_noise))
        alpha = min(1.0, self.dt / max(self.steering_time_constant, 1e-6))
        self.steering_angle_rad += alpha * (target - self.steering_angle_rad)

        speed_alpha = min(1.0, self.dt / 0.6)
        self.speed_mps += speed_alpha * (self.target_speed_mps - self.speed_mps)

        curvature = self._current_curvature()
        self.heading_error_rad += (self.steering_gain * self.steering_angle_rad - self.speed_mps * curvature) * self.dt
        self.heading_error_rad = float(np.clip(self.heading_error_rad, -0.7, 0.7))
        self.lane_center_offset_m += self.speed_mps * math.sin(self.heading_error_rad) * self.dt

        if self.front_vehicle_distance_m is not None:
            rel_speed = self.task.front_vehicle_speed_mps - self.speed_mps
            self.front_vehicle_relative_speed_mps = rel_speed
            self.front_vehicle_distance_m += rel_speed * self.dt
            if self.front_vehicle_distance_m < 2.0:
                self.collision = True

        if self.distance_to_maneuver_m is not None:
            self.distance_to_maneuver_m -= self.speed_mps * self.dt
            if abs(self.steering_angle_rad) > 0.18 and self.turn_started_distance_m is None:
                self.turn_started_distance_m = self.distance_to_maneuver_m

        departure_limit = self.lane_width_m * (1.45 if self.task.route_command in ("turn_left", "turn_right") else 0.65)
        if abs(self.lane_center_offset_m) > departure_limit:
            self.lane_departure = True

        self.step_count += 1
        self.time_s += self.dt
        self._append_trace("step")
        return self.get_observation()

    def get_observation(self) -> SimObservation:
        rgb, depth = self._render_front_sensor()
        return SimObservation(
            rgb=rgb,
            depth_m=depth,
            speed_mps=self.speed_mps,
            steering_angle_rad=self.steering_angle_rad,
            timestamp=self.time_s,
        )

    def get_oracle_state_values(self) -> dict[str, float | bool | str | None]:
        route = self.task.route_command
        if route in ("turn_left", "turn_right") and self.distance_to_maneuver_m is not None:
            if self.distance_to_maneuver_m > 25.0:
                route = "keep_lane"
            elif self.distance_to_maneuver_m < -14.0:
                route = "keep_lane"
        return {
            "speed_mps": self.speed_mps,
            "steering_angle_rad": self.steering_angle_rad,
            "lane_center_offset_m": self.lane_center_offset_m,
            "heading_error_rad": self.heading_error_rad,
            "lane_curvature": self._current_curvature(),
            "drivable_area_confidence": 1.0,
            "front_vehicle_exists": self.front_vehicle_distance_m is not None,
            "front_vehicle_distance_m": self.front_vehicle_distance_m,
            "front_vehicle_relative_speed_mps": self.front_vehicle_relative_speed_mps,
            "route_command": route,
            "distance_to_maneuver_m": self.distance_to_maneuver_m,
            "perception_confidence": 1.0,
        }

    def evaluate_after_skill(self, skill_name: str, target_angle: float | None) -> str:
        if self.collision:
            return "front_vehicle_too_close"
        risk_limit = 5.0 if self.task.route_command in ("turn_left", "turn_right") else 1.3
        if self.task.name == "curve_following":
            risk_limit = 1.6
        if self.lane_departure or abs(self.lane_center_offset_m) > risk_limit:
            return "lane_departure_risk"
        if target_angle is not None:
            err = self.steering_angle_rad - target_angle
            if err < -0.08:
                return "under_steer"
            if err > 0.08:
                return "over_steer"
        if skill_name in {"execute_turn", "steer_to"} and self.task.route_command in ("turn_left", "turn_right"):
            if self.distance_to_maneuver_m is not None:
                if self.turn_started_distance_m is not None and self.turn_started_distance_m > 13.0:
                    return "turn_too_early"
                if self.distance_to_maneuver_m < -4.0 and self.turn_started_distance_m is None:
                    return "turn_too_late"
        return "none"

    def task_finished(self) -> bool:
        if self.collision or self.lane_departure:
            return True
        return self.step_count >= self.task.horizon_steps

    def metrics(self) -> dict[str, float | int | bool | str]:
        offsets = [abs(float(row["lane_center_offset_m"])) for row in self.trace if row["event"] != "reset"]
        min_front = [
            float(row["front_vehicle_distance_m"])
            for row in self.trace
            if row["front_vehicle_distance_m"] is not None
        ]
        turn_ok = True
        if self.task.route_command in ("turn_left", "turn_right"):
            turn_ok = self.turn_started_distance_m is not None and 0.0 <= self.turn_started_distance_m <= 12.0
        max_offset_limit = 5.1 if self.task.route_command in ("turn_left", "turn_right") else 1.4
        if self.task.name == "curve_following":
            max_offset_limit = 1.6
        success = (
            not self.collision
            and not self.lane_departure
            and (max(offsets) if offsets else 0.0) < max_offset_limit
            and turn_ok
        )
        if self.task.has_front_vehicle:
            success = success and (min(min_front) if min_front else 99.0) > 4.0
        return {
            "task": self.task.name,
            "steps": self.step_count,
            "success": bool(success),
            "collision": bool(self.collision),
            "lane_departure": bool(self.lane_departure),
            "mean_lane_center_offset_m": float(np.mean(offsets)) if offsets else 0.0,
            "max_lane_center_offset_m": float(max(offsets)) if offsets else 0.0,
            "front_vehicle_min_distance_m": float(min(min_front)) if min_front else -1.0,
            "turn_started_distance_m": float(self.turn_started_distance_m) if self.turn_started_distance_m is not None else -1.0,
        }

    def _current_curvature(self) -> float:
        if self.task.route_command in ("turn_left", "turn_right") and self.distance_to_maneuver_m is not None:
            if -18.0 <= self.distance_to_maneuver_m <= 12.0:
                sign = -1.0 if self.task.route_command == "turn_left" else 1.0
                return sign * 0.052
        if self.task.name == "curve_following":
            return self.task.road_curvature * (0.65 + 0.35 * math.sin(self.time_s * 0.25))
        return self.task.road_curvature

    def _append_trace(self, event: str) -> None:
        self.trace.append(
            {
                "step": self.step_count,
                "time_s": self.time_s,
                "event": event,
                "skill_name": self.trace_skill_name,
                "steering_primitive": self.trace_steering_primitive,
                "target_angle_rad": self.trace_target_angle_rad,
                "speed_mps": self.speed_mps,
                "steering_angle_rad": self.steering_angle_rad,
                "lane_center_offset_m": self.lane_center_offset_m,
                "heading_error_rad": self.heading_error_rad,
                "lane_curvature": self._current_curvature(),
                "front_vehicle_distance_m": self.front_vehicle_distance_m,
                "distance_to_maneuver_m": self.distance_to_maneuver_m,
            }
        )

    def _render_front_sensor(self) -> tuple[np.ndarray, np.ndarray]:
        h, w = self.height, self.width
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:] = (35, 45, 45)
        depth = np.full((h, w), 80.0, dtype=np.float32)

        bottom_y = h - 1
        horizon_y = int(h * 0.38)
        px_per_m_bottom = 64.0
        px_per_m_horizon = 22.0
        lane_px_bottom = self.lane_width_m * px_per_m_bottom
        lane_px_horizon = self.lane_width_m * px_per_m_horizon
        center_bottom = w / 2.0 - self.lane_center_offset_m * px_per_m_bottom
        center_horizon = (
            w / 2.0
            - self.lane_center_offset_m * px_per_m_horizon
            - self.heading_error_rad * 320.0
        )

        road_poly = np.array(
            [
                [int(center_bottom - lane_px_bottom * 0.8), bottom_y],
                [int(center_bottom + lane_px_bottom * 0.8), bottom_y],
                [int(center_horizon + lane_px_horizon * 0.9), horizon_y],
                [int(center_horizon - lane_px_horizon * 0.9), horizon_y],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(rgb, [road_poly], (70, 75, 72))

        for sign, color in [(-1.0, (245, 245, 245)), (1.0, (245, 245, 245))]:
            x0 = int(center_bottom + sign * lane_px_bottom / 2.0)
            x1 = int(center_horizon + sign * lane_px_horizon / 2.0)
            cv2.line(rgb, (x0, bottom_y), (x1, horizon_y), color, 6)

        cv2.line(rgb, (0, horizon_y), (w, horizon_y), (55, 70, 80), 2)

        if self.front_vehicle_distance_m is not None and 0.0 < self.front_vehicle_distance_m < 70.0:
            dist = max(self.front_vehicle_distance_m, 3.0)
            box_h = int(np.clip(800.0 / dist, 16, 110))
            box_w = int(box_h * 1.6)
            cy = int(np.clip(h * 0.72 - dist * 2.0, horizon_y + 20, h - 45))
            cx = int(center_horizon + (center_bottom - center_horizon) * 0.65)
            x0, y0 = max(0, cx - box_w // 2), max(0, cy - box_h // 2)
            x1, y1 = min(w - 1, cx + box_w // 2), min(h - 1, cy + box_h // 2)
            cv2.rectangle(rgb, (x0, y0), (x1, y1), (30, 40, 210), -1)
            cv2.rectangle(rgb, (x0, y0), (x1, y1), (230, 230, 230), 2)
            depth[y0:y1, x0:x1] = float(dist)

        return rgb, depth


def make_stage1_tasks() -> list[TaskConfig]:
    return [
        TaskConfig(name="lane_keeping", horizon_steps=180, initial_offset_m=0.15),
        TaskConfig(name="offset_recovery_right", horizon_steps=180, initial_offset_m=0.85),
        TaskConfig(name="offset_recovery_left", horizon_steps=180, initial_offset_m=-0.85),
        TaskConfig(name="curve_following", horizon_steps=220, road_curvature=0.018),
        TaskConfig(
            name="front_vehicle_following",
            horizon_steps=220,
            has_front_vehicle=True,
            front_vehicle_distance_m=28.0,
            front_vehicle_speed_mps=5.2,
        ),
        TaskConfig(
            name="route_turn_left",
            horizon_steps=220,
            route_command="turn_left",
            distance_to_maneuver_m=45.0,
            road_curvature=0.0,
        ),
        TaskConfig(
            name="route_turn_right",
            horizon_steps=220,
            route_command="turn_right",
            distance_to_maneuver_m=45.0,
            road_curvature=0.0,
        ),
    ]
