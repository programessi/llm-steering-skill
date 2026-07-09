from __future__ import annotations

import math
import queue
from dataclasses import dataclass

import numpy as np

from envs.simulator_adapter import SimObservation, TaskConfig


@dataclass
class CarlaAdapterConfig:
    host: str = "127.0.0.1"
    port: int = 2000
    timeout_s: float = 10.0
    town: str | None = None
    spawn_index: int = 0
    width: int = 1280
    height: int = 720
    fov: float = 90.0
    fixed_delta_seconds: float = 0.05
    max_steering_rad: float = 0.45
    lane_departure_limit_m: float = 2.0
    enable_sync_mode: bool = True


class CarlaAdapter:
    """CARLA-backed SimulatorAdapter.

    This first CARLA path intentionally uses CARLA/map oracle fields for
    `DrivingState` while returning real front RGB frames for video. That keeps
    the experiment boundary clear: the policy still sees structured state and
    controls only through `OracleSteeringSkill`; perception-model replacement
    can happen later behind the same adapter contract.
    """

    def __init__(self, config: CarlaAdapterConfig | None = None):
        self.config = config or CarlaAdapterConfig()
        try:
            import carla
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CARLA Python API is not installed. Install CARLA, expose its "
                "PythonAPI/carla package on PYTHONPATH, and start CarlaUE4 "
                "before running the CARLA rendered demo."
            ) from exc

        self.carla = carla
        self.client = carla.Client(self.config.host, self.config.port)
        self.client.set_timeout(self.config.timeout_s)
        self.world = self._load_or_get_world()
        self.original_settings = self.world.get_settings()
        self.blueprints = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        self.task = TaskConfig(name="carla_rendered_lane_keep", horizon_steps=300)
        self.vehicle = None
        self.camera = None
        self.actors = []
        self.image_queue: queue.Queue = queue.Queue(maxsize=4)
        self.latest_rgb = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
        self.camera_frames: list[np.ndarray] = []
        self.step_count = 0
        self.time_s = 0.0
        self.target_speed_mps = 8.0
        self.last_steering_target_rad = 0.0
        self.last_lane_departure = False
        self.trace_skill_name: str | None = None
        self.trace_steering_primitive: str | None = None
        self.trace_target_angle_rad: float | None = None
        self.visual_marker_actor = None
        self.trace: list[dict[str, float | int | bool | str | None]] = []

    def reset(self, task: TaskConfig) -> SimObservation:
        self.task = task
        self.step_count = 0
        self.time_s = 0.0
        self.target_speed_mps = 8.0
        self.last_steering_target_rad = 0.0
        self.last_lane_departure = False
        self.trace = []
        self.camera_frames = []
        self._configure_world()
        self._destroy_actors()
        self._spawn_ego_vehicle()
        self._spawn_front_camera()
        self._tick_and_read_camera()
        self._append_trace("reset")
        return self.get_observation()

    def spawn_visual_maneuver_marker(
        self,
        distance_m: float,
        lateral_offset_m: float = 2.2,
        blueprint_id: str = "vehicle.mini.cooper_s",
    ) -> str:
        """Spawn a visible RGB marker for model-backed maneuver-distance perception.

        The marker is intentionally a common vehicle class so off-the-shelf COCO
        detectors can recognize it without task-specific training. It is placed
        near the planned maneuver marker but offset from the ego lane.
        """
        if self.vehicle is None:
            raise RuntimeError("CARLA adapter must be reset before spawning a visual marker.")
        ego = self.vehicle.get_transform()
        forward = ego.get_forward_vector()
        right = ego.get_right_vector()
        location = self.carla.Location(
            x=ego.location.x + float(distance_m) * forward.x + float(lateral_offset_m) * right.x,
            y=ego.location.y + float(distance_m) * forward.y + float(lateral_offset_m) * right.y,
            z=ego.location.z + 0.15,
        )
        rotation = self.carla.Rotation(
            pitch=ego.rotation.pitch,
            yaw=ego.rotation.yaw + 90.0,
            roll=ego.rotation.roll,
        )
        transform = self.carla.Transform(location, rotation)
        bp = self._find_blueprint(blueprint_id)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "stage1_visual_maneuver_marker")
        actor = self.world.try_spawn_actor(bp, transform)
        if actor is None:
            transform.location.z += 0.4
            actor = self.world.try_spawn_actor(bp, transform)
        if actor is None:
            raise RuntimeError(f"Failed to spawn visual maneuver marker with blueprint {blueprint_id!r}.")
        if hasattr(actor, "apply_control"):
            actor.apply_control(self.carla.VehicleControl(hand_brake=True, brake=1.0))
        self.visual_marker_actor = actor
        self.actors.append(actor)
        self._tick_and_read_camera()
        return str(bp.id)

    def step(self, steering_target_rad: float, target_speed_mps: float | None = None) -> SimObservation:
        if self.vehicle is None:
            raise RuntimeError("CARLA adapter must be reset before step().")
        if target_speed_mps is not None:
            self.target_speed_mps = float(np.clip(target_speed_mps, 0.0, 15.0))

        self.last_steering_target_rad = float(
            np.clip(steering_target_rad, -self.config.max_steering_rad, self.config.max_steering_rad)
        )
        steer = float(np.clip(self.last_steering_target_rad / self.config.max_steering_rad, -1.0, 1.0))
        speed = self._speed_mps()
        speed_error = self.target_speed_mps - speed
        throttle = float(np.clip(speed_error / 5.0, 0.0, 0.65))
        brake = float(np.clip(-speed_error / 4.0, 0.0, 0.7))
        self.vehicle.apply_control(
            self.carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        )

        self._tick_and_read_camera()
        self.step_count += 1
        self.time_s += self.config.fixed_delta_seconds
        values = self.get_oracle_state_values()
        self.last_lane_departure = (
            abs(float(values["lane_center_offset_m"] or 0.0)) > self.config.lane_departure_limit_m
        )
        self._append_trace("step")
        return self.get_observation()

    def get_observation(self) -> SimObservation:
        return SimObservation(
            rgb=self.latest_rgb.copy(),
            depth_m=None,
            speed_mps=self._speed_mps(),
            steering_angle_rad=self.last_steering_target_rad,
            timestamp=self.time_s,
        )

    def get_oracle_state_values(self) -> dict[str, float | bool | str | None]:
        if self.vehicle is None:
            return self._empty_state()
        transform = self.vehicle.get_transform()
        location = transform.location
        waypoint = self.map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=self.carla.LaneType.Driving,
        )
        lane_center_offset_m = 0.0
        heading_error_rad = 0.0
        lane_curvature = 0.0
        if waypoint is not None:
            wp_location = waypoint.transform.location
            right = waypoint.transform.get_right_vector()
            lane_center_offset_m = (
                (location.x - wp_location.x) * right.x
                + (location.y - wp_location.y) * right.y
                + (location.z - wp_location.z) * right.z
            )
            heading_error_rad = _wrap_angle(math.radians(transform.rotation.yaw - waypoint.transform.rotation.yaw))
            lane_curvature = self._estimate_lane_curvature(waypoint)

        front_exists, front_distance, front_rel_speed = self._front_vehicle_state()
        return {
            "speed_mps": self._speed_mps(),
            "steering_angle_rad": self.last_steering_target_rad,
            "lane_center_offset_m": float(lane_center_offset_m),
            "heading_error_rad": float(heading_error_rad),
            "lane_curvature": float(lane_curvature),
            "drivable_area_confidence": 1.0,
            "front_vehicle_exists": front_exists,
            "front_vehicle_distance_m": front_distance,
            "front_vehicle_relative_speed_mps": front_rel_speed,
            "route_command": "keep_lane",
            "distance_to_maneuver_m": None,
            "perception_confidence": 1.0,
        }

    def evaluate_after_skill(self, skill_name: str, target_angle: float | None) -> str:
        values = self.get_oracle_state_values()
        if values["front_vehicle_exists"] and values["front_vehicle_distance_m"] is not None:
            if float(values["front_vehicle_distance_m"]) < 3.0:
                return "front_vehicle_too_close"
        if abs(float(values["lane_center_offset_m"] or 0.0)) > 2.0:
            return "lane_departure_risk"
        return "none"

    def task_finished(self) -> bool:
        return self.step_count >= self.task.horizon_steps or self.last_lane_departure

    def metrics(self) -> dict[str, float | int | bool | str]:
        offsets = [abs(float(row["lane_center_offset_m"])) for row in self.trace if row["event"] != "reset"]
        min_front = [
            float(row["front_vehicle_distance_m"])
            for row in self.trace
            if row["front_vehicle_distance_m"] is not None
        ]
        max_offset = max(offsets) if offsets else 0.0
        lane_departure = max_offset > self.config.lane_departure_limit_m
        return {
            "task": self.task.name,
            "steps": self.step_count,
            "success": bool(not lane_departure),
            "collision": False,
            "lane_departure": bool(lane_departure),
            "mean_lane_center_offset_m": float(np.mean(offsets)) if offsets else 0.0,
            "max_lane_center_offset_m": float(max_offset),
            "front_vehicle_min_distance_m": float(min(min_front)) if min_front else -1.0,
            "turn_started_distance_m": -1.0,
            "lane_departure_limit_m": float(self.config.lane_departure_limit_m),
        }

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

    def close(self) -> None:
        self._destroy_actors()
        try:
            self.world.apply_settings(self.original_settings)
        except Exception:
            pass

    def _load_or_get_world(self):
        if self.config.town:
            return self.client.load_world(self.config.town)
        return self.client.get_world()

    def _configure_world(self) -> None:
        settings = self.world.get_settings()
        if self.config.enable_sync_mode:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.config.fixed_delta_seconds
            self.world.apply_settings(settings)

    def _spawn_ego_vehicle(self) -> None:
        vehicle_bp = self.blueprints.find("vehicle.tesla.model3")
        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", "stage1_ego")
        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("CARLA map has no spawn points.")
        spawn = spawn_points[self.config.spawn_index % len(spawn_points)]
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn)
        if self.vehicle is None:
            raise RuntimeError(f"Failed to spawn ego vehicle at spawn index {self.config.spawn_index}.")
        self.actors.append(self.vehicle)
        spectator = self.world.get_spectator()
        spectator.set_transform(self._spectator_transform())

    def _spawn_front_camera(self) -> None:
        if self.vehicle is None:
            raise RuntimeError("Cannot spawn camera before ego vehicle.")
        camera_bp = self.blueprints.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.config.width))
        camera_bp.set_attribute("image_size_y", str(self.config.height))
        camera_bp.set_attribute("fov", str(self.config.fov))
        transform = self.carla.Transform(
            self.carla.Location(x=1.6, z=1.55),
            self.carla.Rotation(pitch=-2.0),
        )
        self.camera = self.world.spawn_actor(camera_bp, transform, attach_to=self.vehicle)
        self.actors.append(self.camera)
        self.camera.listen(self._on_camera_image)

    def _find_blueprint(self, blueprint_id: str):
        try:
            return self.blueprints.find(blueprint_id)
        except Exception:
            matches = list(self.blueprints.filter("vehicle.*"))
            if not matches:
                raise
            return matches[0]

    def _on_camera_image(self, image) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        rgb = array[:, :, :3][:, :, ::-1].copy()
        while not self.image_queue.empty():
            try:
                self.image_queue.get_nowait()
            except queue.Empty:
                break
        self.image_queue.put(rgb)

    def _tick_and_read_camera(self) -> None:
        if self.config.enable_sync_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()
        try:
            self.latest_rgb = self.image_queue.get(timeout=2.0)
        except queue.Empty:
            pass
        self.camera_frames.append(self.latest_rgb.copy())

    def _speed_mps(self) -> float:
        if self.vehicle is None:
            return 0.0
        velocity = self.vehicle.get_velocity()
        return float(math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))

    def _estimate_lane_curvature(self, waypoint) -> float:
        next_wps = waypoint.next(15.0)
        if not next_wps:
            return 0.0
        yaw0 = waypoint.transform.rotation.yaw
        yaw1 = next_wps[0].transform.rotation.yaw
        return float(_wrap_angle(math.radians(yaw1 - yaw0)) / 15.0)

    def _front_vehicle_state(self) -> tuple[bool, float | None, float | None]:
        if self.vehicle is None:
            return False, None, None
        ego_transform = self.vehicle.get_transform()
        ego_location = ego_transform.location
        ego_forward = ego_transform.get_forward_vector()
        ego_waypoint = self.map.get_waypoint(
            ego_location,
            project_to_road=True,
            lane_type=self.carla.LaneType.Driving,
        )
        best_distance = None
        best_actor = None
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.id == self.vehicle.id:
                continue
            loc = actor.get_location()
            rel_x = loc.x - ego_location.x
            rel_y = loc.y - ego_location.y
            rel_z = loc.z - ego_location.z
            longitudinal = rel_x * ego_forward.x + rel_y * ego_forward.y + rel_z * ego_forward.z
            if longitudinal <= 0.0 or longitudinal > 70.0:
                continue
            actor_wp = self.map.get_waypoint(loc, project_to_road=True, lane_type=self.carla.LaneType.Driving)
            if ego_waypoint and actor_wp:
                if actor_wp.road_id != ego_waypoint.road_id or actor_wp.lane_id != ego_waypoint.lane_id:
                    continue
            if best_distance is None or longitudinal < best_distance:
                best_distance = float(longitudinal)
                best_actor = actor
        if best_actor is None or best_distance is None:
            return False, None, None
        rel_speed = self._actor_speed_mps(best_actor) - self._speed_mps()
        return True, best_distance, float(rel_speed)

    def _actor_speed_mps(self, actor) -> float:
        velocity = actor.get_velocity()
        return float(math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))

    def _append_trace(self, event: str) -> None:
        values = self.get_oracle_state_values()
        self.trace.append(
            {
                "step": self.step_count,
                "time_s": self.time_s,
                "event": event,
                "state_source": "carla_map_oracle",
                "camera_frame": "carla_front_rgb",
                "perception_adapter": "CarlaMapOracleStateAdapter",
                "control_source": "SteeringSkill",
                "skill_name": self.trace_skill_name,
                "steering_primitive": self.trace_steering_primitive,
                "target_angle_rad": self.trace_target_angle_rad,
                "speed_mps": values["speed_mps"],
                "steering_angle_rad": values["steering_angle_rad"],
                "lane_center_offset_m": values["lane_center_offset_m"],
                "heading_error_rad": values["heading_error_rad"],
                "lane_curvature": values["lane_curvature"],
                "front_vehicle_distance_m": values["front_vehicle_distance_m"],
                "distance_to_maneuver_m": values["distance_to_maneuver_m"],
            }
        )

    def _spectator_transform(self):
        if self.vehicle is None:
            return self.carla.Transform()
        transform = self.vehicle.get_transform()
        forward = transform.get_forward_vector()
        location = self.carla.Location(
            x=transform.location.x - 8.0 * forward.x,
            y=transform.location.y - 8.0 * forward.y,
            z=transform.location.z + 4.0,
        )
        rotation = self.carla.Rotation(pitch=-18.0, yaw=transform.rotation.yaw)
        return self.carla.Transform(location, rotation)

    def _destroy_actors(self) -> None:
        for actor in reversed(self.actors):
            try:
                if actor.is_alive:
                    stop = getattr(actor, "stop", None)
                    if callable(stop):
                        stop()
                    actor.destroy()
            except Exception:
                pass
        self.actors = []
        self.vehicle = None
        self.camera = None

    def _empty_state(self) -> dict[str, float | bool | str | None]:
        return {
            "speed_mps": 0.0,
            "steering_angle_rad": 0.0,
            "lane_center_offset_m": 0.0,
            "heading_error_rad": 0.0,
            "lane_curvature": 0.0,
            "drivable_area_confidence": 0.0,
            "front_vehicle_exists": False,
            "front_vehicle_distance_m": None,
            "front_vehicle_relative_speed_mps": None,
            "route_command": "keep_lane",
            "distance_to_maneuver_m": None,
            "perception_confidence": 0.0,
        }


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle
