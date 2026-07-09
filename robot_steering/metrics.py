from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean

from robot_steering.schema import SteeringTrajectorySample


@dataclass(frozen=True)
class SteeringSkillMetrics:
    episode_count: int
    success_count: int
    success_rate: float
    mean_final_angle_error_rad: float
    max_final_angle_error_rad: float
    mean_overshoot_rad: float
    max_overshoot_rad: float
    mean_settling_time_s: float
    hold_stability_rad: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def episode_overshoot(sample: SteeringTrajectorySample) -> float:
    target = sample.command.target_angle_rad
    start = sample.command.current_angle_rad
    direction = 1.0 if target >= start else -1.0
    signed_progress = [
        direction * (obs.wheel_angle_rad - target)
        for obs in sample.observations
    ]
    return max(0.0, max(signed_progress) if signed_progress else 0.0)


def episode_settling_time(sample: SteeringTrajectorySample, tolerance_rad: float) -> float:
    if not sample.observations:
        return 0.0
    target = sample.command.target_angle_rad
    for obs in sample.observations:
        if abs(obs.wheel_angle_rad - target) <= tolerance_rad:
            return obs.timestamp_s
    return sample.observations[-1].timestamp_s


def episode_hold_stability(sample: SteeringTrajectorySample, tail_s: float = 0.5) -> float:
    if not sample.observations:
        return 0.0
    end_t = sample.observations[-1].timestamp_s
    tail = [obs.wheel_angle_rad for obs in sample.observations if obs.timestamp_s >= end_t - tail_s]
    if len(tail) <= 1:
        return 0.0
    avg = mean(tail)
    return max(abs(value - avg) for value in tail)


def aggregate_metrics(samples: list[SteeringTrajectorySample], tolerance_rad: float = 0.035) -> SteeringSkillMetrics:
    if not samples:
        return SteeringSkillMetrics(
            episode_count=0,
            success_count=0,
            success_rate=0.0,
            mean_final_angle_error_rad=0.0,
            max_final_angle_error_rad=0.0,
            mean_overshoot_rad=0.0,
            max_overshoot_rad=0.0,
            mean_settling_time_s=0.0,
            hold_stability_rad=0.0,
        )
    errors = [abs(sample.final_angle_error_rad) for sample in samples]
    overshoots = [episode_overshoot(sample) for sample in samples]
    settling = [episode_settling_time(sample, tolerance_rad) for sample in samples]
    stability = [episode_hold_stability(sample) for sample in samples]
    success_count = sum(1 for sample in samples if sample.success)
    return SteeringSkillMetrics(
        episode_count=len(samples),
        success_count=success_count,
        success_rate=success_count / len(samples),
        mean_final_angle_error_rad=mean(errors),
        max_final_angle_error_rad=max(errors),
        mean_overshoot_rad=mean(overshoots),
        max_overshoot_rad=max(overshoots),
        mean_settling_time_s=mean(settling),
        hold_stability_rad=mean(stability),
    )

