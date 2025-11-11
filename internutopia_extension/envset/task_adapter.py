from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Dict, List

from .scenario_types import EnvsetScenarioData, RobotSpec

# Import pre-defined controller configurations for complex robots
try:
    from internutopia_extension.configs.robots.aliengo import move_to_point_cfg as aliengo_move_to_point_cfg
    from internutopia_extension.configs.robots.h1 import move_to_point_cfg as h1_move_to_point_cfg
    from internutopia_extension.configs.robots.g1 import move_to_point_cfg as g1_move_to_point_cfg
    from internutopia_extension.configs.robots.gr1 import move_to_point_cfg as gr1_move_to_point_cfg
except ImportError:
    aliengo_move_to_point_cfg = None
    h1_move_to_point_cfg = None
    g1_move_to_point_cfg = None
    gr1_move_to_point_cfg = None


class EnvsetTaskAugmentor:
    @staticmethod
    def apply(config: Dict[str, Any], scenario_data: EnvsetScenarioData, scenario_id: str = None) -> Dict[str, Any]:
        tasks = config.get("task_configs")
        if not isinstance(tasks, list):
            return config
        payload = EnvsetTaskAugmentor._build_envset_payload(scenario_data, scenario_id)
        for idx, task in enumerate(tasks):
            EnvsetTaskAugmentor._inject_task(task, payload, scenario_data, robot_prefix=f"envset_{idx}")
        return config

    @staticmethod
    def _build_envset_payload(scenario_data: EnvsetScenarioData, scenario_id: str = None) -> Dict[str, Any]:
        payload = {
            "scene": scenario_data.scene.raw,
            "navmesh": scenario_data.navmesh.raw if scenario_data.navmesh else None,
            "logging": scenario_data.logging,
            "scenario_id": scenario_id,  # Add scenario id to payload
        }
        if scenario_data.virtual_humans:
            vh = scenario_data.virtual_humans
            payload["virtual_humans"] = {
                "category": vh.category,
                "count": vh.count,
                "name_sequence": list(vh.name_sequence),
                "assets": vh.assets,
                "asset_root": vh.asset_root,
                "spawn_points": [EnvsetTaskAugmentor._spawn_point_to_dict(sp) for sp in vh.spawn_points],
                "routes": [EnvsetTaskAugmentor._route_to_dict(rt) for rt in vh.routes],
                "options": vh.options,
            }
        robots_payload = [EnvsetTaskAugmentor._robot_to_payload(rb) for rb in scenario_data.robots]
        payload["robots"] = robots_payload
        return payload

    @staticmethod
    def _inject_task(task: Dict[str, Any], payload: Dict[str, Any], scenario_data: EnvsetScenarioData, robot_prefix: str):
        envset_entry = task.setdefault("envset", {})
        envset_entry.update(payload)
        scene_usd = scenario_data.scene.usd_path
        if scene_usd:
            task["scene_asset_path"] = scene_usd
        EnvsetTaskAugmentor._inject_robots(task, scenario_data.robots, robot_prefix)

    @staticmethod
    def _spawn_point_to_dict(spec) -> Dict[str, Any]:
        return {
            "name": spec.name,
            "position": list(spec.position),
            "orientation_deg": spec.orientation_deg,
        }

    @staticmethod
    def _route_to_dict(spec) -> Dict[str, Any]:
        return {
            "name": spec.name,
            "commands": list(spec.commands),
        }

    @staticmethod
    def _robot_to_payload(spec: RobotSpec) -> Dict[str, Any]:
        payload = asdict(spec)
        # remove nested dataclass conversion for control raw data already handled
        return payload

    @staticmethod
    def _inject_robots(task: Dict[str, Any], robots: tuple[RobotSpec, ...], robot_prefix: str):
        """Inject robot configurations into task dict as RobotCfg objects."""
        if not robots:
            return
        robot_list = task.setdefault("robots", [])

        # Extract existing names (handle both dict and RobotCfg objects for compatibility)
        existing_names = set()
        for entry in robot_list:
            if isinstance(entry, dict):
                existing_names.add(str(entry.get("name")))
            else:
                existing_names.add(str(entry.name))

        for idx, spec in enumerate(robots):
            robot_cfg = EnvsetTaskAugmentor._build_robot_entry(spec, robot_prefix, idx)
            if not robot_cfg:
                continue

            # Check for name conflicts and resolve
            if robot_cfg.name in existing_names:
                robot_cfg.name = f"{robot_cfg.name}_{idx}"

            existing_names.add(robot_cfg.name)
            robot_list.append(robot_cfg)  # Append RobotCfg object directly

    @staticmethod
    def _build_robot_entry(spec: RobotSpec, robot_prefix: str, idx: int):
        """Build RobotCfg object from envset RobotSpec.

        Returns: RobotCfg object (Pydantic model) or None
        """
        from internutopia.core.config import RobotCfg

        robot_type = EnvsetTaskAugmentor._resolve_robot_type(spec)
        if not robot_type:
            return None

        name = spec.label or f"{robot_prefix}_{idx}"
        controllers = EnvsetTaskAugmentor._build_robot_controllers(spec, name)

        # Build RobotCfg object directly
        robot_cfg = RobotCfg(
            name=name,
            type=robot_type,
            prim_path=spec.spawn_path or f"/World/Robots/{name}",
            usd_path=spec.usd_path,
            position=tuple(spec.initial_position),
            orientation=EnvsetTaskAugmentor._orientation_from_deg(spec.initial_orientation_deg),
            controllers=controllers if controllers else None,
        )

        # Store envset control metadata if needed (via extra fields - Pydantic allows this)
        if spec.control and hasattr(robot_cfg, '__pydantic_extra__'):
            robot_cfg.__pydantic_extra__ = {"envset_control": asdict(spec.control)}

        return robot_cfg

    @staticmethod
    def _resolve_robot_type(spec: RobotSpec) -> str | None:
        type_name = (spec.type or "").lower()

        # Differential drive robots
        if type_name == "carter_v1":
            return "CarterV1Robot"
        if type_name in {"carter", "jetbot", "differential_drive"}:
            return "JetbotRobot"

        # Quadruped robots
        if type_name in {"aliengo"}:
            return "AliengoRobot"

        # Humanoid robots
        if type_name in {"h1", "human"}:
            return "H1Robot"
        if type_name in {"g1"}:
            return "G1Robot"
        if type_name in {"gr1"}:
            return "GR1Robot"

        # Manipulation robots
        if type_name in {"franka"}:
            return "FrankaRobot"

        # Fallback to JetbotRobot for backward compatibility
        return "JetbotRobot" if spec.control else None

    @staticmethod
    def _build_robot_controllers(spec: RobotSpec, name: str) -> List:  # Returns List[ControllerCfg] objects
        """Build controller configuration objects for the robot."""
        from internutopia.core.config.robot import ControllerCfg
        from internutopia_extension.configs.controllers import (
            DifferentialDriveControllerCfg,
            MoveToPointBySpeedControllerCfg,
        )

        params = spec.control.params if spec.control else {}
        robot_type = (spec.type or "").lower()

        # Differential drive robots (jetbot, carter, etc.)
        if robot_type in {"carter", "carter_v1", "jetbot", "differential_drive"}:
            # Carter V1 has different default wheel parameters than Jetbot
            if robot_type == "carter_v1":
                default_wheel_radius = 0.24
                default_wheel_base = 0.54
            else:
                default_wheel_radius = 0.03
                default_wheel_base = 0.1125

            wheel_radius = EnvsetTaskAugmentor._safe_float(params.get("wheel_radius"), fallback=default_wheel_radius)
            wheel_base = EnvsetTaskAugmentor._safe_float(params.get("track_width"), fallback=default_wheel_base)
            forward_speed = EnvsetTaskAugmentor._safe_float(params.get("base_velocity"), fallback=1.0)
            rotation_speed = EnvsetTaskAugmentor._safe_float(params.get("base_turn_rate"), fallback=1.0)

            # Build Pydantic objects directly
            drive_cfg = DifferentialDriveControllerCfg(
                name=f"{name}_drive",
                wheel_radius=wheel_radius,
                wheel_base=wheel_base,
            )
            goto_cfg = MoveToPointBySpeedControllerCfg(
                name=f"{name}_move",
                forward_speed=forward_speed,
                rotation_speed=rotation_speed,
                threshold=0.1,
                sub_controllers=[drive_cfg],
            )
            return [goto_cfg]

        # Legged and humanoid robots - use pre-defined configurations with parameter overrides
        if robot_type == "aliengo" and aliengo_move_to_point_cfg is not None:
            return [EnvsetTaskAugmentor._clone_and_override_controller(aliengo_move_to_point_cfg, params)]

        if robot_type in {"h1", "human"} and h1_move_to_point_cfg is not None:
            return [EnvsetTaskAugmentor._clone_and_override_controller(h1_move_to_point_cfg, params)]

        if robot_type == "g1" and g1_move_to_point_cfg is not None:
            return [EnvsetTaskAugmentor._clone_and_override_controller(g1_move_to_point_cfg, params)]

        if robot_type == "gr1" and gr1_move_to_point_cfg is not None:
            return [EnvsetTaskAugmentor._clone_and_override_controller(gr1_move_to_point_cfg, params)]

        # Manipulation robots (franka) - no controllers
        if robot_type in {"franka"}:
            return []

        # Fallback: return empty
        return []

    @staticmethod
    def _clone_and_override_controller(controller_cfg, params: Dict[str, Any]):
        """Clone controller config and override parameters from envset.

        Returns: ControllerCfg object (Pydantic model)
        """
        # Deep clone to avoid modifying the original predefined config
        cloned = controller_cfg.model_copy(deep=True)

        # Override speed parameters from envset if provided
        forward_speed = EnvsetTaskAugmentor._safe_float(params.get("base_velocity"), fallback=None)
        rotation_speed = EnvsetTaskAugmentor._safe_float(params.get("base_turn_rate"), fallback=None)

        if forward_speed is not None and hasattr(cloned, "forward_speed"):
            cloned.forward_speed = forward_speed
        if rotation_speed is not None and hasattr(cloned, "rotation_speed"):
            cloned.rotation_speed = rotation_speed

        return cloned

    @staticmethod
    def _orientation_from_deg(yaw_deg: float | None):
        if yaw_deg is None:
            return None
        rad = math.radians(yaw_deg)
        half = rad / 2.0
        return (math.cos(half), 0.0, 0.0, math.sin(half))

    @staticmethod
    def _safe_float(value: Any, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(fallback)
