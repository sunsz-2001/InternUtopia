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
        if not robots:
            return
        robot_list = task.setdefault("robots", [])
        existing_names = {str(entry.get("name")) for entry in robot_list}
        for idx, spec in enumerate(robots):
            entry = EnvsetTaskAugmentor._build_robot_entry(spec, robot_prefix, idx)
            if not entry:
                continue
            name = entry.get("name")
            if name in existing_names:
                entry["name"] = f"{name}_{idx}"
            existing_names.add(entry["name"])
            robot_list.append(entry)

    @staticmethod
    def _build_robot_entry(spec: RobotSpec, robot_prefix: str, idx: int) -> Dict[str, Any] | None:
        robot_type = EnvsetTaskAugmentor._resolve_robot_type(spec)
        if not robot_type:
            return None
        name = spec.label or f"{robot_prefix}_{idx}"
        entry: Dict[str, Any] = {
            "name": name,
            "type": robot_type,
            "prim_path": spec.spawn_path or f"/World/Robots/{name}",
            "usd_path": spec.usd_path,
            "position": list(spec.initial_position),
        }
        quat = EnvsetTaskAugmentor._orientation_from_deg(spec.initial_orientation_deg)
        if quat:
            entry["orientation"] = quat
        controllers = EnvsetTaskAugmentor._build_robot_controllers(spec, name)
        if controllers:
            entry["controllers"] = controllers
        if spec.control:
            entry.setdefault("extra", {})["envset_control"] = asdict(spec.control)
        return entry

    @staticmethod
    def _resolve_robot_type(spec: RobotSpec) -> str | None:
        type_name = (spec.type or "").lower()

        # Differential drive robots
        if type_name in {"carter", "carter_v1", "jetbot", "differential_drive"}:
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
    def _build_robot_controllers(spec: RobotSpec, name: str) -> List[Dict[str, Any]]:
        params = spec.control.params if spec.control else {}
        robot_type = (spec.type or "").lower()

        # Differential drive robots (jetbot, carter, etc.)
        if robot_type in {"carter", "carter_v1", "jetbot", "differential_drive"}:
            wheel_radius = EnvsetTaskAugmentor._safe_float(params.get("wheel_radius"), fallback=0.03)
            wheel_base = EnvsetTaskAugmentor._safe_float(params.get("track_width"), fallback=0.1125)
            forward_speed = EnvsetTaskAugmentor._safe_float(params.get("base_velocity"), fallback=1.0)
            rotation_speed = EnvsetTaskAugmentor._safe_float(params.get("base_turn_rate"), fallback=1.0)

            drive_cfg = {
                "name": f"{name}_drive",
                "type": "DifferentialDriveController",
                "wheel_radius": wheel_radius,
                "wheel_base": wheel_base,
            }
            goto_cfg = {
                "name": f"{name}_move",
                "type": "MoveToPointBySpeedController",
                "forward_speed": forward_speed,
                "rotation_speed": rotation_speed,
                "threshold": 0.1,
                "sub_controllers": [drive_cfg],
            }
            return [goto_cfg]

        # Legged and humanoid robots - use pre-defined controller configurations
        # These include the full policy-based controllers with correct weights and joint mappings
        if robot_type == "aliengo" and aliengo_move_to_point_cfg is not None:
            return [EnvsetTaskAugmentor._controller_cfg_to_dict(aliengo_move_to_point_cfg, params)]

        if robot_type in {"h1", "human"} and h1_move_to_point_cfg is not None:
            return [EnvsetTaskAugmentor._controller_cfg_to_dict(h1_move_to_point_cfg, params)]

        if robot_type == "g1" and g1_move_to_point_cfg is not None:
            return [EnvsetTaskAugmentor._controller_cfg_to_dict(g1_move_to_point_cfg, params)]

        if robot_type == "gr1" and gr1_move_to_point_cfg is not None:
            return [EnvsetTaskAugmentor._controller_cfg_to_dict(gr1_move_to_point_cfg, params)]

        # Manipulation robots (franka) - no controllers, use robot's built-in
        if robot_type in {"franka"}:
            return []

        # Fallback: return empty to avoid misconfiguration
        # If pre-defined configs are not available, robot will have no controllers
        return []

    @staticmethod
    def _controller_cfg_to_dict(controller_cfg, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ControllerCfg object to dictionary and optionally override parameters."""
        # Use pydantic's dict() or model_dump() depending on version
        try:
            cfg_dict = controller_cfg.model_dump()  # pydantic v2
        except AttributeError:
            cfg_dict = controller_cfg.dict()  # pydantic v1

        # Override speed parameters from envset if provided
        forward_speed = EnvsetTaskAugmentor._safe_float(params.get("base_velocity"), fallback=None)
        rotation_speed = EnvsetTaskAugmentor._safe_float(params.get("base_turn_rate"), fallback=None)

        if forward_speed is not None and "forward_speed" in cfg_dict:
            cfg_dict["forward_speed"] = forward_speed
        if rotation_speed is not None and "rotation_speed" in cfg_dict:
            cfg_dict["rotation_speed"] = rotation_speed

        return cfg_dict

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
