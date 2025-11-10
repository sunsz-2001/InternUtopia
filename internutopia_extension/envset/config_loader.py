"""Utilities to merge InternUtopia config with envset scenario data."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from .scenario_types import (
    EnvsetScenarioData,
    NavmeshSpec,
    RobotControlSpec,
    RobotSpec,
    RouteSpec,
    SceneSpec,
    SpawnPointSpec,
    VirtualHumanSpec,
)
from .task_adapter import EnvsetTaskAugmentor


@dataclass
class EnvsetConfigBundle:
    """Result of merging the base config with an envset scenario."""

    merged_config_path: Path
    config: Dict[str, Any]
    envset: Dict[str, Any]
    scenario_id: str
    scenario: Dict[str, Any]
    scenario_data: EnvsetScenarioData


class EnvsetConfigLoader:
    def __init__(self, config_path: Path, envset_path: Path, scenario_id: str | None = None):
        self._config_path = config_path
        self._envset_path = envset_path
        self._scenario_id = scenario_id

    def load(self) -> EnvsetConfigBundle:
        config = self._load_config_yaml()
        envset = self._load_envset_json()
        scenario_id, scenario = self._select_scenario(envset)
        scenario_data = self._build_scenario_data(scenario)
        # Store scenario_id for use in _apply_envset
        self._selected_scenario_id = scenario_id
        merged_config = self._apply_envset(config, scenario_data)
        merged_path = self._write_temp_yaml(merged_config)
        return EnvsetConfigBundle(
            merged_config_path=merged_path,
            config=merged_config,
            envset=envset,
            scenario_id=scenario_id,
            scenario=scenario,
            scenario_data=scenario_data,
        )

    def _load_config_yaml(self) -> Dict[str, Any]:
        if not self._config_path.exists():
            raise FileNotFoundError(f"Config YAML not found: {self._config_path}")
        with self._config_path.open("r", encoding="utf-8") as fp:
            return yaml.safe_load(fp) or {}

    def _load_envset_json(self) -> Dict[str, Any]:
        if not self._envset_path.exists():
            raise FileNotFoundError(f"Envset JSON not found: {self._envset_path}")
        with self._envset_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    def _select_scenario(self, envset: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        scenarios = envset.get("scenarios") or []
        if not scenarios:
            raise ValueError("Envset file does not define any scenarios")
        if self._scenario_id:
            for item in scenarios:
                if item.get("id") == self._scenario_id:
                    return self._scenario_id, item
            raise ValueError(f"Scenario '{self._scenario_id}' not found in envset")
        first = scenarios[0]
        scenario_id = first.get("id") or "scenario_0"
        return scenario_id, first

    def _apply_envset(self, base: Dict[str, Any], scenario_data: EnvsetScenarioData) -> Dict[str, Any]:
        merged = dict(base or {})
        scene_section = dict(merged.get("scene") or {})
        scene_cfg = scenario_data.scene.raw

        usd_path = scene_cfg.get("usd_path") or scene_cfg.get("asset_path")
        if usd_path:
            scene_section["asset_path"] = usd_path

        use_matterport = scene_cfg.get("use_matterport")
        if use_matterport is None:
            category = str(scene_cfg.get("category") or "").lower()
            use_matterport = category == "mp3d"
        scene_section["use_matterport"] = bool(use_matterport)

        merged["scene"] = scene_section
        # Use the selected scenario_id (from _select_scenario) instead of self._scenario_id
        scenario_id = getattr(self, '_selected_scenario_id', self._scenario_id)
        merged = EnvsetTaskAugmentor.apply(merged, scenario_data, scenario_id=scenario_id)
        return merged

    def _build_scenario_data(self, scenario: Dict[str, Any]) -> EnvsetScenarioData:
        scene = self._build_scene_spec(scenario.get("scene") or {})
        navmesh = self._build_navmesh_spec(scenario.get("navmesh"))
        vh = self._build_virtual_humans_spec(scenario.get("virtual_humans"))
        robots = tuple(self._build_robot_spec(entry) for entry in scenario.get("robots", {}).get("entries", []))
        logging = scenario.get("logging") or {}
        return EnvsetScenarioData(
            scene=scene,
            navmesh=navmesh,
            virtual_humans=vh,
            robots=robots,
            logging=logging,
            raw=scenario,
        )

    def _build_scene_spec(self, data: Dict[str, Any]) -> SceneSpec:
        return SceneSpec(
            usd_path=data.get("usd_path"),
            scene_type=data.get("type"),
            category=data.get("category"),
            root_prim_path=data.get("root_prim_path"),
            navmesh_root_prim_path=data.get("navmesh_root_prim_path"),
            notes=data.get("notes"),
            raw=data,
        )

    def _build_navmesh_spec(self, data: Optional[Dict[str, Any]]) -> Optional[NavmeshSpec]:
        if not data:
            return None
        min_size = data.get("min_include_volume_size") or {}
        return NavmeshSpec(
            bake_root_prim_path=data.get("bake_root_prim_path"),
            include_volume_parent=data.get("include_volume_parent"),
            z_padding=data.get("z_padding"),
            agent_radius=data.get("agent_radius"),
            min_include_xy=min_size.get("xy"),
            min_include_z=min_size.get("z"),
            spawn_min_separation_m=data.get("spawn_min_separation_m"),
            raw=data,
        )

    def _build_virtual_humans_spec(self, data: Optional[Dict[str, Any]]) -> Optional[VirtualHumanSpec]:
        if not data:
            return None
        name_sequence = tuple(str(name) for name in data.get("name_sequence", []) if name is not None)
        spawn_points = tuple(self._build_spawn_point_spec(item) for item in data.get("spawn_points", []))
        routes = tuple(self._build_route_spec(item) for item in data.get("routes", []))
        assets = {}
        raw_assets = data.get("assets") or {}
        for key, value in raw_assets.items():
            assets[str(key)] = str(value)
        return VirtualHumanSpec(
            category=data.get("category"),
            count=data.get("count"),
            name_sequence=name_sequence,
            assets=assets,
            asset_root=data.get("asset_root") or {},
            spawn_points=spawn_points,
            routes=routes,
            options=data,
        )

    def _build_spawn_point_spec(self, data: Dict[str, Any]) -> SpawnPointSpec:
        pos = tuple(float(v) for v in data.get("position", (0.0, 0.0, 0.0)))
        return SpawnPointSpec(
            name=data.get("name"),
            position=(pos + (0.0, 0.0, 0.0))[:3],
            orientation_deg=self._safe_float(data.get("orientation_deg")),
            raw=data,
        )

    def _build_route_spec(self, data: Dict[str, Any]) -> RouteSpec:
        commands = tuple(str(cmd) for cmd in data.get("commands", []) if cmd)
        return RouteSpec(name=data.get("name"), commands=commands, raw=data)

    def _build_robot_spec(self, data: Dict[str, Any]) -> RobotSpec:
        initial_pose = data.get("initial_pose") or {}
        position = tuple(float(v) for v in initial_pose.get("position", (0.0, 0.0, 0.0)))
        orientation_deg = self._safe_float(initial_pose.get("orientation_deg"))
        control_cfg = data.get("control") or None
        control = None
        if control_cfg:
            control = RobotControlSpec(
                mode=control_cfg.get("mode"),
                module=control_cfg.get("module"),
                entry=control_cfg.get("entry"),
                params=control_cfg.get("params") or {},
            )
        return RobotSpec(
            label=data.get("label"),
            type=data.get("type"),
            spawn_path=data.get("spawn_path"),
            usd_path=data.get("usd_path"),
            initial_position=(position + (0.0, 0.0, 0.0))[:3],
            initial_orientation_deg=orientation_deg,
            control=control,
            raw=data,
        )

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _write_temp_yaml(self, data: Dict[str, Any]) -> Path:
        fd, tmp_path = tempfile.mkstemp(prefix="envset_cfg_", suffix=".yaml")
        os.close(fd)
        path = Path(tmp_path)
        with path.open("w", encoding="utf-8") as fp:
            yaml.safe_dump(data, fp, sort_keys=False)
        return path
