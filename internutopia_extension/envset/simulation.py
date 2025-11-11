# flake8: noqa
import asyncio
import json
import os
from pathlib import Path
from typing import List

import carb
import numpy as np
import omni.anim.navigation.core as nav

import omni.anim.navigation.meshtools as meshtools
import omni.client
import omni.kit
import omni.usd
from omni.metropolis.utils.debug_util import DebugPrint
from omni.metropolis.utils.file_util import CommandFileUtil, FileUtil
from omni.metropolis.utils.semantics_util import SemanticsUtils
from pxr import Gf, Sdf, UsdGeom, Usd
import NavSchema

from omni.metropolis.utils.config_file.core import ConfigFile
from omni.metropolis.utils.config_file.util import ConfigFileError

# Optional import: matterport may not be available in all Isaac Sim versions
try:
    from omni.isaac.matterport.scripts import import_matterport_asset
    MATTERPORT_AVAILABLE = True
except ImportError:
    MATTERPORT_AVAILABLE = False
    import_matterport_asset = None
    carb.log_warn("omni.isaac.matterport is not available. Matterport scene import will be disabled.")

# Removed redundant direct imports; collision/ground handled in importer
from internutopia_extension.data_generation.data_generation import DataGeneration
from internutopia_extension.randomization.camera_randomizer import CameraRandomizer, LidarCameraRandomizer
from internutopia_extension.randomization.carter_randomizer import CarterRandomizer
from internutopia_extension.randomization.character_randomizer import CharacterRandomizer
from internutopia_extension.randomization.randomizer_util import RandomizerUtil
from internutopia_extension.randomization.iw_hub_randomizer import IwHubRandomizer
from internutopia_extension.response.core import AgentResponseManager
from internutopia_extension.envset.robot_control import RobotControlManager
from internutopia_extension.envset.settings import AssetPaths, PrimPaths, BehaviorScriptPaths, Settings, GlobalValues
from .stage_util import (
    AgentUtil,
    CameraUtil,
    CharacterUtil,
    LidarCamUtil,
    RobotUtil,
    StageUtil,
    UnitScaleService,
)
from .navmesh_utils import ensure_navmesh_async
from .incident_bridge import IncidentBridge
from .virtual_human_colliders import ColliderConfig, VirtualHumanColliderApplier
from isaacsim.core.utils import prims
from internutopia_extension.guards.arrival_guard import ArrivalGuard

FRAME_RATE = 30

OMNI_ANIM_PEOPLE_COMMAND_PATH = "/exts/omni.anim.people/command_settings/command_file_path"
ANIM_ROBOT_COMMAND_PATH = "/exts/isaacsim.anim.robot/command_settings/command_file_path"
ENVSET_PATH_SETTING = "/exts/isaacsim.replicator.agent/envset/path"
ENVSET_SCENARIO_SETTING = "/exts/isaacsim.replicator.agent/envset/scenario_id"
ENVSET_AUTOSTART_SETTING = "/exts/isaacsim.replicator.agent/envset/autostart"

dp = DebugPrint(Settings.DEBUG_PRINT, "SimulationManager")


class SimulationManager:
    """
    Simulation Manager class that takes in config file to set up simulation accordingly.
    """

    SET_UP_SIMULATION_DONE_EVENT = "isaacsim.replicator.agent.SET_UP_SIMULATION_DONE"
    DATA_GENERATION_DONE_EVENT = "isaacsim.replicator.agent.DATA_GENERATION_DONE_EVENT"

    def __init__(self):
        self.character_assets_list = (
            []
        )  # List of all characters inside the character asset folders, provided by config file
        self.available_character_list = []  # Character list after filtering and shuffling
        # Config file variables
        self.config_file: ConfigFile = None
        # Randomizers
        self._character_randomizer = CharacterRandomizer(0)
        self._nova_carter_randomizer = CarterRandomizer(0)
        self._iw_hub_randomizer = IwHubRandomizer(0)
        self._camera_randomizer = CameraRandomizer(0)
        self._lidar_camera_randomizer = LidarCameraRandomizer(0)
        self._agent_positions = []
        # State variables for assets loading
        self._load_stage_handle = None
        # Incident bridge
        self._incident_bridge = IncidentBridge()
        # Need to acquire the meshtools interface and hold it throughout the simulation
        self._imeshtools = meshtools.acquire_interface()
        self._dg = None
        self._dg_task = None
        # Envset-driven robot control
        self._robot_control_mgr = RobotControlManager()
        self._env_scene_catalog = None
        self._env_scene_config = None
        self._envset_autostart_done = False
        self._refresh_envset_catalog()
        self._virtual_human_collider_mgr = None
        # Arrival guard (only for GRScenes/cm-scale)
        self._arrival_guard: ArrivalGuard | None = None
        
        # NavMesh synchronization primitives
        self._navmesh_baking_complete = None
        # removed unused _navmesh_baking_task
        self._navmesh_status = "idle"  # idle, baking, ready, failed
        self._matterport_import_future = None
        self._navmesh_ready = False
        

    # ========= Set Up Characters/Robots =========

    def _refresh_envset_catalog(self):
        try:
            envset_path_setting = carb.settings.get_settings().get(ENVSET_PATH_SETTING)
        except Exception:
            envset_path_setting = None
        catalog = self._load_envset_catalog(envset_path_setting)
        self._env_scene_catalog = catalog
        if catalog is None:
            self.apply_env_scene_config(None)
        return catalog

    def _load_envset_catalog(self, envset_path_setting):
        if not envset_path_setting:
            return None
        try:
            envset_path = Path(str(envset_path_setting)).expanduser()
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"[EnvSet] Invalid envset path '{envset_path_setting}': {exc}")
            return None
        if not envset_path.exists():
            carb.log_warn(f"[EnvSet] Envset file not found at '{envset_path}'.")
            return None
        try:
            with envset_path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"[EnvSet] Failed to load envset '{envset_path}': {exc}")
            return None
        scenarios = {}
        for idx, item in enumerate(data.get("scenarios", [])):
            scenario_id = item.get("id")
            if not scenario_id:
                scenario_id = f"scenario_{idx}"
            scenarios[scenario_id] = item
        if not scenarios:
            carb.log_warn(f"[EnvSet] No scenarios defined in '{envset_path}'.")
            return None
        carb.log_info(f"[EnvSet] Loaded {len(scenarios)} scenario(s) from '{envset_path}'.")
        return {"path": envset_path, "data": data, "scenarios": scenarios}

    @staticmethod
    def _paths_match(path_a, path_b) -> bool:
        if not path_a or not path_b:
            return False
        a = str(path_a).rstrip("/").lower()
        b = str(path_b).rstrip("/").lower()
        return a == b

    def apply_env_scene_config(self, scene_cfg):
        self._env_scene_config = scene_cfg
        self._robot_control_mgr.set_scene_config(scene_cfg)

    def _get_env_scene_config(self):
        return self._env_scene_config or {}

    def _get_env_scene(self):
        cfg = self._get_env_scene_config()
        return cfg.get("scene") or {}

    def _get_env_scene_usd_path(self):
        scene_cfg = self._get_env_scene()
        usd_path = scene_cfg.get("usd_path")
        if usd_path:
            return usd_path
        return None

    def _get_env_scene_type(self):
        scene_cfg = self._get_env_scene()
        return scene_cfg.get("type")

    def _get_env_scene_category(self):
        scene_cfg = self._get_env_scene()
        category = scene_cfg.get("category")
        if category:
            category_str = str(category).strip()
            if category_str:
                return category_str

        # Fallbacks for legacy configs
        try:
            use_matterport = scene_cfg.get("use_matterport")
        except Exception:
            use_matterport = None
        if use_matterport is not None:
            if bool(use_matterport):
                return "MP3D"
        scene_type = scene_cfg.get("type")
        if scene_type and str(scene_type).lower() in ("matterport", "mp3d"):
            return "MP3D"

        usd_path = scene_cfg.get("usd_path") or ""
        if isinstance(usd_path, str) and "grscene" in usd_path.lower():
            return "GRScenes"

        navmesh_root = scene_cfg.get("navmesh_root_prim_path")
        if navmesh_root and str(navmesh_root).startswith("/Root"):
            return "GRScenes"

        return None

    def load_env_scene_by_id(self, scenario_id: str) -> bool:
        if not self._env_scene_catalog:
            carb.log_warn("[EnvSet] No envset catalog available.")
            return False
        scenario = self._env_scene_catalog.get("scenarios", {}).get(scenario_id)
        if not scenario:
            carb.log_warn(f"[EnvSet] Scenario '{scenario_id}' not found.")
            return False
        carb.log_info(f"[EnvSet] Manually apply scenario '{scenario_id}'.")
        self.apply_env_scene_config(scenario)
        self._maybe_autostart()
        return True

    def _select_env_scene_for_config(self):
        if not self._env_scene_catalog:
            self.apply_env_scene_config(None)
            return

        scenarios = self._env_scene_catalog.get("scenarios", {})
        chosen = None

        try:
            scenario_id = carb.settings.get_settings().get(ENVSET_SCENARIO_SETTING)
        except Exception:
            scenario_id = None

        if scenario_id and scenario_id in scenarios:
            chosen = scenarios[scenario_id]
        elif scenarios:
            # Single-scenario envset: pick the first one.
            first_key = next(iter(scenarios))
            chosen = scenarios[first_key]
            carb.log_info(f"[EnvSet] Defaulting to scenario '{first_key}'.")
        else:
            carb.log_info("[EnvSet] No scenario available in envset catalog.")
        if chosen:
            self.apply_env_scene_config(chosen)
        else:
            self.apply_env_scene_config(None)
        self._maybe_autostart()

    def _should_autostart(self) -> bool:
        try:
            value = carb.settings.get_settings().get(ENVSET_AUTOSTART_SETTING)
        except Exception:
            return False
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _maybe_autostart(self):
        if self._envset_autostart_done:
            return
        if not self._should_autostart():
            return
        self._envset_autostart_done = True
        carb.log_info("[EnvSet] Autostart enabled; starting simulation setup.")
        try:
            self.set_up_simulation_from_config_file()
        except Exception as exc:  # noqa: BLE001
            carb.log_error(f"[EnvSet] Autostart failed: {exc}")
            self._envset_autostart_done = False

    def _get_env_navmesh_config(self):
        if not self._env_scene_config:
            return {}
        return self._env_scene_config.get("navmesh") or {}

    def _get_navmesh_runtime_params(self):
        cfg = self._get_env_navmesh_config()
        min_size = cfg.get("min_include_volume_size") or {}
        return {
            "bake_root": cfg.get("bake_root_prim_path"),
            "include_parent": cfg.get("include_volume_parent"),
            "z_padding": cfg.get("z_padding"),
            "agent_radius": cfg.get("agent_radius"),
            "min_xy": min_size.get("xy"),
            "min_z": min_size.get("z"),
            "navmesh_root_override": cfg.get("navmesh_root_prim_path"),
        }

    def _get_env_character_spawn_specs(self):
        if not self._env_scene_config:
            return []
        vh_cfg = self._env_scene_config.get("virtual_humans") or {}
        return vh_cfg.get("spawn_points") or []

    def _get_env_character_options(self):
        if not self._env_scene_config:
            return {}
        return self._env_scene_config.get("virtual_humans") or {}

    def _get_env_character_names_and_assets(self, count: int):
        """Return (names, assets_map) for characters based on envset virtual_humans.

        - names: list of character names length==count, preferring virtual_humans.name_sequence，
                 否则回退为 CharacterUtil.get_character_name_by_index(i)。
        - assets_map: 仅从 virtual_humans.assets 读取的 name->usd 映射；
                 若 assets 的某些值为相对路径且提供了 asset_root.settings_key，则做前缀拼接。
        """
        vh_opts = self._get_env_character_options() or {}
        # names
        names = []
        seq = vh_opts.get("name_sequence") or []
        for i in range(count):
            try:
                nm = str(seq[i]).strip()
            except Exception:
                nm = ""
            if nm:
                names.append(nm)
            else:
                names.append(CharacterUtil.get_character_name_by_index(i))

        # assets map（仅支持 virtual_humans.assets）
        assets_map = {}
        raw_assets = vh_opts.get("assets")
        if isinstance(raw_assets, dict):
            for k, v in raw_assets.items():
                try:
                    name = str(k)
                    path = str(v)
                    assets_map[name] = path
                except Exception:
                    continue

        asset_root = vh_opts.get("asset_root") or {}
        root_prefix = None
        if isinstance(asset_root, dict):
            root_prefix = asset_root.get("settings_key") or None

        # resolve relative paths using root_prefix if available
        def _resolve(p: str) -> str:
            s = str(p)
            if s.startswith("omniverse://") or s.startswith("/") or s.startswith("http://") or s.startswith("https://"):
                return s
            if root_prefix:
                return f"{str(root_prefix).rstrip('/')}/{s.lstrip('/')}"
            return s

        assets_map = {k: _resolve(v) for k, v in assets_map.items()}
        return names, assets_map

    def _maybe_enable_arrival_guard(self):
        """Enable arrival guard only for GRScenes (或厘米制舞台 mpu<=0.02)。

        到达守护按“米”为单位计算水平距离，在进入容差时结束当前 GoTo 类命令，避免毫米级阈值导致的“看似到点仍行走”。
        不修改 People 内部逻辑，不影响 MP3D/内置场景（通常 mpu=1.0）。
        """
        scene_category = self._get_env_scene_category()
        if self._arrival_guard is None:
            self._arrival_guard = ArrivalGuard()
        # 可选：从 envset 读取容差（米），否则使用默认 0.5 m
        tol_m = 0.5
        try:
            vh_opts = self._get_env_character_options() or {}
            if isinstance(vh_opts, dict):
                val = vh_opts.get("arrival_tolerance_m")
                if val is not None:
                    tol_m = float(val)
        except Exception:
            pass
        self._arrival_guard.enable_if_grscenes(scene_category, tolerance_m=tol_m)

    def _build_virtual_human_paths(self, vh_options: dict) -> List[str]:
        parent_path = str(vh_options.get("collider_parent_path") or PrimPaths.characters_parent_path()).rstrip("/")
        if not parent_path:
            parent_path = "/World"
        name_sequence = vh_options.get("name_sequence") or []
        cleaned_names: List[str] = []
        seen: Set[str] = set()
        for raw in name_sequence:
            if not raw:
                continue
            name = str(raw).strip()
            if not name or name in seen:
                continue
            cleaned_names.append(name)
            seen.add(name)
        count_val = vh_options.get("count")
        try:
            count = int(count_val) if count_val is not None else len(cleaned_names)
        except Exception:
            count = len(cleaned_names)
        count = max(count, 0)
        if not cleaned_names or count == 0:
            return []
        if count > len(cleaned_names):
            carb.log_warn(
                "[EnvSet] virtual_humans.count larger than name_sequence entries; trimming to available names."
            )
            count = len(cleaned_names)
        usable = cleaned_names[:count]
        paths: List[str] = []
        for name in usable:
            paths.append(f"{parent_path}/{name}")
        return paths

    def _get_env_character_routes_commands(self) -> List[str]:
        vh_options = self._get_env_character_options()
        routes = vh_options.get("routes") if isinstance(vh_options, dict) else None
        if not routes:
            return []
        commands: List[str] = []
        for entry in routes:
            if not isinstance(entry, dict):
                carb.log_warn("[EnvSet] Ignoring route entry because it is not a dict.")
                continue
            raw_cmds = entry.get("commands") or []
            for cmd in raw_cmds:
                if isinstance(cmd, str):
                    cmd_text = cmd.strip()
                    if cmd_text:
                        commands.append(cmd_text)
                else:
                    carb.log_warn(f"[EnvSet] Route command '{cmd}' is not a string; skipping.")
        return commands

    def _setup_virtual_human_colliders(self):
        vh_options = self._get_env_character_options()
        if self._virtual_human_collider_mgr:
            self._virtual_human_collider_mgr.deactivate()
            self._virtual_human_collider_mgr = None
        character_paths = self._build_virtual_human_paths(vh_options)
        if not character_paths:
            return
        approx_shape = vh_options.get("collider_shape") or vh_options.get("approximation_shape") or "convexHull"
        kinematic_flag = vh_options.get("collider_kinematic")
        if kinematic_flag is None:
            kinematic_flag = True
        collider_cfg = ColliderConfig(approximation_shape=str(approx_shape), kinematic=bool(kinematic_flag))
        self._virtual_human_collider_mgr = VirtualHumanColliderApplier(
            character_paths=character_paths,
            collider_config=collider_cfg,
        )
        # 立即应用碰撞体，不等待timeline启动，避免虚拟人物在物理启动时下滑
        self._virtual_human_collider_mgr.activate(apply_immediately=True)

    async def _spawn_envset_robots(self):
        if not self._env_scene_config:
            return
        robots_cfg = self._env_scene_config.get("robots", {})
        entries = robots_cfg.get("entries", [])
        if not entries:
            return
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        # Find existing env_id by checking /World/env_X paths
        # This matches InternUtopia's path structure
        env_id = None
        robots_root_path = "/robots"  # Default robots_root_path from TaskCfg
        for check_id in range(10):  # Check up to 10 environments
            env_path = f"/World/env_{check_id}"
            env_prim = stage.GetPrimAtPath(env_path)
            if env_prim and env_prim.IsValid():
                env_id = check_id
                break
        
        # If no env found, default to env_0 (InternUtopia typically starts with env_0)
        if env_id is None:
            env_id = 0
            carb.log_info(f"[EnvSet] No existing env found, defaulting to env_{env_id}")

        spawned = False
        for entry in entries:
            spawn_path = entry.get("spawn_path")
            if not spawn_path:
                continue
            
            # Convert spawn_path to InternUtopia's path format
            # spawn_path is relative (e.g., "/aliengo"), we need to build full path
            # Format: /World/env_{env_id}/robots{spawn_path}
            # This matches InternUtopia's setup_offset_for_assets logic
            full_prim_path = f"/World/env_{env_id}{robots_root_path}{spawn_path}"
            
            print(f"[DEBUG] [EnvSet] Processing robot entry: spawn_path={spawn_path}, full_path={full_prim_path}")
            prim = stage.GetPrimAtPath(full_prim_path)
            prim_valid = prim.IsValid()

            # Resolve USD path only when we need to create
            if not prim_valid:
                usd_path = entry.get("usd_path")
                if not usd_path:
                    asset_path = entry.get("asset_path", {})
                    usd_path = asset_path.get("resolved") if isinstance(asset_path, dict) else None
                if not usd_path:
                    carb.log_warn(f"[EnvSet] Missing usd_path for robot entry at {spawn_path}.")
                    continue
                # Create prim at InternUtopia's expected path
                prim = prims.create_prim(full_prim_path, "Xform", usd_path=usd_path)
                prim_valid = prim.IsValid()

            # Apply/override initial pose even when prim already exists
            initial_pose = entry.get("initial_pose", {})
            position = initial_pose.get("position")
            orientation_deg = float(initial_pose.get("orientation_deg", 0.0))
            try:
                if prim_valid:
                    # Align with StageUtil.spawn_robot: enforce SOT order and set attributes directly
                    original_xform_order_setting = StageUtil.set_xformOpType_SOT()
                    if position:
                        prim.GetAttribute("xformOp:translate").Set(
                            Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
                        )
                    # Set orientation about Z axis in degrees
                    rot_quat = Gf.Rotation(Gf.Vec3d(0, 0, 1), float(orientation_deg)).GetQuat()
                    if type(prim.GetAttribute("xformOp:orient").Get()) == Gf.Quatf:
                        prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(rot_quat))
                    else:
                        prim.GetAttribute("xformOp:orient").Set(rot_quat)

                    scale_override = entry.get("scale")
                    if scale_override is not None:
                        parsed_scale = scale_override
                        if isinstance(scale_override, str):
                            try:
                                parsed_scale = float(scale_override)
                            except Exception:
                                parsed_scale = scale_override
                        if not StageUtil.set_prim_scale(prim, parsed_scale):
                            carb.log_warn(f"[EnvSet] Invalid scale override for {full_prim_path}: {scale_override}")

                    StageUtil.recover_xformOpType(original_xform_order_setting)
                    spawned = True
            except Exception as exc:  # noqa: BLE001
                carb.log_warn(f"[EnvSet] Failed to set transform for {full_prim_path}: {exc}")

        if spawned:
            print("[DEBUG] [EnvSet] Robots spawned; waiting a frame before registration.")
            try:
                await omni.kit.app.get_app().next_update_async()
            except Exception:
                pass

    def load_filters(self):
        """
        Load the filters from the asset folder
        The filter must be a json file named "filter" and located in the asset root directory
        """
        if not self.config_file:
            return None
        prop = self.config_file.get_property("character", "asset_path")
        if not prop:
            carb.log_error("Unable to get character asset path. Will not load filter file.")
            return None
        if prop.is_value_error():
            carb.log_error("Character asset path has error. Will not load filter file.")
            return None
        file_path = prop.get_resolved_value()
        # Making sure that the path ends with a slash
        if file_path[-1] != "/":
            file_path += "/"
        file_path += "filter.json"
        result, _, content = omni.client.read_file(file_path)
        data = {}
        if result == omni.client.Result.OK:
            data = json.loads(memoryview(content).tobytes().decode("utf-8"))
        # Handling the case if the file does not exist
        else:
            carb.log_warn("Filter file does not exist. Asset filtering will not function.")
            return None
        return data  # noqa

    def spawn_character_by_idx(self, spawn_location, spawn_rotation, idx):
        """
        Spawns character according to index in the character folder list at provided spawn_location and spawn_rotation.
        Ensures duplicate characters are not spawned, until all character assets have been utilized.
        If all character assets have been utilized, duplicates will be spawned.
        """
        # Character name
        char_name = CharacterUtil.get_character_name_by_index(idx)
        # Characters will be spawned in the same order again if all unique assets are used
        list_len = len(self.available_character_list)
        if list_len == 0:
            carb.log_error("Unable to spawn character due to no character assets found.")
            return None
        # Loop the list if there are multiple characters
        idx = idx % list_len  # noqa
        # The character assets are randomly sorted by global seed when the assets is selected
        # This draws the character based on the index, producing a deterministic result
        char_asset_name = self.available_character_list[idx]
        prop = self.config_file.get_property("character", "asset_path")
        if prop.is_value_error():
            carb.log_error("Unable to spawn character due to invalid character asset path.")
            return None
        asset_root_path = prop.get_resolved_value()
        character_folder = f"{asset_root_path}/{char_asset_name}"
        # Get the usd present in the character folder
        character_usd_name = self._get_character_usd_in_folder(character_folder)
        if not character_usd_name:
            carb.log_error("Unable to spawn character due to no character usd present in folder.")
            return None
        character_usd_path = f"{character_folder}/{character_usd_name}"
        # Spawn character
        return CharacterUtil.load_character_usd_to_stage(character_usd_path, spawn_location, spawn_rotation, char_name)

    def _get_character_usd_in_folder(self, character_folder_path):
        result, folder_list = omni.client.list(character_folder_path)
        if result != omni.client.Result.OK:
            carb.log_error(f"Unable to read character folder path at {character_folder_path}")
            return None
        for item in folder_list:
            if item.relative_path.endswith(".usd"):
                return item.relative_path
        carb.log_error(f"Unable to file a .usd file in {character_folder_path} character folder")
        return None

    def read_character_asset_list(self):
        """
        Read character assets into list according to the character asset path in config file
        """
        prop = self.config_file.get_property("character", "asset_path")
        if not prop:
            return
        if prop.is_value_error():
            carb.log_error("Unable to get character assets from character asset path.")
            return
        assets_root_path = prop.get_resolved_value()
        # List all files in characters directory
        result, folder_list = omni.client.list(f"{assets_root_path}/")
        if result != omni.client.Result.OK:
            carb.log_error("Unable to get character assets from character asset path.")
            self.character_assets_list = []
            return
        # Prune items from folder list that are not directories.
        pruned_folder_list = [
            folder.relative_path
            for folder in folder_list
            if (folder.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN) and not folder.relative_path.startswith(".")
        ]
        if pruned_folder_list is None or len(pruned_folder_list) == 0:
            self.character_assets_list = []
            return
        # Prune folders that do not have usd inside
        pruned_usd_folder_list = []
        for folder in pruned_folder_list:
            result, file_list = omni.client.list(f"{assets_root_path}/{folder}/")
            for file in file_list:
                post_fix = file.relative_path[file.relative_path.rfind(".") + 1 :].lower()
                if post_fix in ("usd", "usda"):
                    pruned_usd_folder_list.append(folder)
                    break
        # Prune the default biped character
        biped_name = AssetPaths.default_biped_asset_name()
        if biped_name in pruned_usd_folder_list:
            pruned_usd_folder_list.remove(biped_name)
        # Prune exclusive folders
        exclusive_character_folders = AssetPaths.exclusive_character_folders()
        for folder in exclusive_character_folders:
            if folder in pruned_usd_folder_list:
                pruned_usd_folder_list.remove(folder)
        self.character_assets_list = pruned_usd_folder_list

    def refresh_available_character_asset_list(self):
        """
        Set avaliable character asset list by filtering and shuffling the character assets list.
        """
        if len(self.character_assets_list) == 0:
            self.available_character_list = []
            return
        prop = self.config_file.get_property("character", "filters")
        labels = prop.get_resolved_value()
        self.available_character_list = self.character_assets_list.copy()
        filters = self.load_filters()
        self.filter_character_asset_list(filters, labels)
        self.shuffle_character_asset_list()

    def shuffle_character_asset_list(self):
        """
        Shuffles the order of characters in the avaliable asset list.
        """
        prop = self.config_file.get_property("global", "seed")
        if prop.is_value_error():
            carb.log_warn("Shuffle character asset list fails due to invalid global seed.")
            return
        np.random.seed(RandomizerUtil.handle_overflow(prop.get_resolved_value()))
        self.available_character_list = np.random.choice(
            self.available_character_list, size=len(self.available_character_list), replace=False
        )

    def filter_character_asset_list(self, filters, labels):
        """
        Given labels, return character assets with these labels
        """
        # Filter file does not exist, skip the filtering
        if filters is not None:
            filtered = self.available_character_list
            for label in labels:
                if label in filters:
                    filtered = [char for char in filtered if char in filters[label]]
                # Handle non-existent labels
                else:
                    if label != "" and label != " ":  # noqa
                        carb.log_warn(
                            f'Invalid character filter label: "{label}". Available labels: {", ".join(filters.keys())}'
                        )
                        labels.remove(label)
            self.available_character_list = filtered

    @dp.debug_func
    def setup_python_scripts_to_robot(self, robot_list, robot_type):
        """
        Add behavior script to all characters in stage
        """
        script_path = BehaviorScriptPaths.robot_behavior_script_path(robot_type)
        dp.print(f"To use behavior script: {script_path}.")
        for prim in robot_list:
            omni.kit.commands.execute("ApplyScriptingAPICommand", paths=[Sdf.Path(prim.GetPrimPath())])
            attr = prim.GetAttribute("omni:scripting:scripts")
            # Get the corresponding robot script
            attr.Set([f"{script_path}"])
            dp.print(f"Set up python script for robot, prim = {prim.GetPrimPath()}.")

    def refresh_randomizers(self):
        """
        Refresh randomizers with global seed.
        """
        prop = self.config_file.get_property("global", "seed")
        if prop.is_value_error():
            carb.log_error("Refresh randomizers fails due to invalid global seed.")
            return
        seed = prop.get_resolved_value()
        self._character_randomizer.update_seed(seed)
        self._camera_randomizer.update_seed(seed)
        self._lidar_camera_randomizer.update_seed(seed)
        self._nova_carter_randomizer.update_seed(seed)
        self._iw_hub_randomizer.update_seed(seed)

    # ========= Config File =========

    def load_config_file(self, file_path):
        """
        Load config file object by input file path.
        """
        try:
            self.config_file = GlobalValues.config_file_format.load_config_file(file_path)
        except ConfigFileError as exc:
            carb.log_error(f"Failed to load config file '{file_path}': {exc}")
            self.config_file = None
            return False
        except Exception as exc:
            carb.log_error(f"Unexpected error while loading config file '{file_path}': {exc}")
            self.config_file = None
            return False
        if not self.config_file:
            carb.log_error(f"Config file cannot be loaded from: {file_path}.")
            return False
        self._on_config_file_loaded()
        return True

    def _on_config_file_loaded(self):
        # Register property listeners
        self.register_property_listeners()
        # Self refresh
        self.refresh_randomizers()
        self.read_character_asset_list()
        self.refresh_available_character_asset_list()
        # Envset scenario binding
        self._refresh_envset_catalog()
        self._select_env_scene_for_config()
        # Response refresh
        AgentResponseManager.get_instance().reset()

    def save_config_file(self):
        if not self.config_file:
            carb.log_error("Unable to save config file due to no loaded config file.")
            return False
        if not self.config_file.save():
            return False
        self._on_config_file_loaded()
        return True

    def save_as_config_file(self, folder_path, commands_list, robot_commands_list):
        if not self.config_file:
            carb.log_error("Unable to save as config file due to no config file is loaded.")
            return False
        # New file paths
        folder_path = Path(folder_path)
        new_config_file_path = str(folder_path / "config.yaml")
        new_cmd_file_path = str(folder_path / "command.txt")
        new_robot_cmd_file_path = str(folder_path / "robot_command.txt")
        # Create and point to new command files
        cmd_prop = self.get_config_file_property("character", "command_file")
        if cmd_prop:
            CommandFileUtil.save_command_file(new_config_file_path, new_cmd_file_path, commands_list)
            cmd_prop.set_value(new_cmd_file_path)
        robot_cmd_prop = self.get_config_file_property("robot", "command_file")
        if robot_cmd_prop:
            CommandFileUtil.save_command_file(new_config_file_path, new_robot_cmd_file_path, robot_commands_list)
            robot_cmd_prop.set_value(new_robot_cmd_file_path)
        # Save this current config file
        if not self.config_file.save_as(new_config_file_path):
            return False
        # Load the new config file
        return self.load_config_file(new_config_file_path)


    def get_config_file(self):
        return self.config_file

    def clear_config_file(self):
        self.config_file = None

    def register_property_listeners(self):
        """
        Register listeners for config file properties to update internal states accordingly.
        """

        def on_global_seed_update(new_val):
            self.refresh_randomizers()
            self.refresh_available_character_asset_list()

        def on_character_asset_update(new_val):
            self._character_randomizer.reset()
            self.read_character_asset_list()
            self.refresh_available_character_asset_list()

        def on_character_filter_update(new_val):
            self._character_randomizer.reset()
            self.refresh_available_character_asset_list()

        def try_register_prop_update(section_name, prop_name, func):
            prop = self.config_file.get_property(section_name, prop_name)
            if prop:
                prop.register_update_func(func)

        def on_character_command_file_update(new_val):
            self.setup_anim_people_command_from_config_file()

        def on_robot_command_file_update(new_val):
            self.setup_anim_people_robot_command_from_config_file()

        # Global section
        try_register_prop_update("global", "seed", on_global_seed_update)
        # Character section
        try_register_prop_update("character", "asset_path", on_character_asset_update)
        try_register_prop_update("character", "filters", on_character_filter_update)
        try_register_prop_update("character", "command_file", on_character_command_file_update)
        # Robot section
        try_register_prop_update("robot", "command_file", on_robot_command_file_update)

    def get_config_file_property(self, section_name, property_name):
        """
        Get Property from loaded config file.
        Return None if config file is not loaded.
        """
        if not self.config_file:
            return None
        return self.config_file.get_property(section_name, property_name)

    def get_config_file_property_group(self, section_name, property_group_name):
        """
        Get PropertyGroup from loaded config file.
        Return None if config file is not loaded.
        """
        if not self.config_file:
            return None
        return self.config_file.get_property_group(section_name, property_group_name)

    def get_config_file_valid_value(self, section_name, property_name):
        """
        Get a valid Property value from loaded config file.
        Return None if Property cannot be fetched or its value is error.
        """
        prop = self.get_config_file_property(section_name, property_name)
        if not prop or prop.is_value_error() or not prop.is_setup():
            return None
        return prop.get_resolved_value()

    def get_config_file_section(self, section_name):
        if not self.config_file:
            return None
        return self.config_file.get_section(section_name)

    # ========= Characters/Robots Commands =========

    def load_commands(self):
        command_file_path = self.get_config_file_valid_value("character", "command_file")
        if command_file_path:
            return CommandFileUtil.load_command_file(self.config_file.file_path, command_file_path)
        else:
            return []

    def load_robot_commands(self):
        command_file_path = self.get_config_file_valid_value("robot", "command_file")
        if command_file_path:
            return CommandFileUtil.load_command_file(self.config_file.file_path, command_file_path)
        else:
            return []

    def save_commands(self, commands_list):
        command_file_path = self.get_config_file_valid_value("character", "command_file")
        if command_file_path:
            return CommandFileUtil.save_command_file(self.config_file.file_path, command_file_path, commands_list)
        else:
            carb.log_warn(
                "Unable to save character commands due to not getting character command file from config file."
            )
            return False

    def save_robot_commands(self, robot_commands_list):
        command_file_path = self.get_config_file_valid_value("robot", "command_file")
        if command_file_path:
            return CommandFileUtil.save_command_file(self.config_file.file_path, command_file_path, robot_commands_list)
        else:
            carb.log_warn("Unable to save robot commands due to not getting robot command file from config file.")
            return False

    async def generate_random_commands(self):
        """
        Generate random character commands by the current config file.
        """
        global_seed = self.get_config_file_valid_value("global", "seed")
        duration: float = self.get_config_file_valid_value("global", "simulation_length") / FRAME_RATE
        agent_count = self.get_config_file_valid_value("character", "num")
        navigation_area = self.get_config_file_valid_value("character", "navigation_area")
        if not global_seed:
            carb.log_error("Unable to generate random commands due to invalid seed in config file.")
            return []
        if not duration:
            carb.log_error("Unable to generate random commands due to invalid duration in config file.")
            return []
        if agent_count is None:
            carb.log_error("Unable to generate random commands due to invalid character number in config file.")
            return []
        else:
            # 直接调用async方法，避免不必要的task包装
            return await self._character_randomizer.generate_character_commands(
                global_seed, duration, agent_count, navigation_area
            )

    # @dp.debug_func
    async def generate_random_robot_commands(self):
        """
        Generate random robot commands by the current config file.
        """
        seed = self.get_config_file_valid_value("global", "seed")
        duration: float = self.get_config_file_valid_value("global", "simulation_length") / FRAME_RATE
        nova_carter_count = self.get_config_file_valid_value("robot", "nova_carter_num")
        iw_hub_count = self.get_config_file_valid_value("robot", "iw_hub_num")
        navigation_area = self.get_config_file_valid_value("robot", "navigation_area")
        if not seed:
            carb.log_error("Unable to generate robot commands due to invalid seed in config file.")
            return []
        if not duration:
            carb.log_error("Unable to generate robot commands due to invalid simulation length in config file.")
            return []
        commands = []
        if nova_carter_count is not None:
            # 直接调用async方法，避免不必要的task包装
            nova_commands = await self._nova_carter_randomizer.generate_robot_commands(
                seed, duration, "Nova_Carter", nova_carter_count, navigation_area
            )
            commands += nova_commands
        else:
            carb.log_error("Unable to generate nova careter commands due to invalid nova carter number.")
        if iw_hub_count is not None:
            # 直接调用async方法，避免不必要的task包装
            iw_commands = await self._iw_hub_randomizer.generate_robot_commands(
                seed, duration, "iw_hub", iw_hub_count, navigation_area
            )
            commands += iw_commands
        else:
            carb.log_error("Unable to generate iw.hub commands due to invalid iw.hub number.")
        # dp.print(f"Commands = {commands}")
        return commands

    # ========= Data Generation =========

    def register_data_generation_callback(self, on_event: callable):
        return carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=SimulationManager.DATA_GENERATION_DONE_EVENT, on_event=on_event,
            observer_name="isaacsim/replicator/agent/core/simulation/ON_DATA_GENERATION_DONE"
        )
    
    async def run_data_generation_async(self, will_wait_until_complete):
        if not self.config_file:
            carb.log_error("Config file is not loaded. Start data generation fails.")
            return

        sim_length = self.get_config_file_valid_value("global", "seed")
        if not sim_length:
            carb.log_error("Simulation Length is invalid. Start data generation fails.")
            return

        self._dg = DataGeneration(self.config_file)
        self._dg.register_recorder_done_callback(self._data_generation_done_callback)

        writer_selection_group = self.config_file.get_property_group( "replicator", "writer_selection")
        output_dir = writer_selection_group.content_prop.get_resolved_value()["output_dir"]

        # use the empty output dir to stand for the none output writer.
        if output_dir :
            self._incident_bridge.start_recording(output_dir)

        await self._dg.run_async(will_wait_until_complete)

    def _data_generation_done_callback(self):
        """
        Release handle when data generation is finished.
        """
        # Clean up reference
        self._dg = None
        self._dg_task = None
        # Incident report
        # Check whether recording is activated:
        if self._incident_bridge.is_recording():
            self._incident_bridge.end_recording()
        # Mark complete
        carb.eventdispatcher.get_eventdispatcher().dispatch_event(
            event_name=SimulationManager.DATA_GENERATION_DONE_EVENT,
            payload={}
        )
        carb.log_info("One data generation completes.")

    # ===== Utilities =====

    async def _wait_for_navmesh_if_needed(self):
        """Wait for NavMesh baking to complete if needed for Matterport scenes."""
        if not self._is_matterport_scene():
            return

        if self._navmesh_status == "ready":
            carb.log_info("NavMesh is already ready, proceeding with data generation.")
            return
        elif self._navmesh_status == "failed":
            carb.log_warn("NavMesh baking failed, proceeding with data generation anyway.")
            return
        elif self._navmesh_status == "baking":
            carb.log_info("NavMesh is baking, waiting for completion...")
            if self._navmesh_baking_complete:
                try:
                    await self._navmesh_baking_complete.wait()
                    carb.log_info("NavMesh baking completed, proceeding with data generation.")
                except Exception as e:
                    carb.log_warn(f"NavMesh baking wait interrupted: {e}, proceeding anyway.")
            else:
                carb.log_warn("NavMesh baking event not set, proceeding with data generation.")
        elif self._navmesh_status == "idle":
            carb.log_info("NavMesh status is idle, proceeding with data generation.")

    def _is_matterport_scene(self):
        """Check if the current scene contains Matterport assets."""
        try:
            scene_section = self.get_config_file_section("scene")
            if scene_section and hasattr(scene_section, "use_matterport") and scene_section.use_matterport:
                if not scene_section.use_matterport.is_value_error():
                    if bool(scene_section.use_matterport.get_resolved_value()):
                        return True

            # Also check if there are any Matterport prims in the stage
            stage = omni.usd.get_context().get_stage()
            if stage:
                for prim in stage.Traverse():
                    if "matterport" in str(prim.GetPath()).lower():
                        return True
            return False
        except Exception:
            return False

    def get_navmesh_status(self):
        """Get current NavMesh baking status."""
        return self._navmesh_status

    # ========= Set Up Simulation by Config File =========

    def set_up_simulation_from_config_file(self):
        self.load_scene_from_config_file()

    def register_set_up_simulation_done_callback(self, on_event: callable):
        return carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=SimulationManager.SET_UP_SIMULATION_DONE_EVENT, on_event=on_event,
            observer_name="isaacsim/replicator/agent/core/simulation/ON_SET_UP_SIMULATION_DONE"
        )

    def load_scene_from_config_file(self):
        """
        Load scene by config file and triggers load assets when scene is loaded.
        """
        ctx = omni.usd.get_context()
        scene_section = self.get_config_file_section("scene")
        if not scene_section:
            carb.log_error("Scene section is missing in config file. Scene loading aborted.")
            return

        def _resolved(prop):
            if not prop:
                return None
            if hasattr(prop, "is_value_error") and prop.is_value_error():
                return None
            if hasattr(prop, "is_setup") and not prop.is_setup():
                return None
            if hasattr(prop, "get_resolved_value"):
                return prop.get_resolved_value()
            return None

        scene_cfg_env = self._get_env_scene()
        scene_category = self._get_env_scene_category()
        scene_category_lc = scene_category.lower() if scene_category else ""

        use_matterport_cfg = self.get_config_file_valid_value("scene", "use_matterport")
        use_matterport_cfg = bool(use_matterport_cfg)
        use_matterport_env = scene_cfg_env.get("use_matterport")
        if use_matterport_env is not None:
            use_matterport_env = bool(use_matterport_env)

        if scene_category_lc == "mp3d":
            use_matterport = True
        elif scene_category_lc in ("grscenes", "builtin"):
            use_matterport = False
        elif use_matterport_env is not None:
            use_matterport = use_matterport_env
        else:
            use_matterport = use_matterport_cfg

        if use_matterport != use_matterport_cfg:
            resolved_category = scene_category or "(unspecified)"
            carb.log_info(
                f"[EnvSet] scene.use_matterport overridden to {use_matterport} based on envset category '{resolved_category}'."
            )

        if not use_matterport:
            env_scene_path = self._get_env_scene_usd_path()
            env_scene_type = self._get_env_scene_type()
            config_scene_path = self.get_config_file_valid_value("scene", "asset_path")

            scene_path = None
            if env_scene_path and (not env_scene_type or env_scene_type == "usd_stage"):
                scene_path = env_scene_path
                log_msg = f"[EnvSet] Using scene path from envset: {scene_path}"
                if config_scene_path and config_scene_path != scene_path:
                    log_msg += f" (overriding config path {config_scene_path})"
                carb.log_info(log_msg)
            elif config_scene_path:
                scene_path = config_scene_path

            if not scene_path:
                carb.log_error("Unable to load scene due to missing scene path in config file.")
                return
            stage_url = ctx.get_stage_url()
            if scene_path.lower().endswith((".usd", ".usda", ".usdc")):
                if scene_path != stage_url:
                    self.refresh_randomizers()
                    self.refresh_available_character_asset_list()

                    def load_scene_from_config_file_callback(event):
                        self._load_stage_handle = None
                        self.load_assets_to_scene()

                    self._load_stage_handle = carb.eventdispatcher.get_eventdispatcher().observe_event(
                        event_name=ctx.stage_event_name(omni.usd.StageEventType.ASSETS_LOADED),
                        on_event=load_scene_from_config_file_callback,
                        observer_name="isaacsim/replicator/agent/core/simulation/ON_ASSETS_LOADED",
                    )
                    try:
                        carb.log_info(f"To load scene from config file: {scene_path}")
                        StageUtil.open_stage(scene_path)
                    except Exception:
                        carb.log_error(f"Load scene ({scene_path}) fails. No assets will be loaded.")
                        self._load_stage_handle = None
                else:
                    carb.log_info("The current scene matches the scene path in config file. Scene loading is skipped.")
                    self._character_randomizer.reset()
                    self._nova_carter_randomizer.reset()
                    self._iw_hub_randomizer.reset()
                    self.load_assets_to_scene()
            else:
                carb.log_error(
                    "Scene asset path must point to a USD file when use_matterport=False. Please update the configuration."
                )
            return

        matterport_section = getattr(scene_section, "matterport", None)
        matterport_cfg_env = scene_cfg_env.get("matterport") or {}

        usd_path = _resolved(getattr(matterport_section, "usd_path", None))
        if not usd_path:
            usd_path = matterport_cfg_env.get("usd_path")
        obj_path = _resolved(getattr(matterport_section, "obj_path", None))
        if not obj_path:
            obj_path = matterport_cfg_env.get("obj_path")
        root_prim_path = _resolved(getattr(matterport_section, "root_prim_path", None))
        if not root_prim_path:
            root_prim_path = matterport_cfg_env.get("root_prim_path")
        root_prim_path = root_prim_path or "/World/terrain/Matterport"
        import_path = usd_path or obj_path
        if not import_path:
            carb.log_error("Matterport import requested but no USD or OBJ path provided.")
            return

        # Determine the container prim path in a way compatible with Isaac Sim 5.0 (avoid Sdf.Path.GetName).
        # If the provided root path ends with '/Matterport', use its parent as container; otherwise use the path itself.
        container_prim_path = root_prim_path or "/World"
        if not container_prim_path.startswith("/"):
            container_prim_path = "/" + container_prim_path
        if container_prim_path.endswith("/Matterport"):
            parent = container_prim_path.rsplit("/", 1)[0]
            container_prim_path = parent if parent and parent != "" else "/World"
        if container_prim_path == "/":
            container_prim_path = "/World"

        self.refresh_randomizers()
        self.refresh_available_character_asset_list()
        ctx.new_stage()

        if self._matterport_import_future:
            self._matterport_import_future.cancel()
            self._matterport_import_future = None

        resolve_root = os.path.dirname(self.config_file.file_path) if self.config_file else None

        # 使用新的事件驱动方式处理Matterport
        self._matterport_import_future = asyncio.ensure_future(
            self._prepare_matterport_scene_complete(container_prim_path, import_path, resolve_root)
        )

    async def _prepare_matterport_scene_complete(self, container_prim_path, import_path, resolve_root):
        """
        完整准备Matterport场景，包括导入、碰撞、地面、NavMesh
        完成后触发ASSETS_LOADED事件，复用现有的资产加载流程
        """
        # Check if Matterport is available
        if not MATTERPORT_AVAILABLE:
            carb.log_error("Cannot import Matterport scene: omni.isaac.matterport extension is not available in this Isaac Sim version.")
            return

        try:
            # Step 1: Import Matterport asset (honor ground plane config directly)
            gp_enabled = True
            try:
                scene_section = self.get_config_file_section("scene")
                if scene_section and hasattr(scene_section, "matterport") and scene_section.matterport:
                    gp_prop = getattr(scene_section.matterport, "add_ground_plane", None)
                    if gp_prop and not gp_prop.is_value_error():
                        gp_enabled = bool(gp_prop.get_resolved_value())
                else:
                    env_scene_cfg = self._get_env_scene()
                    mp_cfg_env = env_scene_cfg.get("matterport") if env_scene_cfg else {}
                    if mp_cfg_env and "add_ground_plane" in mp_cfg_env:
                        gp_enabled = bool(mp_cfg_env.get("add_ground_plane"))
            except Exception:
                gp_enabled = True

            matterport_prim = await import_matterport_asset(
                prim_path=container_prim_path,
                input_path=import_path,
                groundplane=gp_enabled,
                manage_simulation=True,
                resolve_relative_to=resolve_root,
            )
            
            # Step 1: import done
            carb.log_info(f"[MP] Import completed at: {matterport_prim}")

            # Collision is already applied by importer; skipping redundant step

            # Ground plane handled by importer when configured; skipping redundant step

            # Step 4: bake navmesh around the actual imported prim path
            carb.log_info(f"[MP] Baking NavMesh around: {matterport_prim}")
            # rely on ensure_navmesh_async's internal frame syncs

            # Set up NavMesh baking status tracking
            self._navmesh_status = "baking"
            self._navmesh_baking_complete = asyncio.Event()

            def on_navmesh_status_change(status):
                """Callback for NavMesh status updates."""
                self._navmesh_status = status
                if status in ["ready", "failed"]:
                    if self._navmesh_baking_complete:
                        self._navmesh_baking_complete.set()

            navmesh_params = self._get_navmesh_runtime_params()
            bake_root_path = navmesh_params["bake_root"] or matterport_prim
            z_padding_cfg = navmesh_params["z_padding"]
            z_padding_value = float(z_padding_cfg) if z_padding_cfg is not None else 2.0
            navmesh = await ensure_navmesh_async(
                bake_root_path,
                z_padding=z_padding_value,
                status_callback=on_navmesh_status_change,
                include_volume_parent=navmesh_params["include_parent"],
                min_xy=navmesh_params["min_xy"],
                min_z=navmesh_params["min_z"],
                agent_radius=navmesh_params["agent_radius"],
            )
            if navmesh is None:
                carb.log_error("[MP] NavMesh baking failed after Matterport import.")
                self._navmesh_status = "failed"
                return

            self._navmesh_ready = True
            self._navmesh_status = "ready"
            carb.log_info("[MP] NavMesh ready: True")

            # Import complete; proceed to loading workflow

            carb.log_info("[MP] Matterport scene preparation complete, triggering asset loading")

            # 关键：触发标准的ASSETS_LOADED事件，让现有流程接管
            self._trigger_assets_loaded_event()
        except Exception as exc:
            # 捕获所有异常，包括 NavMesh baking 的异常
            carb.log_error(f"Matterport scene preparation failed: {exc}")
            self._navmesh_status = "failed"

        finally:
            self._matterport_import_future = None

    def _trigger_assets_loaded_event(self):
        """手动触发ASSETS_LOADED事件，复用现有的事件驱动机制"""
        ctx = omni.usd.get_context()

        def assets_loaded_callback(event):
            self._load_stage_handle = None
            self.load_assets_to_scene()

        self._load_stage_handle = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=ctx.stage_event_name(omni.usd.StageEventType.ASSETS_LOADED),
            on_event=assets_loaded_callback,
            observer_name="isaacsim/replicator/agent/core/simulation/ON_MATTERPORT_ASSETS_LOADED",
        )

        # 手动分发事件
        carb.eventdispatcher.get_eventdispatcher().dispatch_event(
            event_name=ctx.stage_event_name(omni.usd.StageEventType.ASSETS_LOADED),
            payload={}
        )

    def load_assets_to_scene(self):
        """
        Trigger navmesh baking and load characters, robots and cameras.
        """
        # 物理场景路径由会话设置控制，这里不再进行事后移动

        try:
            stage = omni.usd.get_context().get_stage()
        except Exception:
            stage = None
        try:
            UnitScaleService.configure(stage)
            UnitScaleService.apply_people_unit_scaling()
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"[UnitScale] Failed to initialize metersPerUnit from stage: {exc}")

        async def _load_assets():
            """实际加载资产的函数（先设置命令，再挂脚本），确保第一次即可读取到指令文件"""
            await self._robot_control_mgr.stop_all()
            await self._spawn_envset_robots()
            # Robot: 先设置命令文件，再加载并挂脚本
            if self.get_config_file_section("robot"):
                self.setup_anim_people_robot_command_from_config_file()
                # 等待一帧确保设置可见
                try:
                    await omni.kit.app.get_app().next_update_async()
                except Exception:
                    pass
                self.load_robot_from_config_file()
            else:
                carb.log_info("No robot section in the config file. Skip robot setup.")
            await self._robot_control_mgr.start_controls()

            # Character: 先设置命令文件，再加载并挂脚本/动画图
            if self.get_config_file_section("character"):
                self.setup_anim_people_command_from_config_file()
                env_route_commands = self._get_env_character_routes_commands()
                if env_route_commands:
                    if self.save_commands(env_route_commands):
                        carb.log_info(f"[EnvSet] Overrode character command file with routes: {env_route_commands}")
                    else:
                        carb.log_warn("[EnvSet] Failed to save character routes to command file.")
                # 等待一帧确保设置可见
                try:
                    await omni.kit.app.get_app().next_update_async()
                except Exception:
                    pass
                self.load_characters_from_config_file()
                self.setup_all_characters()
                # Enable external arrival guard only for GRScenes/cm-scale scenes
                try:
                    self._maybe_enable_arrival_guard()
                except Exception:
                    pass
            else:
                carb.log_info("No character section in the config file. Skip character setup.")

            if self.get_config_file_section("event"):
                self.setup_incidents_from_config_file()
            else:
                carb.log_info("No incident section in the config file. Skip incident setup.")

            response_section = self.get_config_file_section("response")
            if response_section:
                AgentResponseManager.get_instance().reset()
                AgentResponseManager.get_instance().setup_responses_from_config_file(response_section)
            else:
                carb.log_info("No response section in the config file. Skip agent response setup.")

            self.load_camera_from_config_file()
            self.load_lidar_from_config_file()
            carb.eventdispatcher.get_eventdispatcher().dispatch_event(
                event_name=SimulationManager.SET_UP_SIMULATION_DONE_EVENT,
                payload={}
            )

        async def try_bake_navmesh():
            """异步确保存在NavMesh（若需则创建体素并烘焙），完成后加载资产。"""
            _inav = nav.acquire_interface()

            # 若Matterport导入阶段已准备好NavMesh且仍可获取，则直接加载资产
            if self._navmesh_ready:
                navmesh0 = _inav.get_navmesh()
                if navmesh0 is not None:
                    carb.log_info("[MP] Using NavMesh from Matterport import")
                    self._navmesh_ready = False
                    await _load_assets()
                    return
                # 否则重置标志，继续走常规异步保证流程
                self._navmesh_ready = False

            # 使用通用的异步保证函数：自动创建IncludeVolume并烘焙（必要时放大重试）
            try:
                stage = omni.usd.get_context().get_stage()
                if stage is None:
                    raise RuntimeError("USD stage is not available before NavMesh baking.")
                world_prim = stage.GetPrimAtPath("/World")
                if not world_prim or not world_prim.IsValid():
                    omni.kit.commands.execute(
                        "CreatePrimCommand",
                        prim_type="Xform",
                        prim_path="/World",
                        select_new_prim=False,
                    )
                    world_prim = stage.GetPrimAtPath("/World")
                    if not world_prim or not world_prim.IsValid():
                        raise RuntimeError("Failed to create /World prim required for NavMesh baking.")
                def _ensure_runtime_node(path: str):
                    if not stage.GetPrimAtPath(path).IsValid():
                        stage.DefinePrim(path, "Xform")

                runtime_nodes = [
                    PrimPaths.characters_parent_path(),
                    PrimPaths.robots_parent_path(),
                    PrimPaths.cameras_parent_path(),
                    PrimPaths.lidar_cameras_parent_path(),
                ]
                for node in runtime_nodes:
                    _ensure_runtime_node(node)

                # 选择 NavMesh 的根：优先使用 scene.navmesh_root_prim_path（GRScenes），否则退回 /World
                navmesh_root_path = "/World"
                cfg_nav_root = None
                try:
                    cfg_nav_root = self.get_config_file_valid_value("scene", "navmesh_root_prim_path")
                except Exception as exc:
                    cfg_nav_root = None
                    carb.log_warn(f"[NavMesh] Failed to read navmesh_root_prim_path: {exc}")
                if cfg_nav_root:
                    navmesh_root_path = str(cfg_nav_root)

                if navmesh_root_path.startswith("/Root"):
                    carb.log_warn(
                        f"[NavMesh] Using navmesh_root_prim_path='{navmesh_root_path}'. "
                        "Prefer /World to avoid multiple physics scenes."
                    )
                    root_prim = stage.GetPrimAtPath(navmesh_root_path)
                    if not root_prim or not root_prim.IsValid():
                        raise RuntimeError(f"NavMesh root '{navmesh_root_path}' does not exist on the stage.")
                    # 注：仅在 legacy 场景仍依赖 /Root 时才会触发以下缩放传播逻辑。
                    try:
                        import carb.settings as _cs

                        _propagate = bool(
                            _cs.get_settings().get(
                                "/exts/isaacsim.replicator.agent/navmesh/propagate_root_scale"
                            )
                        )
                    except Exception as exc:
                        carb.log_warn(f"[NavMesh] Failed to read root-scale propagation flag: {exc}")
                        _propagate = False
                    if _propagate:
                        root_xform = UsdGeom.Xformable(root_prim)
                        root_scale = None
                        for op in root_xform.GetOrderedXformOps():
                            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                                value = op.Get()
                                if isinstance(value, (tuple, list)):
                                    root_scale = value
                                else:
                                    root_scale = (value[0], value[1], value[2])
                                break
                        if root_scale is not None:
                            for node in runtime_nodes:
                                prim = stage.GetPrimAtPath(node)
                                if not prim or not prim.IsValid():
                                    continue
                                xform = UsdGeom.Xformable(prim)
                                scale_op = None
                                for op in xform.GetOrderedXformOps():
                                    if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                                        scale_op = op
                                        break
                                if scale_op is None:
                                    scale_op = xform.AddScaleOp()
                                scale_op.Set(root_scale)
                # 设置“角色生成是否需要做归一化”的开关：仅 GRScenes（提供了 navmesh_root_prim_path 且未使用 MP）启用
                try:
                    import carb.settings as _cs
                    use_mp = bool(self.get_config_file_valid_value("scene", "use_matterport"))
                except Exception:
                    use_mp = False
                try:
                    import carb.settings as _cs
                    character_opts = self._get_env_character_options()
                    normalize_override = character_opts.get("normalize_on_spawn")
                    if normalize_override is None:
                        normalize_override = character_opts.get("normalize_characters")
                    if normalize_override is None:
                        normalize_flag = bool(cfg_nav_root) and (not use_mp)
                    else:
                        normalize_flag = bool(normalize_override)
                    _cs.get_settings().set(
                        "/exts/isaacsim.replicator.agent/character/normalize_on_spawn",
                        normalize_flag,
                    )
                except Exception:
                    pass

                navmesh_params = self._get_navmesh_runtime_params()
                navmesh_root_override = navmesh_params["navmesh_root_override"]
                if navmesh_root_override:
                    navmesh_root_path = navmesh_root_override
                bake_root_path = navmesh_params["bake_root"] or navmesh_root_path
                z_padding_cfg = navmesh_params["z_padding"]
                z_padding_value = float(z_padding_cfg) if z_padding_cfg is not None else 2.0
                navmesh = await ensure_navmesh_async(
                    bake_root_path,
                    z_padding=z_padding_value,
                    include_volume_parent=navmesh_params["include_parent"],
                    min_xy=navmesh_params["min_xy"],
                    min_z=navmesh_params["min_z"],
                    agent_radius=navmesh_params["agent_radius"],
                )
            except Exception as _exc:
                navmesh = None
                carb.log_error(f"NavMesh ensure failed: {_exc}")

            if navmesh is None:
                carb.log_error(
                    "NavMesh building failed. Please ensure a valid NavMeshIncludeVolume encloses walkable geometry."
                )
                return

            carb.log_info("NavMesh ready. Proceeding to load assets.")
            await _load_assets()

        # 如果是Matterport场景且NavMesh已经ready，直接加载资产，避免重复baking
        if self._is_matterport_scene() and self._navmesh_ready:
            carb.log_info("[MP] NavMesh already ready from Matterport import, loading assets directly.")
            self._navmesh_ready = False  # 关键：使用后立即重置标志
            asyncio.ensure_future(_load_assets())
            return

        # 启动异步navmesh烘焙任务
        asyncio.ensure_future(try_bake_navmesh())

    @dp.debug_func
    def load_robot_by_type(self, robot_type, randomizer):
        property_name = robot_type.lower() + "_num"
        # Skipping the rest of the code when there is no loading happening
        # Early out if requested robots already exist
        robot_count = self.get_config_file_valid_value("robot", property_name)
        if robot_count is None:
            carb.log_error(f"Unable to load {robot_type} due to invalid robot count.")
            return
        robot_count_in_stage = len(RobotUtil.get_robots_in_stage(robot_type_name=robot_type))
        if robot_count <= robot_count_in_stage:
            return
        # Set up agent randomizer
        all_agents_pos = AgentUtil.get_all_agents_positions()
        randomizer.update_agent_positions(all_agents_pos)
        # Make sure all required robots exist
        stage = omni.usd.get_context().get_stage()
        parent_path = PrimPaths.robots_parent_path()
        spawn_area = self.get_config_file_valid_value("robot", "spawn_area")
        robot_pos_list = [randomizer.get_random_position(i, spawn_area) for i in range(robot_count)]
        for i in range(robot_count):
            robot_name = RobotUtil.get_robot_name_by_index(robot_type, i)
            robot_path = parent_path + "/" + robot_name
            robot_prim = stage.GetPrimAtPath(robot_path)
            if not robot_prim.IsValid():
                robot_prim = RobotUtil.spawn_robot(robot_type, robot_pos_list[i], 0, robot_path)
            omni.kit.commands.execute("ApplyNavMeshAPICommand", prim_path=robot_path, api=NavSchema.NavMeshExcludeAPI)
            dp.print(f"Robot is spawned, type = {robot_type}, prim = {robot_path}")

    def load_robot_from_config_file(self):
        """
        Load to enough robots by config file. Return if no load is needed.
        """
        seed = self.get_config_file_valid_value("global", "seed")
        if not seed:
            carb.log_error("Unable to load robots due to invalid seed in config file.")
            return
        self.load_robot_by_type("Nova_Carter", self._nova_carter_randomizer)
        self.load_robot_by_type("iw_hub", self._iw_hub_randomizer)
        self.setup_robot_by_type("Nova_Carter")
        self.setup_robot_by_type("iw_hub")

    def setup_robot_by_type(self, robot_type):
        """
        Set up all robots in stage (python script, semantic)
        """
        robot_prims_list = RobotUtil.get_robots_in_stage(count=-1, robot_type_name=robot_type)
        self.setup_python_scripts_to_robot(robot_prims_list, robot_type)
        SemanticsUtils.add_update_prim_metrosim_semantics(robot_prims_list, type_value="class", name=robot_type.lower())

    def load_camera_from_config_file(self):
        """
        Load to enough cameras by config file.
        Loaded camera will aim to one of the character if it is present.
        Return if no load is needed.
        """
        camera_group = self.get_config_file_property_group("sensor", "camera_group")
        if not camera_group:
            return
        cam_prop = camera_group.get_property("camera_num")
        if not cam_prop:
            return
        cam_count = cam_prop.get_value() if cam_prop.is_setup() else None
        if cam_count is None:
            # Show warning here since camera number can be optional
            carb.log_warn("Unable to load cameras due to no camera number in config file.")
            return
        if cam_count == 0:
            return
        seed = self.get_config_file_valid_value("global", "seed")
        if seed is None:
            carb.log_error("Unable to load cameras due to invalid seed in config file.")
            return
        # Make sure camera root prim exist
        stage = omni.usd.get_context().get_stage()
        parent_path = PrimPaths.cameras_parent_path()
        if not stage.GetPrimAtPath(parent_path).IsValid():
            omni.kit.commands.execute(
                "CreatePrimCommand", prim_type="Xform", prim_path=parent_path, select_new_prim=False
            )
        # Get all required camera infos from randomizer
        character_prim_list = CharacterUtil.get_characters_root_in_stage(count_invisible=False)
        character_list = [prim.GetPath() for prim in character_prim_list]
        all_camera_transforms = self._camera_randomizer.get_random_position_rotation(cam_count, character_list)
        all_focallengths = self._camera_randomizer.get_random_camera_focallength_list(cam_count)
        # Make sure all required cameras exist
        for i in range(cam_count):
            cam_name = CameraUtil.get_camera_name_by_index(i)
            cam_path = f"{parent_path}/{cam_name}"
            if not stage.GetPrimAtPath(cam_path).IsValid():
                cam_prim = CameraUtil.spawn_camera(cam_path)
                pos, rot = all_camera_transforms[i]
                CameraUtil.set_camera(cam_prim, pos, rot, all_focallengths[i])

    def load_lidar_from_config_file(self):
        """
        Load to enough lidar cameras by config file.
        Lidar camera transformation will follow the matching camera.
        Return if no load is needed.
        """
        # Temporary disable lidar generation
        return

    def setup_anim_people_command_from_config_file(self):
        """
        Link character command file to omni.anim.people.
        """
        command_file_path = self.get_config_file_valid_value("character", "command_file")
        if command_file_path:
            target_path = FileUtil.get_absolute_path(self.config_file.file_path, command_file_path)
            carb.settings.get_settings().set(OMNI_ANIM_PEOPLE_COMMAND_PATH, target_path)
            carb.log_info(f"Character command file is set to: {command_file_path}.")
        else:
            carb.log_error(f"Unable to set up character command file: {command_file_path}.")

    def setup_incidents_from_config_file(self):
        self._incident_bridge.setup_incident_from_config(self.config_file)

    def setup_anim_people_robot_command_from_config_file(self):
        """
        Link the robot command file to IAR.
        """
        command_file_path = self.get_config_file_valid_value("robot", "command_file")
        if command_file_path:
            target_path = FileUtil.get_absolute_path(self.config_file.file_path, command_file_path)
            carb.settings.get_settings().set(ANIM_ROBOT_COMMAND_PATH, target_path)
            carb.log_info(f"Robot command file is set to: {command_file_path}.")
        else:
            carb.log_error(f"Unable to set up robot command file: {command_file_path}.")

    def load_characters_from_config_file(self):
        """
        Load to enough characters by config file.
        Load default biped character when needed.
        """

        seed = self.get_config_file_valid_value("global", "seed")
        if not seed:
            carb.log_error("Unable to load character due to invalid seed in config file.")
            return
        character_opts = self._get_env_character_options()
        character_count = character_opts.get("count")
        if character_count is None:
            character_count = self.get_config_file_valid_value("character", "num")
        if character_count is not None:
            try:
                character_count = int(character_count)
            except Exception:
                carb.log_error("Character count is not a valid integer.")
                character_count = None
        if character_count is None:
            carb.log_error("Unable to load character due to invalid character count in config file.")
            return
        # Make sure skeleton and animation loaded
        # Set up agent randomizer
        agents_pos = AgentUtil.get_all_agents_positions()
        self._character_randomizer.update_agent_positions(agents_pos)
        # Make sure all required characters exist
        stage = omni.usd.get_context().get_stage()
        parent_path = PrimPaths.characters_parent_path()
        spawn_area = character_opts.get("spawn_area")
        if spawn_area is None:
            spawn_area = self.get_config_file_valid_value("character", "spawn_area")
        spawn_specs = self._get_env_character_spawn_specs()
        character_pos_list = []
        character_rot_list = []
        for i in range(character_count):
            spec = spawn_specs[i] if i < len(spawn_specs) else None
            custom_pos = None
            custom_rot = 0.0
            if spec and isinstance(spec, dict):
                pos_val = spec.get("position")
                if pos_val and len(pos_val) >= 3:
                    try:
                        custom_pos = carb.Float3(float(pos_val[0]), float(pos_val[1]), float(pos_val[2]))
                        custom_rot = float(spec.get("orientation_deg", 0.0))
                    except Exception as exc:  # noqa: BLE001
                        carb.log_warn(f"Invalid custom spawn position at index {i}: {exc}")
                        custom_pos = None
            if custom_pos is None:
                random_pos = self._character_randomizer.get_random_position(i, spawn_area)
                if random_pos is None:
                    carb.log_error(f"Unable to determine spawn position for character index {i}.")
                    continue
                character_pos_list.append(random_pos)
                character_rot_list.append(0.0)
            else:
                character_pos_list.append(custom_pos)
                character_rot_list.append(custom_rot)
        # Determine final character names and any explicit USD assets from envset
        names_list, assets_map = self._get_env_character_names_and_assets(character_count)

        for i in range(character_count):
            character_name = names_list[i] if i < len(names_list) else CharacterUtil.get_character_name_by_index(i)
            character_path = f"{parent_path}/{character_name}"
            character_prim = stage.GetPrimAtPath(character_path)
            if not character_prim.IsValid():
                if i >= len(character_pos_list):
                    carb.log_error(f"Missing spawn data for {character_name}.")
                    continue
                # Prefer explicit USD from envset assets map; fallback to folder-based spawn
                explicit_usd = assets_map.get(character_name)
                if explicit_usd:
                    try:
                        character_prim = CharacterUtil.load_character_usd_to_stage(
                            explicit_usd, character_pos_list[i], character_rot_list[i], character_name
                        )
                        if character_prim and character_prim.IsValid():
                            carb.log_info(
                                f"[EnvSet] Spawned character '{character_name}' from assets map: {explicit_usd}"
                            )
                        else:
                            carb.log_warn(
                                f"[EnvSet] Failed to spawn '{character_name}' from assets map; falling back to folder selection."
                            )
                            character_prim = self.spawn_character_by_idx(
                                character_pos_list[i], character_rot_list[i], i
                            )
                    except Exception as _exc:
                        carb.log_warn(
                            f"[EnvSet] Exception while spawning '{character_name}' from assets map: {_exc}; falling back."
                        )
                        character_prim = self.spawn_character_by_idx(character_pos_list[i], character_rot_list[i], i)
                else:
                    character_prim = self.spawn_character_by_idx(character_pos_list[i], character_rot_list[i], i)
                if not character_prim:
                    carb.log_error(f"Unable to spawn {character_name}.")
                carb.log_info(f"Character {character_name} is spawned at {character_pos_list[i]}.")
            else:
                carb.log_info(f"Character exists in stage: {character_path}, skip spawning.")
            omni.kit.commands.execute(
                "ApplyNavMeshAPICommand", prim_path=character_path, api=NavSchema.NavMeshExcludeAPI
            )

    def setup_all_characters(self):
        """
        Set up all characters in stage (anim graph, python script, semantic)
        """
        biped_prim = CharacterUtil.load_default_biped_to_stage()
        character_list = CharacterUtil.get_characters_in_stage()
        CharacterUtil.setup_animation_graph_to_character(
            character_list, CharacterUtil.get_anim_graph_from_character(biped_prim)
        )
        CharacterUtil.setup_python_scripts_to_character(character_list, BehaviorScriptPaths.behavior_script_path())
        SemanticsUtils.add_update_prim_metrosim_semantics(character_list, type_value="class", name="character")

    def get_character_randomizer(self) -> CharacterRandomizer:
        return self._character_randomizer
    
